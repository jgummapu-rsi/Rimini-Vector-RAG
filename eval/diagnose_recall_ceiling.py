"""Diagnostic: WHERE is doc-level recall@10 lost, and what is the ceiling?

Reranking (cross-encoder or LLM) can only REORDER the candidate pool retrieval
hands it -- it can never surface a document whose chunks were never fetched. So
recall@10 is bounded above by RETRIEVAL, not by the reranker. This script
measures that ceiling directly on the already-embedded bge-base corpus (no LLM,
no re-embedding -- just embeds the 300 queries and hits pgvector at varying
depths).

For each query it finds the gold document's best rank in each independent
channel (dense HNSW, lexical FTS), deduped to documents, then reports:

  oracle_recall@10 at pool depth N
    = fraction of queries whose gold doc appears ANYWHERE in
      (dense dedup top-N  UNION  lexical dedup top-N).
    This is the MAX recall@10 any reranker could reach if it fetched N per
    channel and ranked perfectly -- an oracle rerank puts a reachable gold doc
    at rank 1, hence inside top-10.

Compare oracle_recall@10 at N=20 (the current POOL_TO_RERANKER) against the
measured live recall@10 (~0.88) to see how much the reranker leaves on the
table at the current depth; compare across N to see how much a WIDER pool buys.

Run:  python -m eval.diagnose_recall_ceiling
"""
from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

import ir_datasets
from app.adapters.embedders.onnx_embedder import OnnxEmbedder
from app.adapters.pgvector.db import transaction
from app.config import settings
from app.container import build_container

EVAL_TENANT_NAME = "scifact-eval-postgres-benchmark"
EMBED_REPO = "Xenova/bge-base-en-v1.5"
EMBED_DIM = 768
EMBED_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
DEPTHS = [10, 20, 50, 100, 200]
DEEP = max(DEPTHS)
TS_CONFIG = "english"


def _dedupe_docs(rows):
    seen, out = set(), []
    for r in rows:
        d = r["payload"]["_id"]
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _dense_deep(cur, tid, qvec, limit):
    cur.execute(
        "SELECT payload FROM vector_chunks "
        "WHERE deleted=false AND (tenant_id=%s OR scope='global') "
        "ORDER BY embedding <=> %s::vector LIMIT %s",
        (tid, qvec, limit),
    )
    return cur.fetchall()


def _lexical_deep(cur, tid, qtext, limit):
    cur.execute(
        "SELECT payload, ts_rank_cd(tsv, q) AS lex "
        "FROM vector_chunks, websearch_to_tsquery(%s, %s) AS q "
        "WHERE deleted=false AND (tenant_id=%s OR scope='global') AND tsv @@ q "
        "ORDER BY lex DESC LIMIT %s",
        (TS_CONFIG, qtext, tid, limit),
    )
    return cur.fetchall()


def _first_rank(doc_list, gold):
    """1-indexed rank of the first gold doc in a deduped doc list; None if absent."""
    for i, d in enumerate(doc_list, start=1):
        if d in gold:
            return i
    return None


def main() -> None:
    embedder = OnnxEmbedder(EMBED_REPO, EMBED_DIM, pooling="cls",
                            query_instruction=EMBED_QUERY_INSTRUCTION, max_length=512)
    container = build_container(embedder=embedder)

    with transaction(settings.postgres_dsn) as cur:
        cur.execute("SELECT id FROM tenants WHERE name = %s", (EVAL_TENANT_NAME,))
        tid = cur.fetchone()["id"]
        cur.execute("SELECT COUNT(*) AS n, COUNT(DISTINCT document_id) AS d "
                    "FROM vector_chunks WHERE tenant_id=%s AND deleted=false", (tid,))
        row = cur.fetchone()
    n_chunks, n_docs = row["n"], row["d"]
    print(f"tenant {tid} | {n_chunks} chunks / {n_docs} docs "
          f"({n_chunks / n_docs:.2f} chunks/doc)\n")

    ds = ir_datasets.load("beir/scifact/test")
    qrels = defaultdict(dict)
    for q in ds.qrels_iter():
        qrels[q.query_id][q.doc_id] = q.relevance
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    corpus_docs = {d.doc_id for d in ds.docs_iter()}
    query_ids = [qid for qid in qrels if qid in queries
                 and any(d in corpus_docs for d in qrels[qid])]
    print(f"queries: {len(query_ids)}\n")

    # per-depth reachability counts, per channel
    reach = {ch: {N: 0 for N in DEPTHS} for ch in ("dense", "lexical", "union")}
    n_rel = []           # gold docs per query (SciFact is usually 1)
    dense_ranks, lex_ranks, best_ranks = [], [], []  # gold doc's rank (None=missed)
    scored = 0

    t0 = time.time()
    with transaction(settings.postgres_dsn) as cur:
        for n, qid in enumerate(query_ids, 1):
            qtext = queries[qid]
            gold = {d for d, r in qrels[qid].items() if r > 0}
            n_rel.append(len(gold))
            qvec = list(embedder.embed([qtext], is_query=True)[0])

            dense_docs = _dedupe_docs(_dense_deep(cur, tid, qvec, DEEP))
            lex_docs = _dedupe_docs(_lexical_deep(cur, tid, qtext, DEEP))

            dr = _first_rank(dense_docs, gold)
            lr = _first_rank(lex_docs, gold)
            dense_ranks.append(dr)
            lex_ranks.append(lr)
            best = min([r for r in (dr, lr) if r is not None], default=None)
            best_ranks.append(best)
            scored += 1

            for N in DEPTHS:
                d_hit = dr is not None and dr <= N
                l_hit = lr is not None and lr <= N
                reach["dense"][N] += d_hit
                reach["lexical"][N] += l_hit
                reach["union"][N] += (d_hit or l_hit)

            if n % 25 == 0:
                print(f"  {n}/{len(query_ids)}  ({time.time()-t0:.0f}s)", end="\r")
    print(f"\n  measured in {time.time()-t0:.0f}s\n")

    q = scored
    print(f"avg gold docs/query: {np.mean(n_rel):.2f} "
          f"(SciFact is mostly single-answer)\n")

    print("=== oracle recall@10 ceiling by pool depth N (gold doc reachable in channel) ===")
    print("  interpretation: max recall@10 a PERFECT reranker could reach if it")
    print("  fetched N candidates/channel. Current live pipeline fetches N=20.\n")
    print(f"{'depth N':>8} {'dense':>9} {'lexical':>9} {'union':>9}")
    for N in DEPTHS:
        print(f"{N:>8} {reach['dense'][N]/q:>9.4f} "
              f"{reach['lexical'][N]/q:>9.4f} {reach['union'][N]/q:>9.4f}")

    missed = sum(1 for b in best_ranks if b is None)
    print(f"\ngold doc UNREACHABLE in either channel within top-{DEEP}: "
          f"{missed}/{q} ({missed/q:.4f}) "
          f"-> hard ceiling recall@10 <= {1 - missed/q:.4f} at this depth")

    # where the gold doc sits when it IS reachable (union best rank)
    reachable = [b for b in best_ranks if b is not None]
    if reachable:
        arr = np.array(reachable)
        print(f"\nwhen reachable, gold doc's best dedup rank across channels: "
              f"median={int(np.median(arr))}, p90={int(np.percentile(arr,90))}, "
              f"max={int(arr.max())}")
        for cut in (10, 20, 50):
            print(f"  reachable within top-{cut}: {(arr <= cut).mean():.4f}")

    dense_only = sum(1 for dr, lr in zip(dense_ranks, lex_ranks)
                     if dr is not None and dr <= 20 and (lr is None or lr > 20))
    lex_only = sum(1 for dr, lr in zip(dense_ranks, lex_ranks)
                   if lr is not None and lr <= 20 and (dr is None or dr > 20))
    print(f"\nat the live depth N=20, gold reachable via:")
    print(f"  dense only (lexical missed it): {dense_only}")
    print(f"  lexical only (dense missed it): {lex_only}   "
          f"<- lexical channel's unique contribution")


if __name__ == "__main__":
    main()
