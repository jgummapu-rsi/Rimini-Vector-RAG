"""Same benchmark as `eval/run_retrieval_postgres.py`, but querying through the
REAL production retrieval path including the cross-encoder reranker
(`app.rag.query._retrieve` + `_rerank`) instead of a bare `vectors.search()`
call at a fixed top_k=10.

`run_retrieval_postgres.py` fetches only the top-10 chunks and never reranks --
its own printed caveat says this caps recall@10 below what a wider candidate
pool would find. Production, when a reranker is configured, fetches a WIDER
pool first (`_fetch_k`: top_k * rerank_candidate_multiplier, floor
rerank_min_candidates) and reranks down to top_k -- this script measures that
real behavior instead of the narrower shortcut, using the SAME already-ingested
eval tenant/corpus (no re-ingestion) so it's a direct, isolated comparison of
just this one lever.

Run:  python -m eval.run_retrieval_postgres_rerank
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import ir_datasets
from app.adapters.postgres.db import transaction
from app.config import settings
from app.container import build_container
from app.rag.query import _rerank, _retrieve

K_VALUES = [1, 3, 5, 10]
FETCH_K = 20  # chunks retrieved+reranked BEFORE dedup to docs -- wider than the
              # largest scored K (10) so a document-dominant chunk cluster can't
              # crowd other relevant docs out of the ranked list (see chunk-vs-doc
              # dedup note in run_retrieval_postgres.py)
EVAL_TENANT_NAME = "scifact-eval-postgres-benchmark"
OUT_XLSX = "eval/retrieval_postgres_rerank_results.xlsx"


def _dedupe_docs(ranked_chunk_doc_ids):
    seen, ranked = set(), []
    for d in ranked_chunk_doc_ids:
        if d not in seen:
            seen.add(d)
            ranked.append(d)
    return ranked


def _recall_at_k(ranked, rel, k):
    if not rel:
        return None
    return len(set(ranked[:k]) & rel) / len(rel)


def _ndcg_at_k(ranked, rel, k):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in rel)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg else 0.0


def _rr_at_10(ranked, rel):
    for i, d in enumerate(ranked[:10]):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def main() -> None:
    if settings.metadata_backend != "postgres" or settings.vector_backend != "pgvector":
        raise RuntimeError(
            "Requires METADATA_BACKEND=postgres and VECTOR_BACKEND=pgvector in .env"
        )
    if settings.reranker_provider == "none":
        raise RuntimeError(
            "RERANKER_PROVIDER=none in .env -- set it to cross_encoder to measure "
            "the reranker's effect (this script's whole point)."
        )

    ds = ir_datasets.load("beir/scifact/test")
    qrels = defaultdict(dict)
    for q in ds.qrels_iter():
        qrels[q.query_id][q.doc_id] = q.relevance
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    query_ids = [qid for qid in qrels if qid in queries]

    doc_ids_in_corpus = {d.doc_id for d in ds.docs_iter()}
    query_ids = [qid for qid in query_ids if any(d in doc_ids_in_corpus for d in qrels[qid])]
    print(f"queries: {len(query_ids)}\n")

    container = build_container()
    print(f"reranker: {settings.reranker_provider} ({settings.reranker_model}), "
          f"candidate_multiplier={settings.rerank_candidate_multiplier}, "
          f"min_candidates={settings.rerank_min_candidates}")

    with transaction(settings.postgres_dsn) as cur:
        cur.execute("SELECT id FROM tenants WHERE name = %s", (EVAL_TENANT_NAME,))
        row = cur.fetchone()
    if not row:
        raise RuntimeError(
            f"eval tenant '{EVAL_TENANT_NAME}' not found -- run "
            "eval/run_retrieval_postgres.py first to ingest the corpus."
        )
    tid = row["id"]
    print(f"reusing eval tenant {tid} ({container.vectors.count(tid)} vectors, "
          "no re-ingestion)\n")

    print(f"running {len(query_ids)} queries through the real production path "
          f"(_retrieve wide pool + _rerank down to {FETCH_K} chunks, then dedup "
          f"to docs, scored at k in {K_VALUES})...")
    agg = defaultdict(list)
    per_query_rows = []
    t0 = time.time()
    for n, qid in enumerate(query_ids, 1):
        qtext = queries[qid]
        rel = {d for d, r in qrels[qid].items() if r > 0}

        hits = _retrieve(container, tid, qtext, FETCH_K, access=None)
        hits, _scores = _rerank(container, qtext, hits, FETCH_K)
        ranked = _dedupe_docs(h.payload["_id"] for h in hits)

        row = {"query_id": qid, "question": qtext}
        for k in K_VALUES:
            r = _recall_at_k(ranked, rel, k)
            nd = _ndcg_at_k(ranked, rel, k)
            agg[f"recall@{k}"].append(r)
            agg[f"ndcg@{k}"].append(nd)
            row[f"recall@{k}"] = r
            row[f"ndcg@{k}"] = nd
        rr = _rr_at_10(ranked, rel)
        agg["mrr@10"].append(rr)
        row["rr@10"] = rr
        per_query_rows.append(row)
        if n % 20 == 0:
            print(f"  {n}/{len(query_ids)}", end="\r")
    print(f"\n  queried in {time.time()-t0:.0f}s\n")

    summary = {"metric": [], "value": []}
    print("=== HYBRID + CROSS-ENCODER RERANK (real production path) ===")
    print(f"{'metric':12} {'score':>7}")
    for k in K_VALUES:
        v = float(np.mean(agg[f"recall@{k}"]))
        print(f"recall@{k:<5} {v:>7.4f}")
        summary["metric"].append(f"recall@{k}")
        summary["value"].append(v)
    for k in K_VALUES:
        v = float(np.mean(agg[f"ndcg@{k}"]))
        print(f"ndcg@{k:<7} {v:>7.4f}")
        summary["metric"].append(f"ndcg@{k}")
        summary["value"].append(v)
    mrr = float(np.mean(agg["mrr@10"]))
    print(f"{'mrr@10':12} {mrr:>7.4f}")
    summary["metric"].append("mrr@10")
    summary["value"].append(mrr)

    for k2, v2 in [
        ("num_queries", len(query_ids)),
        ("fetch_k", FETCH_K),
        ("candidate_multiplier", settings.rerank_candidate_multiplier),
        ("min_candidates", settings.rerank_min_candidates),
        ("reranker_model", settings.reranker_model),
        ("tenant_id", tid),
        ("run_at_utc", datetime.now(timezone.utc).isoformat()),
    ]:
        summary["metric"].append(k2)
        summary["value"].append(v2)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        pd.DataFrame(summary).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(per_query_rows).to_excel(writer, sheet_name="per_query", index=False)
    print(f"\nresults written -> {OUT_XLSX}")


if __name__ == "__main__":
    main()
