"""Real end-to-end retrieval benchmark on gold qrels — Postgres/pgvector, no
in-memory shortcut, no LLM judge.

Unlike `eval/run_retrieval.py` (which ranks with an in-memory numpy matmul),
this script goes through the REAL storage + retrieval path: the actual
pgvector adapter (`app.adapters.pgvector.vector_store.PgVectorStore`) backed by
a real Postgres database, with the exact same `VectorStore.search()` method
production traffic uses (SQL HNSW ANN query + real RRF fusion against BM25 via
`app.adapters.shared.bm25`).

Dataset : BEIR SciFact (via ir_datasets) — 5,183 docs, 300 queries, gold qrels.
Pipeline under test : real chunker + real MiniLM embeddings + real Postgres/
pgvector storage + real hybrid (dense+BM25 RRF) search — the actual
production code path, not a shortcut.
Metrics : recall@k, nDCG@k (k in 1,3,5,10), MRR@10 — computed directly from
qrels. All pure math, no LLM involved.

Requires .env: METADATA_BACKEND=postgres, VECTOR_BACKEND=pgvector, and a
reachable DATABASE_URL with the pgvector extension available.

Run:  python -m eval.run_retrieval_postgres  [max_docs]
"""
from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import ir_datasets
from app.adapters.postgres.db import transaction
from app.config import settings
from app.container import build_container
from app.domain.models import Modality
from app.pipeline.chunker import chunk_elements
from app.pipeline.elements import Element
from app.ports.vector_store import VectorPoint

K_VALUES = [1, 3, 5, 10]
FETCH_K = 20  # chunks fetched BEFORE dedup to docs -- wider than the largest
              # scored K (10) so a document-dominant chunk cluster can't crowd
              # other relevant docs out of the ranked doc list
EVAL_TENANT_NAME = "scifact-eval-postgres-benchmark"
OUT_XLSX = "eval/retrieval_postgres_results.xlsx"


def _embed_batched(embedder, texts, batch=128):
    out = []
    for i in range(0, len(texts), batch):
        out.extend(embedder.embed(texts[i:i + batch]))
        if (i // batch) % 10 == 0:
            print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}", end="\r")
    print()
    return out


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


def _ensure_eval_tenant(container) -> str:
    dsn = settings.postgres_dsn
    with transaction(dsn) as cur:
        cur.execute("SELECT id FROM tenants WHERE name = %s", (EVAL_TENANT_NAME,))
        row = cur.fetchone()
    if row:
        tid = row["id"]
        print(f"reusing eval tenant {tid} ({EVAL_TENANT_NAME})")
    else:
        tid = container.metadata.create_tenant(EVAL_TENANT_NAME)
        print(f"created eval tenant {tid} ({EVAL_TENANT_NAME})")

    with transaction(dsn) as cur:
        cur.execute("DELETE FROM vector_chunks WHERE tenant_id = %s", (tid,))
        print(f"wiped {cur.rowcount} stale vector_chunks rows for this tenant")
    return tid


def main() -> None:
    if settings.metadata_backend != "postgres" or settings.vector_backend != "pgvector":
        raise RuntimeError(
            "This benchmark requires METADATA_BACKEND=postgres and "
            "VECTOR_BACKEND=pgvector in .env (real Postgres/pgvector, no "
            f"shortcut) — got METADATA_BACKEND={settings.metadata_backend} "
            f"VECTOR_BACKEND={settings.vector_backend}"
        )

    max_docs = int(sys.argv[1]) if len(sys.argv) > 1 else None
    ds = ir_datasets.load("beir/scifact/test")

    qrels = defaultdict(dict)
    for q in ds.qrels_iter():
        qrels[q.query_id][q.doc_id] = q.relevance
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    query_ids = [qid for qid in qrels if qid in queries]

    corpus = []
    for i, d in enumerate(ds.docs_iter()):
        if max_docs and i >= max_docs:
            break
        corpus.append((d.doc_id, (d.title + "\n\n" + d.text).strip()))
    print(f"corpus: {len(corpus)} docs | queries (pre-filter): {len(query_ids)}\n")

    doc_set = {d for d, _ in corpus}
    query_ids = [qid for qid in query_ids if any(d in doc_set for d in qrels[qid])]
    print(f"queries (with a relevant doc in corpus): {len(query_ids)}\n")

    container = build_container()
    tid = _ensure_eval_tenant(container)

    print("chunking corpus (real chunker)...")
    chunk_texts, chunk_doc_ids = [], []
    for doc_id, text in corpus:
        for ch in chunk_elements([Element(text, Modality.TEXT.value,
                                          "text", "text_layer", 0, {})]):
            chunk_texts.append(ch.text)
            chunk_doc_ids.append(doc_id)
    print(f"  {len(chunk_texts)} chunks\n")

    print("embedding chunks (real MiniLM)...")
    t0 = time.time()
    vectors = _embed_batched(container.embedder, chunk_texts)
    print(f"  embedded in {time.time()-t0:.0f}s\n")

    print("upserting into real Postgres/pgvector...")
    t0 = time.time()
    ordinals: dict[str, int] = defaultdict(int)
    batch: list[VectorPoint] = []
    upserted = 0
    for doc_id, text, vec in zip(chunk_doc_ids, chunk_texts, vectors):
        ordinal = ordinals[doc_id]
        ordinals[doc_id] += 1
        batch.append(VectorPoint(
            chunk_id=f"{doc_id}::{ordinal:05d}",
            tenant_id=tid,
            vector=list(vec),
            payload={"_id": doc_id, "scope": "tenant", "content": text, "modality": "text"},
        ))
        if len(batch) >= 500:
            container.vectors.upsert(batch)
            upserted += len(batch)
            print(f"  upserted {upserted}/{len(chunk_texts)}", end="\r")
            batch = []
    if batch:
        container.vectors.upsert(batch)
        upserted += len(batch)
    print(f"\n  {upserted} chunks upserted in {time.time()-t0:.0f}s "
          f"(pgvector count for tenant: {container.vectors.count(tid)})\n")

    print(f"running {len(query_ids)} queries through real search() "
          f"(fetch_k={FETCH_K} chunks, hybrid, then dedup to docs)...")
    agg = defaultdict(list)
    per_query_rows = []
    t0 = time.time()
    for n, qid in enumerate(query_ids, 1):
        qtext = queries[qid]
        rel = {d for d, r in qrels[qid].items() if r > 0}
        qvec = container.embedder.embed([qtext])[0]
        hits = container.vectors.search(tid, list(qvec), top_k=FETCH_K, query_text=qtext)
        ranked = _dedupe_docs(hit.payload["_id"] for hit in hits)

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
    print("=== HYBRID (real pgvector search: dense HNSW + BM25 RRF fusion) ===")
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
        ("corpus_docs", len(corpus)),
        ("corpus_chunks", len(chunk_texts)),
        ("num_queries", len(query_ids)),
        ("fetch_k", FETCH_K),
        ("tenant_id", tid),
        ("run_at_utc", datetime.now(timezone.utc).isoformat()),
    ]:
        summary["metric"].append(k2)
        summary["value"].append(v2)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        pd.DataFrame(summary).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(per_query_rows).to_excel(writer, sheet_name="per_query", index=False)
    print(f"\nresults written -> {OUT_XLSX}")

    print(f"\nNote: search() fetches the top-{FETCH_K} CHUNKS per query, then "
          "dedups to parent documents before scoring at k<=10. If >1 chunk "
          f"per document still crowds the {FETCH_K}-chunk pool, the deduped "
          "doc list can be shorter than 10 -- less likely than at fetch_k=10, "
          "but not eliminated.")
    print("Reference: published all-MiniLM-L6-v2 on SciFact is ~nDCG@10 0.64; "
          "BM25 is ~nDCG@10 0.66. Numbers in that range mean the harness is honest.")


if __name__ == "__main__":
    main()
