"""A/B retrieval benchmark: complex (gated/weighted) vs simplified (plain 50/50)
hybrid fusion, both feeding the same cross-encoder reranker, on full BEIR SciFact.

Isolates the fusion change from every other knob by calling
`container.vectors.search()` DIRECTLY with an explicit `top_k=N` -- deliberately
bypassing `app.rag.query._retrieve`'s `_fetch_k` widening -- so N is exactly the
number of fused candidates handed to the reranker (the user's "pass N to the
reranker"). Then `_rerank` reorders that pool and we dedupe chunks to parent
documents and score doc-level recall/nDCG@{1,3,5,10} + MRR@10 against SciFact's
gold qrels. Identical scoring to eval/run_retrieval_postgres_rerank.py.

Which fusion runs is determined by the CURRENT code in
app/adapters/pgvector/vector_store.py -- this script does not know or care; the
`label` arg is only used to name the output file. So the protocol is:
  1. run with label=complex   BEFORE editing search()  (captures the baseline)
  2. edit search() -> simplified
  3. run with label=simplified (N=20, the user's spec) and again at N=40

Reuses the already-ingested eval tenant (14,722 vectors); NO re-ingestion, so the
only thing that changes between runs is the retrieval code + N.

Run:  python -m eval.run_retrieval_ab <N> <label>
      python -m eval.run_retrieval_ab 20 complex
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
from app.adapters.pgvector.db import transaction
from app.config import settings
from app.container import build_container
from app.rag.query import _rerank

K_VALUES = [1, 3, 5, 10]
EVAL_TENANT_NAME = "scifact-eval-postgres-benchmark"


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
        raise RuntimeError("Requires METADATA_BACKEND=postgres and VECTOR_BACKEND=pgvector")
    if settings.reranker_provider == "none":
        raise RuntimeError("RERANKER_PROVIDER=none -- set cross_encoder; this A/B feeds a reranker")

    if len(sys.argv) < 3:
        raise SystemExit("usage: python -m eval.run_retrieval_ab <N> <label>")
    n_pool = int(sys.argv[1])          # fused candidates handed to the reranker (= per-channel fetch)
    label = sys.argv[2]                # 'complex' | 'simplified' -- names the output only
    out_xlsx = f"eval/retrieval_ab_{label}_n{n_pool}.xlsx"

    ds = ir_datasets.load("beir/scifact/test")
    qrels = defaultdict(dict)
    for q in ds.qrels_iter():
        qrels[q.query_id][q.doc_id] = q.relevance
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    query_ids = [qid for qid in qrels if qid in queries]
    doc_ids = {d.doc_id for d in ds.docs_iter()}
    query_ids = [qid for qid in query_ids if any(d in doc_ids for d in qrels[qid])]
    print(f"queries: {len(query_ids)} | N (pool to reranker): {n_pool} | label: {label}\n")

    container = build_container()
    with transaction(settings.postgres_dsn) as cur:
        cur.execute("SELECT id FROM tenants WHERE name = %s", (EVAL_TENANT_NAME,))
        row = cur.fetchone()
    if not row:
        raise RuntimeError(f"eval tenant '{EVAL_TENANT_NAME}' not found -- ingest first "
                           "via eval/run_retrieval_postgres.py")
    tid = row["id"]
    print(f"reusing eval tenant {tid} ({container.vectors.count(tid)} vectors, no re-ingestion)")
    print(f"reranker: {settings.reranker_provider} ({settings.reranker_model})\n")

    agg = defaultdict(list)
    per_query_rows = []
    t0 = time.time()
    for n, qid in enumerate(query_ids, 1):
        qtext = queries[qid]
        rel = {d for d, r in qrels[qid].items() if r > 0}

        qvec = container.embedder.embed([qtext])[0]
        # DIRECT search: top_k=n_pool is exactly the fused pool size (no _fetch_k widening)
        hits = container.vectors.search(tid, list(qvec), top_k=n_pool, access=None, query_text=qtext)
        # rerank reorders the whole fused pool (top_k=n_pool => no truncation, just reorder)
        hits, _scores = _rerank(container, qtext, hits, n_pool)
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
    print(f"=== {label.upper()} fusion | pool_to_reranker={n_pool} | rerank -> dedupe -> score ===")
    print(f"{'metric':12} {'score':>7}")
    for k in K_VALUES:
        v = float(np.mean(agg[f"recall@{k}"]))
        print(f"recall@{k:<5} {v:>7.4f}")
        summary["metric"].append(f"recall@{k}"); summary["value"].append(v)
    for k in K_VALUES:
        v = float(np.mean(agg[f"ndcg@{k}"]))
        print(f"ndcg@{k:<7} {v:>7.4f}")
        summary["metric"].append(f"ndcg@{k}"); summary["value"].append(v)
    mrr = float(np.mean(agg["mrr@10"]))
    print(f"{'mrr@10':12} {mrr:>7.4f}")
    summary["metric"].append("mrr@10"); summary["value"].append(mrr)

    for k2, v2 in [
        ("label", label), ("pool_to_reranker", n_pool), ("num_queries", len(query_ids)),
        ("reranker_model", settings.reranker_model), ("tenant_id", tid),
        ("run_at_utc", datetime.now(timezone.utc).isoformat()),
    ]:
        summary["metric"].append(k2); summary["value"].append(v2)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        pd.DataFrame(summary).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(per_query_rows).to_excel(writer, sheet_name="per_query", index=False)
    print(f"\nresults written -> {out_xlsx}")


if __name__ == "__main__":
    main()
