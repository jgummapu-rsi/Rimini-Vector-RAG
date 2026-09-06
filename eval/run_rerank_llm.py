"""Query-only reranker A/B over the CURRENT (bge-base, 768-dim) pgvector table
-- same shape as `eval/run_rerank_ab.py`, but with the cross-encoder swapped
for `LLMReranker`, a listwise "LLM-as-reranker" adapter (cf. RankGPT): it
sends the whole candidate pool to the gateway chat model in ONE prompt per
query and asks it to score every passage's relevance in one shot, instead of
a dedicated cross-encoder scoring one (query, passage) pair per forward pass.

Does NOT re-ingest -- the corpus is already embedded with bge-base-en-v1.5 by
`eval/run_retrieval_sota.py`, so this re-queries the same 300 SciFact queries
against the same table/tenant, isolating the reranker with the embedder held
fixed. Comparable directly against `eval/retrieval_sota_results.xlsx`
(bge-base + bge-reranker-base cross-encoder), the current live baseline.

Run:  python -m eval.run_rerank_llm
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import ir_datasets
from app.shared.adapters.embedders.onnx_embedder import OnnxEmbedder
from app.shared.adapters.pgvector.db import transaction
from app.retrieval.adapters.rerankers.llm_reranker import LLMReranker
from app.shared.config import settings
from app.shared.container import build_container
from app.retrieval.rag.query import _rerank

K_VALUES = [1, 3, 5, 10]
POOL_TO_RERANKER = 20
EVAL_TENANT_NAME = "scifact-eval-postgres-benchmark"
OUT_XLSX = "eval/retrieval_rerank_llm_results.xlsx"

EMBED_REPO = "Xenova/bge-base-en-v1.5"
EMBED_DIM = 768
EMBED_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# bge-base embed + bge-reranker-base cross-encoder, current live baseline
# (eval/retrieval_sota_results.xlsx, run 2026-08-12T09:39:42Z)
BASELINE = {
    "recall@1": 0.548444, "recall@3": 0.725222, "recall@5": 0.777, "recall@10": 0.882333,
    "ndcg@1": 0.573333, "ndcg@3": 0.661558, "ndcg@5": 0.684128, "ndcg@10": 0.720797,
    "mrr@10": 0.67572,
}


def _recall_at_k(ranked, rel, k):
    return None if not rel else len(set(ranked[:k]) & rel) / len(rel)


def _ndcg_at_k(ranked, rel, k):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in rel)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg else 0.0


def _rr_at_10(ranked, rel):
    for i, d in enumerate(ranked[:10]):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def _dedupe(doc_ids):
    seen, out = set(), []
    for d in doc_ids:
        if d not in seen:
            seen.add(d); out.append(d)
    return out


def main() -> None:
    embedder = OnnxEmbedder(EMBED_REPO, EMBED_DIM, pooling="cls",
                            query_instruction=EMBED_QUERY_INSTRUCTION, max_length=512)
    container = build_container(embedder=embedder)     # dim 768 matches existing table (no rebuild)
    container.reranker = LLMReranker(container.gateway, container.settings.chat_model)

    with transaction(settings.postgres_dsn) as cur:
        cur.execute("SELECT id FROM tenants WHERE name = %s", (EVAL_TENANT_NAME,))
        row = cur.fetchone()
    if not row:
        raise RuntimeError(
            f"eval tenant '{EVAL_TENANT_NAME}' not found -- run "
            "eval/run_retrieval_sota.py first to ingest+embed the corpus."
        )
    tid = row["id"]
    n_vec = container.vectors.count(tid)
    print(f"tenant {tid} | {n_vec} vectors (bge-base 768) | "
          f"reranker=llm ({container.settings.chat_model}, listwise)\n")

    ds = ir_datasets.load("beir/scifact/test")
    qrels = defaultdict(dict)
    for q in ds.qrels_iter():
        qrels[q.query_id][q.doc_id] = q.relevance
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    doc_ids = {d.doc_id for d in ds.docs_iter()}
    query_ids = [qid for qid in qrels if qid in queries and any(d in doc_ids for d in qrels[qid])]

    agg = defaultdict(list)
    rows = []
    t0 = time.time()
    for n, qid in enumerate(query_ids, 1):
        qtext = queries[qid]
        rel = {d for d, r in qrels[qid].items() if r > 0}
        qvec = embedder.embed([qtext], is_query=True)[0]
        hits = container.vectors.search(tid, list(qvec), top_k=POOL_TO_RERANKER,
                                        access=None, query_text=qtext)
        hits, _ = _rerank(container, qtext, hits, POOL_TO_RERANKER)
        ranked = _dedupe(h.payload["_id"] for h in hits)
        row = {"query_id": qid}
        for k in K_VALUES:
            r = _recall_at_k(ranked, rel, k); nd = _ndcg_at_k(ranked, rel, k)
            agg[f"recall@{k}"].append(r); agg[f"ndcg@{k}"].append(nd)
            row[f"recall@{k}"] = r; row[f"ndcg@{k}"] = nd
        rr = _rr_at_10(ranked, rel); agg["mrr@10"].append(rr); row["rr@10"] = rr
        rows.append(row)
        if n % 20 == 0:
            print(f"  {n}/{len(query_ids)}  ({time.time()-t0:.0f}s elapsed)", end="\r")
    print(f"\n  queried in {time.time()-t0:.0f}s\n")

    def mean(k):
        return float(np.mean(agg[k]))

    print("=== bge-base embed + llm (listwise) rerank  vs  bge-base + bge-reranker-base baseline ===")
    print(f"{'metric':12} {'baseline':>9} {'this':>9} {'delta':>9}")
    order = [f"recall@{k}" for k in K_VALUES] + [f"ndcg@{k}" for k in K_VALUES] + ["mrr@10"]
    summary = {"metric": [], "value": []}
    for m in order:
        v = mean(m); b = BASELINE[m]
        print(f"{m:12} {b:>9.4f} {v:>9.4f} {v-b:>+9.4f}")
        summary["metric"].append(m); summary["value"].append(v)

    for k2, v2 in [
        ("embedder", EMBED_REPO), ("embed_dim", EMBED_DIM),
        ("reranker", f"llm:{container.settings.chat_model}"),
        ("pool_to_reranker", POOL_TO_RERANKER), ("num_queries", len(query_ids)),
        ("tenant_id", tid), ("run_at_utc", datetime.now(timezone.utc).isoformat()),
    ]:
        summary["metric"].append(k2); summary["value"].append(v2)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        pd.DataFrame(summary).to_excel(w, sheet_name="summary", index=False)
        pd.DataFrame(rows).to_excel(w, sheet_name="per_query", index=False)
    print(f"\nresults -> {OUT_XLSX}")


if __name__ == "__main__":
    main()
