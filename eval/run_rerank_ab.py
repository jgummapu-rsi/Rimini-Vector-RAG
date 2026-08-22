"""Query-only reranker A/B over the CURRENT (bge-base, 768-dim) pgvector table.

Isolates the reranker from the embedder: the corpus is already embedded with
bge-base-en-v1.5 (run_retrieval_sota.py), so this does NOT re-ingest. It just
re-queries all 300 SciFact queries with a chosen reranker, so comparing two runs
tells us the reranker's effect with the embedder held fixed.

Run:  python -m eval.run_rerank_ab <reranker_repo> <label>
      python -m eval.run_rerank_ab Xenova/ms-marco-MiniLM-L-6-v2 msmarco
      python -m eval.run_rerank_ab Xenova/bge-reranker-base       bge
"""
from __future__ import annotations

import math
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

import ir_datasets
from app.adapters.embedders.onnx_embedder import OnnxEmbedder
from app.adapters.pgvector.db import transaction
from app.adapters.rerankers.cross_encoder import CrossEncoderReranker
from app.config import settings
from app.container import build_container
from app.rag.query import _rerank

K_VALUES = [1, 3, 5, 10]
POOL_TO_RERANKER = 20
EVAL_TENANT_NAME = "scifact-eval-postgres-benchmark"
EMBED_REPO = "Xenova/bge-base-en-v1.5"
EMBED_DIM = 768
EMBED_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

BASELINE_MINILM = {  # MiniLM embed + ms-marco rerank, simplified N=20 (original baseline)
    "recall@1": 0.5694, "recall@3": 0.7021, "recall@5": 0.7717, "recall@10": 0.8307,
    "ndcg@1": 0.5967, "ndcg@3": 0.6605, "ndcg@5": 0.6885, "ndcg@10": 0.7100, "mrr@10": 0.6796,
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
    if len(sys.argv) < 3:
        raise SystemExit("usage: python -m eval.run_rerank_ab <reranker_repo> <label>")
    rr_repo, label = sys.argv[1], sys.argv[2]

    embedder = OnnxEmbedder(EMBED_REPO, EMBED_DIM, pooling="cls",
                            query_instruction=EMBED_QUERY_INSTRUCTION, max_length=512)
    container = build_container(embedder=embedder)     # dim 768 matches existing table (no rebuild)
    container.reranker = CrossEncoderReranker(rr_repo)

    with transaction(settings.postgres_dsn) as cur:
        cur.execute("SELECT id FROM tenants WHERE name = %s", (EVAL_TENANT_NAME,))
        tid = cur.fetchone()["id"]
    n_vec = container.vectors.count(tid)
    print(f"tenant {tid} | {n_vec} vectors (bge-base 768) | reranker={rr_repo} | label={label}\n")

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
            print(f"  {n}/{len(query_ids)}", end="\r")
    print(f"\n  queried in {time.time()-t0:.0f}s\n")

    def mean(k):
        return float(np.mean(agg[k]))

    print(f"=== bge-base embed + {label} rerank  vs  MiniLM+ms-marco baseline ===")
    print(f"{'metric':12} {'baseline':>9} {'this':>9} {'delta':>9}")
    order = [f"recall@{k}" for k in K_VALUES] + [f"ndcg@{k}" for k in K_VALUES] + ["mrr@10"]
    summary = {"metric": [], "value": []}
    for m in order:
        v = mean(m); b = BASELINE_MINILM[m]
        print(f"{m:12} {b:>9.4f} {v:>9.4f} {v-b:>+9.4f}")
        summary["metric"].append(m); summary["value"].append(v)
    out = f"eval/retrieval_rerank_{label}.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        pd.DataFrame(summary).to_excel(w, sheet_name="summary", index=False)
        pd.DataFrame(rows).to_excel(w, sheet_name="per_query", index=False)
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
