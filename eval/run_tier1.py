"""Tier 1 retrieval-quality levers, measured on the CURRENT bge-base corpus
(no re-ingestion), reranked by the LOCAL cross-encoder. Fast: cross-encoder is
local ONNX, no gateway calls.

Reranker = ms-marco-MiniLM-L-6-v2 (the app's config default and the reranker
recommended as production standard in the model-swap report). The pool=20 hybrid
cell reproduces this reranker's own baseline, so every delta in the matrix is
internally consistent (recall@10 is pool-dominated, so the reranker choice
barely moves it -- what we're isolating here is pool depth and fusion mode).

Two levers, tested as a matrix so their independent effect is visible:

  1. POOL DEPTH fed to the reranker (20 = current, 50, 100). Reranking can only
     REORDER the pool retrieval fetched, so a deeper pool raises the recall@10
     ceiling. (diagnose_recall_ceiling.py measured the ceiling; this measures
     what the real cross-encoder actually realizes.)
  2. FUSION MODE: 'hybrid' (dense + lexical FTS, 50/50 RRF -- the live path) vs
     'dense' (dense-only, query_text="" bypasses lexical in PgVectorStore.search).
     On identifier-free prose (SciFact) the lexical channel has ~0.06 recall and
     can DISPLACE genuine dense hits from the fused top-k; dense-only isolates
     whether that dilution costs recall.

Run:  python -m eval.run_tier1
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
from app.adapters.embedders.onnx_embedder import OnnxEmbedder
from app.adapters.pgvector.db import transaction
from app.adapters.rerankers.cross_encoder import CrossEncoderReranker
from app.config import settings
from app.container import build_container
from app.rag.query import _rerank

K_VALUES = [1, 3, 5, 10]
EVAL_TENANT_NAME = "scifact-eval-postgres-benchmark"
EMBED_REPO = "Xenova/bge-base-en-v1.5"
EMBED_DIM = 768
EMBED_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# CLI: python -m eval.run_tier1 [reranker_repo] [modes] [pools]
#   reranker_repo : HF repo of the cross-encoder (default ms-marco)
#   modes         : comma list of hybrid|dense (default "hybrid,dense")
#   pools         : comma list of pool depths  (default "20,50,100")
RERANK_REPO = sys.argv[1] if len(sys.argv) > 1 else "Xenova/ms-marco-MiniLM-L-6-v2"
FUSION_MODES = (sys.argv[2] if len(sys.argv) > 2 else "hybrid,dense").split(",")
POOLS = [int(p) for p in (sys.argv[3] if len(sys.argv) > 3 else "20,50,100").split(",")]
_TAG = RERANK_REPO.rsplit("/", 1)[-1].replace(".", "_")
OUT_XLSX = f"eval/retrieval_tier1_{_TAG}.xlsx"

# anchor: the pool=20 hybrid cell measured in THIS run is the true same-reranker
# baseline; ~0.878 was the bge-base + ms-marco recall@10 in the model-swap report.
BASELINE_R10 = 0.8780


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
    container = build_container(embedder=embedder)
    container.reranker = CrossEncoderReranker(RERANK_REPO)

    with transaction(settings.postgres_dsn) as cur:
        cur.execute("SELECT id FROM tenants WHERE name = %s", (EVAL_TENANT_NAME,))
        tid = cur.fetchone()["id"]
    print(f"tenant {tid} | {container.vectors.count(tid)} vectors (bge-base 768) | "
          f"reranker={RERANK_REPO}\n")

    ds = ir_datasets.load("beir/scifact/test")
    qrels = defaultdict(dict)
    for q in ds.qrels_iter():
        qrels[q.query_id][q.doc_id] = q.relevance
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    corpus = {d.doc_id for d in ds.docs_iter()}
    qids = [q for q in qrels if q in queries and any(d in corpus for d in qrels[q])]
    print(f"queries: {len(qids)}\n")

    # precompute query embeddings once (shared across all matrix cells)
    qvecs = {q: list(embedder.embed([queries[q]], is_query=True)[0]) for q in qids}
    golds = {q: {d for d, r in qrels[q].items() if r > 0} for q in qids}

    results = {}  # (mode, pool) -> {metric: value}
    rows = []
    for mode in FUSION_MODES:
        for pool in POOLS:
            agg = defaultdict(list)
            t0 = time.time()
            for q in qids:
                qtext = queries[q]
                # dense-only: pass query_text="" so PgVectorStore.search takes the
                # pure-dense path (no lexical fusion); hybrid: pass the real text.
                fts = "" if mode == "dense" else qtext
                hits = container.vectors.search(tid, qvecs[q], top_k=pool,
                                                access=None, query_text=fts)
                hits, _ = _rerank(container, qtext, hits, pool)
                ranked = _dedupe(h.payload["_id"] for h in hits)
                rel = golds[q]
                for k in K_VALUES:
                    agg[f"recall@{k}"].append(_recall_at_k(ranked, rel, k))
                    agg[f"ndcg@{k}"].append(_ndcg_at_k(ranked, rel, k))
                agg["mrr@10"].append(_rr_at_10(ranked, rel))
            cell = {m: float(np.mean(agg[m])) for m in agg}
            results[(mode, pool)] = cell
            rows.append({"mode": mode, "pool": pool, **cell})
            print(f"{mode:6} pool={pool:<4} recall@10={cell['recall@10']:.4f} "
                  f"ndcg@10={cell['ndcg@10']:.4f} mrr@10={cell['mrr@10']:.4f} "
                  f"({time.time()-t0:.0f}s)")

    print(f"\n=== recall@10 matrix (baseline pool=20 hybrid = {BASELINE_R10:.4f}) ===")
    print(f"{'pool':>6} " + " ".join(f"{m:>10}" for m in FUSION_MODES))
    for pool in POOLS:
        print(f"{pool:>6} " + " ".join(
            f"{results[(m, pool)]['recall@10']:>10.4f}" for m in FUSION_MODES))

    best = max(results, key=lambda k: results[k]["recall@10"])
    print(f"\nbest recall@10: {results[best]['recall@10']:.4f} "
          f"at mode={best[0]} pool={best[1]} "
          f"({results[best]['recall@10'] - BASELINE_R10:+.4f} vs baseline)")
    print("clears 90%!" if results[best]["recall@10"] >= 0.90 else "still under 90%")

    df = pd.DataFrame(rows)
    df.attrs["run_at_utc"] = datetime.now(timezone.utc).isoformat()
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="matrix", index=False)
    print(f"\nresults -> {OUT_XLSX}")


if __name__ == "__main__":
    main()
