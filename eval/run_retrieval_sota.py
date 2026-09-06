"""End-to-end SOTA-model benchmark: rebuild the pgvector table at a new embedding
dim, re-embed the full SciFact corpus with a stronger local ONNX embedder, and
query through the simplified hybrid + a stronger local ONNX reranker.

Models (both local ONNX, no torch -- same recipe as the existing adapters):
  embedder : BAAI/bge-base-en-v1.5  (Xenova ONNX, 768-dim, CLS pooling, query
             instruction) -- replaces all-MiniLM-L6-v2 (384-dim, mean pooling).
  reranker : BAAI/bge-reranker-base (Xenova ONNX) -- replaces ms-marco-MiniLM-L-6-v2.

Everything else is held fixed vs the baseline: same chunker, same corpus, same
300 queries/qrels, same simplified 50/50 RRF search, same doc-level scoring
(rerank the fused pool -> dedupe chunks to docs -> recall/nDCG@{1,3,5,10}+MRR@10).
So the delta isolates the model upgrade.

REBUILDS vector_chunks (drops + recreates at dim 768 via ensure_collection) --
required because the embedding column is a fixed-width vector(dim). This wipes
the current MiniLM (384-dim) vectors; the MiniLM baseline numbers are already
saved in eval/retrieval_ab_*.xlsx.

Run:  python -m eval.run_retrieval_sota [max_docs]
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
from app.shared.adapters.embedders.onnx_embedder import OnnxEmbedder
from app.shared.adapters.pgvector.db import transaction
from app.retrieval.adapters.rerankers.cross_encoder import CrossEncoderReranker
from app.shared.config import settings
from app.shared.container import build_container
from app.shared.domain.models import Modality
from app.ingest.pipeline.chunker import chunk_elements
from app.ingest.pipeline.elements import Element
from app.shared.ports.vector_store import VectorPoint
from app.retrieval.rag.query import _rerank

K_VALUES = [1, 3, 5, 10]
POOL_TO_RERANKER = 20          # fused candidates fed to the reranker (user's spec)
EVAL_TENANT_NAME = "scifact-eval-postgres-benchmark"
OUT_XLSX = "eval/retrieval_sota_results.xlsx"

EMBED_REPO = "Xenova/bge-base-en-v1.5"
EMBED_DIM = 768
EMBED_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
RERANK_REPO = "Xenova/bge-reranker-base"

# MiniLM + ms-marco baseline (simplified fusion, N=20) for side-by-side print.
BASELINE = {
    "recall@1": 0.5694, "recall@3": 0.7021, "recall@5": 0.7717, "recall@10": 0.8307,
    "ndcg@1": 0.5967, "ndcg@3": 0.6605, "ndcg@5": 0.6885, "ndcg@10": 0.7100,
    "mrr@10": 0.6796,
}


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


def _dedupe_docs(doc_ids):
    seen, out = set(), []
    for d in doc_ids:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _ensure_tenant(container) -> str:
    dsn = settings.postgres_dsn
    with transaction(dsn) as cur:
        cur.execute("SELECT id FROM tenants WHERE name = %s", (EVAL_TENANT_NAME,))
        row = cur.fetchone()
    if row:
        return row["id"]
    return container.metadata.create_tenant(EVAL_TENANT_NAME)


def main() -> None:
    if settings.metadata_backend != "postgres" or settings.vector_backend != "pgvector":
        raise RuntimeError("Requires METADATA_BACKEND=postgres and VECTOR_BACKEND=pgvector")

    max_docs = int(sys.argv[1]) if len(sys.argv) > 1 else None

    # --- REBUILD: drop vector_chunks so ensure_collection recreates it at dim 768 ---
    dsn = settings.postgres_dsn
    with transaction(dsn) as cur:
        cur.execute("SELECT to_regclass('vector_chunks') AS reg")
        if cur.fetchone()["reg"] is not None:
            cur.execute("SELECT COUNT(*) AS n FROM vector_chunks")
            print(f"dropping vector_chunks (had {cur.fetchone()['n']} rows, 384-dim MiniLM)")
        cur.execute("DROP TABLE IF EXISTS vector_chunks CASCADE")
    print(f"rebuilding at dim {EMBED_DIM} with {EMBED_REPO}\n")

    embedder = OnnxEmbedder(EMBED_REPO, EMBED_DIM, pooling="cls",
                            query_instruction=EMBED_QUERY_INSTRUCTION,
                            max_length=512, normalize=True)
    container = build_container(embedder=embedder)          # ensure_collection(768) recreates table
    container.reranker = CrossEncoderReranker(RERANK_REPO)
    tid = _ensure_tenant(container)
    print(f"eval tenant {tid} | embedder={EMBED_REPO} dim={container.embedder.dim} | "
          f"reranker={RERANK_REPO}\n")

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
    doc_set = {d for d, _ in corpus}
    query_ids = [qid for qid in query_ids if any(d in doc_set for d in qrels[qid])]
    print(f"corpus: {len(corpus)} docs | queries: {len(query_ids)}\n")

    print("chunking (real chunker)...")
    chunk_texts, chunk_doc_ids = [], []
    for doc_id, text in corpus:
        for ch in chunk_elements([Element(text, Modality.TEXT.value, "text", "text_layer", 0, {})]):
            chunk_texts.append(ch.text)
            chunk_doc_ids.append(doc_id)
    print(f"  {len(chunk_texts)} chunks\n")

    # --- embed passages (is_query=False, no instruction) + upsert ---
    print(f"embedding {len(chunk_texts)} chunks with {EMBED_REPO} (passage mode)...")
    t0 = time.time()
    vectors = []
    B = 64
    for i in range(0, len(chunk_texts), B):
        vectors.extend(embedder.embed(chunk_texts[i:i + B], is_query=False))
        if (i // B) % 20 == 0:
            print(f"  embedded {min(i + B, len(chunk_texts))}/{len(chunk_texts)}", end="\r")
    print(f"\n  embedded in {time.time()-t0:.0f}s\n")

    print("upserting into rebuilt pgvector table...")
    t0 = time.time()
    ordinals = defaultdict(int)
    batch, upserted = [], 0
    for doc_id, text, vec in zip(chunk_doc_ids, chunk_texts, vectors):
        o = ordinals[doc_id]; ordinals[doc_id] += 1
        batch.append(VectorPoint(
            chunk_id=f"{doc_id}::{o:05d}", tenant_id=tid, vector=list(vec),
            payload={"_id": doc_id, "scope": "tenant", "content": text, "modality": "text"},
        ))
        if len(batch) >= 500:
            container.vectors.upsert(batch); upserted += len(batch); batch = []
            print(f"  upserted {upserted}/{len(chunk_texts)}", end="\r")
    if batch:
        container.vectors.upsert(batch); upserted += len(batch)
    print(f"\n  {upserted} chunks upserted in {time.time()-t0:.0f}s "
          f"(count={container.vectors.count(tid)})\n")

    print(f"querying {len(query_ids)} queries (pool_to_reranker={POOL_TO_RERANKER})...")
    agg = defaultdict(list)
    per_query_rows = []
    t0 = time.time()
    for n, qid in enumerate(query_ids, 1):
        qtext = queries[qid]
        rel = {d for d, r in qrels[qid].items() if r > 0}
        qvec = embedder.embed([qtext], is_query=True)[0]      # query instruction applied
        hits = container.vectors.search(tid, list(qvec), top_k=POOL_TO_RERANKER,
                                        access=None, query_text=qtext)
        hits, _ = _rerank(container, qtext, hits, POOL_TO_RERANKER)
        ranked = _dedupe_docs(h.payload["_id"] for h in hits)

        row = {"query_id": qid, "question": qtext}
        for k in K_VALUES:
            r = _recall_at_k(ranked, rel, k); nd = _ndcg_at_k(ranked, rel, k)
            agg[f"recall@{k}"].append(r); agg[f"ndcg@{k}"].append(nd)
            row[f"recall@{k}"] = r; row[f"ndcg@{k}"] = nd
        rr = _rr_at_10(ranked, rel)
        agg["mrr@10"].append(rr); row["rr@10"] = rr
        per_query_rows.append(row)
        if n % 20 == 0:
            print(f"  {n}/{len(query_ids)}", end="\r")
    print(f"\n  queried in {time.time()-t0:.0f}s\n")

    def mean(key):
        return float(np.mean(agg[key]))

    summary = {"metric": [], "value": []}
    print(f"=== SOTA (bge-base-en-v1.5 + bge-reranker-base) vs MiniLM+ms-marco baseline ===")
    print(f"{'metric':12} {'baseline':>9} {'sota':>9} {'delta':>9}")
    for k in K_VALUES:
        v = mean(f"recall@{k}"); b = BASELINE[f"recall@{k}"]
        print(f"recall@{k:<5} {b:>9.4f} {v:>9.4f} {v-b:>+9.4f}")
        summary["metric"].append(f"recall@{k}"); summary["value"].append(v)
    for k in K_VALUES:
        v = mean(f"ndcg@{k}"); b = BASELINE[f"ndcg@{k}"]
        print(f"ndcg@{k:<7} {b:>9.4f} {v:>9.4f} {v-b:>+9.4f}")
        summary["metric"].append(f"ndcg@{k}"); summary["value"].append(v)
    v = mean("mrr@10"); b = BASELINE["mrr@10"]
    print(f"{'mrr@10':12} {b:>9.4f} {v:>9.4f} {v-b:>+9.4f}")
    summary["metric"].append("mrr@10"); summary["value"].append(v)

    for k2, v2 in [
        ("embedder", EMBED_REPO), ("embed_dim", EMBED_DIM), ("reranker", RERANK_REPO),
        ("pool_to_reranker", POOL_TO_RERANKER), ("corpus_docs", len(corpus)),
        ("corpus_chunks", len(chunk_texts)), ("num_queries", len(query_ids)),
        ("tenant_id", tid), ("run_at_utc", datetime.now(timezone.utc).isoformat()),
    ]:
        summary["metric"].append(k2); summary["value"].append(v2)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        pd.DataFrame(summary).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(per_query_rows).to_excel(writer, sheet_name="per_query", index=False)
    print(f"\nresults written -> {OUT_XLSX}")


if __name__ == "__main__":
    main()
