"""Tier 2: re-chunk the corpus against bge-base's REAL tokenizer + limit (via the
now-embedder-driven chunker), re-embed, rebuild the pgvector table, then
benchmark with the local cross-encoder at a chosen pool depth.

Why: the chunker now sizes chunks against the ACTIVE embedder's tokenizer and
hard limit (app.shared.ports.embedder.Embedder.max_tokens/count_tokens +
ChunkSpec.auto). With bge-base (512-token limit) that yields ~390/476-token
chunks instead of MiniLM's 180/220 -- fewer, richer chunks (was 2.84 chunks/doc
at 256), so a document's relevant content is less fragmented and its best chunk
ranks higher. This uses the SAME production code path the runner uses (no
tokenizer monkeypatching): build the container with the bge embedder and call
chunk_elements with embedder.count_tokens / embedder.max_tokens.

REBUILDS vector_chunks (drops + recreates at dim 768) -- wipes the current
256-token-chunk corpus; those baselines are saved in eval/retrieval_*.xlsx.

Run:  python -m eval.run_tier2 [pool] [mode] [reranker_repo]
      python -m eval.run_tier2 50 hybrid Xenova/bge-reranker-base   (defaults)
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
from app.ingest.pipeline.chunker import ChunkSpec, chunk_elements
from app.ingest.pipeline.elements import Element
from app.shared.ports.vector_store import VectorPoint
from app.retrieval.rag.query import _rerank

K_VALUES = [1, 3, 5, 10]
EVAL_TENANT_NAME = "scifact-eval-postgres-benchmark"
EMBED_REPO = "Xenova/bge-base-en-v1.5"
EMBED_DIM = 768
EMBED_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
OUT_XLSX = "eval/retrieval_tier2_results.xlsx"

BASELINE_R10 = 0.8823  # bge-base + bge-reranker-base, 256-tok MiniLM-sized chunks, pool=20


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


def _ensure_tenant(container) -> str:
    with transaction(settings.postgres_dsn) as cur:
        cur.execute("SELECT id FROM tenants WHERE name = %s", (EVAL_TENANT_NAME,))
        row = cur.fetchone()
    return row["id"] if row else container.metadata.create_tenant(EVAL_TENANT_NAME)


def main() -> None:
    pool = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    mode = sys.argv[2] if len(sys.argv) > 2 else "hybrid"
    rr_repo = sys.argv[3] if len(sys.argv) > 3 else "Xenova/bge-reranker-base"
    if settings.metadata_backend != "postgres" or settings.vector_backend != "pgvector":
        raise RuntimeError("Requires METADATA_BACKEND=postgres and VECTOR_BACKEND=pgvector")

    embedder = OnnxEmbedder(EMBED_REPO, EMBED_DIM, pooling="cls",
                            query_instruction=EMBED_QUERY_INSTRUCTION,
                            max_length=512, normalize=True)
    # Production sizing: derive the spec from the embedder's real limit. bge's
    # 512 -> ChunkSpec.auto ~390/476 (vs MiniLM's 180/220 at 256).
    spec = ChunkSpec.auto(embedder.max_tokens)
    print(f"embedder={EMBED_REPO} max_tokens={embedder.max_tokens} | "
          f"spec target={spec.target_tokens} max={spec.max_tokens} overlap={spec.overlap_tokens} | "
          f"rerank pool={pool} mode={mode} reranker={rr_repo}\n")

    # --- REBUILD table at dim 768 ---
    with transaction(settings.postgres_dsn) as cur:
        cur.execute("SELECT to_regclass('vector_chunks') AS reg")
        if cur.fetchone()["reg"] is not None:
            cur.execute("SELECT COUNT(*) AS n FROM vector_chunks")
            print(f"dropping vector_chunks (had {cur.fetchone()['n']} rows)")
        cur.execute("DROP TABLE IF EXISTS vector_chunks CASCADE")

    container = build_container(embedder=embedder)   # ensure_collection(768) recreates table
    container.reranker = CrossEncoderReranker(rr_repo)
    tid = _ensure_tenant(container)

    ds = ir_datasets.load("beir/scifact/test")
    qrels = defaultdict(dict)
    for q in ds.qrels_iter():
        qrels[q.query_id][q.doc_id] = q.relevance
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    corpus = [(d.doc_id, (d.title + "\n\n" + d.text).strip()) for d in ds.docs_iter()]
    doc_set = {d for d, _ in corpus}
    qids = [q for q in qrels if q in queries and any(d in doc_set for d in qrels[q])]
    print(f"corpus: {len(corpus)} docs | queries: {len(qids)}")

    chunk_texts, chunk_doc_ids = [], []
    for doc_id, text in corpus:
        el = Element(text, Modality.TEXT.value, "text", "text_layer", 0, {})
        for ch in chunk_elements([el], spec, count=embedder.count_tokens,
                                 embed_max=embedder.max_tokens):
            chunk_texts.append(ch.text)
            chunk_doc_ids.append(doc_id)
    print(f"chunks: {len(chunk_texts)} ({len(chunk_texts)/len(corpus):.2f} chunks/doc "
          f"-- was 2.84 at MiniLM 256-tok sizing)\n")

    print(f"embedding {len(chunk_texts)} chunks (passage mode)...")
    t0 = time.time()
    vectors = []
    B = 64
    for i in range(0, len(chunk_texts), B):
        vectors.extend(embedder.embed(chunk_texts[i:i + B], is_query=False))
    print(f"  embedded in {time.time()-t0:.0f}s")

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
    if batch:
        container.vectors.upsert(batch); upserted += len(batch)
    print(f"  {upserted} chunks upserted (count={container.vectors.count(tid)})\n")

    print(f"querying {len(qids)} queries (pool={pool}, mode={mode})...")
    agg = defaultdict(list)
    t0 = time.time()
    for n, q in enumerate(qids, 1):
        qtext = queries[q]
        qvec = list(embedder.embed([qtext], is_query=True)[0])
        fts = "" if mode == "dense" else qtext
        hits = container.vectors.search(tid, qvec, top_k=pool, access=None, query_text=fts)
        hits, _ = _rerank(container, qtext, hits, pool)
        ranked = _dedupe(h.payload["_id"] for h in hits)
        rel = {d for d, r in qrels[q].items() if r > 0}
        for k in K_VALUES:
            agg[f"recall@{k}"].append(_recall_at_k(ranked, rel, k))
            agg[f"ndcg@{k}"].append(_ndcg_at_k(ranked, rel, k))
        agg["mrr@10"].append(_rr_at_10(ranked, rel))
        if n % 20 == 0:
            print(f"  {n}/{len(qids)} ({time.time()-t0:.0f}s)", end="\r")
    print(f"\n  queried in {time.time()-t0:.0f}s\n")

    def mean(m):
        return float(np.mean(agg[m]))

    print(f"=== Tier 2 (bge auto-sized chunks + {mode} pool={pool} + {rr_repo.rsplit('/',1)[-1]}) "
          f" vs baseline recall@10={BASELINE_R10:.4f} ===")
    order = [f"recall@{k}" for k in K_VALUES] + [f"ndcg@{k}" for k in K_VALUES] + ["mrr@10"]
    summary = {"metric": [], "value": []}
    for m in order:
        v = mean(m)
        tag = f"   ({v-BASELINE_R10:+.4f} vs base)" if m == "recall@10" else ""
        print(f"{m:12} {v:.4f}{tag}")
        summary["metric"].append(m); summary["value"].append(v)
    print("\nclears 90%!" if mean("recall@10") >= 0.90 else "\nstill under 90%")

    for k2, v2 in [
        ("chunks", len(chunk_texts)), ("chunks_per_doc", round(len(chunk_texts)/len(corpus), 2)),
        ("spec_target", spec.target_tokens), ("spec_max", spec.max_tokens),
        ("pool", pool), ("mode", mode), ("reranker", rr_repo),
        ("num_queries", len(qids)), ("tenant_id", tid),
        ("run_at_utc", datetime.now(timezone.utc).isoformat()),
    ]:
        summary["metric"].append(k2); summary["value"].append(v2)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        pd.DataFrame(summary).to_excel(w, sheet_name="summary", index=False)
    print(f"\nresults -> {OUT_XLSX}")


if __name__ == "__main__":
    main()
