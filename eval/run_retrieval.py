"""Deterministic retrieval benchmark on gold qrels (no LLM judge).

Dataset : BEIR SciFact (via ir_datasets) — 5,183 docs, 300 queries, gold qrels.
Pipelines under test (this is the point of the rewrite): the SAME retrieval the
app actually ships, not a hand-rolled cosine. The corpus is ingested through the
real chunker + MiniLM embeddings into a throwaway localfs `container.vectors`,
and each ranker below is exactly what `/query` runs:

    DENSE           in-memory cosine (reference for the embedder alone)
    BM25            lexical baseline (reference for lexical alone)
    HYBRID          container.vectors.search(query_text=...)  -> dense+BM25 RRF
    HYBRID+RERANK   HYBRID candidate pool -> container.reranker (cross-encoder)

DENSE/BM25 stay as honesty references; HYBRID and HYBRID+RERANK are the shipped
pipeline and the numbers to compare against khub.

Metrics : recall@k, nDCG@k, MRR@10 — computed directly from qrels.

Run:  python -m eval.run_retrieval [max_docs] [--max-queries N]
"""
from __future__ import annotations

import math
import sys
import time
from collections import defaultdict

import numpy as np

import ir_datasets
from app.container import build_container
from app.domain.models import Modality
from app.pipeline.chunker import chunk_elements
from app.pipeline.elements import Element
from app.ports.vector_store import VectorPoint
from app.rag.query import _fetch_k, _rerank

K_VALUES = [1, 3, 5, 10, 20, 100]
_TENANT = "evaltenant"


def _embed_batched(embedder, texts, batch=128):
    out = []
    for i in range(0, len(texts), batch):
        out.extend(embedder.embed(texts[i:i + batch]))
        if (i // batch) % 10 == 0:
            print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.asarray(out, dtype=np.float32)


def _tokens(s: str):
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in s).split() if t]


def _ranked_docs(chunk_doc_ids, order, cutoff):
    """Dedup a chunk-level ranking (indices into chunk_doc_ids) into a doc-level
    ranking, keeping first (best) appearance, up to `cutoff` docs."""
    seen, ranked = set(), []
    for idx in order:
        d = chunk_doc_ids[idx]
        if d not in seen:
            seen.add(d)
            ranked.append(d)
            if len(ranked) >= cutoff:
                break
    return ranked


def _dedup_hit_docs(hits, cutoff):
    """Dedup a list of SearchHit into a doc-level ranking (payload `_id`)."""
    seen, ranked = set(), []
    for h in hits:
        d = h.payload.get("_id")
        if d not in seen:
            seen.add(d)
            ranked.append(d)
            if len(ranked) >= cutoff:
                break
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


def _evaluate(name, rank_fn, query_ids, queries, qrels):
    agg = defaultdict(list)
    t0 = time.time()
    for qid in query_ids:
        rel = {d for d, r in qrels[qid].items() if r > 0}
        ranked = rank_fn(queries[qid])
        for k in K_VALUES:
            agg[f"recall@{k}"].append(_recall_at_k(ranked, rel, k))
            agg[f"ndcg@{k}"].append(_ndcg_at_k(ranked, rel, k))
        agg["mrr@10"].append(_rr_at_10(ranked, rel))
    print(f"\n=== {name} ===   ({len(query_ids)} queries in {time.time()-t0:.0f}s)")
    print(f"{'metric':12} {'score':>7}")
    for k in K_VALUES:
        print(f"recall@{k:<5} {np.mean(agg[f'recall@{k}']):>7.4f}")
    for k in K_VALUES:
        print(f"ndcg@{k:<7} {np.mean(agg[f'ndcg@{k}']):>7.4f}")
    print(f"{'mrr@10':12} {np.mean(agg['mrr@10']):>7.4f}")


def _parse_args(argv):
    max_docs, max_queries, no_rerank = None, None, False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--max-queries":
            max_queries = int(argv[i + 1]); i += 2
        elif a == "--no-rerank":
            no_rerank = True; i += 1
        else:
            max_docs = int(a); i += 1
    return max_docs, max_queries, no_rerank


def main() -> None:
    max_docs, max_queries, no_rerank = _parse_args(sys.argv[1:])
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
    print(f"corpus: {len(corpus)} docs | queries: {len(query_ids)} | qrels loaded\n")

    # keep only qrels whose relevant docs are in the (possibly truncated) corpus
    doc_set = {d for d, _ in corpus}
    query_ids = [qid for qid in query_ids
                 if any(d in doc_set for d in qrels[qid])]
    if max_queries:
        query_ids = query_ids[:max_queries]
        print(f"(evaluating first {len(query_ids)} queries)\n")

    container = build_container()

    print("chunking + embedding corpus (real pipeline)...")
    chunk_texts, chunk_doc_ids = [], []
    for doc_id, text in corpus:
        for ch in chunk_elements([Element(text, Modality.TEXT.value,
                                          "text", "text_layer", 0, {})]):
            chunk_texts.append(ch.text)
            chunk_doc_ids.append(doc_id)
    t0 = time.time()
    mat = _embed_batched(container.embedder, chunk_texts)
    print(f"  {len(chunk_texts)} chunks embedded in {time.time()-t0:.0f}s "
          f"(dim={mat.shape[1]})")

    # upsert into container.vectors so HYBRID/RERANK exercise the shipped search
    print("upserting into vector store...")
    per_doc: dict[str, list[VectorPoint]] = defaultdict(list)
    for i, (doc_id, text) in enumerate(zip(chunk_doc_ids, chunk_texts)):
        cid = f"{doc_id}_{len(per_doc[doc_id]):03d}"
        per_doc[doc_id].append(VectorPoint(
            chunk_id=cid, tenant_id=_TENANT, vector=mat[i].tolist(),
            payload={"_id": doc_id, "content": text, "user_id": "eval",
                     "visibility": "tenant", "source_type": "text"},
        ))
    for pts in per_doc.values():
        container.vectors.upsert(pts)
    print(f"  {container.vectors.count(_TENANT)} chunk-vectors stored\n")

    def dense_rank(qtext):
        qv = np.asarray(container.embedder.embed([qtext])[0], dtype=np.float32)
        scores = mat @ qv
        order = np.argsort(-scores)[:2000]
        return _ranked_docs(chunk_doc_ids, order, max(K_VALUES))

    from rank_bm25 import BM25Okapi
    bm25_ref = BM25Okapi([_tokens(t) for _, t in corpus])
    corpus_ids = [d for d, _ in corpus]

    def bm25_rank(qtext):
        scores = bm25_ref.get_scores(_tokens(qtext))
        order = np.argsort(-scores)[:max(K_VALUES)]
        return [corpus_ids[i] for i in order]

    # Chunk pool for doc-level recall. DENSE ranks the whole corpus before
    # deduping chunks->docs; HYBRID must pull an equally wide chunk pool or
    # same-doc chunks collapse and doc-recall is unfairly starved. The localfs
    # store scores every chunk regardless of top_k, so a wide pool is free.
    doc_pool = min(2000, len(chunk_texts))

    def hybrid_rank(qtext):
        qv = container.embedder.embed([qtext])[0]
        hits = container.vectors.search(_TENANT, qv, top_k=doc_pool, query_text=qtext)
        return _dedup_hit_docs(hits, max(K_VALUES))

    # --- HYBRID+RERANK (shipped cross-encoder second pass), mirroring
    # app.rag.query: fetch a wider fused pool, then reorder with the reranker.
    # The cross-encoder runs one forward pass per candidate, so its pool is
    # bounded (rerank_pool) for tractable runtime -- recall beyond that pool's
    # doc coverage is capped by design; judge RERANK on nDCG@10/MRR (precision),
    # not recall@100. ---
    rerank_pool = max(_fetch_k(container, max(K_VALUES)), 400)

    def rerank_rank(qtext):
        qv = container.embedder.embed([qtext])[0]
        hits = container.vectors.search(_TENANT, qv, top_k=rerank_pool, query_text=qtext)
        hits, _ = _rerank(container, qtext, hits, max(K_VALUES))
        return _dedup_hit_docs(hits, max(K_VALUES))

    _evaluate("DENSE  (embedder alone: MiniLM + cosine)", dense_rank,
              query_ids, queries, qrels)
    _evaluate("BM25   (lexical baseline)", bm25_rank, query_ids, queries, qrels)
    _evaluate("HYBRID (shipped: dense+BM25 RRF)", hybrid_rank,
              query_ids, queries, qrels)
    if no_rerank:
        print("\n(--no-rerank: skipping the cross-encoder pass)")
    elif container.reranker is not None:
        _evaluate("HYBRID+RERANK (shipped: + cross-encoder)", rerank_rank,
                  query_ids, queries, qrels)
    else:
        print("\n(reranker disabled: RERANKER_PROVIDER=none — skipping RERANK run)")

    print("\nReference: published all-MiniLM-L6-v2 on SciFact is ~nDCG@10 0.64;")
    print("BM25 is ~nDCG@10 0.66. DENSE/BM25 in that range mean the harness is honest;")
    print("HYBRID / HYBRID+RERANK are the shipped pipeline and should meet or beat both.")


if __name__ == "__main__":
    main()
