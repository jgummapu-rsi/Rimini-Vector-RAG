"""Compare hybrid, reranked, and score-filtered retrieval on a saved subset.

Run with explicit EVAL_DATABASE_URL, EVAL_REDIS_URL and evaluation gateway
credentials. Embedding checkpoints allow interrupted evaluations to resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import ir_datasets
import numpy as np
import pandas as pd

from app.ingest.pipeline.chunker import ChunkSpec, chunk_elements
from app.ingest.pipeline.elements import Element
from app.shared.container import build_container
from app.shared.gateway.client import GatewayError
from app.shared.ports.vector_store import VectorPoint
from eval.run_retrieval import _dedup_hit_docs, _ndcg_at_k, _recall_at_k, _rr_at_10
from scripts.disposable_storage import evaluation_settings


def rank_variants(hits, scores, keep=20, floor=-3.0):
    if len(hits) != len(scores):
        raise ValueError("Reranker scores must align with candidates")
    order = sorted(range(len(hits)), key=lambda index: scores[index], reverse=True)[:keep]
    return {
        "hybrid": hits[:keep],
        "rerank_no_floor": [hits[index] for index in order],
        "rerank_floor": [hits[index] for index in order if scores[index] >= floor],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset", type=Path, default=Path("eval/retrieval_gateway_large_subset.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("eval/retrieval_gateway_large_comparison.json")
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pause", type=float, default=3.0)
    args = parser.parse_args()
    previous = json.loads(args.subset.read_text())
    wanted = set(previous["indexed_document_ids"])
    dataset = ir_datasets.load("beir/scifact/test")
    with evaluation_settings() as settings:
        settings = settings.model_copy(
            update={"reranker_provider": "cross_encoder", "metadata_extraction_enabled": False}
        )
        container = build_container(settings)
        try:
            run(container, dataset, wanted, previous, args)
        finally:
            container.close()


def run(container, dataset, wanted, previous, args):
    profile = container.embedder.profile
    if profile.id != previous["summary"]["embedding_profile_id"]:
        raise ValueError("Comparison must use the original embedding profile")
    chunks = []
    spec = ChunkSpec.auto(container.embedder.max_tokens)
    for source in dataset.docs_iter():
        if source.doc_id not in wanted:
            continue
        records = chunk_elements(
            [
                Element(
                    (source.title + "\n\n" + source.text).strip(),
                    "text",
                    "text",
                    "text_layer",
                    0,
                    {},
                )
            ],
            spec,
            count=container.embedder.count_tokens,
            embed_max=container.embedder.max_tokens,
        )
        chunks.extend((source.doc_id, index, chunk.text) for index, chunk in enumerate(records))

    chunks = chunks[: previous["summary"]["corpus_chunks"]]
    if len(chunks) != previous["summary"]["corpus_chunks"] or {d for d, _, _ in chunks} != wanted:
        raise ValueError("Cannot reproduce the original subset")
    cache = json.loads(args.checkpoint.read_text()) if args.checkpoint.exists() else {}

    def vector(text):
        key = hashlib.sha256((profile.id + "\0" + text).encode()).hexdigest()
        return key, cache.get(key)

    def embed(texts):
        keys = [vector(text)[0] for text in texts]
        missing = [index for index, key in enumerate(keys) if key not in cache]
        for offset in range(0, len(missing), 8):
            indices = missing[offset : offset + 8]
            for attempt in range(6):
                time.sleep(args.pause)
                try:
                    vectors = container.embedder.embed_documents(
                        [texts[index] for index in indices]
                    )
                    break
                except GatewayError as exc:
                    if exc.status_code not in (408, 429, 502, 503, 504) or attempt == 5:
                        raise
                    print(f"Gateway status {exc.status_code}; waiting 60 seconds", flush=True)
                    time.sleep(60)
            cache.update(
                (keys[index], value) for index, value in zip(indices, vectors, strict=True)
            )
            args.checkpoint.write_text(json.dumps(cache))
            print(
                f"Embedded {min(offset + 8, len(missing))}/{len(missing)} missing texts", flush=True
            )
        return [cache[key] for key in keys]

    print(f"Rebuilding {len(wanted)} documents / {len(chunks)} chunks", flush=True)
    vectors = embed([text for _, _, text in chunks])
    tenant = container.metadata.create_tenant("controlled-subset-comparison")
    points = [
        VectorPoint(
            f"{doc}::{ordinal:05d}",
            tenant,
            vec,
            {"_id": doc, "scope": "tenant", "content": text, "modality": "text"},
        )
        for (doc, ordinal, text), vec in zip(chunks, vectors, strict=True)
    ]
    for offset in range(0, len(points), 64):
        container.vectors.upsert(points[offset : offset + 64])
    cases = previous["queries"]
    qvectors = embed([case["question"] for case in cases])
    observations = []
    for number, (case, qvector) in enumerate(zip(cases, qvectors, strict=True), 1):
        started = time.monotonic()
        hits = container.vectors.search(tenant, qvector, top_k=80, query_text=case["question"])
        search_ms = (time.monotonic() - started) * 1000
        started = time.monotonic()
        scores = container.reranker.score(
            case["question"], [hit.payload["content"] for hit in hits]
        )
        rerank_ms = (time.monotonic() - started) * 1000
        relevant = set(case["relevant_in_subset"])
        variants = {}
        for mode, selected in rank_variants(hits, scores).items():
            ranked = _dedup_hit_docs(selected, 10)
            metrics = {f"recall@{k}": _recall_at_k(ranked, relevant, k) for k in (1, 3, 5, 10)}
            metrics.update({f"ndcg@{k}": _ndcg_at_k(ranked, relevant, k) for k in (1, 3, 5, 10)})
            variants[mode] = dict(
                metrics,
                **{"mrr@10": _rr_at_10(ranked, relevant)},
                ranked_documents=ranked,
                empty=not selected,
            )
        observations.append(
            dict(
                query_id=case["query_id"],
                question=case["question"],
                relevant_documents=sorted(relevant),
                search_ms=search_ms,
                rerank_ms=rerank_ms,
                candidate_recall=_recall_at_k(_dedup_hit_docs(hits, 80), relevant, 80),
                candidates=[
                    dict(
                        chunk_id=h.chunk_id,
                        document_id=h.payload["_id"],
                        hybrid_rank=i + 1,
                        hybrid_score=h.score,
                        rerank_score=scores[i],
                    )
                    for i, h in enumerate(hits)
                ],
                variants=variants,
            )
        )
        print(
            f"Query {number}/{len(cases)} recall@10: "
            + str({m: v["recall@10"] for m, v in variants.items()}),
            flush=True,
        )
    summaries = {}
    for mode in ("hybrid", "rerank_no_floor", "rerank_floor"):
        summaries[mode] = {
            metric: float(np.mean([row["variants"][mode][metric] for row in observations]))
            for metric in [f"{m}@{k}" for k in (1, 3, 5, 10) for m in ("recall", "ndcg")]
            + ["mrr@10"]
        }
        summaries[mode]["empty_results"] = sum(
            row["variants"][mode]["empty"] for row in observations
        )
    result = dict(
        corpus_documents=len(wanted),
        corpus_chunks=len(chunks),
        questions=len(cases),
        candidate_pool=80,
        retained_chunks=20,
        floor=-3.0,
        embedding_profile=asdict(profile),
        candidate_recall=float(np.mean([r["candidate_recall"] for r in observations])),
        generation_calls=0,
        summaries=summaries,
        queries=observations,
        latency_note="Search/rerank timings exclude precomputed gateway query embeddings; not end-to-end latency.",
    )
    args.output.write_text(json.dumps(result, indent=2))
    with pd.ExcelWriter(args.output.with_suffix(".xlsx")) as writer:
        pd.DataFrame(summaries).to_excel(writer, sheet_name="summary")
        pd.DataFrame(
            [
                dict(query_id=r["query_id"], mode=m, **v)
                for r in observations
                for m, v in r["variants"].items()
            ]
        ).to_excel(writer, sheet_name="per_query", index=False)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
