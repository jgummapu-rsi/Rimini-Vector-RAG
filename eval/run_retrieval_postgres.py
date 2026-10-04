"""SciFact retrieval through real PostgreSQL/pgvector, with optional reranking.

Uses disposable evaluation storage and the active embedder's tokenizer. Scores
document-level recall/nDCG at 1, 3, 5, 10 and MRR@10 after deduplicating chunks.

Run: python -m eval.run_retrieval_postgres [max_docs] [--rerank] [--fetch-k N]
Requires EVAL_DATABASE_URL and EVAL_REDIS_URL.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import ir_datasets
import numpy as np
import pandas as pd

from app.retrieval.rag.access import access_predicate
from app.retrieval.rag.query import _rerank, _retrieve
from app.shared.container import build_container
from app.shared.domain.models import Principal, Role
from eval.run_retrieval import _dedup_hit_docs, _ndcg_at_k, _recall_at_k, _rr_at_10
from eval.storage import isolated_evaluation, populate_scifact, populate_scifact_pipeline

K_VALUES = [1, 3, 5, 10]


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("max_docs", nargs="?", type=_positive_int)
    parser.add_argument(
        "--rerank", action="store_true", help="Use production candidate widening and reranking"
    )
    parser.add_argument(
        "--fetch-k",
        type=_positive_int,
        default=20,
        help="Chunks retained before document deduplication (default: 20)",
    )
    parser.add_argument(
        "--output", type=Path, help="Output workbook; defaults to the selected mode's result file"
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Run original text through blobs, queue, parsing and atomic publication",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=4,
        help="Concurrent ingestion workers for --pipeline",
    )
    return parser.parse_args(argv)


@isolated_evaluation
def main(settings, args=None) -> None:
    args = args if args is not None else _parse_args()
    if args.rerank and settings.reranker_provider == "none":
        raise ValueError("--rerank requires RERANKER_PROVIDER=cross_encoder")
    if not args.rerank:
        settings = settings.model_copy(update={"reranker_provider": "none"})
    settings = settings.model_copy(update={"metadata_extraction_enabled": False})

    dataset = ir_datasets.load("beir/scifact/test")
    corpus_ids = set()
    for index, document in enumerate(dataset.docs_iter()):
        if args.max_docs is not None and index >= args.max_docs:
            break
        corpus_ids.add(document.doc_id)
    qrels = defaultdict(set)
    for relevance in dataset.qrels_iter():
        if relevance.relevance > 0:
            qrels[relevance.query_id].add(relevance.doc_id)
    queries = [query for query in dataset.queries_iter() if qrels[query.query_id] & corpus_ids]
    if not queries:
        raise ValueError("Selected corpus has no queries with relevant documents")

    container = build_container(settings)
    try:
        gateway_calls = defaultdict(int)
        original_post = container.gateway._post

        def embeddings_only(path, payload):
            if path != "/v1/embeddings":
                raise AssertionError("Non-LLM evaluation attempted a generation call")
            gateway_calls[path] += 1
            return original_post(path, payload)

        container.gateway._post = embeddings_only
        access = None
        ingest = {}
        if args.pipeline:
            tenant_id, owner, ingest = populate_scifact_pipeline(
                container, dataset, "scifact-eval-postgres-benchmark", args.max_docs, args.workers
            )
            access = access_predicate(Principal(tenant_id, owner, Role.ADMIN))
        else:
            tenant_id = populate_scifact(
                container, dataset, "scifact-eval-postgres-benchmark", max_docs=args.max_docs
            )
        print(f"corpus: {len(corpus_ids)} docs | queries: {len(queries)} | rerank: {args.rerank}")
        rows = []
        started = time.monotonic()
        for index, query in enumerate(queries, 1):
            query_started = time.monotonic()
            hits = _retrieve(container, tenant_id, query.text, args.fetch_k, access=access)
            candidates = len(hits)
            if args.rerank:
                hits, _ = _rerank(container, query.text, hits, args.fetch_k)
            ranked = _dedup_hit_docs(hits, max(K_VALUES))
            relevant = qrels[query.query_id]
            row = {
                "query_id": query.query_id,
                "question": query.text,
                "latency_ms": (time.monotonic() - query_started) * 1000,
                "candidates": candidates,
                "returned_chunks": len(hits),
                "ranked_doc_ids": json.dumps(ranked),
                "relevant_doc_ids": json.dumps(sorted(relevant)),
            }
            for k in K_VALUES:
                row[f"recall@{k}"] = _recall_at_k(ranked, relevant, k)
                row[f"ndcg@{k}"] = _ndcg_at_k(ranked, relevant, k)
            row["rr@10"] = _rr_at_10(ranked, relevant)
            rows.append(row)
            if index % 25 == 0 or index == len(queries):
                print(
                    f"Queried {index}/{len(queries)} in {time.monotonic() - started:.1f}s",
                    flush=True,
                )

        summary = {}
        for k in K_VALUES:
            for metric in (f"recall@{k}", f"ndcg@{k}"):
                summary[metric] = float(np.mean([row[metric] for row in rows]))
        summary["mrr@10"] = float(np.mean([row["rr@10"] for row in rows]))
        print(summary)
        summary.update(
            corpus_docs=len(corpus_ids),
            corpus_chunks=container.vectors.count(tenant_id),
            num_queries=len(queries),
            fetch_k=args.fetch_k,
            rerank=args.rerank,
            reranker_model=settings.reranker_model if args.rerank else "none",
            candidate_multiplier=settings.rerank_candidate_multiplier if args.rerank else 1,
            min_candidates=settings.rerank_min_candidates if args.rerank else args.fetch_k,
            embedding_profile_id=container.embedder.profile.id,
            embedding_model=container.embedder.profile.model,
            embedding_dimensions=container.embedder.dim,
            pipeline_ingestion=args.pipeline,
            generation_calls=0,
            embedding_requests=gateway_calls["/v1/embeddings"],
            latency_p50_ms=float(np.percentile([r["latency_ms"] for r in rows], 50)),
            latency_p95_ms=float(np.percentile([r["latency_ms"] for r in rows], 95)),
            latency_mean_ms=float(np.mean([r["latency_ms"] for r in rows])),
            empty_results=sum(row["returned_chunks"] == 0 for row in rows),
            tenant_id=tenant_id,
            query_seconds=time.monotonic() - started,
            run_at_utc=datetime.now(UTC).isoformat(),
        )
        summary.update(ingest)
        output = args.output or Path(
            "eval/retrieval_postgres_rerank_results.xlsx"
            if args.rerank
            else "eval/retrieval_postgres_results.xlsx"
        )
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            pd.DataFrame(summary.items(), columns=["metric", "value"]).to_excel(
                writer, sheet_name="summary", index=False
            )
            pd.DataFrame(rows).to_excel(writer, sheet_name="per_query", index=False)
        output.with_suffix(".json").write_text(
            json.dumps(
                {
                    "summary": summary,
                    "embedding_profile": asdict(container.embedder.profile),
                    "queries": rows,
                },
                indent=2,
            )
        )
        print(
            f"Results written to {output}; chunk deduplication can return fewer than fetch-k documents."
        )
    finally:
        container.close()


if __name__ == "__main__":
    main(_parse_args())
