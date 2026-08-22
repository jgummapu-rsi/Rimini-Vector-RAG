"""Composition root: build adapters from config (the flip point).

Change the *_BACKEND settings and only this file selects a different adapter;
the rest of the app depends on ports, not implementations.
"""
from __future__ import annotations

from dataclasses import dataclass

from typing import Optional

from app.adapters.embedders.gateway import GatewayEmbedder
from app.adapters.embedders.minilm import MiniLMEmbedder
from app.adapters.localfs.blob_store import LocalFsBlobStore
from app.adapters.localfs.vector_store import LocalFileVectorStore
from app.adapters.pgvector.vector_store import PgVectorStore
from app.adapters.postgres.metadata_store import PostgresMetadataStore
from app.adapters.postgres.metrics import PostgresMetrics
from app.adapters.postgres.queue import PostgresTaskQueue
from app.adapters.queue.sqlite_queue import SqliteTaskQueue
from app.adapters.rerankers.cross_encoder import CrossEncoderReranker
from app.adapters.sqlite.metadata_store import SqliteMetadataStore
from app.adapters.sqlite.metrics import SqliteMetrics
from app.config import Settings, settings
from app.gateway.client import LiteLLMClient
from app.ports.answer_cache import AnswerCache
from app.ports.blob_store import BlobStore
from app.ports.embedder import Embedder
from app.ports.metadata_store import MetadataStore
from app.ports.reranker import Reranker
from app.ports.task_queue import TaskQueue
from app.ports.vector_store import VectorStore


@dataclass
class Container:
    settings: Settings
    metadata: MetadataStore
    blob: BlobStore
    queue: TaskQueue
    vectors: VectorStore
    gateway: LiteLLMClient
    embedder: Embedder
    metrics: "SqliteMetrics | PostgresMetrics"
    reranker: Optional[Reranker] = None
    cache: Optional[AnswerCache] = None


def build_container(cfg: Settings = settings, embedder: Optional[Embedder] = None) -> Container:
    """Compose adapters. `embedder` can be injected (tests use a fast fake); when
    None it is selected from EMBEDDING_PROVIDER."""
    cfg.ensure_dirs()

    if cfg.metadata_backend == "sqlite":
        metadata: MetadataStore = SqliteMetadataStore(cfg.sqlite_path)
    elif cfg.metadata_backend == "postgres":
        metadata = PostgresMetadataStore(cfg.postgres_dsn)
    else:
        raise ValueError(f"unsupported METADATA_BACKEND={cfg.metadata_backend}")

    if cfg.blob_backend == "localfs":
        blob: BlobStore = LocalFsBlobStore(cfg.blob_dir)
    else:
        raise ValueError(f"unsupported BLOB_BACKEND={cfg.blob_backend}")

    if cfg.queue_backend == "sqlite":
        queue: TaskQueue = SqliteTaskQueue(cfg.sqlite_path)
    elif cfg.queue_backend == "postgres":
        queue = PostgresTaskQueue(cfg.postgres_dsn)
    else:
        raise ValueError(f"unsupported QUEUE_BACKEND={cfg.queue_backend}")

    gateway = LiteLLMClient(
        base_url=cfg.litellm_base_url,
        api_key=cfg.litellm_api_key,
        vision_model=cfg.vision_model,
        embedding_model=cfg.embedding_model,
    )

    # embedder selection (float embeddings; binarization deferred)
    if embedder is None:
        if cfg.embedding_provider == "minilm":
            embedder = MiniLMEmbedder()
        elif cfg.embedding_provider == "gateway":
            embedder = GatewayEmbedder(gateway, cfg.embedding_dim, cfg.embed_batch_size)
        else:
            raise ValueError(f"unsupported EMBEDDING_PROVIDER={cfg.embedding_provider}")

    if cfg.vector_backend == "localfile":
        vectors: VectorStore = LocalFileVectorStore(
            cfg.vector_dir, cfg.sqlite_path, embedder.dim
        )
    elif cfg.vector_backend == "pgvector":
        vectors = PgVectorStore(
            cfg.postgres_dsn, embedder.dim,
            candidate_multiplier=cfg.pgvector_candidate_multiplier,
            min_candidates=cfg.pgvector_min_candidates,
            hnsw_m=cfg.pgvector_hnsw_m,
            hnsw_ef_construction=cfg.pgvector_hnsw_ef_construction,
            hnsw_ef_search=cfg.pgvector_hnsw_ef_search,
            lexical_only_cap=cfg.pgvector_lexical_only_cap,
        )
    else:
        raise ValueError(f"unsupported VECTOR_BACKEND={cfg.vector_backend}")

    metadata.init_schema()
    vectors.ensure_collection(embedder.dim)
    # metrics share metadata's database (no separate METRICS_BACKEND knob)
    metrics = (
        PostgresMetrics(cfg.postgres_dsn) if cfg.metadata_backend == "postgres"
        else SqliteMetrics(cfg.sqlite_path)
    )

    # reranker selection (no rerank model on the gateway, so "none" is the only
    # other option today -- local cross-encoder or skip reranking entirely)
    if cfg.reranker_provider == "none":
        reranker: Optional[Reranker] = None
    elif cfg.reranker_provider == "cross_encoder":
        reranker = CrossEncoderReranker(cfg.reranker_model)
    else:
        raise ValueError(f"unsupported RERANKER_PROVIDER={cfg.reranker_provider}")

    # semantic answer cache (optional). Import + `redis` dependency are lazy: with
    # no REDIS_URL the adapter module is never imported, so redis stays optional.
    cache: Optional[AnswerCache] = None
    if cfg.redis_url:
        from app.adapters.cache.redis_stack import RedisStackAnswerCache
        cache = RedisStackAnswerCache(
            cfg.redis_url, cfg.cache_index_name, embedder.dim,
            cfg.cache_similarity_threshold, cfg.cache_ttl_seconds,
        )
        cache.init_index()

    return Container(
        settings=cfg, metadata=metadata, blob=blob,
        queue=queue, vectors=vectors, gateway=gateway, embedder=embedder,
        metrics=metrics, reranker=reranker, cache=cache,
    )