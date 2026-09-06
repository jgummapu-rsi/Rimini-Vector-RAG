"""Composition root: build adapters from config (the flip point).

Change the *_BACKEND settings and only this file selects a different adapter;
the rest of the app depends on ports, not implementations.
"""
from __future__ import annotations

from dataclasses import dataclass

from typing import Optional

from app.shared.adapters.embedders.gateway import GatewayEmbedder
from app.shared.adapters.embedders.minilm import MiniLMEmbedder
from app.ingest.adapters.localfs.blob_store import LocalFsBlobStore
from app.shared.adapters.localfs.vector_store import LocalFileVectorStore
from app.shared.adapters.pgvector.vector_store import PgVectorStore
from app.shared.adapters.postgres.metadata_store import PostgresMetadataStore
from app.shared.adapters.postgres.metrics import PostgresMetrics
from app.ingest.adapters.queue.postgres import PostgresTaskQueue
from app.ingest.adapters.queue.sqlite import SqliteTaskQueue
from app.retrieval.adapters.rerankers.cross_encoder import CrossEncoderReranker
from app.shared.adapters.sqlite.metadata_store import SqliteMetadataStore
from app.shared.adapters.sqlite.metrics import SqliteMetrics
from app.shared.config import Settings, settings
from app.shared.gateway.client import LiteLLMClient
from app.shared.rate_limit import RateLimiter
from app.retrieval.ports.answer_cache import AnswerCache
from app.ingest.ports.blob_store import BlobStore
from app.shared.ports.embedder import Embedder
from app.shared.ports.metadata_store import MetadataStore
from app.retrieval.ports.reranker import Reranker
from app.ingest.ports.task_queue import TaskQueue
from app.shared.ports.vector_store import VectorStore


@dataclass
class Container:
    """The wired set of adapters + settings a request or job runs against."""
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
    # Abuse guards for app.api.onboarding_routes (see app.shared.rate_limit).
    # Always set by build_container; typed Optional only so these fields can
    # follow the ones above that already carry a default (dataclass field-
    # ordering rule), not because a Container is ever meant to have none.
    login_email_limiter: Optional[RateLimiter] = None
    login_ip_limiter: Optional[RateLimiter] = None
    register_ip_limiter: Optional[RateLimiter] = None


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
        queue: TaskQueue = SqliteTaskQueue(cfg.sqlite_path, lease_seconds=cfg.job_lease_seconds)
    elif cfg.queue_backend == "postgres":
        queue = PostgresTaskQueue(cfg.postgres_dsn, lease_seconds=cfg.job_lease_seconds)
    else:
        raise ValueError(f"unsupported QUEUE_BACKEND={cfg.queue_backend}")

    gateway = LiteLLMClient(
        base_url=cfg.litellm_base_url,
        api_key=cfg.litellm_api_key,
        vision_model=cfg.vision_model,
        embedding_model=cfg.embedding_model,
        # Lets POST /onboarding/gateway-config override the gateway's
        # base_url/api_key live, no restart -- see LiteLLMClient._resolve_config.
        # Only container.py may reference a concrete adapter, so the closure
        # (not app.shared.gateway.client) is what knows `metadata` is one.
        config_provider=lambda: (
            metadata.get_system_config("litellm_base_url") or "",
            metadata.get_system_config("litellm_api_key") or "",
        ),
    )

    # embedder selection (float embeddings; binarization deferred).
    # embed_batch_size applies to BOTH providers: for the gateway it bounds
    # request size, for the local ONNX model it bounds peak memory per forward
    # pass (a whole document's chunks arrive in one embed() call).
    if embedder is None:
        if cfg.embedding_provider == "minilm":
            embedder = MiniLMEmbedder(batch_size=cfg.embed_batch_size)
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
        from app.retrieval.adapters.cache.redis_stack import RedisStackAnswerCache
        cache = RedisStackAnswerCache(
            cfg.redis_url, cfg.cache_index_name, embedder.dim,
            cfg.cache_similarity_threshold, cfg.cache_ttl_seconds,
        )
        cache.init_index()

    return Container(
        settings=cfg, metadata=metadata, blob=blob,
        queue=queue, vectors=vectors, gateway=gateway, embedder=embedder,
        metrics=metrics, reranker=reranker, cache=cache,
        login_email_limiter=RateLimiter(
            cfg.login_rate_limit_per_email, cfg.login_rate_limit_window_seconds),
        login_ip_limiter=RateLimiter(
            cfg.login_rate_limit_per_ip, cfg.login_rate_limit_window_seconds),
        register_ip_limiter=RateLimiter(
            cfg.register_rate_limit_per_ip, cfg.register_rate_limit_window_seconds),
    )