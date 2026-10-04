from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from app.ingest.adapters.localfs.blob_store import LocalFsBlobStore
from app.ingest.adapters.publication import PostgresPublicationStore
from app.ingest.adapters.queue.postgres import PostgresTaskQueue
from app.ingest.ports.blob_store import BlobStore
from app.ingest.ports.publication import PublicationStore
from app.ingest.ports.task_queue import TaskQueue
from app.retrieval.adapters.cache.redis_stack import RedisStackAnswerCache
from app.retrieval.adapters.rerankers.cross_encoder import CrossEncoderReranker
from app.retrieval.ports.answer_cache import AnswerCache
from app.retrieval.ports.reranker import Reranker
from app.retrieval.rag.grounding import pack_evidence
from app.shared.adapters.embedders.gateway import GatewayEmbedder
from app.shared.adapters.embedders.minilm import MiniLMEmbedder
from app.shared.adapters.pgvector.profile import check_profile
from app.shared.adapters.pgvector.vector_store import PgVectorStore
from app.shared.adapters.postgres.db import close_pool, get_pool, transaction
from app.shared.adapters.postgres.metadata_store import PostgresMetadataStore
from app.shared.adapters.postgres.metrics import PostgresMetrics
from app.shared.adapters.request_gate import RedisRequestGate
from app.shared.config import Settings, settings
from app.shared.gateway.client import LiteLLMClient
from app.shared.ports.embedder import Embedder
from app.shared.ports.metadata_store import MetadataStore
from app.shared.ports.request_gate import RequestGate
from app.shared.ports.vector_store import VectorStore
from app.shared.rate_limit import RateLimiter

log = logging.getLogger(__name__)


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
    metrics: PostgresMetrics
    publication: PublicationStore
    request_gate: RequestGate
    readiness_check: Callable[[], None]
    reranker: Reranker | None = None
    cache: AnswerCache | None = None

    login_email_limiter: RateLimiter | None = None
    login_ip_limiter: RateLimiter | None = None
    register_ip_limiter: RateLimiter | None = None

    def close(self) -> None:
        try:
            if self.cache is not None:
                self.cache.close()
        finally:
            try:
                self.request_gate.close()
            finally:
                close_pool(self.settings.postgres_dsn)


def build_container(cfg: Settings = settings, embedder: Embedder | None = None) -> Container:
    """Compose adapters. `embedder` can be injected (tests use a fast fake); when
    None it is selected from EMBEDDING_PROVIDER."""
    if not cfg.redis_url:
        raise ValueError("REDIS_URL is required")
    if not 1 <= cfg.postgres_pool_min <= cfg.postgres_pool_max:
        raise ValueError("Postgres pool limits must satisfy 1 <= min <= max")
    cfg.ensure_dirs()
    get_pool(cfg.postgres_dsn, cfg.postgres_pool_min, cfg.postgres_pool_max)

    metadata = PostgresMetadataStore(cfg.postgres_dsn)
    blob = LocalFsBlobStore(cfg.blob_dir)
    queue = PostgresTaskQueue(cfg.postgres_dsn, lease_seconds=cfg.job_lease_seconds)

    gateway = LiteLLMClient(
        base_url=cfg.litellm_base_url,
        api_key=cfg.litellm_api_key,
        vision_model=cfg.vision_model,
        embedding_model=cfg.embedding_model,
        config_provider=lambda: metadata.get_gateway_config() or ("", ""),
    )

    if embedder is None:
        if cfg.embedding_provider == "minilm":
            embedder = MiniLMEmbedder(batch_size=cfg.embed_batch_size)
        elif cfg.embedding_provider == "gateway":
            embedder = GatewayEmbedder(
                gateway, cfg.embedding_dim, cfg.embed_batch_size, revision=cfg.embedding_revision
            )
        else:
            raise ValueError(f"unsupported EMBEDDING_PROVIDER={cfg.embedding_provider}")

    vectors = PgVectorStore(
        cfg.postgres_dsn,
        embedder.dim,
        candidate_multiplier=cfg.pgvector_candidate_multiplier,
        min_candidates=cfg.pgvector_min_candidates,
        hnsw_m=cfg.pgvector_hnsw_m,
        hnsw_ef_construction=cfg.pgvector_hnsw_ef_construction,
        hnsw_ef_search=cfg.pgvector_hnsw_ef_search,
        lexical_only_cap=cfg.pgvector_lexical_only_cap,
        profile=embedder.profile,
    )

    if cfg.initialize_schema:
        metadata.init_schema()
        vectors.ensure_collection(embedder.dim)
    else:
        with transaction(cfg.postgres_dsn) as cur:
            check_profile(cur, embedder.profile.id)
            cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
            extension = cur.fetchone()
            if extension is None or tuple(map(int, extension["extversion"].split("."))) < (0, 8, 0):
                raise RuntimeError("pgvector >= 0.8.0 is required")
            cur.execute(
                "SELECT atttypmod AS dim FROM pg_attribute WHERE attrelid='vector_chunks'::regclass AND attname='embedding'"
            )
            if cur.fetchone()["dim"] != embedder.dim:
                raise ValueError("Vector dimension differs from configured embedding profile")
    log.info(
        "Embedding profile validated",
        extra={
            "event": "embedding_profile_validated",
            "profile_id": embedder.profile.id,
            "dimensions": embedder.dim,
        },
    )

    metrics = PostgresMetrics(cfg.postgres_dsn)

    if cfg.reranker_provider == "none":
        reranker: Reranker | None = None
    elif cfg.reranker_provider == "cross_encoder":
        reranker = CrossEncoderReranker(cfg.reranker_model)
    else:
        raise ValueError(f"unsupported RERANKER_PROVIDER={cfg.reranker_provider}")

    cache = RedisStackAnswerCache(
        cfg.redis_url,
        cfg.cache_index_name,
        embedder.dim,
        cfg.cache_similarity_threshold,
        cfg.cache_ttl_seconds,
    )
    cache.init_index()
    embedder.prepare()
    if reranker is not None:
        reranker.prepare()
    pack_evidence("startup validation", ["startup validation"], cfg.chat_model)

    def readiness_check() -> None:
        with transaction(cfg.postgres_dsn) as cur:
            check_profile(cur, embedder.profile.id)
            cur.execute("SELECT to_regclass('ingestion_jobs') AS queue")
            if cur.fetchone()["queue"] is None:
                raise RuntimeError("Ingestion queue schema is missing")
        cache.check_ready()

    return Container(
        settings=cfg,
        metadata=metadata,
        blob=blob,
        queue=queue,
        vectors=vectors,
        gateway=gateway,
        embedder=embedder,
        metrics=metrics,
        publication=PostgresPublicationStore(cfg.postgres_dsn, embedder.profile),
        request_gate=RedisRequestGate(
            cfg.redis_url,
            cfg.cache_index_name,
            global_concurrency=cfg.request_global_concurrency,
            tenant_concurrency=cfg.request_tenant_concurrency,
            user_concurrency=cfg.request_user_concurrency,
            tenant_per_minute=cfg.request_tenant_per_minute,
            user_per_minute=cfg.request_user_per_minute,
        ),
        readiness_check=readiness_check,
        reranker=reranker,
        cache=cache,
        login_email_limiter=RateLimiter(
            cfg.login_rate_limit_per_email, cfg.login_rate_limit_window_seconds
        ),
        login_ip_limiter=RateLimiter(
            cfg.login_rate_limit_per_ip, cfg.login_rate_limit_window_seconds
        ),
        register_ip_limiter=RateLimiter(
            cfg.register_rate_limit_per_ip, cfg.register_rate_limit_window_seconds
        ),
    )
