"""Application configuration.

`Settings` reads from environment / `.env`. The *_BACKEND fields are the single
place we switch local <-> production adapters (the "flip"). Nothing else in the
code needs to change when we move to Postgres/Qdrant/Azure Blob/Celery.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every configurable value for both the ingest and retrieval flows, read from
    the environment / `.env` (see `pydantic_settings.BaseSettings`).

    Field groups and the non-obvious rationale behind their defaults:

    - `*_backend`: the single flip points between local and production adapters
      (metadata/blob/vector/queue), each `sqlite`/`localfs`/`localfile`/`sqlite`
      by default, switching to `postgres`/`azure_blob`(later)/`pgvector`/`postgres`.
    - `pgvector_lexical_only_cap`: bounds how many lexical(FTS)-only candidates
      (not found by dense search) a lexical-leaning query's fusion can inject,
      so lexical can't overly perturb a dense-covered ranking.
    - `platform_tenant_id`: only an admin of this one tenant may publish
      `scope=global` documents (visible to every tenant); empty means no
      tenant can publish globally.
    - `vision_model` does OCR (image/scanned-page transcription); `chat_model`
      generates RAG answers at query time; `embedding_model` is only read
      when `embedding_provider=gateway`.
    - `embedding_provider`: `minilm` is a real local ONNX all-MiniLM-L6-v2
      (384-dim, no gateway needed); `gateway` calls LiteLLM (requires an
      embedding model configured there).
    - `reranker_provider`: `cross_encoder` is a real local ONNX cross-encoder
      second pass over the fused dense+BM25 top-k (local-only, no rerank
      model on the gateway); `none` skips reranking.
    - `rerank_candidate_multiplier`/`rerank_min_candidates`: the pool fed to
      the reranker is `requested_k * multiplier`, floored at `min_candidates`
      (same "fetch wider, then narrow" shape as `pgvector_candidate_multiplier`).
      `min_candidates=50` (raised from 20) because on SciFact the gold doc's
      chunk sits in the deduped top-20 pool only ~91% of the time but in the
      top-50 pool ~95% — reranking can only reorder what retrieval fetched, so
      pool depth is the dominant recall@10 lever (see `docs/BENCHMARKS.md`).
    - `rerank_min_score`: an absolute floor on the cross-encoder's raw logit
      score (unbounded, not a 0-1 probability) below which a hit is dropped
      rather than padded into context just to fill `top_k`. Calibrated
      against `ms-marco-MiniLM-L-6-v2` on a real query: a genuinely relevant
      hit scored 3.4 while every irrelevant hit in the same pool clustered at
      -11.3..-11.45 — `-3.0` sits well clear of that noise floor. `None`
      disables the floor (always return exactly `top_k`).
    - `chunk_auto_size=True` (default): chunk sizes are derived from the
      *active* embedder's real token limit (`Embedder.max_tokens`, via
      `ChunkSpec.auto`), so switching embedders resizes chunks automatically —
      MiniLM (256 tokens) yields 180/20/220 (identical to the explicit
      `chunk_*_tokens` fields below), bge-base (512 tokens) yields ~390/43/476
      instead of being needlessly fragmented at 256. The explicit fields are
      only used when `chunk_auto_size=False`. Both paths measure against the
      embedder's real WordPiece tokenizer, never a generic BPE proxy, which
      undercounts and silently truncates (see `app.ingest.pipeline.tokens`).
    - `redis_url=""` (default) disables the semantic answer cache entirely —
      `container.cache` stays `None` and the query path behaves as if the
      cache didn't exist. A non-empty URL (e.g. `redis://localhost:6379`)
      requires a Redis with the RediSearch module (Redis Stack, or Redis 8+
      which bundles it) for vector KNN.
    - `cache_similarity_threshold`: a cached answer is served when the new
      question's embedding is at least this cosine-similar to the stored
      question (1.0 = exact only, lower = fuzzier).
    - `cache_ttl_seconds`: per-entry TTL as a staleness backstop; the
      generation-counter invalidation is the primary freshness mechanism.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "local"

    metadata_backend: str = "sqlite"
    blob_backend: str = "localfs"
    vector_backend: str = "localfile"
    queue_backend: str = "sqlite"

    data_dir: Path = Path("data")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/rag_ingestion"
    postgres_pool_min: int = 1
    postgres_pool_max: int = 10

    pgvector_hnsw_m: int = 16
    pgvector_hnsw_ef_construction: int = 64
    pgvector_hnsw_ef_search: int = 100
    pgvector_candidate_multiplier: int = 4
    pgvector_min_candidates: int = 50
    pgvector_lexical_only_cap: int = 10

    platform_tenant_id: str = ""

    litellm_base_url: str = ""
    litellm_api_key: str = ""
    vision_model: str = "claude-sonnet-5"
    chat_model: str = "gpt-5-nano"
    embedding_model: str = ""
    embedding_dim: int = 384

    embedding_provider: str = "minilm"
    embed_batch_size: int = 64

    reranker_provider: str = "cross_encoder"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_candidate_multiplier: int = 4
    rerank_min_candidates: int = 50
    rerank_min_score: Optional[float] = -3.0

    chunk_auto_size: bool = True
    chunk_target_tokens: int = 180
    chunk_overlap_tokens: int = 20
    chunk_max_tokens: int = 220
    chunk_min_tokens: int = 16

    max_upload_mb: int = 50
    max_attempts: int = 5
    worker_poll_seconds: float = 1.0

    # File-safety limits: bound worst-case CPU/memory on a malicious or
    # malformed upload, independent of MAX_UPLOAD_MB (a small file can still
    # decompress/render into something huge -- a PDF-bomb or a tiny image with
    # an enormous reported resolution).
    max_pdf_pages: int = 500
    max_image_pixels: int = 40_000_000        # ~40 megapixels
    max_table_rows: int = 200_000
    parse_timeout_seconds: float = 120.0
    # How long a claimed job may stay `running` before the reaper reclaims it
    # (worker crash/kill mid-job). Must comfortably exceed the slowest real
    # pipeline run (large-document OCR/embedding), or the reaper will requeue
    # jobs that are still legitimately in progress.
    job_lease_seconds: int = 300

    redis_url: str = ""
    cache_index_name: str = "ans_idx"
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 86400

    # Abuse guards on the unauthenticated onboarding endpoints (app.shared.rate_limit).
    # Per-email is stricter than per-IP: it bounds credential-stuffing against ONE
    # account regardless of how many source IPs an attacker spreads it across;
    # per-IP additionally bounds a single source hammering many different emails.
    login_rate_limit_per_email: int = 10
    login_rate_limit_per_ip: int = 30
    login_rate_limit_window_seconds: float = 900.0     # 15 minutes
    register_rate_limit_per_ip: int = 10
    register_rate_limit_window_seconds: float = 3600.0  # 1 hour

    @property
    def sqlite_path(self) -> Path:
        """Path to the local SQLite database file under `data_dir`."""
        return self.data_dir / "app.db"

    @property
    def blob_dir(self) -> Path:
        """Directory holding locally-stored blobs under `data_dir`."""
        return self.data_dir / "blobs"

    @property
    def vector_dir(self) -> Path:
        """Directory holding local vector-store files under `data_dir`."""
        return self.data_dir / "vectors"

    @property
    def postgres_dsn(self) -> str:
        """Alias for `database_url`, named for readability at Postgres call sites."""
        return self.database_url

    def ensure_dirs(self) -> None:
        """Create `data_dir`, `blob_dir`, and `vector_dir` if they don't already exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
