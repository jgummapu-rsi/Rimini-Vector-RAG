from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every configurable value for both the ingest and retrieval flows, read from
    the environment / `.env` (see `pydantic_settings.BaseSettings`).

    Field groups and the non-obvious rationale behind their defaults:

    - `pgvector_lexical_only_cap`: bounds how many lexical(FTS)-only candidates
      (not found by dense search) a lexical-leaning query's fusion can inject,
      so lexical can't overly perturb a dense-covered ranking.
    - `platform_tenant_id`: only an admin of this one tenant may publish
      `scope=global` documents (visible to every tenant); empty means no
      tenant can publish globally.
    - `vision_model` does OCR (image/scanned-page transcription); `chat_model`
      generates RAG answers at query time; `embedding_model` is only read
      when `embedding_provider=gateway`.
    - `embedding_provider`: `gateway` calls LiteLLM (the default is
      text-embedding-3-large at 1536 dimensions); `minilm` uses local
      all-MiniLM-L6-v2 (384-dim). Gateway mode requires an
      embedding model configured there.
    - `reranker_provider`: `cross_encoder` is a real local ONNX cross-encoder
      second pass over the fused dense+BM25 top-k (local-only, no rerank
      model on the gateway); `none` skips reranking.
    - `rerank_candidate_multiplier`/`rerank_min_candidates`: the pool fed to
      the reranker is `requested_k * multiplier`, floored at `min_candidates`
      (same "fetch wider, then narrow" shape as `pgvector_candidate_multiplier`).
      `min_candidates=50` (raised from 20) because on SciFact the gold doc's
      chunk sits in the deduped top-20 pool only ~91% of the time but in the
      top-50 pool ~95% — reranking can only reorder what retrieval fetched, so
       pool depth affects candidate availability.
    - `rerank_min_score`: optional absolute floor on the cross-encoder's raw
      relevance score (not a probability). Disabled by default: the previous
      -3.0 floor rejected valid scientific evidence. With None, retain up to
      `top_k` available candidates in reranked order, including negative scores.
    - `chunk_auto_size=True` (default): chunk sizes are derived from the
      *active* embedder's real token limit (`Embedder.max_tokens`, via
      `ChunkSpec.auto`), so switching embedders resizes chunks automatically —
      MiniLM (256 tokens) yields 180/20/220 (identical to the explicit
      `chunk_*_tokens` fields below), bge-base (512 tokens) yields ~390/43/476
      instead of being needlessly fragmented at 256. The explicit fields are
      only used when `chunk_auto_size=False`. Both paths measure against the
      embedder's real WordPiece tokenizer, never a generic BPE proxy, which
      undercounts and silently truncates (see `app.ingest.pipeline.tokens`).
    - `cache_similarity_threshold`: a cached answer is served when the new
      question's embedding is at least this cosine-similar to the stored
      question (1.0 = exact only, lower = fuzzier).
    - `cache_ttl_seconds`: per-entry TTL as a staleness backstop; the
      generation-counter invalidation is the primary freshness mechanism.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @classmethod
    def settings_customise_sources(
        cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
    ):
        def mounted_secrets():
            values = {}
            for name in ("database_url", "redis_url", "litellm_api_key"):
                path = os.environ.get(name.upper() + "_FILE")
                if path:
                    value = Path(path).read_text().strip()
                    if not value:
                        raise ValueError(f"Mounted {name} secret is empty")
                    values[name] = value
            return values

        return init_settings, mounted_secrets, env_settings, dotenv_settings, file_secret_settings

    app_env: str = "production"

    data_dir: Path = Path("data")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/rag_ingestion"
    postgres_pool_min: int = 1
    postgres_pool_max: int = 10
    initialize_schema: bool = True

    pgvector_hnsw_m: int = 16
    pgvector_hnsw_ef_construction: int = 64
    pgvector_hnsw_ef_search: int = 100
    pgvector_candidate_multiplier: int = 4
    pgvector_min_candidates: int = 50
    pgvector_lexical_only_cap: int = 10

    platform_tenant_id: str = ""
    operator_user_ids: list[str] = []
    gateway_allowed_origins: list[str] = []

    litellm_base_url: str = ""
    litellm_api_key: str = ""
    vision_model: str = "gpt-6-sol"
    chat_model: str = "gpt-6-sol"
    allowed_chat_models: list[str] = []
    request_global_concurrency: int = Field(default=32, ge=1)
    request_timeout_seconds: float = Field(default=90.0, gt=0, le=110, allow_inf_nan=False)
    request_tenant_concurrency: int = Field(default=8, ge=1)
    request_user_concurrency: int = Field(default=2, ge=1)
    request_tenant_per_minute: int = Field(default=300, ge=1)
    request_user_per_minute: int = Field(default=60, ge=1)
    embedding_model: str = "text-embedding-3-large"
    embedding_revision: str = "text-embedding-3-large"
    embedding_dim: int = 1536

    embedding_provider: str = "gateway"
    embed_batch_size: int = 64
    metadata_extraction_enabled: bool = True

    reranker_provider: str = "cross_encoder"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_candidate_multiplier: int = 4
    rerank_min_candidates: int = 50
    rerank_min_score: float | None = None

    chunk_auto_size: bool = True
    chunk_target_tokens: int = 180
    chunk_overlap_tokens: int = 20
    chunk_max_tokens: int = 220
    chunk_min_tokens: int = 16

    max_upload_mb: int = 50
    max_attempts: int = 5
    worker_poll_seconds: float = 1.0

    max_pdf_pages: int = 500
    layout_enabled: bool = False
    layout_model_path: Path = Path(
        "models/doclayout-yolo/doclayout_yolo_docstructbench_imgsz1024.pt"
    )
    layout_device: str = "cpu"
    layout_image_size: int = Field(default=1024, ge=320, le=2048)
    layout_confidence: float = Field(default=0.2, ge=0, le=1, allow_inf_nan=False)
    max_image_pixels: int = 40_000_000
    parse_timeout_seconds: float = Field(default=120.0, gt=0, allow_inf_nan=False)
    parse_memory_mb: int = Field(default=2048, gt=0)
    max_zip_entries: int = Field(default=10000, gt=0)
    max_zip_expanded_bytes: int = Field(default=256 * 1024 * 1024, gt=0)
    max_zip_expansion_ratio: int = Field(default=200, gt=0)
    max_extracted_bytes: int = Field(default=32 * 1024 * 1024, gt=0)
    max_table_rows: int = Field(default=200000, gt=0)
    max_workbook_cells: int = Field(default=2000000, gt=0)
    max_workbook_sheets: int = Field(default=256, gt=0)
    max_vision_calls: int = Field(default=500, gt=0)
    max_vision_tokens: int = Field(default=512000, gt=0)
    vision_max_tokens: int = Field(default=8192, gt=0)
    vision_timeout_seconds: float = Field(default=90.0, gt=0, allow_inf_nan=False)

    job_lease_seconds: int = 300

    redis_url: str = ""
    cache_index_name: str = "ans_idx"
    cache_similarity_threshold: float = 0.95
    cache_ttl_seconds: int = 86400

    login_rate_limit_per_email: int = 10
    login_rate_limit_per_ip: int = 30
    login_rate_limit_window_seconds: float = 900.0
    register_rate_limit_per_ip: int = 10
    register_rate_limit_window_seconds: float = 3600.0

    @property
    def blob_dir(self) -> Path:
        """Directory holding locally-stored blobs under `data_dir`."""
        return self.data_dir / "blobs"

    @property
    def postgres_dsn(self) -> str:
        """Alias for `database_url`, named for readability at Postgres call sites."""
        return self.database_url

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)

    @model_validator(mode="before")
    @classmethod
    def reject_retired_backends(cls, values):
        retired = {"metadata_backend", "vector_backend", "queue_backend", "blob_backend"}
        if any(key.lower() in retired for key in values):
            raise ValueError(
                "Backend selection has been removed; configure DATABASE_URL and REDIS_URL"
            )
        if any(key.lower() in retired for key in os.environ):
            raise ValueError("Remove retired backend environment variables")
        return values


settings = Settings()
