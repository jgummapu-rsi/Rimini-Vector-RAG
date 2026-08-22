"""Application configuration.

`Settings` reads from environment / `.env`. The *_BACKEND fields are the single
place we switch local <-> production adapters (the "flip"). Nothing else in the
code needs to change when we move to Postgres/Qdrant/Azure Blob/Celery.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "local"

    metadata_backend: str = "sqlite"     # sqlite | postgres
    blob_backend: str = "localfs"        # localfs | azure_blob (later)
    vector_backend: str = "localfile"    # localfile | pgvector
    queue_backend: str = "sqlite"        # sqlite | postgres

    data_dir: Path = Path("data")

    # --- Postgres (metadata_backend/queue_backend=postgres, vector_backend=pgvector) ---
    database_url: str = "postgresql://postgres:postgres@localhost:5432/rag_ingestion"
    postgres_pool_min: int = 1
    postgres_pool_max: int = 10

    # --- pgvector tuning (vector_backend=pgvector only) ---
    pgvector_hnsw_m: int = 16
    pgvector_hnsw_ef_construction: int = 64
    pgvector_hnsw_ef_search: int = 100
    pgvector_candidate_multiplier: int = 4   # candidate pool = top_k * this, floored below
    pgvector_min_candidates: int = 50
    # Max lexical-only (FTS) candidates injected into fusion, and only for
    # lexical-leaning queries (see PgVectorStore.search's gate). Bounds how much
    # the lexical source can perturb a dense-covered ranking.
    pgvector_lexical_only_cap: int = 10

    # Only an admin of this tenant may publish scope=global documents (visible to
    # every tenant). Empty = no tenant can publish globally.
    platform_tenant_id: str = ""

    litellm_base_url: str = ""
    litellm_api_key: str = ""
    # Vision model used for image/scanned-page transcription (OCR done by the LLM).
    vision_model: str = "claude-sonnet-5"
    # Chat model used for RAG answer generation at query time.
    chat_model: str = "gpt-5-nano"
    embedding_model: str = ""       # only used when embedding_provider=gateway
    embedding_dim: int = 384

    # minilm  = real local ONNX all-MiniLM-L6-v2 (384-dim), no gateway needed.
    # gateway = LiteLLM embeddings (requires an embedding model on the gateway).
    embedding_provider: str = "minilm"
    embed_batch_size: int = 64

    # cross_encoder = real local ONNX cross-encoder second pass over the fused
    # dense+BM25 top-k (no rerank model on the gateway, so this is local-only,
    # same reasoning as embeddings). none = skip reranking (today's behavior).
    reranker_provider: str = "cross_encoder"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # candidate pool fed to the reranker = requested_k * this, floored below --
    # same "fetch wider, then narrow" shape as pgvector_candidate_multiplier.
    # min_candidates raised 20 -> 50 (Tier 1): measured on SciFact, the gold
    # doc's chunk sits in the deduped top-20 pool only ~91% of the time but in
    # the top-50 pool ~95% -- reranking can only reorder what retrieval fetched,
    # so a deeper pool is the dominant recall@10 lever (see docs/RECALL_TIER1_TIER2_REPORT.md).
    rerank_candidate_multiplier: int = 4
    rerank_min_candidates: int = 50

    # chunk_auto_size=True (default): derive chunk sizes from the ACTIVE
    # embedder's real token limit (Embedder.max_tokens) via ChunkSpec.auto, so
    # flipping the embedder resizes chunks automatically -- MiniLM (256) yields
    # 180/20/220 (identical to the hand-tuned values below), bge-base (512)
    # yields ~390/43/476 instead of being needlessly fragmented at 256. The
    # explicit values below are used only when chunk_auto_size=False (pin sizes).
    # Both are measured against the embedder's REAL WordPiece tokenizer, never a
    # generic BPE proxy (which undercounts and silently truncates -- the bug this
    # whole path fixes; see app.pipeline.tokens).
    chunk_auto_size: bool = True
    chunk_target_tokens: int = 180
    chunk_overlap_tokens: int = 20
    chunk_max_tokens: int = 220
    chunk_min_tokens: int = 16

    max_upload_mb: int = 50
    max_attempts: int = 5
    worker_poll_seconds: float = 1.0

    # --- semantic answer cache (Redis Stack, optional) ---
    # Empty redis_url = cache disabled (container.cache is None; the query path
    # behaves exactly as if the cache didn't exist). Set e.g.
    # redis://localhost:6379 to enable. Requires a Redis with the RediSearch
    # module (Redis Stack, or Redis 8+ which bundles it) for vector KNN.
    redis_url: str = ""
    cache_index_name: str = "ans_idx"
    # A cached answer is served when the new question's embedding is at least this
    # cosine-similar to the stored question. 1.0 = exact only; lower = fuzzier.
    cache_similarity_threshold: float = 0.95
    # Per-entry TTL (staleness backstop; generation-counter invalidation is the
    # primary freshness mechanism). Default 24h.
    cache_ttl_seconds: int = 86400

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def blob_dir(self) -> Path:
        return self.data_dir / "blobs"

    @property
    def vector_dir(self) -> Path:
        return self.data_dir / "vectors"

    @property
    def postgres_dsn(self) -> str:
        return self.database_url

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.blob_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
