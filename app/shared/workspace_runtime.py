"""Workspace-scoped embedding adapters, initialized only after selection."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from threading import Lock

from psycopg2.extras import Json

from app.ingest.adapters.publication import PostgresPublicationStore
from app.retrieval.adapters.cache.redis_stack import RedisStackAnswerCache
from app.shared.adapters.embedders.gateway import GatewayEmbedder
from app.shared.adapters.embedders.onnx_embedder import OnnxEmbedder
from app.shared.adapters.pgvector.vector_store import PgVectorStore
from app.shared.adapters.postgres.db import transaction
from app.shared.execution import RequestAborted, current_budget
from app.shared.gateway.client import LiteLLMClient
from app.shared.model_catalog import GATEWAY_EMBEDDINGS, LOCAL_EMBEDDINGS

log = logging.getLogger(__name__)


class WorkspaceRuntime:
    def __init__(self, root):
        self.root = root
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embedding-loader")
        self._lock = Lock()
        self._futures = {}

    def prepare(self, selection: str):
        with self._lock:
            future = self._futures.get(selection)
            if future is None or (future.done() and future.exception() is not None):
                future = self._executor.submit(self._build, selection)
                self._futures[selection] = future
            return future

    def for_tenant(self, tenant_id: str):
        models = self.root.metadata.get_workspace_models(tenant_id)
        if not models:
            return self.root
        selected = models["embedding_model"]
        cfg = self.root.settings.model_copy(update={"chat_model": models["chat_model"]})
        if (
            selected == self.root.settings.embedding_model
            and self.root.settings.embedding_provider == "gateway"
        ):
            return replace(self.root, settings=cfg)
        future = self.prepare(selected)
        if not future.done() and current_budget() is not None:
            raise RequestAborted(
                "Your workspace embedding model is loading. Please retry shortly.", 503
            )
        try:
            embedder, vectors, publication, cache = future.result()
        except Exception as exc:
            log.exception("Workspace embedding initialization failed")
            raise RequestAborted(
                "Your workspace embedding model could not be loaded. Please retry.", 503
            ) from exc
        return replace(
            self.root,
            settings=cfg,
            embedder=embedder,
            vectors=vectors,
            publication=publication,
            cache=cache,
        )

    def _build(self, selection):
        cfg = self.root.settings
        if selection in LOCAL_EMBEDDINGS:
            recipe = LOCAL_EMBEDDINGS[selection]
            embedder = OnnxEmbedder(
                recipe["repo"],
                recipe["dimensions"],
                pooling=recipe["pooling"],
                max_length=recipe["max_tokens"],
                revision=recipe["revision"],
                query_instruction=recipe.get("query_instruction", ""),
                batch_size=cfg.embed_batch_size,
            )
        elif selection in GATEWAY_EMBEDDINGS:
            gateway = LiteLLMClient(
                cfg.litellm_base_url,
                cfg.litellm_api_key,
                cfg.vision_model,
                selection,
                config_provider=lambda: self.root.metadata.get_gateway_config() or ("", ""),
            )
            embedder = GatewayEmbedder(gateway, 1536, cfg.embed_batch_size, revision=selection)
        else:
            raise ValueError("Unknown workspace embedding model")
        # The executor has no request context; loading is allowed here. Never
        # weaken the native-runtime guard on the interactive generation path.
        embedder.prepare()
        profile = embedder.profile
        with transaction(cfg.postgres_dsn) as cur:
            cur.execute(
                "INSERT INTO workspace_embedding_profiles(profile_id,manifest) VALUES(%s,%s) "
                "ON CONFLICT(profile_id) DO NOTHING",
                (profile.id, Json(asdict(profile))),
            )
        vectors = PgVectorStore(
            cfg.postgres_dsn,
            embedder.dim,
            profile=profile,
            workspace=True,
            candidate_multiplier=cfg.pgvector_candidate_multiplier,
            min_candidates=cfg.pgvector_min_candidates,
            hnsw_ef_search=cfg.pgvector_hnsw_ef_search,
        )
        cache = RedisStackAnswerCache(
            cfg.redis_url,
            cfg.cache_index_name + "_" + profile.id[:16],
            embedder.dim,
            cfg.cache_similarity_threshold,
            cfg.cache_ttl_seconds,
        )
        cache.init_index()
        return (
            embedder,
            vectors,
            PostgresPublicationStore(cfg.postgres_dsn, profile, workspace=True),
            cache,
        )

    def close(self):
        self._executor.shutdown(wait=True)
        for future in self._futures.values():
            if not future.cancelled() and future.exception() is None:
                future.result()[3].close()
