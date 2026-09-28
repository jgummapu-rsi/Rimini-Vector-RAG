"""Admin model selection for the Knowledge ingestion and retrieval runtime."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.api.auth import get_container, require_admin
from app.retrieval.adapters.cache.redis_stack import RedisStackAnswerCache
from app.retrieval.adapters.rerankers.cross_encoder import CrossEncoderReranker
from app.shared.adapters.pgvector.vector_store import PgVectorStore
from app.shared.container import Container
from app.shared.domain.models import Principal
from app.shared.model_config import (
    CHAT_MODELS,
    EMBEDDING_PRESETS,
    MODEL_CONFIG_KEYS,
    RERANKER_PRESETS,
    VISION_MODELS,
    selected_chat_model,
    selected_value,
    selected_vision_model,
)

router = APIRouter(prefix="/admin/models", tags=["model-governance"])


class ModelSelection(BaseModel):
    chat_model: str
    vision_model: str
    embedding_preset: str
    reranker_preset: str
    confirm_embedding_reset: bool = False


class RetrievalSettings(BaseModel):
    rerank_min_score: float
    rerank_candidate_multiplier: int
    rerank_min_candidates: int
    cache_enabled: bool
    cache_similarity_threshold: float
    cache_ttl_seconds: int
    message: str = ""


class ActivateRetrievalVersion(BaseModel):
    version: int


_RETRIEVAL_VERSIONS_KEY = "rag_retrieval_config_versions"
_ACTIVE_RETRIEVAL_VERSION_KEY = "rag_active_retrieval_config_version"


def _retrieval_values(container: Container) -> dict:
    return {
        "rerank_min_score": container.settings.rerank_min_score,
        "rerank_candidate_multiplier": container.settings.rerank_candidate_multiplier,
        "rerank_min_candidates": container.settings.rerank_min_candidates,
        "cache_enabled": container.cache is not None,
        "cache_similarity_threshold": container.settings.cache_similarity_threshold,
        "cache_ttl_seconds": container.settings.cache_ttl_seconds,
    }


def _retrieval_versions(container: Container) -> list[dict]:
    raw = container.metadata.get_system_config(_RETRIEVAL_VERSIONS_KEY)
    if not raw:
        return []
    try:
        value = json.loads(raw)
        return value if isinstance(value, list) else []
    except (TypeError, ValueError):
        return []


def _apply_retrieval_values(container: Container, values: dict) -> None:
    if container.cache is not None:
        container.cache.invalidate_all()
    for name, value in values.items():
        container.metadata.set_system_config(f"rag_{name}", str(value).lower())
    container.settings.rerank_min_score = float(values["rerank_min_score"])
    container.settings.rerank_candidate_multiplier = int(values["rerank_candidate_multiplier"])
    container.settings.rerank_min_candidates = int(values["rerank_min_candidates"])
    container.settings.cache_similarity_threshold = float(values["cache_similarity_threshold"])
    container.settings.cache_ttl_seconds = int(values["cache_ttl_seconds"])
    container.cache = None
    if values["cache_enabled"]:
        preset = EMBEDDING_PRESETS[_current(container)["embedding_preset"]]
        container.cache = RedisStackAnswerCache(
            container.settings.redis_url,
            f"{container.settings.cache_index_name}_{preset.dim}",
            preset.dim,
            container.settings.cache_similarity_threshold,
            container.settings.cache_ttl_seconds,
        )
        container.cache.init_index()


def _current(container: Container) -> dict[str, str]:
    reranker_default = (
        "none" if container.settings.reranker_provider == "none" else "ms-marco-minilm"
    )
    return {
        "chat_model": selected_chat_model(
            container.metadata, container.settings.chat_model
        ),
        "vision_model": selected_vision_model(
            container.metadata, container.settings.vision_model
        ),
        "embedding_preset": selected_value(
            container.metadata, "embedding_preset", "minilm"
        ),
        "reranker_preset": selected_value(
            container.metadata, "reranker_preset", reranker_default
        ),
    }


def _payload(container: Container, reset: dict[str, int] | None = None) -> dict:
    return {
        "selected": _current(container),
        "options": {
            "chat_models": list(CHAT_MODELS),
            "vision_models": list(VISION_MODELS),
            "embedding_presets": [asdict(preset) for preset in EMBEDDING_PRESETS.values()],
            "reranker_presets": list(RERANKER_PRESETS),
        },
        "retrieval": {
            **_retrieval_values(container),
            "cache_available": bool(container.settings.redis_url),
        },
        "retrieval_versions": _retrieval_versions(container),
        "active_retrieval_version": container.metadata.get_system_config(
            _ACTIVE_RETRIEVAL_VERSION_KEY
        ),
        "reset": reset,
    }


@router.get("")
def get_models(
    _: Principal = Depends(require_admin),
    container: Container = Depends(get_container),
) -> dict:
    return _payload(container)


@router.put("")
def update_models(
    req: ModelSelection,
    principal: Principal = Depends(require_admin),
    container: Container = Depends(get_container),
) -> dict:
    if (
        not container.settings.platform_tenant_id
        or principal.tenant_id != container.settings.platform_tenant_id
    ):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "platform administrator required",
        )
    if req.chat_model not in CHAT_MODELS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "unsupported chat model")
    if req.vision_model not in VISION_MODELS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "unsupported vision model")
    if req.embedding_preset not in EMBEDDING_PRESETS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "unsupported embedding model")
    if req.reranker_preset not in RERANKER_PRESETS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "unsupported reranker")

    current = _current(container)
    embedding_changed = req.embedding_preset != current["embedding_preset"]
    if embedding_changed and not req.confirm_embedding_reset:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Changing embedding dimensions clears all indexed documents. "
            "Confirm the reset to continue.",
        )

    reset = None
    if embedding_changed:
        if not isinstance(container.vectors, PgVectorStore):
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                "Live embedding changes currently require the pgvector backend.",
            )
        preset = EMBEDDING_PRESETS[req.embedding_preset]
        reset = container.vectors.reset_collection(preset.dim)
        container.cache = None

    values = {
        "chat_model": req.chat_model,
        "vision_model": req.vision_model,
        "embedding_preset": req.embedding_preset,
        "reranker_preset": req.reranker_preset,
    }
    for name, value in values.items():
        container.metadata.set_system_config(MODEL_CONFIG_KEYS[name], value)

    container.reranker = (
        None
        if req.reranker_preset == "none"
        else CrossEncoderReranker("Xenova/ms-marco-MiniLM-L-6-v2")
    )
    if container.settings.redis_url:
        preset = EMBEDDING_PRESETS[req.embedding_preset]
        container.cache = RedisStackAnswerCache(
            container.settings.redis_url,
            f"{container.settings.cache_index_name}_{preset.dim}",
            preset.dim,
            container.settings.cache_similarity_threshold,
            container.settings.cache_ttl_seconds,
        )
        container.cache.init_index()

    container.metadata.write_audit(
        principal.tenant_id,
        principal.user_id,
        "model_configuration_updated",
        "knowledge",
        {"previous": current, "selected": values, "reset": reset},
    )
    return _payload(container, reset)


@router.put("/retrieval")
def update_retrieval(
    req: RetrievalSettings,
    principal: Principal = Depends(require_admin),
    container: Container = Depends(get_container),
) -> dict:
    if (
        not container.settings.platform_tenant_id
        or principal.tenant_id != container.settings.platform_tenant_id
    ):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "platform administrator required")
    if not -20 <= req.rerank_min_score <= 20:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "rerank_min_score must be between -20 and 20")
    if not 1 <= req.rerank_candidate_multiplier <= 20:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "rerank_candidate_multiplier must be between 1 and 20")
    if not 1 <= req.rerank_min_candidates <= 500:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "rerank_min_candidates must be between 1 and 500")
    if not 0.0 <= req.cache_similarity_threshold <= 1.0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "cache_similarity_threshold must be between 0 and 1")
    if not 60 <= req.cache_ttl_seconds <= 2_592_000:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "cache_ttl_seconds must be between 60 and 2592000")
    if req.cache_enabled and not container.settings.redis_url:
        raise HTTPException(status.HTTP_409_CONFLICT, "Redis Stack is not configured")

    values = req.model_dump(exclude={"message"})
    _apply_retrieval_values(container, values)
    versions = _retrieval_versions(container)
    version = max((item.get("version", 0) for item in versions), default=0) + 1
    versions.append(
        {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": principal.user_id,
            "message": req.message.strip(),
            "settings": values,
        }
    )
    container.metadata.set_system_config(_RETRIEVAL_VERSIONS_KEY, json.dumps(versions))
    container.metadata.set_system_config(_ACTIVE_RETRIEVAL_VERSION_KEY, str(version))

    container.metadata.write_audit(
        principal.tenant_id,
        principal.user_id,
        "retrieval_configuration_updated",
        "knowledge",
        values,
    )
    return _payload(container)


@router.post("/retrieval/activate")
def activate_retrieval_version(
    req: ActivateRetrievalVersion,
    principal: Principal = Depends(require_admin),
    container: Container = Depends(get_container),
) -> dict:
    if (
        not container.settings.platform_tenant_id
        or principal.tenant_id != container.settings.platform_tenant_id
    ):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "platform administrator required")
    version = next(
        (item for item in _retrieval_versions(container) if item.get("version") == req.version),
        None,
    )
    if version is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "retrieval configuration version not found")
    _apply_retrieval_values(container, version["settings"])
    container.metadata.set_system_config(_ACTIVE_RETRIEVAL_VERSION_KEY, str(req.version))
    container.metadata.write_audit(
        principal.tenant_id,
        principal.user_id,
        "retrieval_configuration_activated",
        "knowledge",
        {"version": req.version},
    )
    return _payload(container)
