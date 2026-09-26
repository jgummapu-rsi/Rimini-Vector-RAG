"""Admin model selection for the Knowledge ingestion and retrieval runtime."""
from __future__ import annotations

from dataclasses import asdict

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
