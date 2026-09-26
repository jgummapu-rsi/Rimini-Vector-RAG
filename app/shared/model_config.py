"""Governed model presets and database-backed runtime selection."""
from __future__ import annotations

from dataclasses import dataclass

from app.shared.adapters.embedders.minilm import MiniLMEmbedder
from app.shared.adapters.embedders.onnx_embedder import OnnxEmbedder
from app.shared.ports.embedder import Embedder
from app.shared.ports.metadata_store import MetadataStore

CHAT_MODELS = (
    "gpt-5-nano",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
)
VISION_MODELS = (
    "gpt-5-nano",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
)
RERANKER_PRESETS = ("ms-marco-minilm", "none")


@dataclass(frozen=True)
class EmbeddingPreset:
    id: str
    name: str
    repo: str
    dim: int
    max_tokens: int
    pooling: str
    query_instruction: str = ""


EMBEDDING_PRESETS = {
    "minilm": EmbeddingPreset(
        "minilm", "all-MiniLM-L6-v2", "Xenova/all-MiniLM-L6-v2", 384, 256, "mean"
    ),
    "bge-base": EmbeddingPreset(
        "bge-base",
        "bge-base-en-v1.5",
        "Xenova/bge-base-en-v1.5",
        768,
        512,
        "cls",
        "Represent this sentence for searching relevant passages: ",
    ),
    "bge-large": EmbeddingPreset(
        "bge-large",
        "bge-large-en-v1.5",
        "Xenova/bge-large-en-v1.5",
        1024,
        512,
        "cls",
        "Represent this sentence for searching relevant passages: ",
    ),
}

MODEL_CONFIG_KEYS = {
    "chat_model": "rag_chat_model",
    "vision_model": "rag_vision_model",
    "embedding_preset": "rag_embedding_preset",
    "reranker_preset": "rag_reranker_preset",
}


def selected_value(metadata: MetadataStore, name: str, default: str) -> str:
    return metadata.get_system_config(MODEL_CONFIG_KEYS[name]) or default


def selected_chat_model(metadata: MetadataStore, default: str = "gpt-5-nano") -> str:
    selected = selected_value(metadata, "chat_model", default)
    return selected if selected in CHAT_MODELS else default


def selected_vision_model(metadata: MetadataStore, default: str = "gpt-5.6-sol") -> str:
    selected = selected_value(metadata, "vision_model", default)
    return selected if selected in VISION_MODELS else default


def create_embedder(preset_id: str, batch_size: int) -> Embedder:
    preset = EMBEDDING_PRESETS[preset_id]
    if preset_id == "minilm":
        return MiniLMEmbedder(batch_size=batch_size)
    return OnnxEmbedder(
        preset.repo,
        preset.dim,
        pooling=preset.pooling,
        query_instruction=preset.query_instruction,
        max_length=preset.max_tokens,
        batch_size=min(batch_size, 32),
    )


class ConfigurableEmbedder(Embedder):
    """Select a cached local embedder from persisted configuration per operation."""

    def __init__(self, metadata: MetadataStore, default: str, batch_size: int):
        self._metadata = metadata
        self._default = default
        self._batch_size = batch_size
        self._models: dict[str, Embedder] = {}

    @property
    def preset_id(self) -> str:
        selected = selected_value(self._metadata, "embedding_preset", self._default)
        return selected if selected in EMBEDDING_PRESETS else self._default

    @property
    def active(self) -> Embedder:
        preset_id = self.preset_id
        if preset_id not in self._models:
            self._models[preset_id] = create_embedder(preset_id, self._batch_size)
        return self._models[preset_id]

    @property
    def dim(self) -> int:
        return EMBEDDING_PRESETS[self.preset_id].dim

    @property
    def max_tokens(self) -> int:
        return self.active.max_tokens

    def count_tokens(self, text: str) -> int:
        return self.active.count_tokens(text)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.active.embed(texts)

    def embed_query(self, texts: list[str]) -> list[list[float]]:
        return self.active.embed_query(texts)
