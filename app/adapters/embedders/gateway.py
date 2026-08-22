"""GatewayEmbedder: real embeddings via the LiteLLM gateway (batched)."""
from __future__ import annotations

from app.gateway.client import LiteLLMClient
from app.ports.embedder import Embedder


class GatewayEmbedder(Embedder):
    def __init__(self, client: LiteLLMClient, dim: int, batch_size: int = 64):
        self._client = client
        self._dim = dim
        self._batch = max(1, batch_size)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            out.extend(self._client.embed(texts[i:i + self._batch]))
        return out
