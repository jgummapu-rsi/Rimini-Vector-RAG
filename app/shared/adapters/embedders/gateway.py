"""GatewayEmbedder: real embeddings via the LiteLLM gateway (batched)."""
from __future__ import annotations

from app.shared.gateway.client import LiteLLMClient
from app.shared.ports.embedder import Embedder


class GatewayEmbedder(Embedder):
    """Embedder backed by the LiteLLM gateway's `/v1/embeddings` endpoint."""

    def __init__(self, client: LiteLLMClient, dim: int, batch_size: int = 64):
        """Store the gateway `client`, the model's `dim`, and request `batch_size`."""
        self._client = client
        self._dim = dim
        self._batch = max(1, batch_size)

    @property
    def dim(self) -> int:
        """Dimensionality of vectors this embedder produces."""
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per text in `texts`, batched to the gateway."""
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            out.extend(self._client.embed(texts[i:i + self._batch]))
        return out
