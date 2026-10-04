"""GatewayEmbedder: real embeddings via the LiteLLM gateway (batched)."""

from __future__ import annotations

import tiktoken

from app.shared.domain.embedding import EmbeddingProfile, validate_vectors
from app.shared.gateway.client import LiteLLMClient
from app.shared.ports.embedder import Embedder


class GatewayEmbedder(Embedder):
    """Embedder backed by the LiteLLM gateway's `/v1/embeddings` endpoint."""

    def __init__(self, client: LiteLLMClient, dim: int, batch_size: int = 64, *, revision: str):
        """Store the gateway `client`, the model's `dim`, and request `batch_size`."""
        self._client = client
        self._dim = dim
        self._batch = max(1, batch_size)
        model = client.embedding_model
        limits = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        if model not in limits or not 1 <= dim <= limits[model]:
            raise ValueError("Unsupported gateway embedding model or dimension")
        if model == "text-embedding-ada-002" and dim != 1536:
            raise ValueError("Ada embeddings do not support reduced dimensions")
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self._profile = EmbeddingProfile(
            "gateway", model, revision, "cl100k_base:tiktoken-0.8.0", dim, 8191, "provider", True
        )

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    @property
    def max_tokens(self) -> int:
        return self.profile.max_tokens

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, disallowed_special=()))

    @property
    def dim(self) -> int:
        """Dimensionality of vectors this embedder produces."""
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding per text in `texts`, batched to the gateway."""
        self.validate_inputs(texts)
        out: list[list[float]] = []
        batch = []
        tokens = 0
        for text in texts:
            size = self.count_tokens(text)
            if batch and (len(batch) >= min(self._batch, 2048) or tokens + size > 300000):
                out.extend(self._request(batch))
                batch, tokens = [], 0
            batch.append(text)
            tokens += size
        if batch:
            out.extend(self._request(batch))
        return out

    def _request(self, texts: list[str]) -> list[list[float]]:
        dimensions = None if self.profile.model == "text-embedding-ada-002" else self.dim
        return validate_vectors(
            self._client.embed(texts, dimensions=dimensions, model=self.profile.model),
            len(texts),
            self.dim,
        )
