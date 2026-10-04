from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from numbers import Real


@dataclass(frozen=True)
class EmbeddingProfile:
    provider: str
    model: str
    revision: str
    tokenizer: str
    dimensions: int
    max_tokens: int
    pooling: str
    normalize: bool
    query_instruction: str = ""
    chunk_policy: str = "actual-token-cell-identity-v1"

    def __post_init__(self):
        if not all((self.provider, self.model, self.revision, self.tokenizer)):
            raise ValueError("Embedding profile identity must be explicit")
        if self.dimensions <= 0 or self.max_tokens <= 0:
            raise ValueError("Embedding profile dimensions and token limit must be positive")

    @property
    def id(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def validate_vectors(vectors, count: int, dimensions: int) -> list[list[float]]:
    if not isinstance(vectors, list) or len(vectors) != count:
        raise ValueError("Embedding response cardinality does not match inputs")
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != dimensions:
            raise ValueError("Embedding response dimension does not match profile")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or abs(value) > 3.4028235e38
            for value in vector
        ):
            raise ValueError("Embedding coordinates must be finite float32 numbers")
        if not any(value != 0 for value in vector):
            raise ValueError("Zero embedding cannot be used for cosine retrieval")
    return vectors
