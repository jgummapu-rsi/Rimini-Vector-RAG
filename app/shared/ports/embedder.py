"""Embedder port: text -> float embedding vectors.

gateway adapter = real embeddings via LiteLLM; local adapter = deterministic
dev embedder. Binarization is deliberately NOT here — embeddings stay float and
are stored as float for now; a binarization step can be inserted later without
touching this interface.

An embedder is also the SOURCE OF TRUTH for tokenization: it knows its own hard
token limit (`max_tokens`) and how to count tokens in its own tokenizer
(`count_tokens`). The chunker sizes chunks against these so a chunk is never
silently truncated at embed time (see app.ingest.pipeline.tokens). Both have concrete
MiniLM-based defaults, so an adapter only overrides them if it uses a different
tokenizer/limit (e.g. app.shared.adapters.embedders.onnx_embedder for bge).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from app.ingest.pipeline.tokens import EMBED_MAX_TOKENS, count_tokens as _default_count


class Embedder(ABC):
    """Port for turning text into float embedding vectors."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of vectors this embedder produces."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one float vector per input text (len == len(texts))."""

    @property
    def max_tokens(self) -> int:
        """This model's hard truncation limit -- inputs longer than this are
        cut before embedding, so chunks must stay within it. Defaults to the
        MiniLM baseline; override for a model with a different limit."""
        return EMBED_MAX_TOKENS

    def count_tokens(self, text: str) -> int:
        """True (untruncated) token length of `text` in THIS model's tokenizer.
        Defaults to the MiniLM tokenizer; override for a different tokenizer."""
        return _default_count(text)
