"""Reranker port: (query, documents) -> per-document relevance scores.

A cross-encoder scores a query and a document jointly, which is more precise
than the embedder's independent-vector similarity but too slow to run over an
entire corpus -- it only runs over the already-narrowed candidate pool that
hybrid retrieval (dense+BM25 RRF) hands it. `reranker_provider=none` means no
adapter is built at all (Container.reranker stays None); callers must treat
that as "skip reranking", not as a no-op adapter.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Reranker(ABC):
    """Port for scoring a query against a candidate pool of documents."""

    @abstractmethod
    def score(self, query: str, documents: list[str]) -> list[float]:
        """Return one relevance score per document, aligned with input order.
        Higher = more relevant. Scores are not necessarily bounded or
        comparable across calls -- only their relative order within one call
        matters to the caller."""
