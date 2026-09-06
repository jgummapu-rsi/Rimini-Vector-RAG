"""VectorStore port: the `knowledgebase` collection of embeddings.

Local adapter = local file: one JSON record per document (sqlite `vector_documents`)
with chunks embedded in a chunks[] array. Production adapter = Qdrant/Mongo.
Vectors are FLOAT for now; binarization is deferred behind this same interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class VectorPoint:
    """One chunk's embedding plus enough payload to filter/render it later.

    `chunk_id` is the point identity (document _id + zero-padded ordinal);
    `payload` carries the document _id, user_id, modality, and other fields
    used for ACL filtering and rendering search hits.
    """
    chunk_id: str
    tenant_id: str
    vector: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchHit:
    """One ranked search result: a chunk id, its score, and its payload."""
    chunk_id: str
    score: float
    payload: dict[str, Any]


class VectorStore(ABC):
    """Port for the `knowledgebase` collection of chunk embeddings."""

    @abstractmethod
    def ensure_collection(self, dim: int) -> None:
        """Create the backing collection/index for `dim`-length vectors if absent."""

    @abstractmethod
    def upsert(self, points: list[VectorPoint]) -> None:
        """Insert or replace the given points."""

    @abstractmethod
    def delete_by_document(self, tenant_id: str, document_id: str) -> int:
        """Delete all points for a document; return count removed."""

    @abstractmethod
    def count(self, tenant_id: str) -> int:
        """Number of points stored for a tenant."""

    @abstractmethod
    def search(
        self,
        tenant_id: str,
        query: list[float],
        top_k: int = 5,
        access: Optional[Callable[[dict], bool]] = None,
        query_text: str = "",
    ) -> list[SearchHit]:
        """Tenant-scoped nearest-neighbour search (cosine), optionally fused with
        BM25 lexical scoring.

        `access`, if given, is a predicate over each document record's payload
        (user_id, visibility, acl_user_ids); non-visible documents are excluded
        BEFORE ranking, so callers still get a full top-k of permitted results.

        `query_text`, if given (non-empty), enables hybrid retrieval: BM25 is
        scored alongside dense cosine similarity and the two rankings are fused
        (reciprocal rank fusion). Omitting it (the default) preserves pure dense
        search — every existing caller that doesn't pass it is unaffected.
        """
