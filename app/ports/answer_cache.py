"""AnswerCache port: a semantic cache of generated query answers.

When a user asks a question and the pipeline generates an answer, we store the
answer keyed by (tenant_id, user_id) alongside the question's embedding. A later
question that is the same or *semantically similar* (cosine similarity above a
threshold) returns the stored answer instantly, skipping retrieval + LLM
generation.

Scoping is PER-USER on purpose: answers are grounded in ACL-filtered chunks
(app.rag.access), so a hit is only ever served back to the same user who
generated it -- this structurally prevents one user's private-document-grounded
answer leaking to another user in the same tenant.

Staleness is handled by a per-tenant generation counter (invalidate_tenant bumps
it; entries built under an older generation stop matching) plus a TTL backstop.

Only a Redis Stack adapter exists today (native RediSearch vector KNN). The cache
is OPTIONAL: when no backend is configured, `container.cache` is None and the
query path behaves exactly as before.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class CacheHit:
    """A cache lookup that cleared the similarity threshold. `payload` is the full
    serialized QueryResult body (answer, contexts, chunk_ids, scores, ...) so the
    caller can reconstruct the exact response it would have generated."""
    payload: dict
    similarity: float          # cosine similarity of the matched question, 0..1


class AnswerCache(ABC):
    @abstractmethod
    def init_index(self) -> None:
        """Create the backing index/collection if absent. Idempotent."""

    @abstractmethod
    def get(self, tenant_id: str, user_id: str, qvec: list[float],
            model: str) -> Optional[CacheHit]:
        """Return the best cached answer for this (tenant, user, model) whose
        question embedding is within the similarity threshold of `qvec`, or None.
        Entries from a superseded tenant generation are ignored."""

    @abstractmethod
    def put(self, tenant_id: str, user_id: str, question: str,
            qvec: list[float], model: str, payload: dict) -> None:
        """Store (or overwrite) the answer for this question under the tenant's
        current generation, with the configured TTL."""

    @abstractmethod
    def invalidate_tenant(self, tenant_id: str) -> None:
        """Advance the tenant's generation so all currently-cached answers for it
        stop matching. O(1); called when the tenant's knowledge base changes."""
