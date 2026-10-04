from __future__ import annotations

from abc import ABC, abstractmethod

from app.shared.domain.models import ChunkRecord, Document, Job
from app.shared.ports.vector_store import VectorPoint


class PublicationStore(ABC):
    @abstractmethod
    def publish(
        self,
        job: Job,
        document: Document,
        chunks: list[ChunkRecord],
        points: list[VectorPoint],
        metadata: dict,
        delta: dict,
    ) -> str:
        raise NotImplementedError
