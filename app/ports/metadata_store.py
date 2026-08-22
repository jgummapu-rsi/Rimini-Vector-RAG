"""MetadataStore port: tenants, users, documents, jobs, chunks, audit.

Local adapter = SQLite. Production adapter = PostgreSQL. Both implement this
interface, so no calling code changes on flip.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from app.domain.models import ChunkRecord, Document, Job, Principal


class MetadataStore(ABC):
    @abstractmethod
    def init_schema(self) -> None: ...

    @abstractmethod
    def create_tenant(self, name: str) -> str: ...

    @abstractmethod
    def create_user(self, tenant_id: str, email: str, role: str, token: str) -> str: ...

    @abstractmethod
    def get_principal_by_token(self, token: str) -> Optional[Principal]: ...

    @abstractmethod
    def get_document_by_hash(self, tenant_id: str, sha256: str) -> Optional[Document]: ...

    @abstractmethod
    def create_document(self, doc: Document) -> None: ...

    @abstractmethod
    def get_document(self, tenant_id: str, document_id: str) -> Optional[Document]: ...

    @abstractmethod
    def set_document_metadata(
        self, tenant_id: str, document_id: str, metadata: dict[str, Any]
    ) -> None:
        """Persist the extracted metadata (author/date/topics/entities) for a document."""

    @abstractmethod
    def delete_document(self, tenant_id: str, document_id: str) -> None: ...

    @abstractmethod
    def create_job(self, job: Job) -> None: ...

    @abstractmethod
    def get_job(self, tenant_id: str, job_id: str) -> Optional[Job]: ...

    @abstractmethod
    def set_route_summary(self, job_id: str, summary: dict[str, Any]) -> None: ...

    @abstractmethod
    def replace_document_chunks(
        self, tenant_id: str, document_id: str, chunks: list[ChunkRecord]
    ) -> list[str]:
        """Delete existing chunks for the document and insert these; return ids."""

    @abstractmethod
    def get_document_chunks(self, tenant_id: str, document_id: str) -> list[ChunkRecord]: ...

    @abstractmethod
    def write_audit(
        self, tenant_id: str, user_id: str, action: str, target: str,
        meta: Optional[dict[str, Any]] = None,
    ) -> None: ...
