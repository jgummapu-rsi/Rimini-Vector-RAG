"""MetadataStore port: tenants, users, documents, jobs, chunks, audit.

Local adapter = SQLite. Production adapter = PostgreSQL. Both implement this
interface, so no calling code changes on flip.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from app.shared.domain.models import AuthUser, ChunkRecord, Document, Job, JobEvent, Principal


class EmailAlreadyRegistered(Exception):
    """Raised by `create_user_with_password` when the email already has a
    password-holding account (in ANY tenant -- see that method's docstring and
    `get_user_by_email`). Callers translate this into a 409, the same response
    the pre-check in `app.api.onboarding_routes.register` already gives for
    the common (non-racing) case; this is the backstop for the race between
    two concurrent registrations with the same email that the pre-check alone
    cannot close."""


class MetadataStore(ABC):
    """Port for tenant/user/document/job/chunk/audit persistence."""

    @abstractmethod
    def init_schema(self) -> None:
        """Create the backing schema if it does not already exist."""

    @abstractmethod
    def create_tenant(self, name: str) -> str:
        """Create a tenant and return its new tenant_id."""

    @abstractmethod
    def create_user(self, tenant_id: str, email: str, role: str, token: str) -> str:
        """Create a bearer-token user in `tenant_id` and return the new user_id.
        `token` is the raw token; it is hashed before being stored, never in
        plaintext -- the caller must show it to the user now, it cannot be
        recovered later."""

    @abstractmethod
    def get_principal_by_token(self, token: str) -> Optional[Principal]:
        """Resolve a raw bearer token (hashed internally before lookup) to its
        Principal, or None if unknown/inactive."""

    @abstractmethod
    def rotate_api_token(self, user_id: str, token: str) -> None:
        """Replace a user's API token with `token` (hashed before storage) --
        e.g. reissued on POST /onboarding/login, invalidating the previous one."""

    @abstractmethod
    def get_user_by_email(self, email: str) -> Optional[AuthUser]:
        """Look up a user by email, across ALL tenants (onboarding login has no
        tenant_id up front -- email is the only key it has).

        There is no database-level UNIQUE constraint on email across tenants,
        so more than one row can match. The one with a non-NULL password_hash
        wins (ties broken by most recent), so a password-less user in one
        tenant can never shadow a real onboarding account with the same email
        in another. Returns None if no row matches."""

    @abstractmethod
    def create_user_with_password(
        self, tenant_id: str, email: str, role: str, token: str, password_hash: str,
    ) -> str:
        """Like create_user (token hashed before storage), but also persists
        password_hash so the account can authenticate via POST /onboarding/login
        as well as by bearer token.

        Raises `EmailAlreadyRegistered` if another password-holding account
        already exists for this email (in any tenant) -- most callers will
        have already checked `get_user_by_email` first and never hit this, but
        two concurrent calls for the same email can both pass that check
        before either commits, so the store itself is the actual backstop."""

    @abstractmethod
    def get_system_config(self, key: str) -> Optional[str]:
        """Current value for `key` (e.g. 'litellm_base_url'), or None if never set."""

    @abstractmethod
    def set_system_config(self, key: str, value: str) -> None:
        """Upsert `key` = `value`."""

    @abstractmethod
    def get_document_by_hash(self, tenant_id: str, sha256: str) -> Optional[Document]:
        """Look up a tenant's document by content hash (dedup check on ingest)."""

    @abstractmethod
    def create_document(self, doc: Document) -> None:
        """Persist a new document row."""

    @abstractmethod
    def get_document(self, tenant_id: str, document_id: str) -> Optional[Document]:
        """Fetch a document by id, or None if not found."""

    @abstractmethod
    def set_document_metadata(
        self, tenant_id: str, document_id: str, metadata: dict[str, Any]
    ) -> None:
        """Persist the extracted metadata (author/date/topics/entities) for a document."""

    @abstractmethod
    def delete_document(self, tenant_id: str, document_id: str) -> None:
        """Delete a document row (chunk/vector/blob cleanup is the caller's job)."""

    @abstractmethod
    def create_job(self, job: Job) -> None:
        """Persist a new ingestion job row."""

    @abstractmethod
    def get_job(self, tenant_id: str, job_id: str) -> Optional[Job]:
        """Fetch a job by id, or None if not found."""

    @abstractmethod
    def set_route_summary(self, job_id: str, summary: dict[str, Any]) -> None:
        """Persist the per-element routing summary produced during parsing."""

    @abstractmethod
    def record_job_event(self, event: JobEvent) -> None:
        """Append one pipeline-stage outcome to the job's trace."""

    @abstractmethod
    def get_job_events(self, tenant_id: str, job_id: str) -> list[JobEvent]:
        """The job's stage trace, in execution order (attempt, then stage)."""

    @abstractmethod
    def list_documents(
        self, tenant_id: str, limit: int = 50, offset: int = 0
    ) -> list[tuple[Document, Optional[Job]]]:
        """Documents readable from this tenant (own tenant, plus scope=global),
        newest first, each paired with its most recent ingestion job (None if the
        job row has since been deleted). Visibility/ACL filtering is the caller's
        job — see `app.retrieval.rag.access.can_view`."""

    @abstractmethod
    def replace_document_chunks(
        self, tenant_id: str, document_id: str, chunks: list[ChunkRecord]
    ) -> list[str]:
        """Delete existing chunks for the document and insert these; return ids."""

    @abstractmethod
    def get_document_chunks(self, tenant_id: str, document_id: str) -> list[ChunkRecord]:
        """All chunks belonging to a document, in stored order."""

    @abstractmethod
    def write_audit(
        self, tenant_id: str, user_id: str, action: str, target: str,
        meta: Optional[dict[str, Any]] = None,
    ) -> None:
        """Append one audit-log row for `action` taken by `user_id` on `target`."""
