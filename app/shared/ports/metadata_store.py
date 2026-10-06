"""MetadataStore port: tenants, users, documents, jobs, chunks, audit."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.shared.domain.models import (
    AuthUser,
    ChunkRecord,
    Document,
    DocumentVersion,
    Job,
    JobEvent,
    Principal,
)


class EmailAlreadyRegistered(Exception):
    """Raised by `create_user_with_password` when the email already has a
    password-holding account (in ANY tenant -- see that method's docstring and
    `get_user_by_email`). Callers translate this into a 409, the same response
    the pre-check in `app.api.onboarding_routes.register` already gives for
    the common (non-racing) case; this is the backstop for the race between
    two concurrent registrations with the same email that the pre-check alone
    cannot close."""


class IngestionConflict(ValueError):
    pass


class MetadataStore(ABC):
    def create_workspace_with_models(
        self,
        name: str,
        email: str,
        token: str,
        password_hash: str,
        chat_model: str,
        embedding_model: str,
    ) -> tuple[str, str]:
        raise NotImplementedError

    def get_workspace_models(self, tenant_id: str) -> dict | None:
        raise NotImplementedError

    def set_workspace_models(self, tenant_id: str, chat_model: str, embedding_model: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_document_access(
        self, tenant_id: str, document_id: str, visibility: str, scope: str
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_gateway_config(self, base_url: str, api_key: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_gateway_config(self) -> tuple[str, str] | None:
        raise NotImplementedError

    @abstractmethod
    def corpus_epoch(self, tenant_id: str) -> str:
        raise NotImplementedError

    """Port for tenant/user/document/job/chunk/audit persistence."""

    @abstractmethod
    def init_schema(self) -> None:
        """Create the backing schema if it does not already exist."""

    @abstractmethod
    def create_tenant(self, name: str) -> str:
        """Create a tenant and return its new tenant_id."""

    @abstractmethod
    def get_workspace_identity(self, principal: Principal) -> dict[str, Any]:
        """Return the authenticated account's email and organization name."""

    @abstractmethod
    def list_tenant_members(self, tenant_id: str, limit: int, offset: int) -> dict[str, Any]:
        """Return a bounded member directory without credentials, scoped to one tenant."""

    @abstractmethod
    def delete_tenant_member(self, principal: Principal, member_id: str) -> str:
        """Revoke a member account atomically, retaining document ownership and audit history.

        Raises PermissionError for an inactive actor, ValueError for self-deletion,
        and LookupError for a missing member in this tenant. Returns the former email.
        """

    @abstractmethod
    def create_user(self, tenant_id: str, email: str, role: str, token: str) -> str:
        """Create a bearer-token user in `tenant_id` and return the new user_id.
        `token` is the raw token; it is hashed before being stored, never in
        plaintext -- the caller must show it to the user now, it cannot be
        recovered later."""

    @abstractmethod
    def get_principal_by_token(self, token: str) -> Principal | None:
        """Resolve a raw bearer token (hashed internally before lookup) to its
        Principal, or None if unknown/inactive."""

    @abstractmethod
    def rotate_api_token(self, user_id: str, token: str) -> None:
        """Replace a user's API token with `token` (hashed before storage) --
        e.g. reissued on POST /onboarding/login, invalidating the previous one."""

    @abstractmethod
    def get_user_by_email(self, email: str) -> AuthUser | None:
        """Look up a user by email, across ALL tenants (onboarding login has no
        tenant_id up front -- email is the only key it has).

        There is no database-level UNIQUE constraint on email across tenants,
        so more than one row can match. The one with a non-NULL password_hash
        wins (ties broken by most recent), so a password-less user in one
        tenant can never shadow a real onboarding account with the same email
        in another. Returns None if no row matches."""

    @abstractmethod
    def create_user_with_password(
        self,
        tenant_id: str,
        email: str,
        role: str,
        token: str,
        password_hash: str,
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
    def get_system_config(self, key: str) -> str | None:
        """Current value for `key` (e.g. 'litellm_base_url'), or None if never set."""

    @abstractmethod
    def set_system_config(self, key: str, value: str) -> None:
        """Upsert `key` = `value`."""

    @abstractmethod
    def get_document_by_hash(
        self, tenant_id: str, sha256: str, owner_user_id: str
    ) -> Document | None:
        raise NotImplementedError

    @abstractmethod
    def create_document(self, doc: Document) -> None:
        """Persist a new document row."""

    @abstractmethod
    def get_document(self, tenant_id: str, document_id: str) -> Document | None:
        """Fetch a document by id, or None if not found."""

    @abstractmethod
    def get_document_by_filename(
        self, tenant_id: str, owner_user_id: str, filename: str
    ) -> Document | None:
        """The caller's own newest document with this filename, or None.

        This is how `POST /ingest` decides "is this an update of something I
        already have?" when no explicit `document_id` was supplied.

        Scoped to `owner_user_id` deliberately, and NOT widened to admins or to
        `scope='global'` the way `get_document` is. Two reasons, both
        correctness rather than tidiness: a shared filename like `report.pdf`
        is entirely unremarkable, so a tenant-wide match would let one member
        silently overwrite a colleague's private document; and even returning
        it would leak the existence of a document the caller may not read. An
        admin who genuinely means to replace someone else's document says so by
        passing `document_id` explicitly.
        """

    @abstractmethod
    def set_document_metadata(
        self, tenant_id: str, document_id: str, metadata: dict[str, Any]
    ) -> None:
        """Persist the extracted metadata (author/date/topics/entities) for a document.
        Also stamps `updated_at` -- this runs as part of a pipeline run, so the
        document genuinely did change."""

    @abstractmethod
    def update_document_content(
        self,
        tenant_id: str,
        document_id: str,
        *,
        blob_path: str,
        content_sha256: str,
        mime: str,
        source_type: str,
        filename: str,
        visibility: str,
        scope: str,
    ) -> int:
        """Point an existing document at newly-uploaded bytes; return its new version.

        Bumps `version`, stamps `updated_at`, and overwrites the storage/identity
        fields. `source_type` and `filename` are included because a replacement
        may legitimately arrive in a different format (`notes.md` -> `notes.pdf`).

        This only rewrites the `documents` row. The chunks and vectors still
        describe the OLD bytes until the job this caller enqueues runs the
        pipeline -- which is why the API creates the job in the same request.
        """

    @abstractmethod
    def create_document_with_job(
        self, doc: Document, job: Job, initial_version: DocumentVersion | None = None
    ) -> None:
        """Atomically insert a new document and its first ingestion job.

        The job is pinned to `doc`'s own `blob_path`/`content_sha256`/`version`
        at insert time (see `Job`'s docstring) -- one transaction, so a crash
        between the two writes can never leave a document with no job queued
        to index it."""

    @abstractmethod
    def update_document_content_and_queue(
        self,
        tenant_id: str,
        document_id: str,
        *,
        blob_path: str,
        content_sha256: str,
        mime: str,
        source_type: str,
        filename: str,
        visibility: str,
        scope: str,
        job: Job,
        uploaded_by: str,
        byte_size: int,
        expected_version: int | None = None,
    ) -> int:
        """Atomically repoint a document at new bytes, queue the re-index job
        (pinned to exactly these new bytes), and append the version-history
        row for them. One transaction covering all three writes, unlike the
        sequential `update_document_content` + `create_job` +
        `add_document_version` calls this replaces at the one call site
        (`app.api.ingest_routes._update_document`) that needs all three to
        succeed or fail together. Returns the new version number."""

    @abstractmethod
    def has_pending_job(self, tenant_id: str, document_id: str) -> bool:
        """True if a job for this document is currently queued or running.

        Single-flight guard: callers refuse a second update/reprocess while
        one is already in flight, so at most one job per document can ever be
        running at once -- see `app.api.ingest_routes`' update and reprocess
        routes."""

    @abstractmethod
    def add_document_version(self, version: DocumentVersion) -> None:
        """Append one content-version history row."""

    @abstractmethod
    def list_document_versions(self, tenant_id: str, document_id: str) -> list[DocumentVersion]:
        """A document's content versions, newest first."""

    @abstractmethod
    def get_document_version(
        self, tenant_id: str, document_id: str, version: int
    ) -> DocumentVersion | None:
        """Fetch one immutable content version, or None."""

    @abstractmethod
    def set_version_delta(self, job_id: str, delta: dict[str, Any]) -> None:
        """Record how the chunk set changed, against the version row `job_id` created.

        A no-op when no version row references this job -- `/reprocess` re-runs
        the pipeline over unchanged bytes and so creates no new version, but it
        still produces a delta worth seeing in the job trace. Callers must treat
        this as observability, never correctness (see `app.ingest.pipeline.runner`).
        """

    @abstractmethod
    def count_blob_references(self, blob_path: str, exclude_document_id: str) -> int:
        """How many OTHER documents (live rows or version history) still point at
        `blob_path`.

        Blobs are content-addressed, so identical bytes uploaded twice resolve to
        one file on disk. Deleting a document must therefore not unlink a blob
        another document still needs -- and versioning makes that reachable: once
        v1's bytes are no longer any document's *current* hash, the ingest dedup
        check stops matching them, so the same bytes can legitimately come back
        as a brand-new document while the old version row still references them.
        """

    @abstractmethod
    def delete_document(self, tenant_id: str, document_id: str) -> None:
        """Delete a document row and its version history (chunk/vector/blob
        cleanup is the caller's job)."""

    @abstractmethod
    def create_job(self, job: Job) -> None:
        """Persist a new ingestion job row."""

    @abstractmethod
    def get_job(self, tenant_id: str, job_id: str) -> Job | None:
        """Fetch a job by id, or None if not found."""

    @abstractmethod
    def set_route_summary(self, job_id: str, summary: dict[str, Any]) -> None:
        """Persist the per-element routing summary produced during parsing."""

    @abstractmethod
    def get_extraction_artifact(self, job_id: str, content_sha256: str, key: str) -> dict | None:
        raise NotImplementedError

    @abstractmethod
    def put_extraction_artifact(
        self, job_id: str, content_sha256: str, key: str, artifact: dict
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def record_job_event(self, event: JobEvent) -> None:
        """Append one pipeline-stage outcome to the job's trace."""

    @abstractmethod
    def get_job_events(self, tenant_id: str, job_id: str) -> list[JobEvent]:
        """The job's stage trace, in execution order (attempt, then stage)."""

    @abstractmethod
    def list_documents(
        self, tenant_id: str, limit: int = 50, offset: int = 0
    ) -> list[tuple[Document, Job | None]]:
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
        self,
        tenant_id: str,
        user_id: str,
        action: str,
        target: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """Append one audit-log row for `action` taken by `user_id` on `target`."""
