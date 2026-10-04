"""Domain enums and dataclasses shared by the ingest and retrieval flows."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.shared.ids import chunk_id


class Role(str, Enum):
    """A principal's permission level: ADMIN (manage+ingest+delete), MEMBER (ingest own), VIEWER (read-only)."""

    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class SourceType(str, Enum):
    """The original file format a document was ingested from. TABLE is a standalone CSV/HTML table upload."""

    PDF = "pdf"
    DOCX = "docx"
    IMAGE = "image"
    TABLE = "table"
    XLSX = "xlsx"


class Modality(str, Enum):
    """The kind of content a single extracted Element/ChunkRecord holds."""

    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


class Visibility(str, Enum):
    """Within-tenant sharing rule for a document: TENANT (any tenant member), PRIVATE (owner only), SHARED (owner + acl_user_ids)."""

    TENANT = "tenant"
    PRIVATE = "private"
    SHARED = "shared"


class Scope(str, Enum):
    """Cross-tenant reach of a document, orthogonal to `Visibility` (which only
    governs sharing *within* a tenant). GLOBAL bypasses visibility/ACL entirely and
    is readable by any authenticated user in any tenant."""

    TENANT = "tenant"
    GLOBAL = "global"


class JobStage(str, Enum):
    """The ingestion pipeline stage an ingestion_job is currently at or has completed."""

    PARSE = "parse"
    ROUTE = "route"
    EXTRACT = "extract"
    CHUNK = "chunk"
    METADATA = "metadata"
    EMBED = "embed"
    BINARIZE = "binarize"
    UPSERT = "upsert"
    DONE = "done"


class JobStatus(str, Enum):
    """The lifecycle status of an ingestion_job row."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    DEAD = "dead"


EXT_TO_SOURCE_TYPE: dict[str, SourceType] = {
    ".pdf": SourceType.PDF,
    ".docx": SourceType.DOCX,
    ".md": SourceType.DOCX,
    ".txt": SourceType.DOCX,
    ".rtf": SourceType.DOCX,
    ".png": SourceType.IMAGE,
    ".jpg": SourceType.IMAGE,
    ".jpeg": SourceType.IMAGE,
    ".tiff": SourceType.IMAGE,
    ".tif": SourceType.IMAGE,
    ".webp": SourceType.IMAGE,
    ".csv": SourceType.TABLE,
    ".html": SourceType.TABLE,
    ".htm": SourceType.TABLE,
    ".xlsx": SourceType.XLSX,
    ".xls": SourceType.XLSX,
}


@dataclass(frozen=True)
class Principal:
    """Authenticated caller resolved from the bearer token."""

    tenant_id: str
    user_id: str
    role: Role

    def can_ingest(self) -> bool:
        """True if this principal's role may create/ingest documents (admin or member)."""
        return self.role in (Role.ADMIN, Role.MEMBER)

    def can_delete(self) -> bool:
        """True if this principal's role may delete documents (admin only)."""
        return self.role == Role.ADMIN


@dataclass
class AuthUser:
    """Full user row needed for password-based auth (app.api.onboarding_routes),
    as opposed to Principal (the token-resolved caller identity used by every
    other route). Not RBAC-checked itself -- callers still go through
    get_principal_by_token for actual request authorization.

    Deliberately has no `api_token` field: the stored token is a one-way hash
    (see app.shared.security.hash_token), never recoverable in usable form, so
    there is nothing meaningful to expose here -- a caller that needs a fresh
    token (e.g. POST /onboarding/login) mints one and stores its hash via
    `MetadataStore.rotate_api_token`."""

    id: str
    tenant_id: str
    email: str
    role: str
    password_hash: str | None = None


@dataclass
class Document:
    """A tenant's uploaded file: identity, storage location, ACL fields, and ingestion-derived metadata.

    `version` counts CONTENT versions, not pipeline runs: re-uploading changed
    bytes for this document bumps it, a `/reprocess` of the same bytes does not.
    `blob_path`/`content_sha256` always describe the CURRENT version; superseded
    ones live in `document_versions` (see `DocumentVersion`).
    """

    id: str
    tenant_id: str
    owner_user_id: str
    source_type: str
    blob_path: str
    content_sha256: str
    mime: str
    filename: str
    visibility: str
    acl_user_ids: list[str] = field(default_factory=list)
    scope: str = Scope.TENANT.value
    extracted_metadata: dict = field(default_factory=dict)
    created_at: str | None = None
    version: int = 1
    active_generation_id: str | None = None
    indexed_version: int | None = None
    updated_at: str | None = None


@dataclass
class DocumentVersion:
    """One historical content version of a document: what was uploaded, by whom,
    and how the chunk set changed as a result.

    Re-ingest replaces searchable chunks and vectors wholesale, while the raw
    bytes remain viewable from the content-addressed blob referenced here.

    The `chunks_*` counts and `delta` are NULL/empty at insert time -- the API
    creates this row, and the worker fills them in once the pipeline has
    actually chunked the new bytes (see `MetadataStore.set_version_delta`).
    """

    id: str
    document_id: str
    tenant_id: str
    version: int
    content_sha256: str
    blob_path: str
    filename: str
    byte_size: int
    uploaded_by: str
    job_id: str | None = None
    chunks_added: int | None = None
    chunks_removed: int | None = None
    chunks_unchanged: int | None = None
    delta: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    mime: str = "application/octet-stream"


@dataclass
class ChunkRecord:
    """One chunk produced by the pipeline's chunker. `id` (set by `finalize_chunks`)
    is `document_id` + a zero-padded ordinal and doubles as the vector store's point id."""

    ordinal: int
    modality: str
    extractor: str
    route_reason: str
    token_count: int
    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    id: str | None = None
    content_sha256: str | None = None


def finalize_chunks(document_id: str, chunks: list[ChunkRecord]) -> list[str]:
    """Stamp each chunk with its deterministic id and content hash; return the ids.

    Both are pure functions of (document_id, ordinal, count, text), so this is
    idempotent -- calling it twice produces the same values. A record that's
    already stamped (both `id` and `content_sha256` set -- e.g. the runner
    stamps before `replace_document_chunks` stamps again as part of its own
    contract) skips recomputation: same result, one fewer SHA-256 pass per
    chunk on that common path.

    It exists as one shared function because the id is the SHARED identity
    between two stores: the metadata store's `chunks.id` primary key and the
    vector store's point id must be the same string or a chunk's text and its
    embedding can no longer be matched up. Previously each metadata-store
    adapter derived it privately as a side effect of persisting, and the
    pipeline read `ChunkRecord.id` back off the objects it had passed in --
    so a store that returned ids without mutating in place would have sent
    `chunk_id=None` to the vector store, silently, with no error anywhere.
    """
    total = len(chunks)
    for r in chunks:
        if r.id is not None and r.content_sha256 is not None:
            continue
        r.id = chunk_id(document_id, r.ordinal, total)
        r.content_sha256 = hashlib.sha256(r.text.encode("utf-8")).hexdigest()
    return [r.id for r in chunks]


@dataclass
class Job:
    """An ingestion_job row: one document's progress through the pipeline stages.

    `blob_path`/`content_sha256`/`version` pin the job to the EXACT content it
    was created for, snapshotted at job-creation time. `run_job` uses these
    (when present) instead of re-fetching the document live, so a job created
    for version N always processes version N's bytes even if the document has
    since been re-ingested to version N+1 while this job was still queued.
    `None` on all three means a legacy, pre-migration job row (or one created
    before this snapshot existed) -- the runner falls back to the live
    document for those.
    """

    id: str
    document_id: str
    tenant_id: str
    stage: str
    status: str
    attempts: int
    error: str | None = None
    route_summary: dict[str, Any] | None = None
    created_at: str | None = None
    updated_at: str | None = None
    blob_path: str | None = None
    content_sha256: str | None = None
    version: int | None = None
    filename: str | None = None
    source_type: str | None = None
    lease_token: str | None = None
    generation_id: str | None = None


class JobEventStatus(str, Enum):
    """Outcome of a single pipeline stage, recorded on the JobEvent it produced."""

    OK = "ok"
    ERROR = "error"


@dataclass
class JobEvent:
    """One pipeline stage's outcome, persisted so a trace can be replayed long
    after the job finishes. The runner already logs per-stage timings, but logs
    aren't queryable — this is the same information, kept as data.

    `detail` is whatever `app.ingest.pipeline.runner._stage_detail` produced for the
    stage (element/chunk/vector counts, routing breakdown), so the trace view
    and the logs never drift apart. `status` holds a `JobEventStatus` value;
    `seq` is the stage's position within the job, for deterministic ordering.
    """

    job_id: str
    document_id: str
    tenant_id: str
    stage: str
    status: str
    duration_ms: float
    seq: int = 0
    detail: dict[str, Any] = field(default_factory=dict)
    attempt: int = 0
    id: str | None = None
    at: str | None = None
