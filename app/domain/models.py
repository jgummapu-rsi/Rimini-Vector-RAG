"""Domain enums and dataclasses shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Role(str, Enum):
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class SourceType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    IMAGE = "image"
    TABLE = "table"   # standalone csv / html table
    XLSX = "xlsx"


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


class Visibility(str, Enum):
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
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    DEAD = "dead"


# Map a file extension -> SourceType (Phase 1 surface).
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
        return self.role in (Role.ADMIN, Role.MEMBER)

    def can_delete(self) -> bool:
        return self.role == Role.ADMIN


@dataclass
class Document:
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
    created_at: Optional[str] = None


@dataclass
class ChunkRecord:
    ordinal: int
    modality: str
    extractor: str
    route_reason: str
    token_count: int
    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None          # = document_id + zero-padded ordinal (also the point id)
    content_sha256: Optional[str] = None


@dataclass
class Job:
    id: str
    document_id: str
    tenant_id: str
    stage: str
    status: str
    attempts: int
    error: Optional[str] = None
    route_summary: Optional[dict[str, Any]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
