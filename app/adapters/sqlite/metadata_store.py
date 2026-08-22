"""SQLite implementation of the MetadataStore port."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from app.adapters.sqlite.db import transaction
from app.domain.models import ChunkRecord, Document, Job, Principal, Role, Scope
from app.ids import new_object_id
from app.ports.metadata_store import MetadataStore

_SCHEMA = Path(__file__).with_name("schema.sql")


class SqliteMetadataStore(MetadataStore):
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def init_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with transaction(self.db_path) as c:
            c.executescript(_SCHEMA.read_text(encoding="utf-8"))
            # lightweight migration: add columns to a pre-existing chunks table
            cols = {r["name"] for r in c.execute("PRAGMA table_info(chunks)")}
            if "text" not in cols:
                c.execute("ALTER TABLE chunks ADD COLUMN text TEXT")
            if "meta" not in cols:
                c.execute("ALTER TABLE chunks ADD COLUMN meta TEXT")
            # lightweight migration: add scope to a pre-existing documents table
            doc_cols = {r["name"] for r in c.execute("PRAGMA table_info(documents)")}
            if "scope" not in doc_cols:
                c.execute("ALTER TABLE documents ADD COLUMN scope TEXT NOT NULL DEFAULT 'tenant'")
            if "extracted_metadata" not in doc_cols:
                c.execute("ALTER TABLE documents ADD COLUMN extracted_metadata TEXT NOT NULL DEFAULT '{}'")
            vd_cols = {r["name"] for r in c.execute("PRAGMA table_info(vector_documents)")}
            if "scope" not in vd_cols:
                c.execute("ALTER TABLE vector_documents ADD COLUMN scope TEXT NOT NULL DEFAULT 'tenant'")

    def create_tenant(self, name: str) -> str:
        tid = new_object_id()
        with transaction(self.db_path) as c:
            c.execute("INSERT INTO tenants (id, name) VALUES (?, ?)", (tid, name))
        return tid

    def create_user(self, tenant_id: str, email: str, role: str, token: str) -> str:
        uid = new_object_id()
        with transaction(self.db_path) as c:
            c.execute(
                "INSERT INTO users (id, tenant_id, email, role, api_token) "
                "VALUES (?, ?, ?, ?, ?)",
                (uid, tenant_id, email, role, token),
            )
        return uid

    def get_principal_by_token(self, token: str) -> Optional[Principal]:
        with transaction(self.db_path) as c:
            row = c.execute(
                "SELECT tenant_id, id, role FROM users "
                "WHERE api_token = ? AND status = 'active'",
                (token,),
            ).fetchone()
        if not row:
            return None
        return Principal(tenant_id=row["tenant_id"], user_id=row["id"], role=Role(row["role"]))

    def get_document_by_hash(self, tenant_id: str, sha256: str) -> Optional[Document]:
        with transaction(self.db_path) as c:
            row = c.execute(
                "SELECT * FROM documents WHERE tenant_id = ? AND content_sha256 = ?",
                (tenant_id, sha256),
            ).fetchone()
        return _row_to_document(row) if row else None

    def create_document(self, doc: Document) -> None:
        with transaction(self.db_path) as c:
            c.execute(
                "INSERT INTO documents (id, tenant_id, owner_user_id, source_type, "
                "blob_path, content_sha256, mime, filename, visibility, acl_user_ids, scope, "
                "extracted_metadata) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    doc.id, doc.tenant_id, doc.owner_user_id, doc.source_type,
                    doc.blob_path, doc.content_sha256, doc.mime, doc.filename,
                    doc.visibility, json.dumps(doc.acl_user_ids), doc.scope,
                    json.dumps(doc.extracted_metadata),
                ),
            )

    def get_document(self, tenant_id: str, document_id: str) -> Optional[Document]:
        """A document is fetchable if it belongs to this tenant, OR it is
        scope=global (readable cross-tenant; write/delete stay tenant-owned —
        enforced by the caller comparing doc.tenant_id, not here)."""
        with transaction(self.db_path) as c:
            row = c.execute(
                "SELECT * FROM documents WHERE id = ? AND (tenant_id = ? OR scope = 'global')",
                (document_id, tenant_id),
            ).fetchone()
        return _row_to_document(row) if row else None

    def set_document_metadata(
        self, tenant_id: str, document_id: str, metadata: dict[str, Any]
    ) -> None:
        with transaction(self.db_path) as c:
            c.execute(
                "UPDATE documents SET extracted_metadata=? WHERE tenant_id=? AND id=?",
                (json.dumps(metadata), tenant_id, document_id),
            )

    def delete_document(self, tenant_id: str, document_id: str) -> None:
        with transaction(self.db_path) as c:
            c.execute(
                "DELETE FROM chunks WHERE tenant_id = ? AND document_id = ?",
                (tenant_id, document_id),
            )
            c.execute(
                "DELETE FROM ingestion_jobs WHERE tenant_id = ? AND document_id = ?",
                (tenant_id, document_id),
            )
            c.execute(
                "DELETE FROM documents WHERE tenant_id = ? AND id = ?",
                (tenant_id, document_id),
            )

    def create_job(self, job: Job) -> None:
        with transaction(self.db_path) as c:
            c.execute(
                "INSERT INTO ingestion_jobs (id, document_id, tenant_id, stage, status, attempts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (job.id, job.document_id, job.tenant_id, job.stage, job.status, job.attempts),
            )

    def get_job(self, tenant_id: str, job_id: str) -> Optional[Job]:
        with transaction(self.db_path) as c:
            row = c.execute(
                "SELECT * FROM ingestion_jobs WHERE tenant_id = ? AND id = ?",
                (tenant_id, job_id),
            ).fetchone()
        return _row_to_job(row) if row else None

    def set_route_summary(self, job_id: str, summary: dict[str, Any]) -> None:
        with transaction(self.db_path) as c:
            c.execute(
                "UPDATE ingestion_jobs SET route_summary=?, updated_at=datetime('now') "
                "WHERE id=?",
                (json.dumps(summary), job_id),
            )

    def replace_document_chunks(
        self, tenant_id: str, document_id: str, chunks: list[ChunkRecord]
    ) -> list[str]:
        with transaction(self.db_path) as c:
            c.execute(
                "DELETE FROM chunks WHERE tenant_id=? AND document_id=?",
                (tenant_id, document_id),
            )
            # width scales with chunk count so ids sort correctly even past 999
            width = max(3, len(str(len(chunks))))
            for r in chunks:
                # deterministic id: document id + 1-based, zero-padded ordinal
                r.id = f"{document_id}{r.ordinal + 1:0{width}d}"
                r.content_sha256 = hashlib.sha256(r.text.encode("utf-8")).hexdigest()
                c.execute(
                    "INSERT INTO chunks (id, document_id, tenant_id, "
                    "ordinal, modality, extractor, route_reason, token_count, "
                    "content_sha256, text, meta) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (r.id, document_id, tenant_id, r.ordinal,
                     r.modality, r.extractor, r.route_reason, r.token_count,
                     r.content_sha256, r.text, json.dumps(r.meta)),
                )
        return [r.id for r in chunks]

    def get_document_chunks(self, tenant_id: str, document_id: str) -> list[ChunkRecord]:
        """Same tenant-or-global relaxation as `get_document` (joined on the
        owning document's scope, since `chunks` itself has no scope column)."""
        with transaction(self.db_path) as c:
            rows = c.execute(
                "SELECT chunks.* FROM chunks JOIN documents ON documents.id = chunks.document_id "
                "WHERE chunks.document_id=? AND (chunks.tenant_id=? OR documents.scope='global') "
                "ORDER BY chunks.ordinal",
                (document_id, tenant_id),
            ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def write_audit(
        self, tenant_id: str, user_id: str, action: str, target: str,
        meta: Optional[dict[str, Any]] = None,
    ) -> None:
        with transaction(self.db_path) as c:
            c.execute(
                "INSERT INTO audit_log (id, tenant_id, user_id, action, target, meta) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (new_object_id(), tenant_id, user_id, action, target,
                 json.dumps(meta or {})),
            )


def _row_to_document(row) -> Document:
    return Document(
        id=row["id"], tenant_id=row["tenant_id"], owner_user_id=row["owner_user_id"],
        source_type=row["source_type"], blob_path=row["blob_path"],
        content_sha256=row["content_sha256"], mime=row["mime"], filename=row["filename"],
        visibility=row["visibility"], acl_user_ids=json.loads(row["acl_user_ids"] or "[]"),
        scope=row["scope"] if "scope" in row.keys() else Scope.TENANT.value,
        extracted_metadata=json.loads(row["extracted_metadata"]) if "extracted_metadata" in row.keys() and row["extracted_metadata"] else {},
        created_at=row["created_at"],
    )


def _row_to_chunk(row) -> ChunkRecord:
    return ChunkRecord(
        ordinal=row["ordinal"], modality=row["modality"], extractor=row["extractor"],
        route_reason=row["route_reason"], token_count=row["token_count"] or 0,
        text=row["text"] or "", meta=json.loads(row["meta"]) if row["meta"] else {},
        id=row["id"], content_sha256=row["content_sha256"],
    )


def _row_to_job(row) -> Job:
    return Job(
        id=row["id"], document_id=row["document_id"], tenant_id=row["tenant_id"],
        stage=row["stage"], status=row["status"], attempts=row["attempts"],
        error=row["error"],
        route_summary=json.loads(row["route_summary"]) if row["route_summary"] else None,
        created_at=row["created_at"], updated_at=row["updated_at"],
    )
