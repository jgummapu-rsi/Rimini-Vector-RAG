"""SQLite implementation of the MetadataStore port."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

from app.shared.adapters.sqlite.db import transaction
from app.shared.domain.models import (
    AuthUser,
    ChunkRecord,
    Document,
    Job,
    JobEvent,
    Principal,
    Role,
    Scope,
    finalize_chunks,
)
from app.shared.ids import new_object_id
from app.shared.ports.metadata_store import EmailAlreadyRegistered, MetadataStore
from app.shared.security import hash_token

_SCHEMA = Path(__file__).with_name("schema.sql")


class SqliteMetadataStore(MetadataStore):
    """SQLite implementation of the MetadataStore port, for local/dev use."""

    def __init__(self, db_path: Path):
        """Bind to the SQLite database file at `db_path` (not created yet)."""
        self.db_path = db_path

    def init_schema(self) -> None:
        """Create the schema if absent, and apply lightweight column migrations
        for databases created before a given column existed."""
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
            # lightweight migration: retry-backoff column on a pre-existing jobs
            # table. A literal default (not datetime('now')) is required -- SQLite
            # rejects a non-constant default in ALTER TABLE ADD COLUMN. Existing
            # rows get a past timestamp, i.e. immediately claimable, which is the
            # correct meaning for jobs queued before backoff existed.
            job_cols = {r["name"] for r in c.execute("PRAGMA table_info(ingestion_jobs)")}
            if "available_at" not in job_cols:
                c.execute("ALTER TABLE ingestion_jobs ADD COLUMN "
                          "available_at TEXT NOT NULL DEFAULT '1970-01-01 00:00:00'")
            if "lease_expires_at" not in job_cols:
                c.execute("ALTER TABLE ingestion_jobs ADD COLUMN lease_expires_at TEXT")
            # lightweight migration: password_hash on a pre-existing users table
            # (added for app.api.onboarding_routes' email/password login).
            user_cols = {r["name"] for r in c.execute("PRAGMA table_info(users)")}
            if "password_hash" not in user_cols:
                c.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")

    def create_tenant(self, name: str) -> str:
        """Create a tenant and return its new id."""
        tid = new_object_id()
        with transaction(self.db_path) as c:
            c.execute("INSERT INTO tenants (id, name) VALUES (?, ?)", (tid, name))
        return tid

    def create_user(self, tenant_id: str, email: str, role: str, token: str) -> str:
        """Create a user with an API token (stored hashed) and return the new user id."""
        uid = new_object_id()
        with transaction(self.db_path) as c:
            c.execute(
                "INSERT INTO users (id, tenant_id, email, role, api_token) "
                "VALUES (?, ?, ?, ?, ?)",
                (uid, tenant_id, email, role, hash_token(token)),
            )
        return uid

    def get_principal_by_token(self, token: str) -> Optional[Principal]:
        """Resolve a bearer token (hashed before lookup) to its active user's Principal, or None."""
        with transaction(self.db_path) as c:
            row = c.execute(
                "SELECT tenant_id, id, role FROM users "
                "WHERE api_token = ? AND status = 'active'",
                (hash_token(token),),
            ).fetchone()
        if not row:
            return None
        return Principal(tenant_id=row["tenant_id"], user_id=row["id"], role=Role(row["role"]))

    def rotate_api_token(self, user_id: str, token: str) -> None:
        """Replace a user's API token (stored hashed), invalidating the previous one."""
        with transaction(self.db_path) as c:
            c.execute("UPDATE users SET api_token = ? WHERE id = ?",
                      (hash_token(token), user_id))

    def get_user_by_email(self, email: str) -> Optional[AuthUser]:
        """Look up an active user by email, preferring the one with a password set."""
        with transaction(self.db_path) as c:
            row = c.execute(
                "SELECT id, tenant_id, email, role, password_hash FROM users "
                "WHERE email = ? AND status = 'active' "
                "ORDER BY (password_hash IS NOT NULL) DESC, created_at DESC LIMIT 1",
                (email,),
            ).fetchone()
        if not row:
            return None
        return AuthUser(id=row["id"], tenant_id=row["tenant_id"], email=row["email"],
                        role=row["role"], password_hash=row["password_hash"])

    def create_user_with_password(
        self, tenant_id: str, email: str, role: str, token: str, password_hash: str,
    ) -> str:
        """Create a user with both an API token (stored hashed) and a password hash (onboarding).

        Raises `EmailAlreadyRegistered` when this violates
        `idx_users_email_password_unique` (schema.sql). SQLite's own
        IntegrityError message names the column(s), not the index
        (`"UNIQUE constraint failed: users.email"` for this partial index,
        verified against SQLite's actual wording -- vs
        `"...users.tenant_id, users.email"` for the composite
        `UNIQUE(tenant_id, email)` and `"...users.api_token"` for the token
        collision), so matching on the exact column-only message tells this
        violation apart from the other two UNIQUE constraints on this table.
        `api_token` colliding is an astronomically unlikely random-token
        clash and `tenant_id` is always freshly created immediately before
        this call, so neither of those two can realistically fire here
        anyway -- but any other IntegrityError is still re-raised as-is
        rather than assumed to be this one.
        """
        uid = new_object_id()
        try:
            with transaction(self.db_path) as c:
                c.execute(
                    "INSERT INTO users (id, tenant_id, email, role, api_token, password_hash) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (uid, tenant_id, email, role, hash_token(token), password_hash),
                )
        except sqlite3.IntegrityError as e:
            if str(e).strip() == "UNIQUE constraint failed: users.email":
                raise EmailAlreadyRegistered(email) from e
            raise
        return uid

    def get_system_config(self, key: str) -> Optional[str]:
        """Return the stored value for a system_config key, or None."""
        with transaction(self.db_path) as c:
            row = c.execute("SELECT value FROM system_config WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_system_config(self, key: str, value: str) -> None:
        """Upsert a system_config key/value pair."""
        with transaction(self.db_path) as c:
            c.execute(
                "INSERT INTO system_config (key, value, updated_at) VALUES (?, ?, datetime('now')) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = datetime('now')",
                (key, value),
            )

    def get_document_by_hash(self, tenant_id: str, sha256: str) -> Optional[Document]:
        """Find an existing document by content hash, for upload de-duplication."""
        with transaction(self.db_path) as c:
            row = c.execute(
                "SELECT * FROM documents WHERE tenant_id = ? AND content_sha256 = ?",
                (tenant_id, sha256),
            ).fetchone()
        return _row_to_document(row) if row else None

    def create_document(self, doc: Document) -> None:
        """Insert a new document record."""
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
        """Overwrite a document's extracted_metadata (author/date/topics/entities)."""
        with transaction(self.db_path) as c:
            c.execute(
                "UPDATE documents SET extracted_metadata=? WHERE tenant_id=? AND id=?",
                (json.dumps(metadata), tenant_id, document_id),
            )

    def delete_document(self, tenant_id: str, document_id: str) -> None:
        """Delete a document and its chunks/job/job-event rows."""
        with transaction(self.db_path) as c:
            c.execute(
                "DELETE FROM chunks WHERE tenant_id = ? AND document_id = ?",
                (tenant_id, document_id),
            )
            c.execute(
                "DELETE FROM job_events WHERE tenant_id = ? AND document_id = ?",
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
        """Insert a new ingestion job record."""
        with transaction(self.db_path) as c:
            c.execute(
                "INSERT INTO ingestion_jobs (id, document_id, tenant_id, stage, status, attempts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (job.id, job.document_id, job.tenant_id, job.stage, job.status, job.attempts),
            )

    def get_job(self, tenant_id: str, job_id: str) -> Optional[Job]:
        """Fetch a job by id within a tenant, or None."""
        with transaction(self.db_path) as c:
            row = c.execute(
                "SELECT * FROM ingestion_jobs WHERE tenant_id = ? AND id = ?",
                (tenant_id, job_id),
            ).fetchone()
        return _row_to_job(row) if row else None

    def set_route_summary(self, job_id: str, summary: dict[str, Any]) -> None:
        """Record which extractor/route each element of a job took."""
        with transaction(self.db_path) as c:
            c.execute(
                "UPDATE ingestion_jobs SET route_summary=?, updated_at=datetime('now') "
                "WHERE id=?",
                (json.dumps(summary), job_id),
            )

    def record_job_event(self, event: JobEvent) -> None:
        """Append one stage-transition event to a job's trace."""
        with transaction(self.db_path) as c:
            c.execute(
                "INSERT INTO job_events (id, job_id, document_id, tenant_id, stage, "
                "seq, status, duration_ms, attempt, detail) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (event.id or new_object_id(), event.job_id, event.document_id,
                 event.tenant_id, event.stage, event.seq, event.status,
                 event.duration_ms, event.attempt, json.dumps(event.detail or {})),
            )

    def get_job_events(self, tenant_id: str, job_id: str) -> list[JobEvent]:
        """Return a job's full stage-transition trace, ordered by attempt/seq."""
        with transaction(self.db_path) as c:
            rows = c.execute(
                "SELECT * FROM job_events WHERE tenant_id = ? AND job_id = ? "
                "ORDER BY attempt, seq",
                (tenant_id, job_id),
            ).fetchall()
        return [_row_to_job_event(r) for r in rows]

    def list_documents(
        self, tenant_id: str, limit: int = 50, offset: int = 0
    ) -> list[tuple[Document, Optional[Job]]]:
        """List a tenant's documents (plus global-scope ones) newest first,
        each paired with its most recent ingestion job if any."""
        with transaction(self.db_path) as c:
            rows = c.execute(
                "SELECT * FROM documents WHERE tenant_id = ? OR scope = 'global' "
                "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (tenant_id, limit, offset),
            ).fetchall()
            out: list[tuple[Document, Optional[Job]]] = []
            for row in rows:
                job_row = c.execute(
                    "SELECT * FROM ingestion_jobs WHERE document_id = ? "
                    "ORDER BY created_at DESC, id DESC LIMIT 1",
                    (row["id"],),
                ).fetchone()
                out.append((_row_to_document(row),
                            _row_to_job(job_row) if job_row else None))
        return out

    def replace_document_chunks(
        self, tenant_id: str, document_id: str, chunks: list[ChunkRecord]
    ) -> list[str]:
        """Replace all of a document's chunks with `chunks`, stamping
        deterministic ids/hashes via the shared domain helper so this adapter
        and the Postgres one cannot drift on chunk identity. Idempotent."""
        ids = finalize_chunks(document_id, chunks)
        with transaction(self.db_path) as c:
            c.execute(
                "DELETE FROM chunks WHERE tenant_id=? AND document_id=?",
                (tenant_id, document_id),
            )
            for r in chunks:
                c.execute(
                    "INSERT INTO chunks (id, document_id, tenant_id, "
                    "ordinal, modality, extractor, route_reason, token_count, "
                    "content_sha256, text, meta) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (r.id, document_id, tenant_id, r.ordinal,
                     r.modality, r.extractor, r.route_reason, r.token_count,
                     r.content_sha256, r.text, json.dumps(r.meta)),
                )
        return ids

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
        """Append one audit-log entry."""
        with transaction(self.db_path) as c:
            c.execute(
                "INSERT INTO audit_log (id, tenant_id, user_id, action, target, meta) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (new_object_id(), tenant_id, user_id, action, target,
                 json.dumps(meta or {})),
            )


def _row_to_document(row) -> Document:
    """Map a `documents` row to a Document."""
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
    """Map a `chunks` row to a ChunkRecord."""
    return ChunkRecord(
        ordinal=row["ordinal"], modality=row["modality"], extractor=row["extractor"],
        route_reason=row["route_reason"], token_count=row["token_count"] or 0,
        text=row["text"] or "", meta=json.loads(row["meta"]) if row["meta"] else {},
        id=row["id"], content_sha256=row["content_sha256"],
    )


def _row_to_job_event(row) -> JobEvent:
    """Map a `job_events` row to a JobEvent."""
    return JobEvent(
        job_id=row["job_id"], document_id=row["document_id"],
        tenant_id=row["tenant_id"], stage=row["stage"], seq=row["seq"],
        status=row["status"], duration_ms=row["duration_ms"] or 0.0,
        attempt=row["attempt"] or 0,
        detail=json.loads(row["detail"]) if row["detail"] else {},
        id=row["id"], at=row["at"],
    )


def _row_to_job(row) -> Job:
    """Map an `ingestion_jobs` row to a Job."""
    return Job(
        id=row["id"], document_id=row["document_id"], tenant_id=row["tenant_id"],
        stage=row["stage"], status=row["status"], attempts=row["attempts"],
        error=row["error"],
        route_summary=json.loads(row["route_summary"]) if row["route_summary"] else None,
        created_at=row["created_at"], updated_at=row["updated_at"],
    )
