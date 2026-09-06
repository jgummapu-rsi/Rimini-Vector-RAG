"""Postgres implementation of the MetadataStore port."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import psycopg2.errors
from psycopg2.extras import Json

from app.shared.adapters.postgres.db import transaction
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


class PostgresMetadataStore(MetadataStore):
    """Postgres implementation of the MetadataStore port, for production use."""

    def __init__(self, dsn: str):
        """Bind to the Postgres database at `dsn` (not connected yet)."""
        self.dsn = dsn

    def init_schema(self) -> None:
        """Create the schema if absent, and apply idempotent column migrations
        for databases created before a given column existed."""
        with transaction(self.dsn) as cur:
            # These ADD COLUMN migrations must run BEFORE schema.sql, guarded by
            # an existence check: schema.sql's CREATE INDEX statements reference
            # these same columns, and CREATE TABLE IF NOT EXISTS is a no-op on a
            # database whose tables predate these columns -- so schema.sql's own
            # CREATE INDEX would fail on the still-missing column if this didn't
            # add it first. On a fresh database the tables don't exist yet, so
            # each IF EXISTS check is false and schema.sql creates the columns
            # itself in one shot.
            cur.execute("""
                DO $$ BEGIN
                    IF EXISTS (SELECT 1 FROM information_schema.tables
                               WHERE table_name = 'ingestion_jobs') THEN
                        ALTER TABLE ingestion_jobs
                            ADD COLUMN IF NOT EXISTS available_at TIMESTAMPTZ
                            NOT NULL DEFAULT now();
                        ALTER TABLE ingestion_jobs
                            ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ;
                    END IF;
                    IF EXISTS (SELECT 1 FROM information_schema.tables
                               WHERE table_name = 'users') THEN
                        ALTER TABLE users
                            ADD COLUMN IF NOT EXISTS password_hash TEXT;
                    END IF;
                END $$;
            """)
            cur.execute(_SCHEMA.read_text(encoding="utf-8"))

    def create_tenant(self, name: str) -> str:
        """Create a tenant and return its new id."""
        tid = new_object_id()
        with transaction(self.dsn) as cur:
            cur.execute("INSERT INTO tenants (id, name) VALUES (%s, %s)", (tid, name))
        return tid

    def create_user(self, tenant_id: str, email: str, role: str, token: str) -> str:
        """Create a user with an API token (stored hashed) and return the new user id."""
        uid = new_object_id()
        with transaction(self.dsn) as cur:
            cur.execute(
                "INSERT INTO users (id, tenant_id, email, role, api_token) "
                "VALUES (%s, %s, %s, %s, %s)",
                (uid, tenant_id, email, role, hash_token(token)),
            )
        return uid

    def get_principal_by_token(self, token: str) -> Optional[Principal]:
        """Resolve a bearer token (hashed before lookup) to its active user's Principal, or None."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT tenant_id, id, role FROM users "
                "WHERE api_token = %s AND status = 'active'",
                (hash_token(token),),
            )
            row = cur.fetchone()
        if not row:
            return None
        return Principal(tenant_id=row["tenant_id"], user_id=row["id"], role=Role(row["role"]))

    def rotate_api_token(self, user_id: str, token: str) -> None:
        """Replace a user's API token (stored hashed), invalidating the previous one."""
        with transaction(self.dsn) as cur:
            cur.execute("UPDATE users SET api_token = %s WHERE id = %s",
                        (hash_token(token), user_id))

    def get_user_by_email(self, email: str) -> Optional[AuthUser]:
        """Look up an active user by email, preferring the one with a password set."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT id, tenant_id, email, role, password_hash FROM users "
                "WHERE email = %s AND status = 'active' "
                "ORDER BY (password_hash IS NOT NULL) DESC, created_at DESC LIMIT 1",
                (email,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return AuthUser(id=row["id"], tenant_id=row["tenant_id"], email=row["email"],
                        role=row["role"], password_hash=row["password_hash"])

    def create_user_with_password(
        self, tenant_id: str, email: str, role: str, token: str, password_hash: str,
    ) -> str:
        """Create a user with both an API token (stored hashed) and a password hash (onboarding).

        Raises `EmailAlreadyRegistered` when this violates
        `idx_users_email_password_unique` (schema.sql). Postgres's
        UniqueViolation exposes the actual index name via
        `e.diag.constraint_name`, so this is an exact check, unlike the SQLite
        adapter (which has to pattern-match SQLite's column-only message).
        `api_token` colliding is an astronomically unlikely random-token
        clash and `tenant_id` is always freshly created immediately before
        this call, so `UNIQUE(tenant_id, email)` can't realistically fire
        here either -- but any other UniqueViolation (or non-unique
        IntegrityError) is still re-raised as-is.
        """
        uid = new_object_id()
        try:
            with transaction(self.dsn) as cur:
                cur.execute(
                    "INSERT INTO users (id, tenant_id, email, role, api_token, password_hash) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (uid, tenant_id, email, role, hash_token(token), password_hash),
                )
        except psycopg2.errors.UniqueViolation as e:
            if getattr(e.diag, "constraint_name", None) == "idx_users_email_password_unique":
                raise EmailAlreadyRegistered(email) from e
            raise
        return uid

    def get_system_config(self, key: str) -> Optional[str]:
        """Return the stored value for a system_config key, or None."""
        with transaction(self.dsn) as cur:
            cur.execute("SELECT value FROM system_config WHERE key = %s", (key,))
            row = cur.fetchone()
        return row["value"] if row else None

    def set_system_config(self, key: str, value: str) -> None:
        """Upsert a system_config key/value pair."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "INSERT INTO system_config (key, value, updated_at) VALUES (%s, %s, now()) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                (key, value),
            )

    def get_document_by_hash(self, tenant_id: str, sha256: str) -> Optional[Document]:
        """Find an existing document by content hash, for upload de-duplication."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT * FROM documents WHERE tenant_id = %s AND content_sha256 = %s",
                (tenant_id, sha256),
            )
            row = cur.fetchone()
        return _row_to_document(row) if row else None

    def create_document(self, doc: Document) -> None:
        """Insert a new document record."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "INSERT INTO documents (id, tenant_id, owner_user_id, source_type, "
                "blob_path, content_sha256, mime, filename, visibility, acl_user_ids, scope, "
                "extracted_metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    doc.id, doc.tenant_id, doc.owner_user_id, doc.source_type,
                    doc.blob_path, doc.content_sha256, doc.mime, doc.filename,
                    doc.visibility, Json(doc.acl_user_ids), doc.scope,
                    Json(doc.extracted_metadata),
                ),
            )

    def get_document(self, tenant_id: str, document_id: str) -> Optional[Document]:
        """A document is fetchable if it belongs to this tenant, OR it is
        scope=global (readable cross-tenant; write/delete stay tenant-owned --
        enforced by the caller comparing doc.tenant_id, not here)."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT * FROM documents WHERE id = %s AND (tenant_id = %s OR scope = 'global')",
                (document_id, tenant_id),
            )
            row = cur.fetchone()
        return _row_to_document(row) if row else None

    def set_document_metadata(
        self, tenant_id: str, document_id: str, metadata: dict[str, Any]
    ) -> None:
        """Overwrite a document's extracted_metadata (author/date/topics/entities)."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "UPDATE documents SET extracted_metadata=%s WHERE tenant_id=%s AND id=%s",
                (Json(metadata), tenant_id, document_id),
            )

    def delete_document(self, tenant_id: str, document_id: str) -> None:
        """Delete a document and its chunks/job/job-event rows."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "DELETE FROM chunks WHERE tenant_id = %s AND document_id = %s",
                (tenant_id, document_id),
            )
            cur.execute(
                "DELETE FROM job_events WHERE tenant_id = %s AND document_id = %s",
                (tenant_id, document_id),
            )
            cur.execute(
                "DELETE FROM ingestion_jobs WHERE tenant_id = %s AND document_id = %s",
                (tenant_id, document_id),
            )
            cur.execute(
                "DELETE FROM documents WHERE tenant_id = %s AND id = %s",
                (tenant_id, document_id),
            )

    def create_job(self, job: Job) -> None:
        """Insert a new ingestion job record."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "INSERT INTO ingestion_jobs (id, document_id, tenant_id, stage, status, attempts) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (job.id, job.document_id, job.tenant_id, job.stage, job.status, job.attempts),
            )

    def get_job(self, tenant_id: str, job_id: str) -> Optional[Job]:
        """Fetch a job by id within a tenant, or None."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT * FROM ingestion_jobs WHERE tenant_id = %s AND id = %s",
                (tenant_id, job_id),
            )
            row = cur.fetchone()
        return _row_to_job(row) if row else None

    def set_route_summary(self, job_id: str, summary: dict[str, Any]) -> None:
        """Record which extractor/route each element of a job took."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "UPDATE ingestion_jobs SET route_summary=%s, updated_at=now() WHERE id=%s",
                (Json(summary), job_id),
            )

    def record_job_event(self, event: JobEvent) -> None:
        """Append one stage-transition event to a job's trace."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "INSERT INTO job_events (id, job_id, document_id, tenant_id, stage, "
                "seq, status, duration_ms, attempt, detail) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (event.id or new_object_id(), event.job_id, event.document_id,
                 event.tenant_id, event.stage, event.seq, event.status,
                 event.duration_ms, event.attempt, Json(event.detail or {})),
            )

    def get_job_events(self, tenant_id: str, job_id: str) -> list[JobEvent]:
        """Return a job's full stage-transition trace, ordered by attempt/seq."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT * FROM job_events WHERE tenant_id = %s AND job_id = %s "
                "ORDER BY attempt, seq",
                (tenant_id, job_id),
            )
            rows = cur.fetchall()
        return [_row_to_job_event(r) for r in rows]

    def list_documents(
        self, tenant_id: str, limit: int = 50, offset: int = 0
    ) -> list[tuple[Document, Optional[Job]]]:
        """List a tenant's documents (plus global-scope ones) newest first,
        each paired with its most recent ingestion job if any.

        Uses LEFT JOIN LATERAL to keep this one round trip. Job columns are
        aliased (j_*) because RealDictCursor would otherwise collapse the
        columns both tables share -- id, tenant_id, created_at.
        """
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT d.*, "
                "       j.id AS j_id, j.stage AS j_stage, j.status AS j_status, "
                "       j.attempts AS j_attempts, j.error AS j_error, "
                "       j.route_summary AS j_route_summary, "
                "       j.created_at AS j_created_at, j.updated_at AS j_updated_at "
                "FROM documents d "
                "LEFT JOIN LATERAL ("
                "    SELECT * FROM ingestion_jobs WHERE document_id = d.id "
                "    ORDER BY created_at DESC, id DESC LIMIT 1"
                ") j ON TRUE "
                "WHERE d.tenant_id = %s OR d.scope = 'global' "
                "ORDER BY d.created_at DESC, d.id DESC LIMIT %s OFFSET %s",
                (tenant_id, limit, offset),
            )
            rows = cur.fetchall()
        return [(_row_to_document(r), _row_to_lateral_job(r)) for r in rows]

    def replace_document_chunks(
        self, tenant_id: str, document_id: str, chunks: list[ChunkRecord]
    ) -> list[str]:
        """Replace all of a document's chunks with `chunks`, stamping
        deterministic ids/hashes via the shared domain helper so this adapter
        and the SQLite one cannot drift on chunk identity. Idempotent."""
        ids = finalize_chunks(document_id, chunks)
        with transaction(self.dsn) as cur:
            cur.execute(
                "DELETE FROM chunks WHERE tenant_id=%s AND document_id=%s",
                (tenant_id, document_id),
            )
            for r in chunks:
                cur.execute(
                    "INSERT INTO chunks (id, document_id, tenant_id, "
                    "ordinal, modality, extractor, route_reason, token_count, "
                    "content_sha256, text, meta) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (r.id, document_id, tenant_id, r.ordinal,
                     r.modality, r.extractor, r.route_reason, r.token_count,
                     r.content_sha256, r.text, Json(r.meta)),
                )
        return ids

    def get_document_chunks(self, tenant_id: str, document_id: str) -> list[ChunkRecord]:
        """Same tenant-or-global relaxation as `get_document` (joined on the
        owning document's scope, since `chunks` itself has no scope column)."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT chunks.* FROM chunks JOIN documents ON documents.id = chunks.document_id "
                "WHERE chunks.document_id=%s AND (chunks.tenant_id=%s OR documents.scope='global') "
                "ORDER BY chunks.ordinal",
                (document_id, tenant_id),
            )
            rows = cur.fetchall()
        return [_row_to_chunk(r) for r in rows]

    def write_audit(
        self, tenant_id: str, user_id: str, action: str, target: str,
        meta: Optional[dict[str, Any]] = None,
    ) -> None:
        """Append one audit-log entry."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "INSERT INTO audit_log (id, tenant_id, user_id, action, target, meta) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (new_object_id(), tenant_id, user_id, action, target, Json(meta or {})),
            )


def _row_to_document(row) -> Document:
    """Map a `documents` row to a Document."""
    return Document(
        id=row["id"], tenant_id=row["tenant_id"], owner_user_id=row["owner_user_id"],
        source_type=row["source_type"], blob_path=row["blob_path"],
        content_sha256=row["content_sha256"], mime=row["mime"], filename=row["filename"],
        visibility=row["visibility"], acl_user_ids=row["acl_user_ids"] or [],
        scope=row["scope"] if "scope" in row.keys() else Scope.TENANT.value,
        extracted_metadata=row["extracted_metadata"] if row.get("extracted_metadata") else {},
        created_at=str(row["created_at"]) if row["created_at"] is not None else None,
    )


def _row_to_chunk(row) -> ChunkRecord:
    """Map a `chunks` row to a ChunkRecord."""
    return ChunkRecord(
        ordinal=row["ordinal"], modality=row["modality"], extractor=row["extractor"],
        route_reason=row["route_reason"], token_count=row["token_count"] or 0,
        text=row["text"] or "", meta=row["meta"] or {},
        id=row["id"], content_sha256=row["content_sha256"],
    )


def _row_to_job_event(row) -> JobEvent:
    """Map a `job_events` row to a JobEvent."""
    return JobEvent(
        job_id=row["job_id"], document_id=row["document_id"],
        tenant_id=row["tenant_id"], stage=row["stage"], seq=row["seq"],
        status=row["status"], duration_ms=row["duration_ms"] or 0.0,
        attempt=row["attempt"] or 0, detail=row["detail"] or {},
        id=row["id"], at=str(row["at"]) if row["at"] is not None else None,
    )


def _row_to_lateral_job(row) -> Optional[Job]:
    """Rebuild the Job from the j_*-aliased columns of `list_documents`' LATERAL
    join. A document with no surviving job row yields None."""
    if row.get("j_id") is None:
        return None
    return Job(
        id=row["j_id"], document_id=row["id"], tenant_id=row["tenant_id"],
        stage=row["j_stage"], status=row["j_status"], attempts=row["j_attempts"],
        error=row["j_error"], route_summary=row["j_route_summary"],
        created_at=str(row["j_created_at"]) if row["j_created_at"] is not None else None,
        updated_at=str(row["j_updated_at"]) if row["j_updated_at"] is not None else None,
    )


def _row_to_job(row) -> Job:
    """Map an `ingestion_jobs` row to a Job."""
    return Job(
        id=row["id"], document_id=row["document_id"], tenant_id=row["tenant_id"],
        stage=row["stage"], status=row["status"], attempts=row["attempts"],
        error=row["error"], route_summary=row["route_summary"],
        created_at=str(row["created_at"]) if row["created_at"] is not None else None,
        updated_at=str(row["updated_at"]) if row["updated_at"] is not None else None,
    )
