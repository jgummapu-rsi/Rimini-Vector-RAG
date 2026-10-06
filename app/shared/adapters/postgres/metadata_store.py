"""Postgres implementation of the MetadataStore port."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import psycopg2.errors
from psycopg2.extras import Json

from app.shared.adapters.postgres.db import transaction
from app.shared.domain.models import (
    AuthUser,
    ChunkRecord,
    Document,
    DocumentVersion,
    Job,
    JobEvent,
    Principal,
    Role,
    Scope,
    finalize_chunks,
)
from app.shared.ids import new_object_id
from app.shared.ports.metadata_store import EmailAlreadyRegistered, IngestionConflict, MetadataStore
from app.shared.security import hash_token

_SCHEMA = Path(__file__).with_name("schema.sql")


class PostgresMetadataStore(MetadataStore):
    """Postgres implementation of the MetadataStore port, for production use."""

    def __init__(self, dsn: str):
        """Bind to the Postgres database at `dsn` (not connected yet)."""
        self.dsn = dsn

    def corpus_epoch(self, tenant_id: str) -> str:
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT scope_key,revision FROM corpus_epochs WHERE scope_key IN (%s,'global')",
                (tenant_id,),
            )
            revisions = {row["scope_key"]: row["revision"] for row in cur.fetchall()}
        return f"{revisions.get(tenant_id, 0)}:{revisions.get('global', 0)}"

    def set_gateway_config(self, base_url: str, api_key: str) -> None:
        with transaction(self.dsn) as cur:
            cur.execute(
                "INSERT INTO system_config(key,value) VALUES ('gateway_config',%s) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=now()",
                (Json({"base_url": base_url, "api_key": api_key}),),
            )

    def get_gateway_config(self) -> tuple[str, str] | None:
        with transaction(self.dsn) as cur:
            cur.execute("SELECT value FROM system_config WHERE key='gateway_config'")
            row = cur.fetchone()
        if row is None:
            return None
        config = json.loads(row["value"])
        return config["base_url"], config["api_key"]

    def update_document_access(
        self, tenant_id: str, document_id: str, visibility: str, scope: str
    ) -> None:
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT scope FROM documents WHERE tenant_id=%s AND id=%s FOR UPDATE",
                (tenant_id, document_id),
            )
            row = cur.fetchone()
            if row is None:
                raise ValueError("Document no longer exists")
            cur.execute(
                "UPDATE documents SET visibility=%s,scope=%s,updated_at=now() WHERE tenant_id=%s AND id=%s",
                (visibility, scope, tenant_id, document_id),
            )
            key = "global" if scope == "global" or row["scope"] == "global" else tenant_id
            cur.execute(
                "INSERT INTO corpus_epochs VALUES(%s,1) ON CONFLICT(scope_key) DO UPDATE SET revision=corpus_epochs.revision+1",
                (key,),
            )

    def init_schema(self) -> None:
        """Create the schema if absent, and apply idempotent column migrations
        for databases created before a given column existed."""
        with transaction(self.dsn) as cur:
            cur.execute("""
                DO $$ BEGIN
                    IF EXISTS (SELECT 1 FROM information_schema.tables
                               WHERE table_name = 'ingestion_jobs') THEN
                        ALTER TABLE ingestion_jobs
                            ADD COLUMN IF NOT EXISTS available_at TIMESTAMPTZ
                            NOT NULL DEFAULT now();
                        ALTER TABLE ingestion_jobs
                            ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ;
                        ALTER TABLE ingestion_jobs
                            ADD COLUMN IF NOT EXISTS blob_path TEXT;
                        ALTER TABLE ingestion_jobs
                            ADD COLUMN IF NOT EXISTS content_sha256 TEXT;
                        ALTER TABLE ingestion_jobs
                            ADD COLUMN IF NOT EXISTS version INTEGER;
                    END IF;
                    IF EXISTS (SELECT 1 FROM information_schema.tables
                               WHERE table_name = 'users') THEN
                        ALTER TABLE users
                            ADD COLUMN IF NOT EXISTS password_hash TEXT;
                    END IF;
                    IF EXISTS (SELECT 1 FROM information_schema.tables
                               WHERE table_name = 'documents') THEN
                        ALTER TABLE documents
                            ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1;
                        -- Deliberately NULLable with no default here, unlike
                        -- schema.sql's `NOT NULL DEFAULT now()`: `now()` would
                        -- claim every pre-existing document had just been
                        -- edited. The backfill below gives the truthful answer
                        -- (never re-ingested => last updated when created), and
                        ALTER TABLE documents
                            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;
                        UPDATE documents SET updated_at = created_at
                            WHERE updated_at IS NULL;
                    END IF;
                    IF EXISTS (SELECT 1 FROM information_schema.tables
                               WHERE table_name = 'document_versions') THEN
                        ALTER TABLE document_versions ADD COLUMN IF NOT EXISTS mime TEXT;
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

    def get_workspace_models(self, tenant_id: str) -> dict | None:
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT chat_model,embedding_model FROM workspace_models WHERE tenant_id=%s",
                (tenant_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def create_workspace_with_models(
        self,
        name: str,
        email: str,
        token: str,
        password_hash: str,
        chat_model: str,
        embedding_model: str,
    ) -> tuple[str, str]:
        tenant_id, user_id = new_object_id(), new_object_id()
        try:
            with transaction(self.dsn) as cur:
                cur.execute("INSERT INTO tenants(id,name) VALUES(%s,%s)", (tenant_id, name))
                cur.execute(
                    "INSERT INTO users(id,tenant_id,email,role,api_token,password_hash) "
                    "VALUES(%s,%s,%s,'admin',%s,%s)",
                    (user_id, tenant_id, email, hash_token(token), password_hash),
                )
                cur.execute(
                    "INSERT INTO workspace_models(tenant_id,chat_model,embedding_model) VALUES(%s,%s,%s)",
                    (tenant_id, chat_model, embedding_model),
                )
        except psycopg2.errors.UniqueViolation as exc:
            if exc.diag.constraint_name == "idx_users_email_password_unique":
                raise EmailAlreadyRegistered(email) from exc
            raise
        return tenant_id, user_id

    def set_workspace_models(self, tenant_id: str, chat_model: str, embedding_model: str) -> None:
        # Immutable after signup: existing document vectors must never silently
        # switch spaces when an account logs in or another member joins.
        with transaction(self.dsn) as cur:
            cur.execute(
                "INSERT INTO workspace_models(tenant_id,chat_model,embedding_model) VALUES (%s,%s,%s)",
                (tenant_id, chat_model, embedding_model),
            )

    def get_workspace_identity(self, principal: Principal) -> dict[str, Any]:
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT u.email, t.name AS organization FROM users u "
                "JOIN tenants t ON t.id=u.tenant_id WHERE u.id=%s AND u.tenant_id=%s",
                (principal.user_id, principal.tenant_id),
            )
            return dict(cur.fetchone())

    def list_tenant_members(self, tenant_id: str, limit: int, offset: int) -> dict[str, Any]:
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT count(*) AS total FROM users WHERE tenant_id=%s AND status<>'deleted'",
                (tenant_id,),
            )
            total = cur.fetchone()["total"]
            cur.execute(
                "SELECT id, email, role, status, created_at FROM users "
                "WHERE tenant_id=%s AND status<>'deleted' "
                "ORDER BY created_at, id LIMIT %s OFFSET %s",
                (tenant_id, limit, offset),
            )
            return {"members": [dict(row) for row in cur.fetchall()], "total": total}

    def delete_tenant_member(self, principal: Principal, member_id: str) -> str:
        with transaction(self.dsn) as cur:
            # Serialize removals within the organization, then recheck the actor.
            # Two administrators cannot concurrently delete each other's access.
            cur.execute("SELECT id FROM tenants WHERE id=%s FOR UPDATE", (principal.tenant_id,))
            cur.execute(
                "SELECT id FROM users WHERE id=%s AND tenant_id=%s "
                "AND status='active' AND role='admin' FOR UPDATE",
                (principal.user_id, principal.tenant_id),
            )
            if cur.fetchone() is None:
                raise PermissionError("Administrator access is no longer active.")
            if member_id == principal.user_id:
                raise ValueError("You cannot delete your own account. Ask another administrator.")
            cur.execute(
                "SELECT email FROM users WHERE id=%s AND tenant_id=%s "
                "AND status<>'deleted' FOR UPDATE",
                (member_id, principal.tenant_id),
            )
            member = cur.fetchone()
            if member is None:
                raise LookupError("This account was not found in your organization.")
            cur.execute(
                "UPDATE users SET status='deleted', password_hash=NULL, api_token=%s, email=%s "
                "WHERE id=%s AND tenant_id=%s",
                (
                    hash_token(new_object_id()),
                    f"deleted-{member_id}@removed.invalid",
                    member_id,
                    principal.tenant_id,
                ),
            )
            cur.execute(
                "INSERT INTO audit_log (id, tenant_id, user_id, action, target, meta) "
                "VALUES (%s,%s,%s,%s,%s,%s)",
                (
                    new_object_id(),
                    principal.tenant_id,
                    principal.user_id,
                    "team_member_deleted",
                    member_id,
                    Json({"email": member["email"]}),
                ),
            )
            return member["email"]

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

    def get_principal_by_token(self, token: str) -> Principal | None:
        """Resolve a bearer token (hashed before lookup) to its active user's Principal, or None."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT u.tenant_id,u.id,u.role FROM users u JOIN tenants t ON t.id=u.tenant_id "
                "WHERE u.api_token = %s AND u.status = 'active' AND t.status='active'",
                (hash_token(token),),
            )
            row = cur.fetchone()
        if not row:
            return None
        return Principal(tenant_id=row["tenant_id"], user_id=row["id"], role=Role(row["role"]))

    def rotate_api_token(self, user_id: str, token: str) -> None:
        """Replace a user's API token (stored hashed), invalidating the previous one."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "UPDATE users SET api_token = %s WHERE id = %s", (hash_token(token), user_id)
            )

    def get_user_by_email(self, email: str) -> AuthUser | None:
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
        return AuthUser(
            id=row["id"],
            tenant_id=row["tenant_id"],
            email=row["email"],
            role=row["role"],
            password_hash=row["password_hash"],
        )

    def create_user_with_password(
        self,
        tenant_id: str,
        email: str,
        role: str,
        token: str,
        password_hash: str,
    ) -> str:
        """Create a user with both an API token (stored hashed) and a password hash (onboarding).

        Raises `EmailAlreadyRegistered` when this violates
        `idx_users_email_password_unique` (schema.sql). Postgres's
        UniqueViolation exposes the actual index name via
        `e.diag.constraint_name`, so this is an exact check.
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
            if getattr(e.diag, "constraint_name", None) in {
                "idx_users_email_password_unique",
                "users_tenant_id_email_key",
            }:
                raise EmailAlreadyRegistered(email) from e
            raise
        return uid

    def get_system_config(self, key: str) -> str | None:
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

    def get_document_by_hash(
        self, tenant_id: str, sha256: str, owner_user_id: str
    ) -> Document | None:
        """Find an existing document by content hash, for upload de-duplication."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT * FROM documents WHERE tenant_id = %s AND content_sha256 = %s AND owner_user_id = %s",
                (tenant_id, sha256, owner_user_id),
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
                    doc.id,
                    doc.tenant_id,
                    doc.owner_user_id,
                    doc.source_type,
                    doc.blob_path,
                    doc.content_sha256,
                    doc.mime,
                    doc.filename,
                    doc.visibility,
                    Json(doc.acl_user_ids),
                    doc.scope,
                    Json(doc.extracted_metadata),
                ),
            )

    def get_document(self, tenant_id: str, document_id: str) -> Document | None:
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

    def get_document_by_filename(
        self, tenant_id: str, owner_user_id: str, filename: str
    ) -> Document | None:
        """The caller's own newest document with this filename (re-ingest target
        resolution). Owner-scoped on purpose -- see the port docstring."""
        if not filename:
            return None
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT * FROM documents "
                "WHERE tenant_id = %s AND owner_user_id = %s AND filename = %s "
                "ORDER BY created_at DESC, id DESC LIMIT 1",
                (tenant_id, owner_user_id, filename),
            )
            row = cur.fetchone()
        return _row_to_document(row) if row else None

    def set_document_metadata(
        self, tenant_id: str, document_id: str, metadata: dict[str, Any]
    ) -> None:
        """Overwrite a document's extracted_metadata (author/date/topics/entities)."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "UPDATE documents SET extracted_metadata=%s, updated_at=now() "
                "WHERE tenant_id=%s AND id=%s",
                (Json(metadata), tenant_id, document_id),
            )

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
        """Repoint a document at new bytes, bumping `version`; return the new version."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "UPDATE documents SET blob_path=%s, content_sha256=%s, mime=%s, "
                "source_type=%s, filename=%s, visibility=%s, scope=%s, "
                "version = version + 1, updated_at = now() "
                "WHERE tenant_id=%s AND id=%s RETURNING version",
                (
                    blob_path,
                    content_sha256,
                    mime,
                    source_type,
                    filename,
                    visibility,
                    scope,
                    tenant_id,
                    document_id,
                ),
            )
            row = cur.fetchone()
        return int(row["version"]) if row else 1

    def add_document_version(self, version: DocumentVersion) -> None:
        """Append one content-version history row."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "INSERT INTO document_versions (id, document_id, tenant_id, version, "
                "content_sha256, blob_path, filename, mime, byte_size, uploaded_by, job_id, "
                "chunks_added, chunks_removed, chunks_unchanged, delta) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    version.id,
                    version.document_id,
                    version.tenant_id,
                    version.version,
                    version.content_sha256,
                    version.blob_path,
                    version.filename,
                    version.mime,
                    version.byte_size,
                    version.uploaded_by,
                    version.job_id,
                    version.chunks_added,
                    version.chunks_removed,
                    version.chunks_unchanged,
                    Json(version.delta or {}),
                ),
            )

    def list_document_versions(self, tenant_id: str, document_id: str) -> list[DocumentVersion]:
        """A document's content versions, newest first."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT * FROM document_versions WHERE tenant_id = %s AND document_id = %s "
                "ORDER BY version DESC",
                (tenant_id, document_id),
            )
            rows = cur.fetchall()
        return [_row_to_document_version(r) for r in rows]

    def get_document_version(
        self, tenant_id: str, document_id: str, version: int
    ) -> DocumentVersion | None:
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT * FROM document_versions WHERE tenant_id=%s AND document_id=%s AND version=%s",
                (tenant_id, document_id, version),
            )
            row = cur.fetchone()
        return _row_to_document_version(row) if row else None

    def set_version_delta(self, job_id: str, delta: dict[str, Any]) -> None:
        """Record the chunk delta against the version row this job created.
        A no-op when no row matches (a /reprocess job creates no new version)."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "UPDATE document_versions SET chunks_added=%s, chunks_removed=%s, "
                "chunks_unchanged=%s, delta=%s WHERE job_id=%s",
                (
                    delta.get("added"),
                    delta.get("removed"),
                    delta.get("unchanged"),
                    Json(delta or {}),
                    job_id,
                ),
            )

    def count_blob_references(self, blob_path: str, exclude_document_id: str) -> int:
        """How many other documents/versions still point at `blob_path`."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT (SELECT COUNT(*) FROM documents "
                "        WHERE blob_path = %s AND id <> %s) "
                "     + (SELECT COUNT(*) FROM document_versions "
                "        WHERE blob_path = %s AND document_id <> %s) AS n",
                (blob_path, exclude_document_id, blob_path, exclude_document_id),
            )
            row = cur.fetchone()
        return int(row["n"]) if row else 0

    def delete_document(self, tenant_id: str, document_id: str) -> None:
        """Delete a document and its chunks/job/job-event/version rows."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT scope FROM documents WHERE id=%s AND tenant_id=%s FOR UPDATE",
                (document_id, tenant_id),
            )
            document = cur.fetchone()
            if document is None:
                return
            cur.execute(
                "INSERT INTO deleted_documents(document_id,tenant_id) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                (document_id, tenant_id),
            )
            cur.execute(
                "INSERT INTO corpus_epochs(scope_key,revision) VALUES (%s,1) "
                "ON CONFLICT(scope_key) DO UPDATE SET revision=corpus_epochs.revision+1",
                ("global" if document["scope"] == "global" else tenant_id,),
            )
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
                "DELETE FROM document_versions WHERE tenant_id = %s AND document_id = %s",
                (tenant_id, document_id),
            )
            cur.execute(
                "DELETE FROM documents WHERE tenant_id = %s AND id = %s",
                (tenant_id, document_id),
            )

    def create_job(self, job: Job) -> None:
        """Insert a new ingestion job record, pinned to whatever content
        snapshot (`blob_path`/`content_sha256`/`version`) the caller set on
        `job` -- None on all three for a legacy-shaped call."""
        with transaction(self.dsn) as cur:
            self._lock_idle_document(cur, job.tenant_id, job.document_id)
            cur.execute(
                "INSERT INTO ingestion_jobs (id, document_id, tenant_id, stage, status, "
                "attempts, blob_path, content_sha256, version) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    job.id,
                    job.document_id,
                    job.tenant_id,
                    job.stage,
                    job.status,
                    job.attempts,
                    job.blob_path,
                    job.content_sha256,
                    job.version,
                ),
            )
            cur.execute(
                "UPDATE ingestion_jobs j SET filename=d.filename, source_type=d.source_type, "
                "blob_path=coalesce(j.blob_path,d.blob_path), "
                "content_sha256=coalesce(j.content_sha256,d.content_sha256), version=coalesce(j.version,d.version) "
                "FROM documents d WHERE j.id=%s AND d.id=j.document_id",
                (job.id,),
            )

    def has_pending_job(self, tenant_id: str, document_id: str) -> bool:
        """True if a job for this document is currently queued or running.

        Single-flight guard (finding 1.2): callers use this to refuse a second
        update/reprocess while one is already in flight, rather than letting
        two jobs for the same document run concurrently and race each other's
        `replace_document_chunks`/vector-upsert calls."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT 1 FROM ingestion_jobs WHERE tenant_id=%s AND document_id=%s "
                "AND status IN ('queued', 'running') LIMIT 1",
                (tenant_id, document_id),
            )
            return cur.fetchone() is not None

    def create_document_with_job(
        self, doc: Document, job: Job, initial_version: DocumentVersion | None = None
    ) -> None:
        """Atomically insert a new document and its first ingestion job, pinned
        to the document's initial content -- so a crash between the two writes
        can never leave a document with no job, or a job pointing at content
        that was never actually persisted."""
        with transaction(self.dsn) as cur:
            if job.document_id != doc.id or job.tenant_id != doc.tenant_id:
                raise ValueError("Initial ingestion job must belong to its document")
            if initial_version is not None:
                if (
                    initial_version.document_id != doc.id
                    or initial_version.tenant_id != doc.tenant_id
                    or initial_version.job_id != job.id
                    or initial_version.version != doc.version
                    or initial_version.content_sha256 != doc.content_sha256
                    or initial_version.blob_path != doc.blob_path
                ):
                    raise ValueError("Initial version must describe the queued source")
                self._lock_owner(cur, doc.tenant_id, doc.owner_user_id)
                cur.execute(
                    "SELECT 1 FROM documents WHERE tenant_id=%s AND owner_user_id=%s "
                    "AND (filename=%s OR content_sha256=%s) LIMIT 1",
                    (doc.tenant_id, doc.owner_user_id, doc.filename, doc.content_sha256),
                )
                if cur.fetchone() is not None:
                    raise IngestionConflict(
                        "An upload for this filename or content was accepted concurrently; retry against the existing document"
                    )
            cur.execute(
                "INSERT INTO documents (id, tenant_id, owner_user_id, source_type, "
                "blob_path, content_sha256, mime, filename, visibility, acl_user_ids, scope, "
                "extracted_metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    doc.id,
                    doc.tenant_id,
                    doc.owner_user_id,
                    doc.source_type,
                    doc.blob_path,
                    doc.content_sha256,
                    doc.mime,
                    doc.filename,
                    doc.visibility,
                    Json(doc.acl_user_ids),
                    doc.scope,
                    Json(doc.extracted_metadata),
                ),
            )
            cur.execute(
                "INSERT INTO ingestion_jobs (id, document_id, tenant_id, stage, status, "
                "attempts, blob_path, content_sha256, version) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    job.id,
                    job.document_id,
                    job.tenant_id,
                    job.stage,
                    job.status,
                    job.attempts,
                    doc.blob_path,
                    doc.content_sha256,
                    doc.version,
                ),
            )
            cur.execute(
                "UPDATE ingestion_jobs SET filename=%s, source_type=%s WHERE id=%s",
                (doc.filename, doc.source_type, job.id),
            )
            if initial_version is not None:
                cur.execute(
                    "INSERT INTO document_versions "
                    "(id,document_id,tenant_id,version,content_sha256,blob_path,filename,mime,byte_size,uploaded_by,job_id) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        initial_version.id,
                        doc.id,
                        doc.tenant_id,
                        doc.version,
                        doc.content_sha256,
                        doc.blob_path,
                        doc.filename,
                        initial_version.mime,
                        initial_version.byte_size,
                        initial_version.uploaded_by,
                        job.id,
                    ),
                )

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
        row -- one transaction, so a crash partway through can never leave the
        document pointing at new content with no job queued to index it, or a
        version-history row for a job that was never created.

        Returns the new version number (needed by the caller to report it and
        to build the version-history entry's own display fields)."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT owner_user_id FROM documents WHERE tenant_id=%s AND id=%s",
                (tenant_id, document_id),
            )
            owner = cur.fetchone()
            if owner is None:
                raise IngestionConflict("Document no longer exists")
            self._lock_owner(cur, tenant_id, owner["owner_user_id"])
            self._lock_idle_document(cur, tenant_id, document_id)
            cur.execute(
                "SELECT version,scope FROM documents WHERE tenant_id=%s AND id=%s",
                (tenant_id, document_id),
            )
            previous = cur.fetchone()
            if expected_version is not None and previous["version"] != expected_version:
                raise IngestionConflict(
                    "Document version changed while this upload was being accepted"
                )
            cur.execute(
                "SELECT 1 FROM documents WHERE tenant_id=%s AND owner_user_id=%s AND content_sha256=%s AND id<>%s",
                (tenant_id, owner["owner_user_id"], content_sha256, document_id),
            )
            if cur.fetchone() is not None:
                raise IngestionConflict("These bytes are already ingested by this document owner")
            if job.document_id != document_id or job.tenant_id != tenant_id:
                raise ValueError("Ingestion job must belong to the updated document")
            cur.execute(
                "UPDATE documents SET blob_path=%s, content_sha256=%s, mime=%s, "
                "source_type=%s, filename=%s, visibility=%s, scope=%s, "
                "version = version + 1, updated_at = now() "
                "WHERE tenant_id=%s AND id=%s RETURNING version",
                (
                    blob_path,
                    content_sha256,
                    mime,
                    source_type,
                    filename,
                    visibility,
                    scope,
                    tenant_id,
                    document_id,
                ),
            )
            row = cur.fetchone()
            version = int(row["version"]) if row else 1
            cur.execute(
                "INSERT INTO ingestion_jobs (id, document_id, tenant_id, stage, status, "
                "attempts, blob_path, content_sha256, version) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    job.id,
                    job.document_id,
                    job.tenant_id,
                    job.stage,
                    job.status,
                    job.attempts,
                    blob_path,
                    content_sha256,
                    version,
                ),
            )
            cur.execute(
                "UPDATE ingestion_jobs SET filename=%s, source_type=%s WHERE id=%s",
                (filename, source_type, job.id),
            )
            cur.execute(
                "INSERT INTO document_versions (id, document_id, tenant_id, version, "
                "content_sha256, blob_path, filename, mime, byte_size, uploaded_by, job_id, "
                "chunks_added, chunks_removed, chunks_unchanged, delta) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    new_object_id(),
                    document_id,
                    tenant_id,
                    version,
                    content_sha256,
                    blob_path,
                    filename,
                    mime,
                    byte_size,
                    uploaded_by,
                    job.id,
                    None,
                    None,
                    None,
                    Json({}),
                ),
            )
            epoch_key = (
                "global" if scope == "global" or previous["scope"] == "global" else tenant_id
            )
            cur.execute(
                "INSERT INTO corpus_epochs VALUES(%s,1) ON CONFLICT(scope_key) DO UPDATE SET revision=corpus_epochs.revision+1",
                (epoch_key,),
            )
        return version

    @staticmethod
    def _lock_owner(cur, tenant_id: str, owner_user_id: str) -> None:
        cur.execute(
            "SELECT id FROM users WHERE tenant_id=%s AND id=%s FOR UPDATE",
            (tenant_id, owner_user_id),
        )
        if cur.fetchone() is None:
            raise IngestionConflict("Document owner no longer exists")

    @staticmethod
    def _lock_idle_document(cur, tenant_id: str, document_id: str) -> None:
        cur.execute(
            "SELECT id FROM documents WHERE tenant_id=%s AND id=%s FOR UPDATE",
            (tenant_id, document_id),
        )
        if cur.fetchone() is None:
            raise IngestionConflict("Document no longer exists")
        cur.execute(
            "SELECT 1 FROM ingestion_jobs WHERE document_id=%s AND status IN ('queued','running') LIMIT 1",
            (document_id,),
        )
        if cur.fetchone() is not None:
            raise IngestionConflict("Document already has an active ingestion job")

    def get_job(self, tenant_id: str, job_id: str) -> Job | None:
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

    def get_extraction_artifact(self, job_id: str, content_sha256: str, key: str) -> dict | None:
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT artifact FROM extraction_artifacts WHERE job_id=%s AND content_sha256=%s AND artifact_key=%s",
                (job_id, content_sha256, key),
            )
            row = cur.fetchone()
            return row["artifact"] if row else None

    def put_extraction_artifact(
        self, job_id: str, content_sha256: str, key: str, artifact: dict
    ) -> None:
        with transaction(self.dsn) as cur:
            cur.execute(
                "INSERT INTO extraction_artifacts (job_id, content_sha256, artifact_key, artifact) "
                "VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                (job_id, content_sha256, key, Json(artifact)),
            )

    def record_job_event(self, event: JobEvent) -> None:
        """Append one stage-transition event to a job's trace."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "INSERT INTO job_events (id, job_id, document_id, tenant_id, stage, "
                "seq, status, duration_ms, attempt, detail) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    event.id or new_object_id(),
                    event.job_id,
                    event.document_id,
                    event.tenant_id,
                    event.stage,
                    event.seq,
                    event.status,
                    event.duration_ms,
                    event.attempt,
                    Json(event.detail or {}),
                ),
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
    ) -> list[tuple[Document, Job | None]]:
        """List a tenant's documents (plus global-scope ones) newest first,
        each paired with its most recent ingestion job if any.

        Uses LEFT JOIN LATERAL to keep this one round trip. Job columns are
        aliased (j_*) because RealDictCursor would otherwise collapse the
        columns both tables share -- id, tenant_id, created_at, and now
        updated_at too. Keep every new job column aliased: an unaliased one
        silently shadows its `documents` namesake in the dict, which surfaces
        as a document reporting the job's timestamp rather than its own.

        Sorted by `created_at`, deliberately not `updated_at`: a document
        shouldn't jump to the top of the list (and shift the pagination window
        under a caller mid-page) just because a new version was ingested.
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
        preserves chunk identity. Idempotent."""
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
                    (
                        r.id,
                        document_id,
                        tenant_id,
                        r.ordinal,
                        r.modality,
                        r.extractor,
                        r.route_reason,
                        r.token_count,
                        r.content_sha256,
                        r.text,
                        Json(r.meta),
                    ),
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
        self,
        tenant_id: str,
        user_id: str,
        action: str,
        target: str,
        meta: dict[str, Any] | None = None,
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
        id=row["id"],
        tenant_id=row["tenant_id"],
        owner_user_id=row["owner_user_id"],
        source_type=row["source_type"],
        blob_path=row["blob_path"],
        content_sha256=row["content_sha256"],
        mime=row["mime"],
        filename=row["filename"],
        visibility=row["visibility"],
        acl_user_ids=row["acl_user_ids"] or [],
        scope=row["scope"] if "scope" in row.keys() else Scope.TENANT.value,
        extracted_metadata=row["extracted_metadata"] if row.get("extracted_metadata") else {},
        created_at=str(row["created_at"]) if row["created_at"] is not None else None,
        version=row.get("version") or 1,
        active_generation_id=row.get("active_generation_id"),
        indexed_version=row.get("indexed_version"),
        updated_at=(
            str(row["updated_at"])
            if row.get("updated_at") is not None
            else (str(row["created_at"]) if row["created_at"] is not None else None)
        ),
    )


def _row_to_document_version(row) -> DocumentVersion:
    """Map a `document_versions` row to a DocumentVersion."""
    return DocumentVersion(
        id=row["id"],
        document_id=row["document_id"],
        tenant_id=row["tenant_id"],
        version=row["version"],
        content_sha256=row["content_sha256"],
        blob_path=row["blob_path"],
        filename=row["filename"] or "",
        mime=row.get("mime") or "application/octet-stream",
        byte_size=row["byte_size"] or 0,
        uploaded_by=row["uploaded_by"],
        job_id=row["job_id"],
        chunks_added=row["chunks_added"],
        chunks_removed=row["chunks_removed"],
        chunks_unchanged=row["chunks_unchanged"],
        delta=row["delta"] or {},
        created_at=str(row["created_at"]) if row["created_at"] is not None else None,
    )


def _row_to_chunk(row) -> ChunkRecord:
    """Map a `chunks` row to a ChunkRecord."""
    return ChunkRecord(
        ordinal=row["ordinal"],
        modality=row["modality"],
        extractor=row["extractor"],
        route_reason=row["route_reason"],
        token_count=row["token_count"] or 0,
        text=row["text"] or "",
        meta=row["meta"] or {},
        id=row["id"],
        content_sha256=row["content_sha256"],
    )


def _row_to_job_event(row) -> JobEvent:
    """Map a `job_events` row to a JobEvent."""
    return JobEvent(
        job_id=row["job_id"],
        document_id=row["document_id"],
        tenant_id=row["tenant_id"],
        stage=row["stage"],
        seq=row["seq"],
        status=row["status"],
        duration_ms=row["duration_ms"] or 0.0,
        attempt=row["attempt"] or 0,
        detail=row["detail"] or {},
        id=row["id"],
        at=str(row["at"]) if row["at"] is not None else None,
    )


def _row_to_lateral_job(row) -> Job | None:
    """Rebuild the Job from the j_*-aliased columns of `list_documents`' LATERAL
    join. A document with no surviving job row yields None."""
    if row.get("j_id") is None:
        return None
    return Job(
        id=row["j_id"],
        document_id=row["id"],
        tenant_id=row["tenant_id"],
        stage=row["j_stage"],
        status=row["j_status"],
        attempts=row["j_attempts"],
        error=row["j_error"],
        route_summary=row["j_route_summary"],
        created_at=str(row["j_created_at"]) if row["j_created_at"] is not None else None,
        updated_at=str(row["j_updated_at"]) if row["j_updated_at"] is not None else None,
    )


def _row_to_job(row) -> Job:
    """Map an `ingestion_jobs` row to a Job."""
    return Job(
        id=row["id"],
        document_id=row["document_id"],
        tenant_id=row["tenant_id"],
        stage=row["stage"],
        status=row["status"],
        attempts=row["attempts"],
        error=row["error"],
        route_summary=row["route_summary"],
        created_at=str(row["created_at"]) if row["created_at"] is not None else None,
        updated_at=str(row["updated_at"]) if row["updated_at"] is not None else None,
        blob_path=row.get("blob_path"),
        content_sha256=row.get("content_sha256"),
        version=row.get("version"),
        filename=row.get("filename"),
        source_type=row.get("source_type"),
        lease_token=row.get("lease_token"),
        generation_id=row.get("generation_id"),
    )
