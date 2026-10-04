from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from uuid import uuid4

from psycopg2 import sql
from psycopg2.extras import Json

from app.shared.adapters.pgvector.db import transaction
from app.shared.adapters.pgvector.profile import refresh_retrieval_view
from app.shared.adapters.postgres.permissions import apply_runtime_grants, validate_runtime_role
from app.shared.domain.embedding import validate_vectors

log = logging.getLogger(__name__)


def _fingerprint(rows) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                {key: str(value) for key, value in row.items()},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    return digest.hexdigest()


def _target_fingerprint(cur, name: str) -> str:
    cur.execute(
        sql.SQL(
            "SELECT chunk_id,tenant_id,document_id,scope,deleted,payload,tsv,created_at,updated_at,generation_id,embedding::text AS embedding "
            "FROM {} ORDER BY chunk_id"
        ).format(sql.Identifier(name))
    )
    return _fingerprint(cur.fetchall())


def build_shadow(dsn: str, embedder) -> str:
    profile = embedder.profile
    if profile.dimensions > 2000:
        raise ValueError("Float-vector HNSW supports at most 2000 dimensions")
    name = "embedding_shadow_" + uuid4().hex
    with transaction(dsn) as cur:
        cur.execute("ALTER TABLE vector_chunks ADD COLUMN IF NOT EXISTS generation_id TEXT")
        cur.execute(
            "CREATE TABLE IF NOT EXISTS embedding_profile (singleton BOOLEAN PRIMARY KEY CHECK(singleton), profile_id TEXT NOT NULL, manifest JSONB NOT NULL)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS embedding_migrations (name TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, profile_id TEXT NOT NULL, manifest JSONB NOT NULL, state TEXT NOT NULL)"
        )
        cur.execute(
            "ALTER TABLE embedding_migrations ADD COLUMN IF NOT EXISTS target_fingerprint TEXT"
        )
        cur.execute(
            "SELECT chunk_id,tenant_id,document_id,scope,deleted,payload,tsv,created_at,updated_at,generation_id "
            "FROM vector_chunks ORDER BY chunk_id"
        )
        rows = cur.fetchall()
        cur.execute(
            sql.SQL(
                "CREATE TABLE {} (LIKE vector_chunks INCLUDING DEFAULTS INCLUDING CONSTRAINTS)"
            ).format(sql.Identifier(name))
        )
        cur.execute(
            sql.SQL("ALTER TABLE {} ALTER COLUMN embedding TYPE vector({})").format(
                sql.Identifier(name), sql.Literal(profile.dimensions)
            )
        )
        cur.execute(
            "INSERT INTO embedding_migrations(name,fingerprint,profile_id,manifest,state) VALUES (%s,%s,%s,%s,'building')",
            (name, _fingerprint(rows), profile.id, Json(asdict(profile))),
        )
    for start in range(0, len(rows), 64):
        batch = rows[start : start + 64]
        vectors = embedder.embed_documents([row["payload"].get("content", "") for row in batch])
        validate_vectors(vectors, len(batch), profile.dimensions)
        with transaction(dsn) as cur:
            for row, vector in zip(batch, vectors, strict=True):
                payload = dict(row["payload"], embedding_profile_id=profile.id)
                cur.execute(
                    sql.SQL(
                        "INSERT INTO {} (chunk_id,tenant_id,document_id,scope,deleted,payload,tsv,created_at,updated_at,generation_id,embedding) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
                    ).format(sql.Identifier(name)),
                    (
                        row["chunk_id"],
                        row["tenant_id"],
                        row["document_id"],
                        row["scope"],
                        row["deleted"],
                        Json(payload),
                        row["tsv"],
                        row["created_at"],
                        row["updated_at"],
                        row["generation_id"],
                        vector,
                    ),
                )
    with transaction(dsn) as cur:
        cur.execute(
            sql.SQL("ALTER TABLE {} ADD PRIMARY KEY (chunk_id)").format(sql.Identifier(name))
        )
        cur.execute(
            sql.SQL("CREATE INDEX ON {} USING hnsw (embedding vector_cosine_ops)").format(
                sql.Identifier(name)
            )
        )
        cur.execute(sql.SQL("CREATE INDEX ON {} USING gin (tsv)").format(sql.Identifier(name)))
        cur.execute(sql.SQL("CREATE INDEX ON {} (tenant_id,deleted)").format(sql.Identifier(name)))
        cur.execute(sql.SQL("CREATE INDEX ON {} (generation_id)").format(sql.Identifier(name)))
        cur.execute(sql.SQL("CREATE INDEX ON {} (document_id)").format(sql.Identifier(name)))
        cur.execute(
            "UPDATE embedding_migrations SET state='ready',target_fingerprint=%s WHERE name=%s",
            (_target_fingerprint(cur, name), name),
        )
    log.info(
        "Shadow embedding index built",
        extra={
            "event": "embedding_shadow_ready",
            "migration": name,
            "profile_id": profile.id,
            "vectors": len(rows),
        },
    )
    return name


def activate_shadow(dsn: str, name: str) -> str:
    previous = "embedding_previous_" + uuid4().hex
    with transaction(dsn) as cur:
        cur.execute("LOCK TABLE embedding_profile IN EXCLUSIVE MODE")
        cur.execute("LOCK TABLE vector_chunks IN ACCESS EXCLUSIVE MODE")
        cur.execute(
            "SELECT * FROM embedding_migrations WHERE name=%s AND state='ready' FOR UPDATE", (name,)
        )
        migration = cur.fetchone()
        if migration is None:
            raise ValueError("A ready shadow migration is required")
        cur.execute(sql.SQL("LOCK TABLE {} IN ACCESS EXCLUSIVE MODE").format(sql.Identifier(name)))
        if (
            not migration.get("target_fingerprint")
            or _target_fingerprint(cur, name) != migration["target_fingerprint"]
        ):
            raise ValueError("Shadow migration integrity mismatch")
        cur.execute(
            "SELECT atttypmod AS dim FROM pg_attribute WHERE attrelid=to_regclass(%s) AND attname='embedding'",
            (name,),
        )
        if cur.fetchone()["dim"] != migration["manifest"]["dimensions"]:
            raise ValueError("Shadow migration dimension mismatch")
        cur.execute(
            "SELECT chunk_id,tenant_id,document_id,scope,deleted,payload,tsv,created_at,updated_at,generation_id "
            "FROM vector_chunks ORDER BY chunk_id"
        )
        rows = cur.fetchall()
        if _fingerprint(rows) != migration["fingerprint"]:
            raise ValueError("Corpus changed during shadow migration; rebuild before switching")
        cur.execute(sql.SQL("SELECT count(*) AS n FROM {}").format(sql.Identifier(name)))
        if cur.fetchone()["n"] != len(rows):
            raise ValueError("Shadow migration coverage mismatch")
        cur.execute("SELECT profile_id,manifest FROM embedding_profile WHERE singleton=true")
        old_profile = cur.fetchone()
        cur.execute("DROP VIEW IF EXISTS retrieval_vectors")
        cur.execute(
            sql.SQL("ALTER TABLE vector_chunks RENAME TO {}").format(sql.Identifier(previous))
        )
        cur.execute(sql.SQL("ALTER TABLE {} RENAME TO vector_chunks").format(sql.Identifier(name)))
        refresh_retrieval_view(cur)
        cur.execute(
            "SELECT chunk_id,tenant_id,document_id,scope,deleted,payload,tsv,created_at,updated_at,generation_id "
            "FROM vector_chunks ORDER BY chunk_id"
        )
        active_fingerprint = _fingerprint(cur.fetchall())
        if old_profile is not None:
            cur.execute(
                "INSERT INTO embedding_migrations(name,fingerprint,profile_id,manifest,state,target_fingerprint) VALUES (%s,%s,%s,%s,'ready',%s)",
                (
                    previous,
                    active_fingerprint,
                    old_profile["profile_id"],
                    Json(old_profile["manifest"]),
                    _target_fingerprint(cur, previous),
                ),
            )
        cur.execute(
            "INSERT INTO embedding_profile VALUES(true,%s,%s) ON CONFLICT(singleton) "
            "DO UPDATE SET profile_id=excluded.profile_id,manifest=excluded.manifest",
            (migration["profile_id"], Json(migration["manifest"])),
        )
        cur.execute("UPDATE embedding_migrations SET state='active' WHERE name=%s", (name,))
        cur.execute("SELECT to_regclass('runtime_database_roles') AS roles")
        if cur.fetchone()["roles"] is not None:
            cur.execute("SELECT current_schema() AS schema")
            schema = cur.fetchone()["schema"]
            cur.execute("SELECT role_name FROM runtime_database_roles")
            for row in cur.fetchall():
                validate_runtime_role(cur, schema, row["role_name"])
                apply_runtime_grants(cur, schema, row["role_name"])
        cur.execute(
            "INSERT INTO corpus_epochs VALUES ('global',1) ON CONFLICT(scope_key) DO UPDATE SET revision=corpus_epochs.revision+1"
        )
    log.info(
        "Shadow embedding index activated",
        extra={
            "event": "embedding_shadow_activated",
            "migration": name,
            "previous_table": previous,
            "profile_id": migration["profile_id"],
        },
    )
    return previous
