from __future__ import annotations

import logging
from dataclasses import asdict

from psycopg2.extras import Json

from app.shared.domain.embedding import EmbeddingProfile

log = logging.getLogger(__name__)


def refresh_retrieval_view(cur) -> None:
    cur.execute(
        "CREATE OR REPLACE VIEW retrieval_vectors AS "
        "SELECT v.chunk_id,v.tenant_id,v.document_id,v.deleted,v.embedding,v.tsv,v.generation_id,"
        "CASE WHEN d.id IS NULL THEN v.scope ELSE d.scope END AS scope,"
        "CASE WHEN d.id IS NULL THEN v.payload ELSE v.payload || jsonb_build_object("
        "'user_id',d.owner_user_id,'visibility',d.visibility,'acl_user_ids',d.acl_user_ids,'scope',d.scope) END AS payload "
        "FROM vector_chunks v LEFT JOIN documents d ON d.id=v.document_id AND d.tenant_id=v.tenant_id "
        "WHERE (v.generation_id IS NULL OR d.active_generation_id=v.generation_id) "
        "AND NOT EXISTS (SELECT 1 FROM deleted_documents dead WHERE dead.document_id=v.document_id AND dead.tenant_id=v.tenant_id)"
    )


def ensure_profile(cur, profile: EmbeddingProfile) -> None:
    cur.execute(
        "CREATE TABLE IF NOT EXISTS embedding_profile (singleton BOOLEAN PRIMARY KEY CHECK(singleton), profile_id TEXT NOT NULL, manifest JSONB NOT NULL)"
    )
    cur.execute("LOCK TABLE embedding_profile IN EXCLUSIVE MODE")
    cur.execute("SELECT profile_id FROM embedding_profile WHERE singleton=true")
    row = cur.fetchone()
    if row is None:
        cur.execute("SELECT EXISTS(SELECT 1 FROM vector_chunks) AS populated")
        if cur.fetchone()["populated"]:
            raise ValueError(
                "Existing vectors have no verified embedding profile; rebuild into a shadow index"
            )
        cur.execute(
            "INSERT INTO embedding_profile VALUES (true,%s,%s)", (profile.id, Json(asdict(profile)))
        )
    elif row["profile_id"] != profile.id:
        raise ValueError("Embedding profile mismatch; migrate using a shadow index")


def check_profile(cur, profile_id: str, workspace: bool = False) -> None:
    if workspace:
        cur.execute(
            "SELECT profile_id FROM workspace_embedding_profiles WHERE profile_id=%s",
            (profile_id,),
        )
        if cur.fetchone() is None:
            raise ValueError("Workspace embedding profile is not registered")
        return
    cur.execute("SELECT profile_id FROM embedding_profile WHERE singleton=true FOR SHARE")
    row = cur.fetchone()
    if row is None or row["profile_id"] != profile_id:
        raise ValueError("Active embedding profile differs from this process")
