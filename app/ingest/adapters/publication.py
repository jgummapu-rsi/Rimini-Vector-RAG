from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import asdict

from psycopg2.extras import Json

from app.ingest.ports.publication import PublicationStore
from app.ingest.ports.task_queue import StaleLeaseError
from app.shared.adapters.bm25 import searchable_text
from app.shared.adapters.pgvector.db import transaction
from app.shared.adapters.pgvector.profile import check_profile

log = logging.getLogger(__name__)


class PostgresPublicationStore(PublicationStore):
    def __init__(self, dsn: str, profile):
        self.dsn = dsn
        self.dim = profile.dimensions
        self.profile = profile

    def publish(self, job, document, chunks, points, metadata, delta) -> str:
        if not job.lease_token or not job.generation_id:
            raise StaleLeaseError("Publication requires a claimed generation and lease")
        if not chunks or len(chunks) != len(points):
            raise ValueError("Publication requires nonempty aligned chunks and vectors")
        for chunk, point in zip(chunks, points, strict=True):
            if chunk.id != point.chunk_id or point.tenant_id != document.tenant_id:
                raise ValueError("Publication chunk/vector identity mismatch")
            if len(point.vector) != self.dim or not all(
                math.isfinite(value) for value in point.vector
            ):
                raise ValueError(
                    "Publication vectors must have the configured dimension and finite values"
                )
            if chunk.content_sha256 != hashlib.sha256(chunk.text.encode()).hexdigest():
                raise ValueError("Publication evidence hash mismatch")
        with transaction(self.dsn) as cur:
            check_profile(cur, self.profile.id)
            cur.execute(
                "SELECT * FROM documents WHERE id=%s AND tenant_id=%s FOR UPDATE",
                (document.id, document.tenant_id),
            )
            current = cur.fetchone()
            if current is None:
                raise StaleLeaseError("Publication rejected: document was deleted")
            cur.execute(
                "SELECT * FROM ingestion_jobs WHERE id=%s AND document_id=%s "
                "AND status='running' AND lease_token=%s AND generation_id=%s "
                "AND lease_expires_at>clock_timestamp() FOR UPDATE",
                (job.id, document.id, job.lease_token, job.generation_id),
            )
            claim = cur.fetchone()
            if claim is None:
                raise StaleLeaseError("Publication rejected: worker lease is stale")
            if (
                current["version"] != document.version
                or current["content_sha256"] != document.content_sha256
            ):
                raise StaleLeaseError("Publication rejected: source version was superseded")
            if current["active_generation_id"] != document.active_generation_id:
                raise StaleLeaseError(
                    "Publication rejected: the expected parent generation changed"
                )
            cur.execute(
                "INSERT INTO ingestion_generations "
                "(id,document_id,tenant_id,job_id,version,content_sha256,blob_path,filename,source_type,"
                "parent_generation_id,chunks,extracted_metadata,embedding_profile_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (
                    job.generation_id,
                    document.id,
                    document.tenant_id,
                    job.id,
                    document.version,
                    document.content_sha256,
                    document.blob_path,
                    document.filename,
                    document.source_type,
                    current["active_generation_id"],
                    Json([asdict(chunk) for chunk in chunks]),
                    Json(metadata),
                    self.profile.id,
                ),
            )
            for chunk, point in zip(chunks, points, strict=True):
                payload = dict(
                    point.payload,
                    embedding_profile_id=self.profile.id,
                    generation_id=job.generation_id,
                    version=document.version,
                    user_id=current["owner_user_id"],
                    visibility=current["visibility"],
                    acl_user_ids=current["acl_user_ids"],
                    scope=current["scope"],
                )
                cur.execute(
                    "INSERT INTO vector_chunks "
                    "(chunk_id,tenant_id,document_id,scope,deleted,embedding,payload,tsv,generation_id) "
                    "VALUES (%s,%s,%s,%s,false,%s,%s,to_tsvector('english',%s),%s)",
                    (
                        chunk.id,
                        document.tenant_id,
                        document.id,
                        current["scope"],
                        point.vector,
                        Json(payload),
                        searchable_text(
                            chunk.text,
                            metadata.get("topics"),
                            metadata.get("entities"),
                            metadata.get("author"),
                        ),
                        job.generation_id,
                    ),
                )
            cur.execute(
                "DELETE FROM chunks WHERE document_id=%s AND tenant_id=%s",
                (document.id, document.tenant_id),
            )
            for chunk in chunks:
                cur.execute(
                    "INSERT INTO chunks (id,document_id,tenant_id,ordinal,modality,extractor,route_reason,"
                    "token_count,content_sha256,text,meta) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (
                        chunk.id,
                        document.id,
                        document.tenant_id,
                        chunk.ordinal,
                        chunk.modality,
                        chunk.extractor,
                        chunk.route_reason,
                        chunk.token_count,
                        chunk.content_sha256,
                        chunk.text,
                        Json(dict(chunk.meta, generation_id=job.generation_id)),
                    ),
                )
            cur.execute(
                "UPDATE vector_chunks SET deleted=true WHERE document_id=%s AND tenant_id=%s "
                "AND generation_id IS NULL",
                (document.id, document.tenant_id),
            )
            cur.execute(
                "UPDATE documents SET active_generation_id=%s,indexed_version=%s,extracted_metadata=%s "
                "WHERE id=%s",
                (job.generation_id, document.version, Json(metadata), document.id),
            )
            cur.execute(
                "UPDATE document_versions SET chunks_added=%s,chunks_removed=%s,chunks_unchanged=%s,delta=%s WHERE job_id=%s",
                (
                    delta.get("added"),
                    delta.get("removed"),
                    delta.get("unchanged"),
                    Json(delta),
                    job.id,
                ),
            )
            cur.execute(
                "INSERT INTO corpus_epochs(scope_key,revision) VALUES (%s,1) "
                "ON CONFLICT(scope_key) DO UPDATE SET revision=corpus_epochs.revision+1",
                ("global" if current["scope"] == "global" else document.tenant_id,),
            )
            cur.execute(
                "UPDATE ingestion_jobs SET status='done',stage='done',error=NULL,lease_token=NULL,"
                "lease_expires_at=NULL,updated_at=now() WHERE id=%s AND lease_token=%s "
                "AND lease_expires_at>clock_timestamp()",
                (job.id, job.lease_token),
            )
            if cur.rowcount != 1:
                raise StaleLeaseError("Publication rejected: lease expired before commit")
        log.info(
            "Document generation published",
            extra={
                "event": "generation_published",
                "tenant_id": document.tenant_id,
                "document_id": document.id,
                "generation_id": job.generation_id,
                "job_id": job.id,
                "chunks": len(chunks),
            },
        )
        return job.generation_id
