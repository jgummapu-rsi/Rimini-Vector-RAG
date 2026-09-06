"""Ingest-side API routes: upload, job/document lifecycle, and ops endpoints."""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.api._common import _owned_or_404, _read_capped, _visible_or_404
from app.api.auth import (
    get_container,
    get_principal,
    require_admin,
    require_delete,
    require_ingest,
)
from app.shared.container import Container
from app.shared.domain.models import (
    EXT_TO_SOURCE_TYPE,
    Document,
    Job,
    JobStage,
    JobStatus,
    Principal,
    Role,
    Scope,
    Visibility,
)
from app.shared.ids import new_object_id
from app.shared.observability import bind
from app.ingest.pipeline.runner import invalidate_cache_for
from app.retrieval.rag.access import can_view

router = APIRouter()
log = logging.getLogger("api")


@router.get("/healthz")
def healthz(container: Container = Depends(get_container)) -> dict:
    """Liveness + active-backend summary, no auth required."""
    return {"status": "ok", "backends": {
        "metadata": container.settings.metadata_backend,
        "blob": container.settings.blob_backend,
        "vector": container.settings.vector_backend,
        "queue": container.settings.queue_backend,
    }}


@router.get("/metrics")
def metrics(
    principal: Principal = Depends(require_admin),
    container: Container = Depends(get_container),
) -> dict:
    """Counters + timings, shared across API and worker processes.

    Admin-only: these counters are deployment-wide, not tenant-scoped, so they
    describe every tenant's ingest/query volume at once. They were previously
    readable with no credentials at all.
    """
    return container.metrics.snapshot()


# Visibility values `/ingest` accepts. `shared` is deliberately excluded: it is
# only meaningful alongside an acl_user_ids list, and there is no way to supply
# one on this endpoint -- accepting it would silently create a document shared
# with nobody, which behaves like `private` but doesn't look like it.
_INGESTABLE_VISIBILITY = (Visibility.PRIVATE.value, Visibility.TENANT.value)


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest(
    file: UploadFile = File(...),
    scope: Optional[str] = Form(None),
    visibility: Optional[str] = Form(None),
    principal: Principal = Depends(require_ingest),
    container: Container = Depends(get_container),
) -> dict:
    """Accept a file for ingestion.

    `visibility` controls who inside the tenant may retrieve it:
      - `private` (default) -- only the uploader (and tenant admins).
      - `tenant`            -- everyone in the tenant.

    The default stays `private` so existing callers are unaffected and a
    personal upload is never exposed by accident; publishing to the whole tenant
    has to be asked for. `scope` is the orthogonal, cross-TENANT control and is
    unchanged.

    `scope`/`visibility` are read as `Optional[str] = Form(None)` rather than
    `str = Form(<default>)` on purpose: FastAPI/Starlette treats an explicitly
    submitted empty string as if the field were absent and silently substitutes
    the declared default, which let `visibility=""` sail through as `"private"`
    instead of being rejected like every other invalid value. Reading the raw
    value and applying our own default only when it is genuinely `None` (field
    omitted) keeps "omitted" and "sent empty" distinguishable.
    """
    cfg = container.settings
    bind(tenant_id=principal.tenant_id, user_id=principal.user_id)

    if visibility is None:
        visibility = Visibility.PRIVATE.value
    if scope is None:
        scope = Scope.TENANT.value

    if visibility not in _INGESTABLE_VISIBILITY:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"invalid visibility '{visibility}'. Expected one of "
            f"{list(_INGESTABLE_VISIBILITY)}",
        )

    if scope == Scope.GLOBAL.value:
        if not (cfg.platform_tenant_id and principal.tenant_id == cfg.platform_tenant_id
                and principal.role == Role.ADMIN):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                "only the platform tenant's admin may publish to global scope",
            )
    elif scope != Scope.TENANT.value:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"invalid scope '{scope}'")

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in EXT_TO_SOURCE_TYPE:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported file type '{ext}'. Phase-1 accepts: "
            f"{sorted(set(EXT_TO_SOURCE_TYPE))}",
        )
    data = await _read_capped(file, cfg.max_upload_mb * 1024 * 1024)
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty file")

    source_type = EXT_TO_SOURCE_TYPE[ext].value
    sha256 = hashlib.sha256(data).hexdigest()

    container.metrics.incr("ingest.requests")

    # --- dedup (idempotency, FR9) ---
    existing = container.metadata.get_document_by_hash(principal.tenant_id, sha256)
    if existing is not None:
        container.metrics.incr("ingest.deduped")
        log.info("ingest deduped", extra={"event": "ingest_deduped",
                 "document_id": existing.id, "sha": sha256[:12]})
        return {"document_id": existing.id, "deduplicated": True,
                "message": "identical content already ingested"}

    blob_path = container.blob.put(principal.tenant_id, sha256, ext, data)

    doc = Document(
        id=new_object_id(),
        tenant_id=principal.tenant_id,
        owner_user_id=principal.user_id,
        source_type=source_type,
        blob_path=blob_path,
        content_sha256=sha256,
        mime=file.content_type or "application/octet-stream",
        filename=file.filename or "",
        visibility=visibility,
        acl_user_ids=[],
        scope=scope,
    )
    container.metadata.create_document(doc)

    job = Job(
        id=new_object_id(), document_id=doc.id, tenant_id=principal.tenant_id,
        stage=JobStage.PARSE.value, status=JobStatus.QUEUED.value, attempts=0,
    )
    container.metadata.create_job(job)
    container.metrics.incr("ingest.accepted")
    log.info("ingest accepted", extra={
        "event": "ingest_accepted", "document_id": doc.id, "job_id": job.id,
        "source_type": source_type, "bytes": len(data), "sha": sha256[:12],
        "file": doc.filename,
    })

    container.metadata.write_audit(
        principal.tenant_id, principal.user_id, "ingest", doc.id,
        {"source_type": source_type, "filename": doc.filename, "job_id": job.id,
         "scope": scope, "visibility": visibility},
    )

    return {"job_id": job.id, "document_id": doc.id, "status": job.status,
            "source_type": source_type, "scope": scope, "visibility": visibility}


@router.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container),
) -> dict:
    """Status of one ingestion job.

    Gated on the DOCUMENT's visibility, not just on tenant membership: the
    response carries the document id, its pipeline stage and any error text, so
    a job is exactly as sensitive as the document behind it. `/jobs/{id}/trace`
    already enforced this; this endpoint did not, which let any member of a
    tenant watch another user's private ingest.
    """
    job = container.metadata.get_job(principal.tenant_id, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    doc = container.metadata.get_document(principal.tenant_id, job.document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    _visible_or_404(doc, principal)
    return {"job_id": job.id, "document_id": job.document_id, "stage": job.stage,
            "status": job.status, "attempts": job.attempts, "error": job.error,
            "route_summary": job.route_summary, "created_at": job.created_at,
            "updated_at": job.updated_at}


@router.get("/jobs/{job_id}/trace")
def get_job_trace(
    job_id: str,
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container),
) -> dict:
    """Everything the trace view needs for one ingestion, in a single call:
    where the job is, how long each stage took and what it produced, how the
    document was routed, and what ended up indexed."""
    job = container.metadata.get_job(principal.tenant_id, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    doc = container.metadata.get_document(principal.tenant_id, job.document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    _visible_or_404(doc, principal)

    events = container.metadata.get_job_events(principal.tenant_id, job_id)
    chunks = container.metadata.get_document_chunks(principal.tenant_id, doc.id)
    return {
        "job": {"job_id": job.id, "document_id": job.document_id, "stage": job.stage,
                "status": job.status, "attempts": job.attempts, "error": job.error,
                "route_summary": job.route_summary, "created_at": job.created_at,
                "updated_at": job.updated_at},
        "document": {"document_id": doc.id, "filename": doc.filename,
                     "source_type": doc.source_type, "mime": doc.mime,
                     "scope": doc.scope, "visibility": doc.visibility,
                     "extracted_metadata": doc.extracted_metadata,
                     "created_at": doc.created_at},
        "stages": [{"stage": e.stage, "seq": e.seq, "status": e.status,
                    "duration_ms": e.duration_ms, "attempt": e.attempt,
                    "detail": e.detail, "at": e.at} for e in events],
        "chunk_count": len(chunks),
        "token_total": sum(c.token_count or 0 for c in chunks),
    }


@router.get("/documents")
def list_documents(
    limit: int = 50,
    offset: int = 0,
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container),
) -> dict:
    """Documents this caller may read, newest first, each with its latest job.
    Backs the trace view's history list.

    Note: the store pages by tenant, then `can_view` filters the page, so a page
    can come back shorter than `limit` without meaning the end was reached. That
    matches how retrieval already filters post-fetch (`access_predicate`), and
    keeps the ACL rule in exactly one place. Proper ACL-aware pagination is part
    of the "response pagination on list endpoints" roadmap item."""
    limit = max(1, min(limit, 200))
    rows = container.metadata.list_documents(principal.tenant_id, limit, max(0, offset))
    items = []
    for doc, job in rows:
        payload = {"user_id": doc.owner_user_id, "visibility": doc.visibility,
                   "acl_user_ids": doc.acl_user_ids, "scope": doc.scope}
        if not can_view(payload, principal.user_id, principal.role.value):
            continue
        items.append({
            "document_id": doc.id, "filename": doc.filename,
            "source_type": doc.source_type, "scope": doc.scope,
            "visibility": doc.visibility, "created_at": doc.created_at,
            "job_id": job.id if job else None,
            "status": job.status if job else None,
            "stage": job.stage if job else None,
        })
    return {"documents": items, "count": len(items),
            "limit": limit, "offset": max(0, offset)}


@router.get("/documents/{document_id}")
def get_document(
    document_id: str,
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container),
) -> dict:
    """One document's metadata plus the tenant's total indexed-vector count."""
    doc = container.metadata.get_document(principal.tenant_id, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    _visible_or_404(doc, principal)
    return {"document_id": doc.id, "source_type": doc.source_type,
            "filename": doc.filename, "visibility": doc.visibility, "scope": doc.scope,
            "extracted_metadata": doc.extracted_metadata,
            "vector_count": container.vectors.count(principal.tenant_id),
            "created_at": doc.created_at}


@router.post("/documents/{document_id}/reprocess", status_code=status.HTTP_202_ACCEPTED)
def reprocess(
    document_id: str,
    principal: Principal = Depends(require_ingest),
    container: Container = Depends(get_container),
) -> dict:
    """Re-run the pipeline on an existing document (e.g. after a chunking change).
    Upsert drops prior vectors first, so it's idempotent.

    Requires BOTH checks: `_visible_or_404` (you may not act on a document you
    are not allowed to see -- tenant membership alone previously let any member
    reprocess another user's private document) and `_owned_or_404` (read access
    to a global document does not grant the right to reprocess it).
    """
    doc = container.metadata.get_document(principal.tenant_id, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    _visible_or_404(doc, principal)
    _owned_or_404(doc, principal)
    job = Job(id=new_object_id(), document_id=doc.id, tenant_id=principal.tenant_id,
              stage=JobStage.PARSE.value, status=JobStatus.QUEUED.value, attempts=0)
    container.metadata.create_job(job)
    container.metrics.incr("reprocess.requests")
    log.info("reprocess queued", extra={"event": "reprocess",
             "document_id": doc.id, "job_id": job.id})
    return {"job_id": job.id, "document_id": doc.id, "status": job.status}


@router.get("/documents/{document_id}/chunks")
def list_chunks(
    document_id: str,
    preview: int = 240,
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container),
) -> dict:
    """Chunk-level breakdown of one document, with a truncated text preview per chunk."""
    doc = container.metadata.get_document(principal.tenant_id, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    _visible_or_404(doc, principal)
    recs = container.metadata.get_document_chunks(principal.tenant_id, document_id)
    return {
        "document_id": document_id,
        "chunk_count": len(recs),
        "chunks": [{
            "ordinal": r.ordinal, "modality": r.modality, "extractor": r.extractor,
            "route_reason": r.route_reason, "tokens": r.token_count,
            "pages": r.meta.get("pages") or r.meta.get("page"),
            "preview": r.text[:preview],
        } for r in recs],
    }


@router.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    principal: Principal = Depends(require_delete),
    container: Container = Depends(get_container),
) -> dict:
    """Delete a document, its vectors, its blob, and invalidate any cached answers it fed."""
    doc = container.metadata.get_document(principal.tenant_id, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    _owned_or_404(doc, principal)
    removed = container.vectors.delete_by_document(principal.tenant_id, document_id)
    container.blob.delete(doc.blob_path)
    container.metadata.delete_document(principal.tenant_id, document_id)
    # Deleting a document changes what's retrievable -> cached answers may be
    # stale. Scope-aware: deleting a global document invalidates every tenant,
    # not just this one (same reasoning as on ingest -- see the runner).
    invalidate_cache_for(container, principal.tenant_id, doc.scope)
    container.metrics.incr("documents.deleted")
    container.metadata.write_audit(
        principal.tenant_id, principal.user_id, "delete", document_id,
        {"vectors_removed": removed},
    )
    return {"document_id": document_id, "deleted": True, "vectors_removed": removed}
