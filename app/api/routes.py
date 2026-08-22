"""Ingestion API routes."""
from __future__ import annotations

import hashlib
import logging
import os
from time import perf_counter

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.api.auth import get_container, get_principal, require_delete, require_ingest
from app.container import Container
from app.domain.models import (
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
from app.ids import new_object_id
from app.observability import bind
from app.rag.access import access_predicate, can_view
from app.rag.query import answer_query

router = APIRouter()
log = logging.getLogger("api")


def _visible_or_404(doc: Document, principal: Principal) -> None:
    payload = {"user_id": doc.owner_user_id, "visibility": doc.visibility,
               "acl_user_ids": doc.acl_user_ids, "scope": doc.scope}
    if not can_view(payload, principal.user_id, principal.role.value):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")


def _owned_or_404(doc: Document, principal: Principal) -> None:
    """Read access to a scope=global document does not imply write/delete access —
    only the owning (platform) tenant may reprocess/delete its own documents."""
    if doc.tenant_id != principal.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@router.get("/healthz")
def healthz(container: Container = Depends(get_container)) -> dict:
    return {"status": "ok", "backends": {
        "metadata": container.settings.metadata_backend,
        "blob": container.settings.blob_backend,
        "vector": container.settings.vector_backend,
        "queue": container.settings.queue_backend,
    }}


@router.get("/metrics")
def metrics(container: Container = Depends(get_container)) -> dict:
    """Counters + timings, shared across API and worker processes."""
    return container.metrics.snapshot()


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest(
    file: UploadFile = File(...),
    scope: str = Form(Scope.TENANT.value),
    principal: Principal = Depends(require_ingest),
    container: Container = Depends(get_container),
) -> dict:
    cfg = container.settings
    bind(tenant_id=principal.tenant_id, user_id=principal.user_id)

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
    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty file")
    if len(data) > cfg.max_upload_mb * 1024 * 1024:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"file exceeds {cfg.max_upload_mb} MB")

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
        visibility=Visibility.PRIVATE.value,
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
         "scope": scope},
    )

    return {"job_id": job.id, "document_id": doc.id, "status": job.status,
            "source_type": source_type, "scope": scope}


@router.post("/query")
def query(
    req: QueryRequest,
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container),
) -> dict:
    bind(tenant_id=principal.tenant_id, user_id=principal.user_id)
    container.metrics.incr("query.requests")
    t0 = perf_counter()
    result = answer_query(
        container, principal.tenant_id, req.question, req.top_k,
        access=access_predicate(principal), user_id=principal.user_id,
    )
    log.info("query answered", extra={
        "event": "query", "top_k": req.top_k, "hits": len(result.chunk_ids),
        "top_score": round(result.scores[0], 3) if result.scores else None,
        "answer_len": len(result.answer),
        "duration_ms": round((perf_counter() - t0) * 1000, 1),
    })
    return {
        "question": result.question,
        "answer": result.answer,
        "contexts": result.contexts,
        "chunk_ids": result.chunk_ids,
        "scores": result.scores,
        "sub_questions": result.sub_questions,
        "citations": result.citations,
        "trace": result.trace,
        "grounded": result.grounded,
    }


@router.get("/jobs/{job_id}")
def get_job(
    job_id: str,
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container),
) -> dict:
    job = container.metadata.get_job(principal.tenant_id, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    return {"job_id": job.id, "document_id": job.document_id, "stage": job.stage,
            "status": job.status, "attempts": job.attempts, "error": job.error,
            "route_summary": job.route_summary, "created_at": job.created_at,
            "updated_at": job.updated_at}


@router.get("/documents/{document_id}")
def get_document(
    document_id: str,
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container),
) -> dict:
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
    Upsert drops prior vectors first, so it's idempotent."""
    doc = container.metadata.get_document(principal.tenant_id, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
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
    doc = container.metadata.get_document(principal.tenant_id, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    _owned_or_404(doc, principal)
    removed = container.vectors.delete_by_document(principal.tenant_id, document_id)
    container.blob.delete(doc.blob_path)
    container.metadata.delete_document(principal.tenant_id, document_id)
    # Deleting a document changes what's retrievable -> cached answers may be
    # stale. No-op when the cache is disabled.
    if container.cache is not None:
        container.cache.invalidate_tenant(principal.tenant_id)
    container.metrics.incr("documents.deleted")
    container.metadata.write_audit(
        principal.tenant_id, principal.user_id, "delete", document_id,
        {"vectors_removed": removed},
    )
    return {"document_id": document_id, "deleted": True, "vectors_removed": removed}
