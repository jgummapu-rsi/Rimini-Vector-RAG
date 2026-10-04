"""Pipeline runner: advances a job through every ingestion stage in order,
from raw bytes to searchable vectors."""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import replace as _replace

from app.ingest.pipeline.chunker import ChunkSpec, chunk_elements
from app.ingest.pipeline.delta import diff_chunks
from app.ingest.pipeline.lease import renewed_lease
from app.ingest.pipeline.metadata_extract import extract_metadata
from app.ingest.pipeline.parse_process import extract_bounded
from app.ingest.pipeline.provenance import location_str
from app.ingest.pipeline.safety import UnsafeContentError
from app.shared.container import Container
from app.shared.domain.models import (
    Job,
    JobEvent,
    JobEventStatus,
    JobStage,
    Scope,
    finalize_chunks,
)
from app.shared.ports.vector_store import VectorPoint

log = logging.getLogger(__name__)

STAGES = [
    JobStage.PARSE,
    JobStage.ROUTE,
    JobStage.EXTRACT,
    JobStage.CHUNK,
    JobStage.METADATA,
    JobStage.EMBED,
    JobStage.BINARIZE,
    JobStage.UPSERT,
]


def run_job(container: Container, job: Job) -> None:
    with renewed_lease(container.queue, job, container.settings.job_lease_seconds) as check:
        _run_claimed_job(container, job, check)


def _run_claimed_job(container: Container, job: Job, check) -> None:
    """Run one claimed job through all stages. Raises on failure (worker handles)."""
    doc = container.metadata.get_document(job.tenant_id, job.document_id)
    if doc is None:
        raise RuntimeError(f"document {job.document_id} missing for job {job.id}")

    if job.blob_path is not None:
        doc = _replace(
            doc,
            blob_path=job.blob_path,
            content_sha256=job.content_sha256,
            version=job.version or doc.version,
            filename=job.filename or doc.filename,
            source_type=job.source_type or doc.source_type,
        )

    data = container.blob.get(doc.blob_path)

    ctx = {
        "document": doc,
        "bytes": data,
        "assets": [],
        "elements": [],
        "chunks": [],
        "embeddings": [],
        "binary": [],
        "chunk_delta": {},
        "check_lease": check,
    }

    log.info(
        "job start",
        extra={
            "event": "job_start",
            "file": doc.filename,
            "source_type": doc.source_type,
            "bytes": len(data),
        },
    )

    t_job = time.perf_counter()
    for seq, stage in enumerate(STAGES):
        check()
        container.queue.set_stage(job.id, stage.value, job.lease_token)
        t0 = time.perf_counter()
        try:
            _HANDLERS[stage](container, job, ctx)
        except Exception as exc:
            _record_event(
                container,
                job,
                stage,
                seq,
                JobEventStatus.ERROR,
                (time.perf_counter() - t0) * 1000,
                {"error": str(exc)},
            )
            raise
        dur_ms = (time.perf_counter() - t0) * 1000
        detail = _stage_detail(stage, ctx)
        _metric(container, f"stage.{stage.value}", dur_ms)
        _record_event(container, job, stage, seq, JobEventStatus.OK, dur_ms, detail)
        log.debug(
            "stage complete",
            extra={
                "event": "stage",
                "stage": stage.value,
                "duration_ms": round(dur_ms, 1),
                **detail,
            },
        )

    try:
        invalidate_cache_for(container, job.tenant_id, doc.scope)
    except Exception:
        log.warning(
            "Published generation cache invalidation failed",
            extra={"event": "cache_invalidation_failed", "job_id": job.id},
            exc_info=True,
        )
    total_ms = (time.perf_counter() - t_job) * 1000
    _metric(container, "jobs.done", total_ms)
    log.info(
        "job done",
        extra={
            "event": "job_done",
            "file": doc.filename,
            "elements": len(ctx["elements"]),
            "chunks": len(ctx["chunks"]),
            "vectors": len(ctx.get("points", [])),
            "duration_ms": round(total_ms, 1),
        },
    )


def _metric(container, name, duration_ms):
    try:
        container.metrics.incr(name, 1, duration_ms)
    except Exception:
        log.warning(
            "Ingestion metric could not be recorded",
            extra={"event": "metric_write_failed", "metric": name},
        )


def invalidate_cache_for(container: Container, tenant_id: str, scope: str) -> None:
    """Drop cached answers made stale by a change to `document`.

    A tenant-scoped document only affects its own tenant. A scope=global one is
    readable from EVERY tenant (app.retrieval.rag.access.can_view), so a per-tenant bump
    would leave every other tenant answering from a knowledge base that no
    longer exists. No-op when the cache is disabled.
    """
    if container.cache is None:
        return
    if scope == Scope.GLOBAL.value:
        container.cache.invalidate_all()
    else:
        container.cache.invalidate_tenant(tenant_id)


def _record_version_delta(container: Container, job: Job, ctx: dict) -> None:
    """Attach this run's chunk delta to the document version this job ingested.

    A no-op in the store when no version row references this job -- `/reprocess`
    re-runs the pipeline over unchanged bytes and so creates no new version. The
    delta is still in the job trace either way (`_stage_detail`), which is where
    a reprocess's "did my chunker change anything?" answer lives.

    Same posture as `_record_event`: this is observability, so a store that
    can't take the write must not fail an ingest that has already succeeded --
    the job is `complete` by the time we get here.
    """
    delta = ctx.get("chunk_delta")
    if not delta:
        return
    try:
        container.metadata.set_version_delta(job.id, delta)
    except Exception:  # noqa: BLE001
        log.warning(
            "version delta not recorded",
            extra={"event": "version_delta_failed", "job_id": job.id},
            exc_info=True,
        )


def _record_event(
    container: Container,
    job: Job,
    stage: JobStage,
    seq: int,
    status: JobEventStatus,
    dur_ms: float,
    detail: dict,
) -> None:
    """Persist one stage outcome for the trace view.

    Trace-keeping is observability, never correctness: a metadata store that
    can't take the write must not fail an otherwise-good ingest, so this
    swallows and logs -- the same posture `_stage_metadata` takes for its
    best-effort LLM call."""
    try:
        container.metadata.record_job_event(
            JobEvent(
                job_id=job.id,
                document_id=job.document_id,
                tenant_id=job.tenant_id,
                stage=stage.value,
                seq=seq,
                status=status.value,
                duration_ms=round(dur_ms, 1),
                attempt=job.attempts,
                detail=detail,
            )
        )
    except Exception:  # noqa: BLE001
        log.warning(
            "job event not recorded",
            extra={"event": "job_event_failed", "stage": stage.value},
            exc_info=True,
        )


def _stage_detail(stage: JobStage, ctx: dict) -> dict:
    """Meaningful per-stage detail for the logs (what the stage actually did)."""
    if stage == JobStage.PARSE:
        rs = ctx.get("route_summary") or {}
        return {
            "elements": len(ctx.get("elements", [])),
            "by_modality": rs.get("by_modality"),
            "by_extractor": rs.get("by_extractor"),
        }
    if stage == JobStage.CHUNK:
        chunks = ctx.get("chunks", [])
        delta = ctx.get("chunk_delta") or {}
        return {
            "chunks": len(chunks),
            "by_modality": dict(Counter(c.modality for c in chunks)),
            "added": delta.get("added"),
            "removed": delta.get("removed"),
            "unchanged": delta.get("unchanged"),
        }
    if stage == JobStage.METADATA:
        meta = ctx.get("extracted_metadata") or {}
        return {
            "author": bool(meta.get("author")),
            "date": bool(meta.get("date")),
            "topics": len(meta.get("topics") or []),
            "entities": len(meta.get("entities") or []),
        }
    if stage == JobStage.EMBED:
        return {"vectors": len(ctx.get("embeddings", []))}
    if stage == JobStage.UPSERT:
        return {"points": len(ctx.get("points", []))}
    return {}


def _stage_parse(c: Container, job: Job, ctx: dict) -> None:
    doc = ctx["document"]
    elements, route_summary = extract_bounded(
        doc.filename,
        ctx["bytes"],
        c.gateway,
        c.settings,
        lambda key: c.metadata.get_extraction_artifact(job.id, doc.content_sha256, key),
        lambda key, artifact: c.metadata.put_extraction_artifact(
            job.id, doc.content_sha256, key, artifact
        ),
        check_lease=ctx["check_lease"],
    )
    ctx["elements"] = elements
    ctx["route_summary"] = route_summary
    c.metadata.set_route_summary(job.id, route_summary)


def _stage_route(c: Container, job: Job, ctx: dict) -> None:

    pass


def _stage_extract(c: Container, job: Job, ctx: dict) -> None:

    pass


def _stage_chunk(c: Container, job: Job, ctx: dict) -> None:
    cfg = c.settings
    emb = c.embedder

    if cfg.chunk_auto_size:
        spec = ChunkSpec.auto(emb.max_tokens, min_tokens=cfg.chunk_min_tokens)
    else:
        spec = ChunkSpec(
            target_tokens=cfg.chunk_target_tokens,
            overlap_tokens=cfg.chunk_overlap_tokens,
            max_tokens=cfg.chunk_max_tokens,
            min_tokens=cfg.chunk_min_tokens,
        )
    records = chunk_elements(
        ctx["elements"], spec, count=emb.count_tokens, embed_max=emb.max_tokens
    )
    if not records and any(element.text.strip() for element in ctx["elements"]):
        raise UnsafeContentError("Meaningful extracted evidence produced no indexable chunks")
    doc = ctx["document"]
    previous = c.metadata.get_document_chunks(doc.tenant_id, doc.id)
    finalize_chunks(job.generation_id or doc.id, records)
    ctx["chunk_delta"] = diff_chunks(previous, records)
    ctx["chunks"] = records
    if not records:
        log.warning(
            "extracted nothing",
            extra={
                "event": "empty_extraction",
                "file": doc.filename,
                "source_type": doc.source_type,
            },
        )


def _stage_metadata(c: Container, job: Job, ctx: dict) -> None:
    """Best-effort document metadata (author/date/topics/entities) via a small LLM.
    Auxiliary step: never fails the job (see app.ingest.pipeline.metadata_extract)."""
    ctx["document"]
    chunks = ctx["chunks"]
    if not chunks or not c.settings.metadata_extraction_enabled:
        ctx["extracted_metadata"] = {}
        return

    sample_text = "\n\n".join(ch.text for ch in chunks)
    result = extract_metadata(c.gateway, c.settings.chat_model, sample_text)
    ctx["extracted_metadata"] = result


def _stage_embed(c: Container, job: Job, ctx: dict) -> None:
    chunks = ctx["chunks"]
    if not chunks:
        ctx["embeddings"] = []
        return
    ctx["embeddings"] = c.embedder.embed_documents([ch.text for ch in chunks])


def _stage_binarize(c: Container, job: Job, ctx: dict) -> None:

    ctx["vectors"] = ctx["embeddings"]


def _stage_upsert(c: Container, job: Job, ctx: dict) -> None:
    doc = ctx["document"]
    chunks = ctx["chunks"]
    vectors = ctx["vectors"]
    if len(vectors) != len(chunks):
        raise RuntimeError(f"embedding count {len(vectors)} != chunk count {len(chunks)}")

    if not chunks:
        raise UnsafeContentError("Empty extraction cannot replace a published generation")

    extracted_metadata = ctx.get("extracted_metadata") or {}
    points: list[VectorPoint] = []
    for ch, vec in zip(chunks, vectors, strict=False):
        payload = {
            "_id": doc.id,
            "user_id": doc.owner_user_id,
            "visibility": doc.visibility,
            "acl_user_ids": doc.acl_user_ids,
            "scope": doc.scope,
            "modality": ch.modality,
            "source_type": doc.source_type,
            "content": ch.text,
            "filename": doc.filename,
            "version": doc.version,
            "location": location_str(ch.meta),
            "meta": ch.meta,
            "topics": extracted_metadata.get("topics") or [],
            "entities": extracted_metadata.get("entities") or [],
            "author": extracted_metadata.get("author"),
        }

        points.append(
            VectorPoint(chunk_id=ch.id, tenant_id=doc.tenant_id, vector=vec, payload=payload)
        )

    ctx["check_lease"]()
    c.publication.publish(job, doc, chunks, points, extracted_metadata, ctx["chunk_delta"])
    ctx["points"] = points


_HANDLERS = {
    JobStage.PARSE: _stage_parse,
    JobStage.ROUTE: _stage_route,
    JobStage.EXTRACT: _stage_extract,
    JobStage.CHUNK: _stage_chunk,
    JobStage.METADATA: _stage_metadata,
    JobStage.EMBED: _stage_embed,
    JobStage.BINARIZE: _stage_binarize,
    JobStage.UPSERT: _stage_upsert,
}
