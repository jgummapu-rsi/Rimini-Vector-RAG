"""Pipeline runner: advances a job through every ingestion stage in order,
from raw bytes to searchable vectors."""
from __future__ import annotations

import logging
import time
from collections import Counter

from app.shared.container import Container
from app.shared.domain.models import (
    Job,
    JobEvent,
    JobEventStatus,
    JobStage,
    Scope,
    finalize_chunks,
)
from app.ingest.pipeline.chunker import ChunkSpec, chunk_elements
from app.ingest.pipeline.loaders import extract_document
from app.ingest.pipeline.metadata_extract import extract_metadata
from app.ingest.pipeline.provenance import location_str
from app.ingest.pipeline.safety import run_with_timeout
from app.shared.ports.vector_store import VectorPoint

log = logging.getLogger("pipeline")

# Order of stages the runner walks (DONE is set by queue.complete()).
STAGES = [
    JobStage.PARSE, JobStage.ROUTE, JobStage.EXTRACT, JobStage.CHUNK,
    JobStage.METADATA, JobStage.EMBED, JobStage.BINARIZE, JobStage.UPSERT,
]


def run_job(container: Container, job: Job) -> None:
    """Run one claimed job through all stages. Raises on failure (worker handles)."""
    doc = container.metadata.get_document(job.tenant_id, job.document_id)
    if doc is None:
        raise RuntimeError(f"document {job.document_id} missing for job {job.id}")

    data = container.blob.get(doc.blob_path)

    ctx = {"document": doc, "bytes": data, "assets": [], "elements": [],
           "chunks": [], "embeddings": [], "binary": []}

    log.info("job start", extra={"event": "job_start", "file": doc.filename,
             "source_type": doc.source_type, "bytes": len(data)})

    t_job = time.perf_counter()
    for seq, stage in enumerate(STAGES):
        container.queue.set_stage(job.id, stage.value)
        t0 = time.perf_counter()
        try:
            _HANDLERS[stage](container, job, ctx)
        except Exception as exc:
            # Record the failure as part of the trace, then let it propagate
            # untouched -- the worker still owns retry/dead-lettering.
            _record_event(container, job, stage, seq, JobEventStatus.ERROR,
                          (time.perf_counter() - t0) * 1000, {"error": str(exc)})
            raise
        dur_ms = (time.perf_counter() - t0) * 1000
        detail = _stage_detail(stage, ctx)
        container.metrics.incr(f"stage.{stage.value}", 1, dur_ms)
        _record_event(container, job, stage, seq, JobEventStatus.OK, dur_ms, detail)
        log.info("stage complete", extra={
            "event": "stage", "stage": stage.value,
            "duration_ms": round(dur_ms, 1), **detail,
        })

    container.queue.complete(job.id)
    # The knowledge base just changed (new/updated vectors are now searchable),
    # so cached answers over it may be stale -- invalidate. This is the single
    # choke point every successful ingest AND reprocess funnels through. No-op
    # when the cache is disabled.
    invalidate_cache_for(container, job.tenant_id, doc.scope)
    total_ms = (time.perf_counter() - t_job) * 1000
    container.metrics.incr("jobs.done", 1, total_ms)
    log.info("job done", extra={
        "event": "job_done", "file": doc.filename,
        "elements": len(ctx["elements"]), "chunks": len(ctx["chunks"]),
        "vectors": len(ctx.get("points", [])), "duration_ms": round(total_ms, 1),
    })


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


def _record_event(container: Container, job: Job, stage: JobStage, seq: int,
                  status: JobEventStatus, dur_ms: float, detail: dict) -> None:
    """Persist one stage outcome for the trace view.

    Trace-keeping is observability, never correctness: a metadata store that
    can't take the write must not fail an otherwise-good ingest, so this
    swallows and logs -- the same posture `_stage_metadata` takes for its
    best-effort LLM call."""
    try:
        container.metadata.record_job_event(JobEvent(
            job_id=job.id, document_id=job.document_id, tenant_id=job.tenant_id,
            stage=stage.value, seq=seq, status=status.value,
            duration_ms=round(dur_ms, 1), attempt=job.attempts, detail=detail,
        ))
    except Exception:  # noqa: BLE001
        log.warning("job event not recorded", extra={
            "event": "job_event_failed", "stage": stage.value}, exc_info=True)


def _stage_detail(stage: JobStage, ctx: dict) -> dict:
    """Meaningful per-stage detail for the logs (what the stage actually did)."""
    if stage == JobStage.PARSE:
        rs = ctx.get("route_summary") or {}
        return {"elements": len(ctx.get("elements", [])),
                "by_modality": rs.get("by_modality"),
                "by_extractor": rs.get("by_extractor")}
    if stage == JobStage.CHUNK:
        chunks = ctx.get("chunks", [])
        return {"chunks": len(chunks),
                "by_modality": dict(Counter(c.modality for c in chunks))}
    if stage == JobStage.METADATA:
        meta = ctx.get("extracted_metadata") or {}
        return {"author": bool(meta.get("author")), "date": bool(meta.get("date")),
                "topics": len(meta.get("topics") or []), "entities": len(meta.get("entities") or [])}
    if stage == JobStage.EMBED:
        return {"vectors": len(ctx.get("embeddings", []))}
    if stage == JobStage.UPSERT:
        return {"points": len(ctx.get("points", []))}
    return {}


def _stage_parse(c: Container, job: Job, ctx: dict) -> None:
    """Parse + route + extract (owned by the per-type loader -- routing is
    inherently type-specific), bounded by a soft timeout so one pathological
    file can't block the worker from ever reaching the next job."""
    doc = ctx["document"]
    elements, route_summary = run_with_timeout(
        lambda: extract_document(doc.filename, ctx["bytes"], c.gateway, c.settings),
        timeout_seconds=c.settings.parse_timeout_seconds,
    )
    ctx["elements"] = elements
    ctx["route_summary"] = route_summary
    c.metadata.set_route_summary(job.id, route_summary)


def _stage_route(c: Container, job: Job, ctx: dict) -> None:
    # Routing already recorded during parse (per-asset, on each element).
    pass


def _stage_extract(c: Container, job: Job, ctx: dict) -> None:
    # Extraction happened in parse (loaders emit final text/markdown elements).
    pass


def _stage_chunk(c: Container, job: Job, ctx: dict) -> None:
    cfg = c.settings
    emb = c.embedder
    # Size chunks against the ACTIVE embedder's real tokenizer + limit, so a
    # chunk is never silently truncated at embed time. auto: derive from the
    # embedder's max_tokens; else use the pinned config values.
    if cfg.chunk_auto_size:
        spec = ChunkSpec.auto(emb.max_tokens, min_tokens=cfg.chunk_min_tokens)
    else:
        spec = ChunkSpec(
            target_tokens=cfg.chunk_target_tokens,
            overlap_tokens=cfg.chunk_overlap_tokens,
            max_tokens=cfg.chunk_max_tokens,
            min_tokens=cfg.chunk_min_tokens,
        )
    records = chunk_elements(ctx["elements"], spec,
                             count=emb.count_tokens, embed_max=emb.max_tokens)
    doc = ctx["document"]
    # Stamp ids/hashes HERE, not implicitly inside the store. `_stage_upsert`
    # uses `ch.id` as the vector-store point id, so the pipeline must own that
    # value rather than read it back off objects a store happened to mutate.
    finalize_chunks(doc.id, records)
    c.metadata.replace_document_chunks(doc.tenant_id, doc.id, records)
    ctx["chunks"] = records
    if not records:
        log.warning("extracted nothing", extra={
            "event": "empty_extraction", "file": doc.filename,
            "source_type": doc.source_type})


def _stage_metadata(c: Container, job: Job, ctx: dict) -> None:
    """Best-effort document metadata (author/date/topics/entities) via a small LLM.
    Auxiliary step: never fails the job (see app.ingest.pipeline.metadata_extract)."""
    doc = ctx["document"]
    chunks = ctx["chunks"]
    if not chunks:
        ctx["extracted_metadata"] = {}
        return

    sample_text = "\n\n".join(ch.text for ch in chunks)   # truncated to budget inside extract_metadata
    result = extract_metadata(c.gateway, c.settings.chat_model, sample_text)
    c.metadata.set_document_metadata(doc.tenant_id, doc.id, result)
    ctx["extracted_metadata"] = result


def _stage_embed(c: Container, job: Job, ctx: dict) -> None:
    chunks = ctx["chunks"]
    if not chunks:
        ctx["embeddings"] = []
        return
    ctx["embeddings"] = c.embedder.embed([ch.text for ch in chunks])


def _stage_binarize(c: Container, job: Job, ctx: dict) -> None:
    # Binarization deferred: we store float embeddings for now. This stage is a
    # passthrough so a sign-threshold + bit-pack step can slot in here later.
    ctx["vectors"] = ctx["embeddings"]


def _stage_upsert(c: Container, job: Job, ctx: dict) -> None:
    doc = ctx["document"]
    chunks = ctx["chunks"]
    vectors = ctx["vectors"]
    if not chunks:
        return
    if len(vectors) != len(chunks):        # never silently drop chunks
        raise RuntimeError(
            f"embedding count {len(vectors)} != chunk count {len(chunks)}")

    # idempotent reprocess: drop any prior vectors for this document first
    c.vectors.delete_by_document(doc.tenant_id, doc.id)

    extracted_metadata = ctx.get("extracted_metadata") or {}
    points: list[VectorPoint] = []
    for ch, vec in zip(chunks, vectors):
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
            "location": location_str(ch.meta),   # e.g. "p.4" or "Business Context" section
            "meta": ch.meta,
            # document-level LLM-extracted signal (app.ingest.pipeline.metadata_extract),
            # folded into BM25's lexical text by the vector store adapters --
            # otherwise this is paid for at ingest and never touches ranking.
            "topics": extracted_metadata.get("topics") or [],
            "entities": extracted_metadata.get("entities") or [],
            "author": extracted_metadata.get("author"),
        }
        # ch.id is the deterministic point identity (= document id + NNN)
        points.append(VectorPoint(chunk_id=ch.id, tenant_id=doc.tenant_id,
                                  vector=vec, payload=payload))

    c.vectors.upsert(points)
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
