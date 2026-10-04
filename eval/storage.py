from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

from app.ingest.pipeline.chunker import ChunkSpec, chunk_elements
from app.ingest.pipeline.elements import Element
from app.ingest.pipeline.runner import run_job
from app.shared.adapters.postgres.db import transaction
from app.shared.domain.models import Document, DocumentVersion, Job
from app.shared.ids import new_object_id
from app.shared.ports.vector_store import VectorPoint
from scripts.disposable_storage import evaluation_settings

log = logging.getLogger(__name__)


def isolated_evaluation(function):
    @wraps(function)
    def run(*args, **kwargs):
        with evaluation_settings() as settings:
            return function(settings, *args, **kwargs)

    return run


def populate_scifact(container, dataset, tenant_name: str, max_docs: int | None = None) -> str:

    tenant_id = container.metadata.create_tenant(tenant_name)
    spec = ChunkSpec.auto(container.embedder.max_tokens)
    points = []
    pending = []

    def flush():
        vectors = container.embedder.embed_documents([chunk.text for _, _, chunk in pending])
        for (document_id, ordinal, chunk), vector in zip(pending, vectors, strict=True):
            points.append(
                VectorPoint(
                    chunk_id=f"{document_id}::{ordinal:05d}",
                    tenant_id=tenant_id,
                    vector=list(vector),
                    payload={
                        "_id": document_id,
                        "scope": "tenant",
                        "content": chunk.text,
                        "modality": "text",
                    },
                )
            )
        container.vectors.upsert(points)
        points.clear()
        pending.clear()

    for index, document in enumerate(dataset.docs_iter()):
        if max_docs is not None and index >= max_docs:
            break
        text = (document.title + "\n\n" + document.text).strip()
        chunks = chunk_elements(
            [Element(text, "text", "text", "text_layer", 0, {})],
            spec,
            count=container.embedder.count_tokens,
            embed_max=container.embedder.max_tokens,
        )
        for ordinal, chunk in enumerate(chunks):
            pending.append((document.doc_id, ordinal, chunk))
            if len(pending) >= 64:
                flush()
        if (index + 1) % 500 == 0:
            print(f"Prepared {index + 1} corpus documents", flush=True)
    if pending:
        flush()
    return tenant_id


def populate_scifact_pipeline(
    container, dataset, tenant_name: str, max_docs: int | None = None, workers: int = 4
) -> tuple[str, str, dict]:
    """Process SciFact title/abstract text through blobs, queued jobs and publication.

    Uses original SciFact document IDs so gold relevance labels stay independent
    of the ingestion process. Optional metadata generation must be disabled.
    """

    if container.settings.metadata_extraction_enabled:
        raise ValueError("Non-LLM pipeline evaluation requires metadata extraction disabled")
    tenant_id = container.metadata.create_tenant(tenant_name)
    owner = container.metadata.create_user(
        tenant_id, "benchmark@example.test", "admin", new_object_id()
    )
    sources = []
    for index, source in enumerate(dataset.docs_iter()):
        if max_docs is not None and index >= max_docs:
            break
        sources.append(source)

    def process(source):
        data = (source.title + "\n\n" + source.text).strip().encode()
        digest = hashlib.sha256(data).hexdigest()
        blob = container.blob.put(tenant_id, digest, ".txt", data)
        document = Document(
            source.doc_id,
            tenant_id,
            owner,
            "text",
            blob,
            digest,
            "text/plain",
            source.doc_id + ".txt",
            "tenant",
            [],
        )
        job = Job(new_object_id(), document.id, tenant_id, "parse", "queued", 0)
        version = DocumentVersion(
            new_object_id(),
            document.id,
            tenant_id,
            1,
            digest,
            blob,
            document.filename,
            len(data),
            owner,
            job.id,
        )
        container.metadata.create_document_with_job(document, job, version)
        claimed = container.queue.claim_next()
        if claimed is None:
            raise RuntimeError("Accepted benchmark job could not be claimed")
        run_job(container, claimed)

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for count, _ in enumerate(executor.map(process, sources), 1):
            if count % 100 == 0 or count == len(sources):
                print(
                    f"Ingested {count}/{len(sources)} documents in {time.monotonic() - started:.1f}s",
                    flush=True,
                )
    with transaction(container.settings.postgres_dsn) as cur:
        cur.execute(
            "SELECT count(*) AS total, count(*) FILTER (WHERE indexed_version=version "
            "AND active_generation_id IS NOT NULL) AS published FROM documents WHERE tenant_id=%s",
            (tenant_id,),
        )
        counts = cur.fetchone()
        cur.execute(
            "SELECT count(*) AS done FROM ingestion_jobs WHERE tenant_id=%s AND status='done'",
            (tenant_id,),
        )
        done = cur.fetchone()["done"]
    if counts["published"] != len(sources) or done != len(sources):
        raise RuntimeError("Not every benchmark source published successfully")
    return (
        tenant_id,
        owner,
        {
            "ingest_seconds": time.monotonic() - started,
            "documents_published": counts["published"],
            "jobs_done": done,
            "ingest_workers": workers,
        },
    )
