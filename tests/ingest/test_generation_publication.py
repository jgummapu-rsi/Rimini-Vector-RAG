import asyncio
import hashlib
import time

import pytest

from app.ingest.pipeline.lease import renewed_lease
from app.ingest.pipeline.parse_process import _vision_with_lease
from app.ingest.pipeline.runner import run_job
from app.ingest.ports.task_queue import StaleLeaseError
from app.retrieval.rag.query import answer_query
from app.shared.adapters.postgres.db import transaction
from app.shared.domain.models import Document, Job
from app.shared.ids import new_object_id
from app.shared.ports.vector_store import VectorPoint
from tests.conftest import structured_answer


def _queue(container, tenant, text, document=None):
    data = text.encode()
    digest = hashlib.sha256(data).hexdigest()
    blob = container.blob.put(tenant["id"], digest, ".txt", data)
    if document is None:
        document = Document(
            new_object_id(),
            tenant["id"],
            tenant["admin_id"],
            "text",
            blob,
            digest,
            "text/plain",
            "report.txt",
            "tenant",
            [],
        )
        job = Job(new_object_id(), document.id, tenant["id"], "parse", "queued", 0)
        container.metadata.create_document_with_job(document, job)
    else:
        job = Job(new_object_id(), document.id, tenant["id"], "parse", "queued", 0)
        container.metadata.update_document_content_and_queue(
            tenant["id"],
            document.id,
            blob_path=blob,
            content_sha256=digest,
            mime="text/plain",
            source_type="text",
            filename="report.txt",
            visibility="tenant",
            scope="tenant",
            job=job,
            uploaded_by=tenant["admin_id"],
            byte_size=len(data),
        )
    return document, container.queue.claim_next()


def _active(container, tenant, document):
    doc = container.metadata.get_document(tenant["id"], document.id)
    chunks = container.metadata.get_document_chunks(tenant["id"], document.id)
    return (
        doc.active_generation_id,
        doc.indexed_version,
        [(chunk.id, chunk.text) for chunk in chunks],
    )


def test_embedding_failure_keeps_last_good_generation(container, tenant, monkeypatch):
    doc, first = _queue(container, tenant, "Invoice 00123 total 007.00 USD")
    run_job(container, first)
    before = _active(container, tenant, doc)
    _, replacement = _queue(container, tenant, "Invoice 00123 total 999.00 USD", doc)
    monkeypatch.setattr(
        container.embedder,
        "embed",
        lambda texts: (_ for _ in ()).throw(RuntimeError("embedding unavailable")),
    )
    with pytest.raises(RuntimeError, match="embedding unavailable"):
        run_job(container, replacement)
    assert _active(container, tenant, doc) == before
    assert container.vectors.count(tenant["id"]) == len(before[2])


def test_failed_publication_transaction_rolls_back_every_write(container, tenant):
    doc, first = _queue(container, tenant, "Invoice 00123 total 007.00 USD")
    run_job(container, first)
    before = _active(container, tenant, doc)
    _, replacement = _queue(container, tenant, "Invoice 00123 total 999.00 USD", doc)
    with transaction(container.settings.postgres_dsn) as cur:
        cur.execute(
            "CREATE FUNCTION reject_epoch() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'injected publication failure'; END $$"
        )
        cur.execute(
            "CREATE TRIGGER reject_epoch BEFORE INSERT OR UPDATE ON corpus_epochs FOR EACH ROW EXECUTE FUNCTION reject_epoch()"
        )
    with pytest.raises(Exception, match="injected publication failure"):
        run_job(container, replacement)
    assert _active(container, tenant, doc) == before
    with transaction(container.settings.postgres_dsn) as cur:
        cur.execute(
            "SELECT count(*) AS n FROM ingestion_generations WHERE document_id=%s", (doc.id,)
        )
        assert cur.fetchone()["n"] == 1


def test_reclaimed_worker_cannot_mutate_job(container, tenant):
    doc, stale = _queue(container, tenant, "Invoice 00123 total 007.00 USD")
    with transaction(container.settings.postgres_dsn) as cur:
        cur.execute(
            "UPDATE ingestion_jobs SET lease_expires_at=now()-interval '1 second' WHERE id=%s",
            (stale.id,),
        )
    container.queue.reap_expired(5)
    container.queue.enqueue(stale.id)
    current = container.queue.claim_next()
    assert current.lease_token != stale.lease_token
    for operation in (
        lambda: container.queue.complete(stale.id, stale.lease_token),
        lambda: container.queue.set_stage(stale.id, "upsert", stale.lease_token),
        lambda: container.queue.retry_or_dead(stale.id, "stale error", 5, stale.lease_token),
        lambda: container.queue.renew(stale.id, stale.lease_token),
        lambda: run_job(container, stale),
    ):
        with pytest.raises(StaleLeaseError):
            operation()
    run_job(container, current)
    assert _active(container, tenant, doc)[0] == current.generation_id


def test_lease_renews_while_work_is_running(container, tenant):

    container.queue.lease_seconds = 1
    _, job = _queue(container, tenant, "Invoice 00123 total 007.00 USD")
    with renewed_lease(container.queue, job, 1) as check:
        time.sleep(1.5)
        check()
        assert container.queue.reap_expired(5) == 0


def test_delete_before_publication_cannot_resurrect_document(container, tenant, monkeypatch):
    doc, job = _queue(container, tenant, "Invoice 00123 total 007.00 USD")
    original = container.publication.publish

    def deleted(*args, **kwargs):
        container.metadata.delete_document(tenant["id"], doc.id)
        return original(*args, **kwargs)

    monkeypatch.setattr(container.publication, "publish", deleted)
    with pytest.raises(StaleLeaseError, match="deleted"):
        run_job(container, job)
    assert container.vectors.count(tenant["id"]) == 0


def test_new_generation_preserves_history_and_hides_previous_vectors(container, tenant):
    doc, first = _queue(container, tenant, "Invoice 00123 total 007.00 USD")
    run_job(container, first)
    _, second = _queue(container, tenant, "Invoice 00123 total 999.00 USD", doc)
    run_job(container, second)
    assert _active(container, tenant, doc)[:2] == (second.generation_id, 2)
    hits = container.vectors.search(
        tenant["id"], container.embedder.embed(["Invoice"])[0], top_k=50
    )
    assert hits and all(hit.payload["generation_id"] == second.generation_id for hit in hits)
    with transaction(container.settings.postgres_dsn) as cur:
        cur.execute("SELECT chunks FROM ingestion_generations WHERE id=%s", (first.generation_id,))
        assert "007.00" in cur.fetchone()["chunks"][0]["text"]


def test_cache_invalidation_failure_does_not_retry_a_published_generation(
    container, tenant, monkeypatch
):

    doc, first = _queue(container, tenant, "Invoice 00123 total 007.00 USD")
    run_job(container, first)
    container.gateway.chat = lambda *args, **kwargs: structured_answer("007.00 USD")
    result = answer_query(container, tenant["id"], "Invoice amount?", user_id=tenant["admin_id"])
    assert result.answer == "007.00 USD [1]"
    _, replacement = _queue(container, tenant, "Invoice 00123 total 999.00 USD", doc)
    monkeypatch.setattr(
        container.cache,
        "invalidate_tenant",
        lambda *args: (_ for _ in ()).throw(RuntimeError("cache outage")),
    )
    run_job(container, replacement)
    assert container.metadata.get_job(tenant["id"], replacement.id).status == "done"
    container.gateway.chat = lambda *args, **kwargs: structured_answer("999.00 USD")
    result = answer_query(container, tenant["id"], "Invoice amount?", user_id=tenant["admin_id"])
    assert result.answer == "999.00 USD [1]"


def test_published_vector_cannot_be_overwritten(container, tenant):

    doc, first = _queue(container, tenant, "Invoice 00123 total 007.00 USD")
    run_job(container, first)
    before = _active(container, tenant, doc)
    with pytest.raises(ValueError, match="immutable"):
        container.vectors.upsert(
            [
                VectorPoint(
                    before[2][0][0],
                    tenant["id"],
                    [1.0] + [0.0] * (container.embedder.dim - 1),
                    {"_id": doc.id, "content": "forged"},
                )
            ]
        )
    assert _active(container, tenant, doc) == before


def test_generation_manifest_rejects_mutation(container, tenant):
    _, first = _queue(container, tenant, "Invoice 00123 total 007.00 USD")
    run_job(container, first)
    with pytest.raises(Exception, match="immutable"):
        with transaction(container.settings.postgres_dsn) as cur:
            cur.execute(
                "UPDATE ingestion_generations SET chunks='[]' WHERE id=%s", (first.generation_id,)
            )


def test_lease_loss_cancels_pending_vision():

    cancelled = []
    checks = []

    class Gateway:
        async def vision_bounded(self, *args, **kwargs):
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.append(True)

    def check():
        checks.append(True)
        if len(checks) > 1:
            raise StaleLeaseError("reclaimed")

    with pytest.raises(StaleLeaseError, match="reclaimed"):
        asyncio.run(_vision_with_lease(Gateway(), b"image", "prompt", "image/png", 20, 32, check))
    assert cancelled == [True]
