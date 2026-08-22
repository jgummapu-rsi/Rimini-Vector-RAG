"""End-to-end pipeline: ingest -> run_job -> chunks + vectors, for every shape,
plus the failure/dead-letter path. Uses the container directly (no HTTP)."""
import hashlib

import pytest

from app.domain.models import Document, Job, JobStage, JobStatus
from app.ids import new_object_id
from app.pipeline.runner import run_job


def _ingest(container, tenant, filename, data):
    """Mimic the API's synchronous half: store blob + create document + job."""
    sha = hashlib.sha256(data).hexdigest()
    ext = "." + filename.rsplit(".", 1)[1]
    from app.domain.models import EXT_TO_SOURCE_TYPE
    blob_path = container.blob.put(tenant["id"], sha, ext, data)
    doc = Document(id=new_object_id(), tenant_id=tenant["id"],
                   owner_user_id=tenant["admin_id"],
                   source_type=EXT_TO_SOURCE_TYPE[ext].value, blob_path=blob_path,
                   content_sha256=sha, mime="x", filename=filename,
                   visibility="private", acl_user_ids=[])
    container.metadata.create_document(doc)
    job = Job(id=new_object_id(), document_id=doc.id, tenant_id=tenant["id"],
              stage=JobStage.PARSE.value, status=JobStatus.QUEUED.value, attempts=0)
    container.metadata.create_job(job)
    return doc, job


@pytest.mark.parametrize("fname", ["data.csv", "notes.txt", "report.docx",
                                   "finance.xlsx", "doc.pdf"])
def test_full_pipeline_each_shape(container, tenant, files, fname):
    doc, job = _ingest(container, tenant, fname, files[fname])
    claimed = container.queue.claim_next()
    run_job(container, claimed)

    assert container.metadata.get_job(tenant["id"], job.id).status == "done"
    # chunks persisted with deterministic ids (document id + zero-padded ordinal)
    chunks = container.metadata.get_document_chunks(tenant["id"], doc.id)
    assert chunks
    assert all(c.id == f"{doc.id}{c.ordinal + 1:03d}" for c in chunks)
    assert container.vectors.count(tenant["id"]) == len(chunks)


def test_metadata_extraction_populates_document(container, tenant, files, monkeypatch):
    monkeypatch.setattr(container.gateway, "chat", lambda messages, model, temperature=0.0: (
        '{"author": "Priya", "date": "2024-03-31", '
        '"topics": ["quarterly review"], "entities": ["APAC", "EMEA"]}'
    ))
    doc, job = _ingest(container, tenant, "notes.txt", files["notes.txt"])
    run_job(container, container.queue.claim_next())

    found = container.metadata.get_document(tenant["id"], doc.id)
    assert found.extracted_metadata == {
        "author": "Priya", "date": "2024-03-31",
        "topics": ["quarterly review"], "entities": ["APAC", "EMEA"],
    }


def test_metadata_extraction_failure_does_not_fail_job(container, tenant, files, monkeypatch):
    def _boom(messages, model, temperature=0.0):
        raise RuntimeError("gateway is down")
    monkeypatch.setattr(container.gateway, "chat", _boom)

    doc, job = _ingest(container, tenant, "notes.txt", files["notes.txt"])
    run_job(container, container.queue.claim_next())

    assert container.metadata.get_job(tenant["id"], job.id).status == "done"
    found = container.metadata.get_document(tenant["id"], doc.id)
    assert found.extracted_metadata == {"author": None, "date": None, "topics": [], "entities": []}


def test_search_finds_table_after_ingest(container, tenant, files):
    _ingest(container, tenant, "finance.xlsx", files["finance.xlsx"])
    run_job(container, container.queue.claim_next())
    q = container.embedder.embed(["regional quarterly sales table"])[0]
    hits = container.vectors.search(tenant["id"], q, top_k=3)
    assert hits and any(h.payload["modality"] == "table" for h in hits)


def test_reprocess_is_idempotent(container, tenant, files):
    doc, job = _ingest(container, tenant, "notes.txt", files["notes.txt"])
    run_job(container, container.queue.claim_next())
    first = container.vectors.count(tenant["id"])
    # re-run the same job: upsert should drop prior vectors, not duplicate
    container.queue.enqueue(job.id)
    run_job(container, container.queue.claim_next())
    assert container.vectors.count(tenant["id"]) == first


def test_dead_letter_on_stage_failure(container, tenant, files, monkeypatch):
    _ingest(container, tenant, "notes.txt", files["notes.txt"])
    claimed = container.queue.claim_next()

    def boom(_texts):
        raise RuntimeError("embed failed")

    monkeypatch.setattr(container.embedder, "embed", boom)
    with pytest.raises(RuntimeError):
        run_job(container, claimed)
    # simulate the worker's handling
    for _ in range(container.settings.max_attempts):
        status = container.queue.retry_or_dead(claimed.id, "embed failed",
                                               container.settings.max_attempts)
    assert status == "dead"
