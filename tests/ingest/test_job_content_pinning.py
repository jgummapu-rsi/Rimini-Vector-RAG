"""Production-readiness finding 1.1: a job must process the exact bytes it was
created for, even if the document is re-ingested to a newer version while that
job is still sitting queued. Without the fix, `run_job` re-fetched the LIVE
document, so a job created for v1 would process v2's bytes against a job
record that thinks it's still indexing v1 -- silently corrupting which content
a job's own trace/version-delta claims to describe.
"""

import hashlib

import pytest

from app.ingest.pipeline.runner import run_job
from app.ingest.ports.task_queue import StaleLeaseError
from app.shared.domain.models import EXT_TO_SOURCE_TYPE, Document, Job, JobStage, JobStatus
from app.shared.ids import new_object_id


def _put(container, tenant, data: bytes, ext=".txt"):
    sha = hashlib.sha256(data).hexdigest()
    blob_path = container.blob.put(tenant["id"], sha, ext, data)
    return blob_path, sha


def test_run_job_processes_the_pinned_snapshot_not_a_later_live_update(container, tenant):
    """Second upload races ahead of the first job's worker pickup: a job
    created for v1 must still index v1's bytes, never v2's."""
    v1_bytes = b"alpha revision body text, quite distinctive"
    v2_bytes = b"bravo revision body text, completely different words"
    v1_path, v1_sha = _put(container, tenant, v1_bytes)

    doc = Document(
        id=new_object_id(),
        tenant_id=tenant["id"],
        owner_user_id=tenant["admin_id"],
        source_type=EXT_TO_SOURCE_TYPE[".txt"].value,
        blob_path=v1_path,
        content_sha256=v1_sha,
        mime="text/plain",
        filename="notes.txt",
        visibility="private",
        acl_user_ids=[],
    )
    job = Job(
        id=new_object_id(),
        document_id=doc.id,
        tenant_id=tenant["id"],
        stage=JobStage.PARSE.value,
        status=JobStatus.QUEUED.value,
        attempts=0,
    )

    container.metadata.create_document_with_job(doc, job)

    v2_path, v2_sha = _put(container, tenant, v2_bytes)
    container.metadata.update_document_content(
        tenant["id"],
        doc.id,
        blob_path=v2_path,
        content_sha256=v2_sha,
        mime="text/plain",
        source_type=EXT_TO_SOURCE_TYPE[".txt"].value,
        filename="changed.pdf",
        visibility="private",
        scope="tenant",
    )

    claimed = container.queue.claim_next()
    assert claimed.id == job.id
    with pytest.raises(StaleLeaseError, match="superseded"):
        run_job(container, claimed)
    events = container.metadata.get_job_events(tenant["id"], job.id)
    assert any(event.stage == "parse" and event.status == "ok" for event in events)
    assert container.metadata.get_document_chunks(tenant["id"], doc.id) == []


def test_run_job_falls_back_to_the_live_document_for_a_legacy_job(container, tenant, files):
    """A job row created before this migration has blob_path/content_sha256/
    version all None -- run_job must fall back to the live document unchanged
    (the existing, pre-fix behavior), not raise or misbehave."""
    data = files["notes.txt"]
    sha = hashlib.sha256(data).hexdigest()
    blob_path = container.blob.put(tenant["id"], sha, ".txt", data)
    doc = Document(
        id=new_object_id(),
        tenant_id=tenant["id"],
        owner_user_id=tenant["admin_id"],
        source_type=EXT_TO_SOURCE_TYPE[".txt"].value,
        blob_path=blob_path,
        content_sha256=sha,
        mime="text/plain",
        filename="notes.txt",
        visibility="private",
        acl_user_ids=[],
    )
    container.metadata.create_document(doc)
    job = Job(
        id=new_object_id(),
        document_id=doc.id,
        tenant_id=tenant["id"],
        stage=JobStage.PARSE.value,
        status=JobStatus.QUEUED.value,
        attempts=0,
    )
    container.metadata.create_job(job)
    assert job.blob_path is None and job.content_sha256 is None and job.version is None

    run_job(container, container.queue.claim_next())

    assert container.metadata.get_job(tenant["id"], job.id).status == "done"
    assert container.metadata.get_document_chunks(tenant["id"], doc.id)
