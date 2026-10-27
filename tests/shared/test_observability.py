import hashlib
import logging

from app.ingest.pipeline.runner import run_job
from app.shared.domain.models import Document, Job, JobStage, JobStatus
from app.shared.ids import new_object_id
from app.shared.logger import TextFormatter, correlation, log_context, stage


def test_text_formatter_shows_stage_timing_and_job():
    rec = logging.LogRecord("pipeline", logging.INFO, __file__, 10, "stage complete", (), None)
    rec.event = "stage"
    rec.job_id = "j1"
    rec.stage = "embed"
    rec.duration_ms = 12.3
    out = TextFormatter().format(rec)
    assert "INFO" in out and "embed | stage complete | 12.300 ms" in out
    assert "job=j1" in out
    assert not out.startswith("{")


def test_metrics_incr_and_snapshot(container):
    m = container.metrics
    m.incr("ingest.requests")
    m.incr("ingest.requests")
    m.incr("stage.embed", 1, ms=30.0)
    m.incr("stage.embed", 1, ms=10.0)
    snap = m.snapshot()
    assert snap["ingest.requests"]["count"] == 2
    assert snap["stage.embed"]["count"] == 2
    assert snap["stage.embed"]["total_ms"] == 40.0
    assert snap["stage.embed"]["avg_ms"] == 20.0


def test_pipeline_records_stage_metrics(container, tenant, files):

    data = files["data.csv"]
    sha = hashlib.sha256(data).hexdigest()
    blob = container.blob.put(tenant["id"], sha, ".csv", data)
    doc = Document(
        id=new_object_id(),
        tenant_id=tenant["id"],
        owner_user_id=tenant["admin_id"],
        source_type="table",
        blob_path=blob,
        content_sha256=sha,
        mime="text/csv",
        filename="data.csv",
        visibility="private",
        acl_user_ids=[],
    )
    container.metadata.create_document(doc)
    container.metadata.create_job(
        Job(
            id=new_object_id(),
            document_id=doc.id,
            tenant_id=tenant["id"],
            stage=JobStage.PARSE.value,
            status=JobStatus.QUEUED.value,
            attempts=0,
        )
    )
    run_job(container, container.queue.claim_next())

    snap = container.metrics.snapshot()
    assert snap["jobs.done"]["count"] == 1
    assert "stage.embed" in snap and snap["stage.embed"]["count"] == 1
    assert snap["stage.upsert"]["count"] == 1


def test_job_correlation_survives_queue_claim(container, tenant, files):
    data = files["data.csv"]
    sha = hashlib.sha256(data).hexdigest()
    doc = Document(
        id=new_object_id(),
        tenant_id=tenant["id"],
        owner_user_id=tenant["admin_id"],
        source_type="table",
        blob_path=container.blob.put(tenant["id"], sha, ".csv", data),
        content_sha256=sha,
        mime="text/csv",
        filename="data.csv",
        visibility="private",
        acl_user_ids=[],
    )
    with log_context(
        request_id="upload-request", session_id="upload-session", tenant_id=tenant["id"]
    ):
        with stage("upload", emit=False):
            fields = correlation()
    job = Job(new_object_id(), doc.id, doc.tenant_id, "parse", "queued", 0, correlation=fields)
    container.metadata.create_document_with_job(doc, job)
    claimed = container.queue.claim_next()
    assert claimed.correlation == fields
    assert claimed.correlation["traceparent"].startswith("00-")
    run_job(container, claimed)
    assert container.metadata.get_job(doc.tenant_id, job.id).correlation == fields
