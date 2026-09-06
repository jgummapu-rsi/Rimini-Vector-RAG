import json
import logging

from app.shared.observability import JsonFormatter


def test_json_formatter_emits_valid_json_with_extra_fields():
    rec = logging.LogRecord("pipeline", logging.INFO, __file__, 10,
                            "stage complete", (), None)
    rec.event = "stage"
    rec.job_id = "j1"
    rec.stage = "embed"
    rec.duration_ms = 12.3
    out = json.loads(JsonFormatter().format(rec))
    assert out["level"] == "INFO"
    assert out["logger"] == "pipeline"
    assert out["msg"] == "stage complete"
    assert out["event"] == "stage" and out["job_id"] == "j1"
    assert out["stage"] == "embed" and out["duration_ms"] == 12.3


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
    import hashlib
    from app.shared.domain.models import Document, Job, JobStage, JobStatus
    from app.shared.ids import new_object_id
    from app.ingest.pipeline.runner import run_job

    data = files["data.csv"]
    sha = hashlib.sha256(data).hexdigest()
    blob = container.blob.put(tenant["id"], sha, ".csv", data)
    doc = Document(id=new_object_id(), tenant_id=tenant["id"],
                   owner_user_id=tenant["admin_id"], source_type="table",
                   blob_path=blob, content_sha256=sha, mime="text/csv",
                   filename="data.csv", visibility="private", acl_user_ids=[])
    container.metadata.create_document(doc)
    container.metadata.create_job(Job(id=new_object_id(), document_id=doc.id,
        tenant_id=tenant["id"], stage=JobStage.PARSE.value,
        status=JobStatus.QUEUED.value, attempts=0))
    run_job(container, container.queue.claim_next())

    snap = container.metrics.snapshot()
    assert snap["jobs.done"]["count"] == 1
    assert "stage.embed" in snap and snap["stage.embed"]["count"] == 1
    assert snap["stage.upsert"]["count"] == 1
