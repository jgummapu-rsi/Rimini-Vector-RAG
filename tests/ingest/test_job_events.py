"""Per-stage trace events: the data behind the Document Trace UI.

The runner has always logged stage timings; these tests cover the part that is
new — that the same information is *persisted*, ordered, and cleaned up, so a
trace can be replayed long after the job finished.
"""
import pytest

from app.shared.domain.models import Document, Job, JobEvent, JobEventStatus, JobStage
from app.shared.ids import new_object_id
from app.ingest.pipeline.runner import STAGES, run_job


def _event(tenant_id, job_id, doc_id, stage, seq, status=JobEventStatus.OK.value,
           duration_ms=1.5, detail=None, attempt=0):
    return JobEvent(job_id=job_id, document_id=doc_id, tenant_id=tenant_id,
                    stage=stage, seq=seq, status=status, duration_ms=duration_ms,
                    detail=detail or {}, attempt=attempt)


def test_events_round_trip_in_execution_order(container, tenant):
    tid, job_id, doc_id = tenant["id"], new_object_id(), new_object_id()
    # insert out of order on purpose - ordering must come from (attempt, seq)
    for seq in (2, 0, 1):
        container.metadata.record_job_event(
            _event(tid, job_id, doc_id, STAGES[seq].value, seq,
                   detail={"n": seq}, duration_ms=seq * 10.0))

    events = container.metadata.get_job_events(tid, job_id)
    assert [e.seq for e in events] == [0, 1, 2]
    assert [e.stage for e in events] == [s.value for s in STAGES[:3]]
    assert events[2].detail == {"n": 2}
    assert events[2].duration_ms == 20.0


def test_retry_attempts_stay_ordered_and_separate(container, tenant):
    tid, job_id, doc_id = tenant["id"], new_object_id(), new_object_id()
    for attempt in (1, 0):
        for seq in (1, 0):
            container.metadata.record_job_event(
                _event(tid, job_id, doc_id, STAGES[seq].value, seq, attempt=attempt))

    events = container.metadata.get_job_events(tid, job_id)
    assert [(e.attempt, e.seq) for e in events] == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_events_are_tenant_scoped(container, tenant, other_tenant):
    tid, job_id, doc_id = tenant["id"], new_object_id(), new_object_id()
    container.metadata.record_job_event(_event(tid, job_id, doc_id, "parse", 0))

    assert len(container.metadata.get_job_events(tid, job_id)) == 1
    assert container.metadata.get_job_events(other_tenant["id"], job_id) == []


def test_full_run_records_one_ok_event_per_stage(container, tenant, files):
    doc_id, job_id = _ingest(container, tenant, "report.docx", files["report.docx"])
    job = container.queue.claim_next()
    run_job(container, job)

    events = container.metadata.get_job_events(tenant["id"], job_id)
    assert [e.stage for e in events] == [s.value for s in STAGES]
    assert all(e.status == JobEventStatus.OK.value for e in events)
    assert all(e.duration_ms >= 0 for e in events)

    # the detail payload is what the UI renders - it must actually carry signal
    by_stage = {e.stage: e.detail for e in events}
    assert by_stage[JobStage.CHUNK.value]["chunks"] > 0
    assert by_stage[JobStage.EMBED.value]["vectors"] > 0
    assert by_stage[JobStage.UPSERT.value]["points"] > 0


def test_failing_stage_records_an_error_event_then_reraises(
    container, tenant, files, monkeypatch
):
    doc_id, job_id = _ingest(container, tenant, "notes.txt", files["notes.txt"])

    import app.ingest.pipeline.runner as runner

    def boom(c, job, ctx):
        raise RuntimeError("embedder exploded")

    monkeypatch.setitem(runner._HANDLERS, JobStage.EMBED, boom)

    job = container.queue.claim_next()
    with pytest.raises(RuntimeError, match="embedder exploded"):
        run_job(container, job)

    events = container.metadata.get_job_events(tenant["id"], job_id)
    assert events[-1].stage == JobStage.EMBED.value
    assert events[-1].status == JobEventStatus.ERROR.value
    assert "embedder exploded" in events[-1].detail["error"]
    # stages before the failure are still recorded as successes
    assert [e.status for e in events[:-1]] == [JobEventStatus.OK.value] * (len(events) - 1)


def test_a_broken_trace_store_never_fails_the_job(container, tenant, files, monkeypatch):
    """Trace-keeping is observability. If it breaks, ingestion must still work."""
    _ingest(container, tenant, "notes.txt", files["notes.txt"])

    def explode(_event):
        raise RuntimeError("trace store down")

    monkeypatch.setattr(container.metadata, "record_job_event", explode)

    job = container.queue.claim_next()
    run_job(container, job)          # must not raise
    assert container.metadata.get_job(tenant["id"], job.id).status == "done"


def test_deleting_a_document_removes_its_events(container, tenant, files):
    doc_id, job_id = _ingest(container, tenant, "notes.txt", files["notes.txt"])
    run_job(container, container.queue.claim_next())
    assert container.metadata.get_job_events(tenant["id"], job_id)

    container.metadata.delete_document(tenant["id"], doc_id)
    assert container.metadata.get_job_events(tenant["id"], job_id) == []


def _ingest(container, tenant, filename, data) -> tuple[str, str]:
    """Create the document + queued job rows an /ingest POST would create."""
    import hashlib

    sha = hashlib.sha256(data).hexdigest()
    ext = "." + filename.rsplit(".", 1)[1]
    doc = Document(
        id=new_object_id(), tenant_id=tenant["id"], owner_user_id=tenant["member_id"],
        source_type="docx" if ext in (".docx", ".txt", ".md") else "table",
        blob_path=container.blob.put(tenant["id"], sha, ext, data),
        content_sha256=sha, mime="application/octet-stream", filename=filename,
        visibility="private",
    )
    container.metadata.create_document(doc)
    job = Job(id=new_object_id(), document_id=doc.id, tenant_id=tenant["id"],
              stage=JobStage.PARSE.value, status="queued", attempts=0)
    container.metadata.create_job(job)
    return doc.id, job.id

