from app.domain.models import Document, Job, JobStatus
from app.ids import new_object_id


def _seed_job(container, tenant):
    doc = Document(id=new_object_id(), tenant_id=tenant["id"],
                   owner_user_id=tenant["admin_id"], source_type="pdf",
                   blob_path="/tmp/x", content_sha256=new_object_id(), mime="x",
                   filename="x", visibility="private", acl_user_ids=[])
    container.metadata.create_document(doc)
    job = Job(id=new_object_id(), document_id=doc.id, tenant_id=tenant["id"],
              stage="parse", status="queued", attempts=0)
    container.metadata.create_job(job)
    return job


def test_claim_marks_running_and_drains(container, tenant):
    q = container.queue
    job = _seed_job(container, tenant)
    claimed = q.claim_next()
    assert claimed.id == job.id
    assert claimed.status == JobStatus.RUNNING.value
    assert q.claim_next() is None


def test_complete_sets_done(container, tenant):
    q = container.queue
    job = _seed_job(container, tenant)
    q.claim_next()
    q.complete(job.id)
    assert container.metadata.get_job(tenant["id"], job.id).status == "done"


def test_retry_then_dead_letter(container, tenant):
    q = container.queue
    job = _seed_job(container, tenant)
    q.claim_next()
    # max_attempts=3 in test settings
    assert q.retry_or_dead(job.id, "boom", 3) == "queued"   # attempt 1
    assert q.retry_or_dead(job.id, "boom", 3) == "queued"   # attempt 2
    assert q.retry_or_dead(job.id, "boom", 3) == "dead"     # attempt 3 -> dead
    j = container.metadata.get_job(tenant["id"], job.id)
    assert j.status == "dead" and j.attempts == 3 and j.error
