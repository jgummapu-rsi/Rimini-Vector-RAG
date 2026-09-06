import pytest

from app.shared.domain.models import Document, Job, JobStatus
from app.shared.ids import new_object_id
from app.ingest.ports.task_queue import RETRY_MAX_SECONDS, retry_delay_seconds


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


# ------------------------------------------------------------- backoff --
# A failed job used to go straight back to `queued` and be re-claimable
# instantly, so a document that always fails burned every attempt -- and every
# LLM gateway call those attempts make -- in milliseconds. The delay lives on
# the ROW, not in the worker's sleep, because a sleeping worker doesn't stop a
# second worker from grabbing the same job immediately.


@pytest.mark.parametrize("attempts,expected", [
    (1, 2), (2, 4), (3, 8), (4, 16), (5, 32), (6, 60), (99, 60),
])
def test_retry_delay_doubles_then_caps(attempts, expected):
    assert retry_delay_seconds(attempts) == expected
    assert retry_delay_seconds(attempts) <= RETRY_MAX_SECONDS


def test_retry_delay_is_zero_for_a_nonsense_attempt_count():
    assert retry_delay_seconds(0) == 0
    assert retry_delay_seconds(-1) == 0


def test_failed_job_is_not_immediately_reclaimable(container, tenant):
    """The regression: retry_or_dead requeued the job and the very next
    claim_next() picked it straight back up."""
    q = container.queue
    job = _seed_job(container, tenant)
    q.claim_next()

    assert q.retry_or_dead(job.id, "boom", 5) == "queued"
    assert container.metadata.get_job(tenant["id"], job.id).status == "queued"
    # queued, but held back -- so there is nothing to claim right now
    assert q.claim_next() is None


def test_backoff_expiry_makes_the_job_claimable_again(container, tenant):
    """The job must come back once its delay elapses -- backoff is a delay, not
    a dead-letter."""
    q = container.queue
    job = _seed_job(container, tenant)
    q.claim_next()
    q.retry_or_dead(job.id, "boom", 5)
    assert q.claim_next() is None

    _expire_backoff(container, job.id)
    reclaimed = q.claim_next()
    assert reclaimed is not None and reclaimed.id == job.id
    assert reclaimed.status == JobStatus.RUNNING.value


def test_enqueue_clears_an_active_backoff(container, tenant):
    """An explicit re-enqueue (what /reprocess does) means run now -- it must
    not inherit a delay left over from an earlier failure."""
    q = container.queue
    job = _seed_job(container, tenant)
    q.claim_next()
    q.retry_or_dead(job.id, "boom", 5)
    assert q.claim_next() is None        # backing off

    q.enqueue(job.id)
    assert q.claim_next() is not None    # available again immediately


def test_dead_job_is_never_claimed_regardless_of_backoff(container, tenant):
    q = container.queue
    job = _seed_job(container, tenant)
    q.claim_next()
    assert q.retry_or_dead(job.id, "boom", 1) == "dead"
    _expire_backoff(container, job.id)
    assert q.claim_next() is None        # dead-lettered, not retried


def _expire_backoff(container, job_id: str) -> None:
    """Fast-forward past a job's retry delay instead of sleeping for it."""
    from app.shared.adapters.sqlite.db import transaction
    with transaction(container.settings.sqlite_path) as c:
        c.execute("UPDATE ingestion_jobs SET available_at=datetime('now','-1 hour') "
                  "WHERE id=?", (job_id,))


# ------------------------------------------------------ stuck-job reaper --
# A worker that crashes (or is killed) mid-job leaves its claimed row
# `running` forever -- nothing else ever re-selects a `running` row. The
# reaper reclaims a job once its lease (set at claim time) expires.


def _expire_lease(container, job_id: str) -> None:
    """Fast-forward past a job's lease instead of waiting JOB_LEASE_SECONDS."""
    from app.shared.adapters.sqlite.db import transaction
    with transaction(container.settings.sqlite_path) as c:
        c.execute("UPDATE ingestion_jobs SET lease_expires_at=datetime('now','-1 hour') "
                  "WHERE id=?", (job_id,))


def test_claim_next_sets_a_future_lease(container, tenant):
    from app.shared.adapters.sqlite.db import transaction
    q = container.queue
    job = _seed_job(container, tenant)
    q.claim_next()
    with transaction(container.settings.sqlite_path) as c:
        row = c.execute("SELECT lease_expires_at FROM ingestion_jobs WHERE id=?",
                         (job.id,)).fetchone()
    assert row["lease_expires_at"] is not None


def test_reap_expired_reclaims_a_stuck_job(container, tenant):
    q = container.queue
    job = _seed_job(container, tenant)
    q.claim_next()
    _expire_lease(container, job.id)

    reclaimed = q.reap_expired(max_attempts=5)
    assert reclaimed == 1
    j = container.metadata.get_job(tenant["id"], job.id)
    assert j.status == "queued" and j.attempts == 1
    # held back by the same backoff schedule as a normal failure
    assert q.claim_next() is None


def test_reap_expired_dead_letters_after_max_attempts(container, tenant):
    q = container.queue
    job = _seed_job(container, tenant)
    q.claim_next()
    _expire_lease(container, job.id)

    assert q.reap_expired(max_attempts=1) == 1
    j = container.metadata.get_job(tenant["id"], job.id)
    assert j.status == "dead"


def test_reap_expired_ignores_jobs_still_within_their_lease(container, tenant):
    q = container.queue
    job = _seed_job(container, tenant)
    q.claim_next()  # lease starts now, far from expired

    assert q.reap_expired(max_attempts=5) == 0
    j = container.metadata.get_job(tenant["id"], job.id)
    assert j.status == "running"


def test_reap_expired_ignores_queued_jobs(container, tenant):
    """Only `running` jobs have a lease to expire -- a merely-queued job must
    never be touched by the reaper."""
    q = container.queue
    _seed_job(container, tenant)
    assert q.reap_expired(max_attempts=5) == 0
