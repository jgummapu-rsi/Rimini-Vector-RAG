from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from app.ingest.ports.task_queue import RETRY_MAX_SECONDS, retry_delay_seconds
from app.shared.adapters.postgres.db import transaction
from app.shared.domain.models import Document, Job, JobStatus
from app.shared.ids import new_object_id
from app.shared.ports.metadata_store import IngestionConflict


def _seed_job(container, tenant):
    doc = Document(
        id=new_object_id(),
        tenant_id=tenant["id"],
        owner_user_id=tenant["admin_id"],
        source_type="pdf",
        blob_path="/tmp/x",
        content_sha256=new_object_id(),
        mime="x",
        filename="x",
        visibility="private",
        acl_user_ids=[],
    )
    container.metadata.create_document(doc)
    job = Job(
        id=new_object_id(),
        document_id=doc.id,
        tenant_id=tenant["id"],
        stage="parse",
        status="queued",
        attempts=0,
    )
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
    claimed = q.claim_next()
    q.complete(job.id, claimed.lease_token)
    assert container.metadata.get_job(tenant["id"], job.id).status == "done"


def test_retry_then_dead_letter(container, tenant):
    q = container.queue
    job = _seed_job(container, tenant)
    for expected in ("queued", "queued", "dead"):
        claimed = q.claim_next()
        assert q.retry_or_dead(job.id, "boom", 3, claimed.lease_token) == expected
        _expire_backoff(container, job.id)
    j = container.metadata.get_job(tenant["id"], job.id)
    assert j.status == "dead" and j.attempts == 3 and j.error


@pytest.mark.parametrize(
    "attempts,expected",
    [
        (1, 2),
        (2, 4),
        (3, 8),
        (4, 16),
        (5, 32),
        (6, 60),
        (99, 60),
    ],
)
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
    claimed = q.claim_next()

    assert q.retry_or_dead(job.id, "boom", 5, claimed.lease_token) == "queued"
    assert container.metadata.get_job(tenant["id"], job.id).status == "queued"

    assert q.claim_next() is None


def test_backoff_expiry_makes_the_job_claimable_again(container, tenant):
    """The job must come back once its delay elapses -- backoff is a delay, not
    a dead-letter."""
    q = container.queue
    job = _seed_job(container, tenant)
    claimed = q.claim_next()
    q.retry_or_dead(job.id, "boom", 5, claimed.lease_token)
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
    claimed = q.claim_next()
    q.retry_or_dead(job.id, "boom", 5, claimed.lease_token)
    assert q.claim_next() is None

    q.enqueue(job.id)
    assert q.claim_next() is not None


def test_dead_job_is_never_claimed_regardless_of_backoff(container, tenant):
    q = container.queue
    job = _seed_job(container, tenant)
    claimed = q.claim_next()
    assert q.retry_or_dead(job.id, "boom", 1, claimed.lease_token) == "dead"
    _expire_backoff(container, job.id)
    assert q.claim_next() is None


def _expire_backoff(container, job_id: str) -> None:
    """Fast-forward past a job's retry delay instead of sleeping for it."""
    with transaction(container.settings.postgres_dsn) as c:
        c.execute(
            "UPDATE ingestion_jobs SET available_at=now() - interval '1 hour' WHERE id=%s",
            (job_id,),
        )


def _expire_lease(container, job_id: str) -> None:
    """Fast-forward past a job's lease instead of waiting JOB_LEASE_SECONDS."""
    with transaction(container.settings.postgres_dsn) as c:
        c.execute(
            "UPDATE ingestion_jobs SET lease_expires_at=now() - interval '1 hour' WHERE id=%s",
            (job_id,),
        )


def test_claim_next_sets_a_future_lease(container, tenant):
    q = container.queue
    job = _seed_job(container, tenant)
    q.claim_next()
    with transaction(container.settings.postgres_dsn) as c:
        c.execute("SELECT lease_expires_at FROM ingestion_jobs WHERE id=%s", (job.id,))
        row = c.fetchone()
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
    q.claim_next()

    assert q.reap_expired(max_attempts=5) == 0
    j = container.metadata.get_job(tenant["id"], job.id)
    assert j.status == "running"


def test_reap_expired_ignores_queued_jobs(container, tenant):
    """Only `running` jobs have a lease to expire -- a merely-queued job must
    never be touched by the reaper."""
    q = container.queue
    _seed_job(container, tenant)
    assert q.reap_expired(max_attempts=5) == 0


def test_concurrent_update_and_reprocess_accept_exactly_one(container, tenant):
    document = Document(
        new_object_id(),
        tenant["id"],
        tenant["admin_id"],
        "text",
        "old",
        "old-hash",
        "text/plain",
        "file.txt",
        "private",
        [],
    )
    container.metadata.create_document(document)
    barrier = Barrier(2)

    def enqueue(update):
        job = Job(new_object_id(), document.id, tenant["id"], "parse", "queued", 0)
        barrier.wait()
        try:
            if update:
                container.metadata.update_document_content_and_queue(
                    tenant["id"],
                    document.id,
                    blob_path="new",
                    content_sha256="new-hash",
                    mime="text/plain",
                    source_type="text",
                    filename="file.txt",
                    visibility="private",
                    scope="tenant",
                    job=job,
                    uploaded_by=tenant["admin_id"],
                    byte_size=3,
                )
            else:
                container.metadata.create_job(job)
            return update, True
        except IngestionConflict:
            return update, False

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(enqueue, [True, False]))
    assert sum(accepted for _, accepted in results) == 1
    update_won = any(update and accepted for update, accepted in results)
    current = container.metadata.get_document(tenant["id"], document.id)
    assert current.version == (2 if update_won else 1)
    assert current.content_sha256 == ("new-hash" if update_won else "old-hash")
    assert len(container.metadata.list_document_versions(tenant["id"], document.id)) == int(
        update_won
    )
