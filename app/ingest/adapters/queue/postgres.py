"""Postgres-backed TaskQueue. `FOR UPDATE SKIP LOCKED` lets N worker processes
each atomically claim a different queued job with zero contention -- strictly
better concurrency than the SQLite `BEGIN IMMEDIATE` single-writer hack this
flips from (that only tolerated one true writer at a time).
"""
from __future__ import annotations

from typing import Optional

from app.shared.adapters.postgres.db import transaction
from app.shared.adapters.postgres.metadata_store import _row_to_job
from app.shared.domain.models import Job, JobStatus
from app.ingest.ports.task_queue import TaskQueue, retry_delay_seconds


class PostgresTaskQueue(TaskQueue):
    """Postgres-backed TaskQueue; `lease_seconds` bounds how long a claimed job
    may stay `running` before the reaper reclaims it."""

    def __init__(self, dsn: str, lease_seconds: int = 300):
        self.dsn = dsn
        self.lease_seconds = lease_seconds

    def enqueue(self, job_id: str) -> None:
        """Mark a job ready for processing now, clearing any retry backoff."""
        # available_at is reset: an explicit re-enqueue (e.g. /reprocess) is a
        # deliberate request to run now, not a continuation of a retry backoff.
        with transaction(self.dsn) as cur:
            cur.execute(
                "UPDATE ingestion_jobs SET status='queued', available_at=now(), "
                "updated_at=now() WHERE id=%s",
                (job_id,),
            )

    def claim_next(self) -> Optional[Job]:
        """Atomically claim the oldest due queued job and start its lease."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT * FROM ingestion_jobs WHERE status='queued' "
                "AND available_at <= now() "
                "ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED"
            )
            row = cur.fetchone()
            if row is None:
                return None
            cur.execute(
                "UPDATE ingestion_jobs SET status='running', "
                "lease_expires_at=now() + (%s * interval '1 second'), "
                "updated_at=now() WHERE id=%s",
                (self.lease_seconds, row["id"]),
            )
        job = _row_to_job(row)
        job.status = JobStatus.RUNNING.value
        return job

    def set_stage(self, job_id: str, stage: str) -> None:
        """Record which pipeline stage a running job is currently on."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "UPDATE ingestion_jobs SET stage=%s, updated_at=now() WHERE id=%s",
                (stage, job_id),
            )

    def complete(self, job_id: str) -> None:
        """Mark a job done, clearing its lease."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "UPDATE ingestion_jobs SET status='done', stage='done', error=NULL, "
                "lease_expires_at=NULL, updated_at=now() WHERE id=%s",
                (job_id,),
            )

    def retry_or_dead(self, job_id: str, error: str, max_attempts: int) -> str:
        """Requeue with incremented attempts (clearing the lease), or dead-letter if exhausted."""
        with transaction(self.dsn) as cur:
            cur.execute("SELECT attempts FROM ingestion_jobs WHERE id=%s FOR UPDATE", (job_id,))
            row = cur.fetchone()
            attempts = (row["attempts"] if row else 0) + 1
            status = JobStatus.QUEUED.value if attempts < max_attempts else JobStatus.DEAD.value
            # Hold the job back before it becomes claimable again, so a
            # deterministically-failing document can't spin through every
            # attempt (and every gateway call those attempts make) instantly.
            delay = retry_delay_seconds(attempts)
            cur.execute(
                "UPDATE ingestion_jobs SET attempts=%s, status=%s, error=%s, "
                "lease_expires_at=NULL, "
                "available_at=now() + (%s * interval '1 second'), "
                "updated_at=now() WHERE id=%s",
                (attempts, status, error[:2000], delay, job_id),
            )
        return status

    def reap_expired(self, max_attempts: int) -> int:
        """Reclaim jobs whose lease expired while `running` (same
        increment-attempts-or-dead-letter logic as retry_or_dead)."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT id, attempts FROM ingestion_jobs WHERE status='running' "
                "AND lease_expires_at IS NOT NULL AND lease_expires_at <= now() "
                "FOR UPDATE SKIP LOCKED"
            )
            rows = cur.fetchall()
            for row in rows:
                attempts = row["attempts"] + 1
                status = JobStatus.QUEUED.value if attempts < max_attempts else JobStatus.DEAD.value
                delay = retry_delay_seconds(attempts)
                cur.execute(
                    "UPDATE ingestion_jobs SET attempts=%s, status=%s, "
                    "error='reclaimed: worker lease expired', lease_expires_at=NULL, "
                    "available_at=now() + (%s * interval '1 second'), "
                    "updated_at=now() WHERE id=%s",
                    (attempts, status, delay, row["id"]),
                )
        return len(rows)
