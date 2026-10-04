from __future__ import annotations

from uuid import uuid4

from app.ingest.ports.task_queue import StaleLeaseError, TaskQueue, retry_delay_seconds
from app.shared.adapters.postgres.db import transaction
from app.shared.adapters.postgres.metadata_store import _row_to_job
from app.shared.domain.models import Job, JobStatus


class PostgresTaskQueue(TaskQueue):
    """Postgres-backed TaskQueue; `lease_seconds` bounds how long a claimed job
    may stay `running` before the reaper reclaims it."""

    def __init__(self, dsn: str, lease_seconds: int = 300):
        self.dsn = dsn
        self.lease_seconds = lease_seconds

    def enqueue(self, job_id: str) -> None:
        """Mark a job ready for processing now, clearing any retry backoff."""

        with transaction(self.dsn) as cur:
            cur.execute(
                "UPDATE ingestion_jobs SET status='queued', available_at=now(), lease_token=NULL, "
                "updated_at=now() WHERE id=%s",
                (job_id,),
            )

    def claim_next(self) -> Job | None:
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
            token = uuid4().hex
            generation = uuid4().hex
            cur.execute(
                "UPDATE ingestion_jobs SET status='running', lease_token=%s, generation_id=%s, "
                "lease_expires_at=now() + (%s * interval '1 second'), "
                "updated_at=now() WHERE id=%s RETURNING *",
                (token, generation, self.lease_seconds, row["id"]),
            )
            row = cur.fetchone()
        job = _row_to_job(row)
        job.status = JobStatus.RUNNING.value
        return job

    def set_stage(self, job_id: str, stage: str, lease_token: str) -> None:
        """Record which pipeline stage a running job is currently on."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "UPDATE ingestion_jobs SET stage=%s, updated_at=now() WHERE id=%s "
                "AND status='running' AND lease_token=%s AND lease_expires_at>clock_timestamp()",
                (stage, job_id, lease_token),
            )
            if cur.rowcount != 1:
                raise StaleLeaseError("Stage update rejected: worker no longer owns a valid lease")

    def complete(self, job_id: str, lease_token: str) -> None:
        """Mark a job done, clearing its lease."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "UPDATE ingestion_jobs SET status='done', stage='done', error=NULL, "
                "lease_expires_at=NULL, lease_token=NULL, updated_at=now() WHERE id=%s "
                "AND status='running' AND lease_token=%s AND lease_expires_at>clock_timestamp()",
                (job_id, lease_token),
            )
            if cur.rowcount != 1:
                raise StaleLeaseError("Completion rejected: worker no longer owns a valid lease")

    def retry_or_dead(self, job_id: str, error: str, max_attempts: int, lease_token: str) -> str:
        """Requeue with incremented attempts (clearing the lease), or dead-letter if exhausted."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT attempts FROM ingestion_jobs WHERE id=%s AND status='running' "
                "AND lease_token=%s AND lease_expires_at>clock_timestamp() FOR UPDATE",
                (job_id, lease_token),
            )
            row = cur.fetchone()
            if row is None:
                raise StaleLeaseError("Retry rejected: worker no longer owns a valid lease")
            attempts = (row["attempts"] if row else 0) + 1
            status = JobStatus.QUEUED.value if attempts < max_attempts else JobStatus.DEAD.value

            delay = retry_delay_seconds(attempts)
            cur.execute(
                "UPDATE ingestion_jobs SET attempts=%s, status=%s, error=%s, "
                "lease_expires_at=NULL, lease_token=NULL, "
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
                    "error='reclaimed: worker lease expired', lease_expires_at=NULL, lease_token=NULL, "
                    "available_at=now() + (%s * interval '1 second'), "
                    "updated_at=now() WHERE id=%s",
                    (attempts, status, delay, row["id"]),
                )
        return len(rows)

    def renew(self, job_id: str, lease_token: str) -> None:
        with transaction(self.dsn) as cur:
            cur.execute("SET LOCAL statement_timeout='4000ms'")
            cur.execute("SET LOCAL lock_timeout='3000ms'")
            cur.execute(
                "UPDATE ingestion_jobs SET lease_expires_at=clock_timestamp() + (%s * interval '1 second') "
                "WHERE id=%s AND status='running' AND lease_token=%s AND lease_expires_at>clock_timestamp()",
                (self.lease_seconds, job_id, lease_token),
            )
            if cur.rowcount != 1:
                raise StaleLeaseError("Renewal rejected: worker no longer owns a valid lease")
