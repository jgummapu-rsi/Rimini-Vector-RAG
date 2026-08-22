"""Postgres-backed TaskQueue. `FOR UPDATE SKIP LOCKED` lets N worker processes
each atomically claim a different queued job with zero contention -- strictly
better concurrency than the SQLite `BEGIN IMMEDIATE` single-writer hack this
flips from (that only tolerated one true writer at a time).
"""
from __future__ import annotations

from typing import Optional

from app.adapters.postgres.db import transaction
from app.adapters.postgres.metadata_store import _row_to_job
from app.domain.models import Job, JobStatus
from app.ports.task_queue import TaskQueue


class PostgresTaskQueue(TaskQueue):
    def __init__(self, dsn: str):
        self.dsn = dsn

    def enqueue(self, job_id: str) -> None:
        with transaction(self.dsn) as cur:
            cur.execute(
                "UPDATE ingestion_jobs SET status='queued', updated_at=now() WHERE id=%s",
                (job_id,),
            )

    def claim_next(self) -> Optional[Job]:
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT * FROM ingestion_jobs WHERE status='queued' "
                "ORDER BY created_at LIMIT 1 FOR UPDATE SKIP LOCKED"
            )
            row = cur.fetchone()
            if row is None:
                return None
            cur.execute(
                "UPDATE ingestion_jobs SET status='running', updated_at=now() WHERE id=%s",
                (row["id"],),
            )
        job = _row_to_job(row)
        job.status = JobStatus.RUNNING.value
        return job

    def set_stage(self, job_id: str, stage: str) -> None:
        with transaction(self.dsn) as cur:
            cur.execute(
                "UPDATE ingestion_jobs SET stage=%s, updated_at=now() WHERE id=%s",
                (stage, job_id),
            )

    def complete(self, job_id: str) -> None:
        with transaction(self.dsn) as cur:
            cur.execute(
                "UPDATE ingestion_jobs SET status='done', stage='done', error=NULL, "
                "updated_at=now() WHERE id=%s",
                (job_id,),
            )

    def retry_or_dead(self, job_id: str, error: str, max_attempts: int) -> str:
        with transaction(self.dsn) as cur:
            cur.execute("SELECT attempts FROM ingestion_jobs WHERE id=%s FOR UPDATE", (job_id,))
            row = cur.fetchone()
            attempts = (row["attempts"] if row else 0) + 1
            status = JobStatus.QUEUED.value if attempts < max_attempts else JobStatus.DEAD.value
            cur.execute(
                "UPDATE ingestion_jobs SET attempts=%s, status=%s, error=%s, "
                "updated_at=now() WHERE id=%s",
                (attempts, status, error[:2000], job_id),
            )
        return status
