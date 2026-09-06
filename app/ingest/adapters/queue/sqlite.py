"""SQLite-backed TaskQueue. The ingestion_jobs table doubles as the queue.

claim_next() uses BEGIN IMMEDIATE so a single write transaction atomically
selects the oldest queued job and flips it to running — safe for one worker
process (and correct-if-slower for a few). Flips to Celery+Redis later.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.shared.adapters.sqlite.db import connect
from app.shared.adapters.sqlite.metadata_store import _row_to_job
from app.shared.domain.models import Job, JobStatus
from app.ingest.ports.task_queue import TaskQueue, retry_delay_seconds


class SqliteTaskQueue(TaskQueue):
    """SQLite-backed TaskQueue; `lease_seconds` bounds how long a claimed job
    may stay `running` before the reaper reclaims it."""

    def __init__(self, db_path: Path, lease_seconds: int = 300):
        self.db_path = db_path
        self.lease_seconds = lease_seconds

    def enqueue(self, job_id: str) -> None:
        """Mark a job ready for processing now, clearing any retry backoff."""
        # available_at is reset: an explicit re-enqueue (e.g. /reprocess) means
        # run now, not "continue whatever backoff a previous failure set".
        conn = connect(self.db_path)
        try:
            conn.execute(
                "UPDATE ingestion_jobs SET status='queued', "
                "available_at=datetime('now'), updated_at=datetime('now') "
                "WHERE id=?",
                (job_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def claim_next(self) -> Optional[Job]:
        """Atomically claim the oldest due queued job and start its lease."""
        conn = connect(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM ingestion_jobs WHERE status='queued' "
                "AND available_at <= datetime('now') "
                "ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            conn.execute(
                "UPDATE ingestion_jobs SET status='running', "
                "lease_expires_at=datetime('now', ? || ' seconds'), "
                "updated_at=datetime('now') WHERE id=?",
                (f"+{self.lease_seconds}", row["id"]),
            )
            conn.commit()
            job = _row_to_job(row)
            job.status = JobStatus.RUNNING.value
            return job
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def set_stage(self, job_id: str, stage: str) -> None:
        """Record which pipeline stage a running job is currently on."""
        conn = connect(self.db_path)
        try:
            conn.execute(
                "UPDATE ingestion_jobs SET stage=?, updated_at=datetime('now') WHERE id=?",
                (stage, job_id),
            )
            conn.commit()
        finally:
            conn.close()

    def complete(self, job_id: str) -> None:
        """Mark a job done, clearing its lease."""
        conn = connect(self.db_path)
        try:
            conn.execute(
                "UPDATE ingestion_jobs SET status='done', stage='done', error=NULL, "
                "lease_expires_at=NULL, updated_at=datetime('now') WHERE id=?",
                (job_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def retry_or_dead(self, job_id: str, error: str, max_attempts: int) -> str:
        """Requeue with incremented attempts (clearing the lease), or dead-letter if exhausted."""
        conn = connect(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT attempts FROM ingestion_jobs WHERE id=?", (job_id,)
            ).fetchone()
            attempts = (row["attempts"] if row else 0) + 1
            status = JobStatus.QUEUED.value if attempts < max_attempts else JobStatus.DEAD.value
            # Hold the job back before it becomes claimable again -- see the
            # matching comment in the Postgres adapter.
            delay = retry_delay_seconds(attempts)
            conn.execute(
                "UPDATE ingestion_jobs SET attempts=?, status=?, error=?, "
                "lease_expires_at=NULL, "
                "available_at=datetime('now', ? || ' seconds'), "
                "updated_at=datetime('now') WHERE id=?",
                (attempts, status, error[:2000], f"+{delay}", job_id),
            )
            conn.commit()
            return status
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def reap_expired(self, max_attempts: int) -> int:
        """Reclaim jobs whose lease expired while `running`, one row at a time
        (same increment-attempts-or-dead-letter logic as retry_or_dead)."""
        conn = connect(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT id, attempts FROM ingestion_jobs WHERE status='running' "
                "AND lease_expires_at IS NOT NULL AND lease_expires_at <= datetime('now')"
            ).fetchall()
            for row in rows:
                attempts = row["attempts"] + 1
                status = JobStatus.QUEUED.value if attempts < max_attempts else JobStatus.DEAD.value
                delay = retry_delay_seconds(attempts)
                conn.execute(
                    "UPDATE ingestion_jobs SET attempts=?, status=?, "
                    "error='reclaimed: worker lease expired', lease_expires_at=NULL, "
                    "available_at=datetime('now', ? || ' seconds'), "
                    "updated_at=datetime('now') WHERE id=?",
                    (attempts, status, f"+{delay}", row["id"]),
                )
            conn.commit()
            return len(rows)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
