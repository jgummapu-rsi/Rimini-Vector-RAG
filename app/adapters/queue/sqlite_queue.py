"""SQLite-backed TaskQueue. The ingestion_jobs table doubles as the queue.

claim_next() uses BEGIN IMMEDIATE so a single write transaction atomically
selects the oldest queued job and flips it to running — safe for one worker
process (and correct-if-slower for a few). Flips to Celery+Redis later.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.adapters.sqlite.db import connect
from app.adapters.sqlite.metadata_store import _row_to_job
from app.domain.models import Job, JobStatus
from app.ports.task_queue import TaskQueue


class SqliteTaskQueue(TaskQueue):
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def enqueue(self, job_id: str) -> None:
        conn = connect(self.db_path)
        try:
            conn.execute(
                "UPDATE ingestion_jobs SET status='queued', updated_at=datetime('now') "
                "WHERE id=?",
                (job_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def claim_next(self) -> Optional[Job]:
        conn = connect(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM ingestion_jobs WHERE status='queued' "
                "ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                conn.commit()
                return None
            conn.execute(
                "UPDATE ingestion_jobs SET status='running', updated_at=datetime('now') "
                "WHERE id=?",
                (row["id"],),
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
        conn = connect(self.db_path)
        try:
            conn.execute(
                "UPDATE ingestion_jobs SET status='done', stage='done', error=NULL, "
                "updated_at=datetime('now') WHERE id=?",
                (job_id,),
            )
            conn.commit()
        finally:
            conn.close()

    def retry_or_dead(self, job_id: str, error: str, max_attempts: int) -> str:
        conn = connect(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT attempts FROM ingestion_jobs WHERE id=?", (job_id,)
            ).fetchone()
            attempts = (row["attempts"] if row else 0) + 1
            status = JobStatus.QUEUED.value if attempts < max_attempts else JobStatus.DEAD.value
            conn.execute(
                "UPDATE ingestion_jobs SET attempts=?, status=?, error=?, "
                "updated_at=datetime('now') WHERE id=?",
                (attempts, status, error[:2000], job_id),
            )
            conn.commit()
            return status
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
