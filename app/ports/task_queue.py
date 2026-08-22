"""TaskQueue port: hand jobs to the worker, record outcomes.

Local adapter = SQLite (ingestion_jobs table doubles as the queue).
Production adapter = Celery + Redis.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from app.domain.models import Job


class TaskQueue(ABC):
    @abstractmethod
    def enqueue(self, job_id: str) -> None:
        """Mark a job ready for processing (status=queued)."""

    @abstractmethod
    def claim_next(self) -> Optional[Job]:
        """Atomically claim the oldest queued job (status->running). None if idle."""

    @abstractmethod
    def complete(self, job_id: str) -> None: ...

    @abstractmethod
    def set_stage(self, job_id: str, stage: str) -> None: ...

    @abstractmethod
    def retry_or_dead(self, job_id: str, error: str, max_attempts: int) -> str:
        """Requeue with incremented attempts, or dead-letter if exhausted.
        Returns the resulting status ('queued' or 'dead')."""
