"""TaskQueue port: hand jobs to the worker, record outcomes.

Local adapter = SQLite (ingestion_jobs table doubles as the queue).
Production adapter = Celery + Redis.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from app.shared.domain.models import Job

# Retry backoff, shared by every queue adapter so the schedule can't drift
# between backends. Doubles per attempt and caps, so a permanently-broken
# document stops hammering the pipeline (and the LLM gateway behind it) without
# a transient failure waiting minutes to be retried.
RETRY_BASE_SECONDS = 2
RETRY_MAX_SECONDS = 60


def retry_delay_seconds(attempts: int) -> int:
    """Seconds to hold a job back before its `attempts`-th retry (1-based).

    1 -> 2s, 2 -> 4s, 3 -> 8s, 4 -> 16s, 5 -> 32s, then capped at 60s.
    """
    if attempts < 1:
        return 0
    return min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS ** attempts)


class TaskQueue(ABC):
    """Port for handing ingestion jobs to the worker and recording outcomes."""

    @abstractmethod
    def enqueue(self, job_id: str) -> None:
        """Mark a job ready for processing now (status=queued), clearing any
        retry backoff still in force."""

    @abstractmethod
    def claim_next(self) -> Optional[Job]:
        """Atomically claim the oldest queued job that is due (status->running).
        None if the queue is idle OR everything left is still backing off."""

    @abstractmethod
    def complete(self, job_id: str) -> None:
        """Mark a job done (status=done)."""

    @abstractmethod
    def set_stage(self, job_id: str, stage: str) -> None:
        """Record which pipeline stage a running job is currently on."""

    @abstractmethod
    def retry_or_dead(self, job_id: str, error: str, max_attempts: int) -> str:
        """Requeue with incremented attempts, or dead-letter if exhausted.
        Returns the resulting status ('queued' or 'dead')."""

    @abstractmethod
    def reap_expired(self, max_attempts: int) -> int:
        """Reclaim jobs stuck `running` past their lease (the worker that
        claimed them crashed or was killed before calling complete/retry_or_dead)
        -- requeue with incremented attempts, or dead-letter if exhausted.
        Returns how many jobs were reclaimed."""
