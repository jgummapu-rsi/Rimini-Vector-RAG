"""TaskQueue port: hand jobs to the worker, record outcomes."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.shared.domain.models import Job


class StaleLeaseError(RuntimeError):
    pass


RETRY_BASE_SECONDS = 2
RETRY_MAX_SECONDS = 60


def retry_delay_seconds(attempts: int) -> int:
    """Seconds to hold a job back before its `attempts`-th retry (1-based).

    1 -> 2s, 2 -> 4s, 3 -> 8s, 4 -> 16s, 5 -> 32s, then capped at 60s.
    """
    if attempts < 1:
        return 0
    return min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS**attempts)


class TaskQueue(ABC):
    """Port for handing ingestion jobs to the worker and recording outcomes."""

    @abstractmethod
    def enqueue(self, job_id: str) -> None:
        """Mark a job ready for processing now (status=queued), clearing any
        retry backoff still in force."""

    @abstractmethod
    def claim_next(self) -> Job | None:
        """Atomically claim the oldest queued job that is due (status->running).
        None if the queue is idle OR everything left is still backing off."""

    @abstractmethod
    def complete(self, job_id: str, lease_token: str) -> None:
        """Mark a job done (status=done)."""

    @abstractmethod
    def set_stage(self, job_id: str, stage: str, lease_token: str) -> None:
        """Record which pipeline stage a running job is currently on."""

    @abstractmethod
    def retry_or_dead(self, job_id: str, error: str, max_attempts: int, lease_token: str) -> str:
        """Requeue with incremented attempts, or dead-letter if exhausted.
        Returns the resulting status ('queued' or 'dead')."""

    @abstractmethod
    def reap_expired(self, max_attempts: int) -> int:
        """Reclaim jobs stuck `running` past their lease (the worker that
        claimed them crashed or was killed before calling complete/retry_or_dead)
        -- requeue with incremented attempts, or dead-letter if exhausted.
        Returns how many jobs were reclaimed."""

    @abstractmethod
    def renew(self, job_id: str, lease_token: str) -> None:
        raise NotImplementedError
