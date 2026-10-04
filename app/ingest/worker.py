"""Ingestion worker: poll the queue, run jobs, retry/dead-letter on failure.

Run:  python -m app.ingest.worker
Restart-safe: job state lives in the metadata store, so a crash mid-job just
leaves the row claimable/retryable. Emits structured JSON logs and records
metrics.

The loop's contract is that it does not exit. Three things can fail and none of
them may take the process down:
  - the QUEUE itself (a database blip): back off and keep polling.
  - the JOB (a bad document, a gateway outage): retry/dead-letter it.
  - recording that failure (the database again): log and move on, rather than
    dying while handling a death.
"""

from __future__ import annotations

import logging
import signal
import threading
import time

from app.ingest.pipeline.runner import run_job
from app.ingest.pipeline.safety import UnsafeContentError
from app.ingest.ports.task_queue import RETRY_MAX_SECONDS, StaleLeaseError, retry_delay_seconds
from app.shared.container import build_container
from app.shared.observability import configure_logging, log_context

log = logging.getLogger(__name__)


def _queue_error_backoff(consecutive: int) -> float:
    """Sleep before re-polling after the queue itself raised.

    Reuses the job retry schedule so an unreachable database produces the same
    doubling-and-capped pattern as a failing job, instead of a hot loop
    hammering a database that is already unhappy.
    """
    return float(min(RETRY_MAX_SECONDS, retry_delay_seconds(consecutive)))


def _reap_expired(container) -> None:
    """Reclaim jobs stuck `running` past their lease -- a crashed/killed worker
    otherwise leaves them unclaimable forever. Tolerates its own failure (a
    database blip) the same way claim_next does: log and keep polling."""
    try:
        reclaimed = container.queue.reap_expired(container.settings.max_attempts)
        if reclaimed:
            log.warning(
                "reclaimed stuck jobs",
                extra={
                    "event": "jobs_reclaimed",
                    "count": reclaimed,
                },
            )
    except Exception:  # noqa: BLE001 - a reaper failure must not kill the worker
        log.error("reaper failed", extra={"event": "reaper_failed"}, exc_info=True)


def _handle_job_failure(container, job, exc: Exception, max_attempts: int) -> None:
    """Retry or dead-letter a job that raised, tolerating a store that is also
    down. If the store write fails, the job stays `running` -- the reaper
    reclaims it once its lease expires -- but the worker keeps serving every
    other job, which is strictly better than exiting."""
    if isinstance(exc, StaleLeaseError):
        log.warning(
            "Stale worker outcome rejected", extra={"event": "stale_lease", "job_id": job.id}
        )
        return
    try:
        limit = 1 if isinstance(exc, UnsafeContentError) else max_attempts
        status = container.queue.retry_or_dead(job.id, str(exc), limit, job.lease_token)
        container.metrics.incr("jobs.failed")
        if status == "dead":
            container.metrics.incr("jobs.dead")
    except Exception:  # noqa: BLE001 - never die while handling a death
        log.error(
            "could not record job failure; job left running",
            extra={
                "event": "job_failure_unrecorded",
                "error": str(exc)[:500],
            },
            exc_info=True,
        )
        return
    log.error(
        "job failed",
        extra={
            "event": "job_failed",
            "status": status,
            "error": str(exc),
            "retry_in_seconds": retry_delay_seconds(job.attempts + 1)
            if status == "queued"
            else None,
        },
        exc_info=exc,
    )


def main() -> None:
    configure_logging()
    container = build_container()
    stopped = threading.Event()
    previous = {}

    def stop(signum, frame):
        stopped.set()

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous[signum] = signal.signal(signum, stop)
    try:
        _poll(container, stopped)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        container.close()


def _poll(container, stopped) -> None:
    cfg = container.settings
    log.info(
        "worker started",
        extra={
            "event": "worker_start",
            "poll_seconds": cfg.worker_poll_seconds,
            "max_attempts": cfg.max_attempts,
        },
    )

    queue_errors = 0
    while not stopped.is_set():
        _reap_expired(container)

        try:
            job = container.queue.claim_next()
        except Exception:  # noqa: BLE001 - a queue outage must not kill the worker
            queue_errors += 1
            delay = _queue_error_backoff(queue_errors)
            log.error(
                "queue unavailable, retrying",
                extra={
                    "event": "queue_unavailable",
                    "consecutive_errors": queue_errors,
                    "retry_in_seconds": delay,
                },
                exc_info=True,
            )
            time.sleep(delay)
            continue

        if queue_errors:
            log.info(
                "queue recovered", extra={"event": "queue_recovered", "after_errors": queue_errors}
            )
            queue_errors = 0

        if job is None:
            time.sleep(cfg.worker_poll_seconds)
            continue

        with log_context(
            job_id=job.id,
            document_id=job.document_id,
            tenant_id=job.tenant_id,
            attempt=job.attempts,
        ):
            log.info("claimed job", extra={"event": "claim"})
            try:
                run_job(container, job)
            except Exception as e:  # noqa: BLE001 - worker must not die on one bad job
                _handle_job_failure(container, job, e, cfg.max_attempts)


if __name__ == "__main__":
    main()
