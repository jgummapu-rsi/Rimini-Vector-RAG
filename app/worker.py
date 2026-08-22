"""Ingestion worker: poll the queue, run jobs, retry/dead-letter on failure.

Run:  python -m app.worker
Restart-safe: job state lives in SQLite, so a crash mid-job just leaves the row
claimable/retryable. Emits structured JSON logs and records metrics.
"""
from __future__ import annotations

import logging
import time

from app.container import build_container
from app.observability import configure_logging, log_context
from app.pipeline.runner import run_job

log = logging.getLogger("worker")


def main() -> None:
    configure_logging()
    container = build_container()
    cfg = container.settings
    log.info("worker started", extra={
        "event": "worker_start", "poll_seconds": cfg.worker_poll_seconds,
        "max_attempts": cfg.max_attempts,
    })

    while True:
        job = container.queue.claim_next()
        if job is None:
            time.sleep(cfg.worker_poll_seconds)
            continue

        # bind job context so every downstream log line correlates to this job
        with log_context(job_id=job.id, document_id=job.document_id,
                         tenant_id=job.tenant_id, attempt=job.attempts):
            log.info("claimed job", extra={"event": "claim"})
            try:
                run_job(container, job)
            except Exception as e:  # noqa: BLE001 - worker must not die on one bad job
                status = container.queue.retry_or_dead(job.id, str(e), cfg.max_attempts)
                container.metrics.incr("jobs.failed")
                if status == "dead":
                    container.metrics.incr("jobs.dead")
                log.error("job failed", extra={
                    "event": "job_failed", "status": status, "error": str(e),
                }, exc_info=True)


if __name__ == "__main__":
    main()
