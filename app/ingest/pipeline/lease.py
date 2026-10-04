from __future__ import annotations

import threading
from contextlib import contextmanager

from app.ingest.ports.task_queue import StaleLeaseError


@contextmanager
def renewed_lease(queue, job, lease_seconds: float):
    if not job.lease_token:
        raise StaleLeaseError("A claimed lease token is required")
    stopped = threading.Event()
    failures = []

    def renew():
        while not stopped.wait(max(0.05, lease_seconds / 3)):
            try:
                queue.renew(job.id, job.lease_token)
            except Exception as exc:
                failures.append(exc)
                return

    def check():
        if failures:
            raise StaleLeaseError("Lease renewal failed; processing stopped") from failures[0]

    queue.renew(job.id, job.lease_token)
    thread = threading.Thread(target=renew, name="ingest-lease-renewal", daemon=True)
    thread.start()
    try:
        yield check
    finally:
        stopped.set()
        thread.join(timeout=5)
