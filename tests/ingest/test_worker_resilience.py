"""The worker loop must not exit.

`claim_next()` used to sit OUTSIDE the try/except that guards `run_job`, so a
single transient failure from the queue (database restart, pool exhausted,
network blip) propagated out of main() and ingestion silently stopped until
someone noticed and restarted the process.

These tests drive `main()` directly with a scripted queue and a stop sentinel,
so no worker process or database outage is needed.
"""
from __future__ import annotations

import pytest

import app.ingest.worker as worker
from app.shared.domain.models import Job


class _Stop(BaseException):
    """Breaks out of the worker's infinite loop. Inherits BaseException so the
    loop's own `except Exception` handlers cannot swallow it -- which is also
    what makes these tests meaningful: if the worker caught it, the test would
    hang rather than pass."""


def _job(jid="j1"):
    return Job(id=jid, document_id="d1", tenant_id="t1",
               stage="parse", status="running", attempts=0)


class _ScriptedQueue:
    """Yields a scripted sequence; each item is either a Job, None (idle), or an
    exception to raise. Raises _Stop once the script is exhausted."""

    def __init__(self, *script):
        self.script = list(script)
        self.claims = 0
        self.retried: list[tuple] = []
        self.reap_calls = 0

    def claim_next(self):
        self.claims += 1
        if not self.script:
            raise _Stop()
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def retry_or_dead(self, job_id, error, max_attempts):
        self.retried.append((job_id, error, max_attempts))
        return "queued"

    def reap_expired(self, max_attempts):
        self.reap_calls += 1
        return 0


class _Metrics:
    def __init__(self):
        self.counts = []

    def incr(self, name, count=1, ms=0.0):
        self.counts.append(name)


class _Container:
    def __init__(self, queue, settings):
        self.queue = queue
        self.settings = settings
        self.metrics = _Metrics()


class _Settings:
    worker_poll_seconds = 0.0
    max_attempts = 3


@pytest.fixture
def run_worker(monkeypatch):
    """Run worker.main() against a scripted queue until the script runs out.

    Sleeps are recorded rather than performed, and the recording list is
    returned on the container as `.slept` -- the fixture owns the sleep patch so
    a test cannot install its own and have the fixture clobber it.
    """
    def _run(queue, run_job=None):
        container = _Container(queue, _Settings())
        container.slept = []
        monkeypatch.setattr(worker, "build_container", lambda: container)
        monkeypatch.setattr(worker, "configure_logging", lambda: None)
        monkeypatch.setattr(worker.time, "sleep", container.slept.append)
        monkeypatch.setattr(worker, "run_job", run_job or (lambda c, j: None))
        with pytest.raises(_Stop):
            worker.main()
        return container
    return _run


# ------------------------------------------------------- queue resilience --


def test_worker_survives_a_queue_error_and_keeps_polling(run_worker):
    """The regression: this exception used to end the process."""
    q = _ScriptedQueue(RuntimeError("connection reset by peer"), _job(), None)
    run_worker(q)
    assert q.claims == 4, "kept polling after the error, then hit the sentinel"


def test_worker_survives_repeated_queue_errors(run_worker):
    q = _ScriptedQueue(*[RuntimeError("db down")] * 5)
    run_worker(q)
    assert q.claims == 6


def test_worker_backs_off_progressively_while_the_queue_is_down(run_worker):
    q = _ScriptedQueue(*[RuntimeError("db down")] * 5)
    slept = run_worker(q).slept
    assert slept == sorted(slept), "delay must not decrease while still failing"
    assert slept[0] < slept[-1], "must actually back off, not poll hot"
    assert max(slept) <= worker.RETRY_MAX_SECONDS


def test_backoff_resets_once_the_queue_recovers(run_worker):
    # fail, fail, recover with a job, then fail again
    q = _ScriptedQueue(RuntimeError("x"), RuntimeError("x"), _job(), RuntimeError("x"))
    slept = run_worker(q).slept
    # the delay after recovery starts from the bottom again, not from where it
    # left off -- a recovered queue must not inherit a long backoff
    assert slept[-1] == slept[0]


# --------------------------------------------------------- job resilience --


def test_a_failing_job_is_retried_and_the_worker_continues(run_worker):
    def boom(_c, _j):
        raise RuntimeError("embed failed")

    q = _ScriptedQueue(_job("bad"), _job("good"))
    container = run_worker(q, run_job=boom)
    assert [r[0] for r in q.retried] == ["bad", "good"]
    assert "jobs.failed" in container.metrics.counts


def test_worker_survives_the_failure_handler_itself_failing(run_worker):
    """If the store is down, recording the failure ALSO throws. The worker must
    still not die -- the job is left running for the reaper, and every other
    job keeps being served."""
    class _BrokenQueue(_ScriptedQueue):
        def retry_or_dead(self, job_id, error, max_attempts):
            raise RuntimeError("database is down too")

    def boom(_c, _j):
        raise RuntimeError("stage failed")

    q = _BrokenQueue(_job(), _job())
    run_worker(q, run_job=boom)
    assert q.claims == 3, "kept going through both jobs despite the store failing"


def test_an_idle_queue_just_polls(run_worker):
    q = _ScriptedQueue(None, None, None)
    run_worker(q)
    assert q.claims == 4
    assert q.retried == []


# --------------------------------------------------------- reaper resilience --


def test_worker_calls_the_reaper_every_poll_cycle(run_worker):
    q = _ScriptedQueue(_job(), None, None)
    run_worker(q)
    assert q.reap_calls == q.claims, "reaper must run once per poll iteration"


def test_worker_survives_the_reaper_itself_failing(run_worker):
    """A reaper failure (store down) must not kill the worker or block normal
    job claiming -- it's a best-effort side task, not on the critical path."""
    class _BrokenReaperQueue(_ScriptedQueue):
        def reap_expired(self, max_attempts):
            self.reap_calls += 1
            raise RuntimeError("database is down")

    q = _BrokenReaperQueue(_job(), None)
    run_worker(q)
    assert q.claims == 3
    assert q.reap_calls == 3
