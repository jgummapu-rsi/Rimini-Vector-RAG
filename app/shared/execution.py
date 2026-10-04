from __future__ import annotations

import asyncio
import logging
import math
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar

log = logging.getLogger(__name__)


class RequestAborted(RuntimeError):
    def __init__(self, reason: str, status_code: int):
        super().__init__(reason)
        self.status_code = status_code


class ExecutionBudget:
    def __init__(self, seconds: float):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Execution deadline must be finite and positive")
        self.deadline = time.monotonic() + seconds
        self._cancelled = threading.Event()
        self._reason = "Request admission lease was lost"

    def cancel(self, reason: str) -> None:
        self._reason = reason
        self._cancelled.set()

    def check(self) -> None:
        if self._cancelled.is_set():
            raise RequestAborted(self._reason, 503)
        if time.monotonic() >= self.deadline:
            raise RequestAborted("Request execution deadline exceeded", 504)

    def remaining(self, maximum: float) -> float:
        self.check()
        return min(maximum, self.deadline - time.monotonic())


_budget: ContextVar[ExecutionBudget | None] = ContextVar("execution_budget", default=None)


@contextmanager
def execution_scope(budget: ExecutionBudget):
    token = _budget.set(budget)
    try:
        yield budget
    finally:
        _budget.reset(token)


def current_budget() -> ExecutionBudget | None:
    return _budget.get()


def check_execution() -> None:
    budget = current_budget()
    if budget is not None:
        budget.check()


def remaining_seconds(maximum: float) -> float:
    budget = current_budget()
    return maximum if budget is None else budget.remaining(maximum)


async def cancellable(awaitable):
    budget = current_budget()
    if budget is None:
        return await awaitable
    request = asyncio.ensure_future(awaitable)

    async def monitor():
        while True:
            budget.check()
            await asyncio.sleep(0.05)

    watcher = asyncio.create_task(monitor())
    try:
        done, _ = await asyncio.wait({request, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if watcher in done:
            await watcher
        result = await request
        budget.check()
        return result
    finally:
        request.cancel()
        watcher.cancel()
        await asyncio.gather(request, watcher, return_exceptions=True)


@contextmanager
def cancel_on_budget(abort):
    budget = current_budget()
    if budget is None:
        yield
        return
    budget.check()
    finished = threading.Event()

    def monitor():
        while not finished.wait(0.02):
            try:
                budget.check()
            except RequestAborted:
                abort()
                return

    watcher = threading.Thread(target=monitor, name="request-cancellation", daemon=True)
    watcher.start()
    try:
        yield
        budget.check()
    except Exception:
        budget.check()
        raise
    finally:
        finished.set()
        watcher.join()


def run_onnx(session, feeds):
    # The model adapter initializes the native runtime before reaching this path.
    import onnxruntime as ort  # noqa: PLC0415

    options = ort.RunOptions()
    with cancel_on_budget(lambda: setattr(options, "terminate", True)):
        return session.run(None, feeds, options)
