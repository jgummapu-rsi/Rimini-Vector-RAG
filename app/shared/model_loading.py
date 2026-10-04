from __future__ import annotations

import logging
from contextlib import contextmanager
from threading import Lock

from app.shared.execution import RequestAborted, check_execution, current_budget

log = logging.getLogger(__name__)


@contextmanager
def model_load_lock(lock: Lock):
    while not lock.acquire(timeout=0.05):
        check_execution()
    try:
        check_execution()
        yield
    finally:
        lock.release()


def require_startup_loading() -> None:
    check_execution()
    if current_budget() is not None:
        raise RequestAborted(
            "Model is not initialized; restart the service to warm it before accepting requests",
            503,
        )
