"""In-process fixed-window rate limiter, for abuse-guarding sensitive
unauthenticated endpoints (onboarding login/register) that have no other
throttle in front of them yet.

This is deliberately simple and in-memory: it resets on process restart and
does not coordinate across multiple API processes/replicas. That matches this
deployment's current single-API-process assumption (see the pool-sizing note
in app.shared.adapters.postgres.db) -- once horizontal API scaling is real
(CLAUDE.md roadmap item 13), this should move to a shared store (e.g. Redis)
so every replica enforces the same limit, or the limit should move to the
ingress/API-gateway layer instead (roadmap item 5).
"""
from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    """Fixed-window limiter: at most `max_hits` calls to `hit(key)` within any
    trailing `window_seconds`, per distinct `key`. Thread-safe (the API server
    is multi-threaded under uvicorn's default worker)."""

    def __init__(self, max_hits: int, window_seconds: float):
        if max_hits < 1:
            raise ValueError("max_hits must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self._max_hits = max_hits
        self._window = window_seconds
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> bool:
        """Record one attempt for `key`. Returns True if it is within the
        limit (the caller may proceed), False if `key` is over the limit
        within the current window (the caller should reject with 429).

        Every call to `hit` -- successful or not -- counts against the
        window; this bounds request VOLUME (the actual abuse surface: a
        script trying thousands of passwords or spinning up thousands of
        accounts), not just failures, so a distributed attacker who
        occasionally succeeds can't reset their own budget.
        """
        now = time.monotonic()
        with self._lock:
            dq = self._hits.get(key)
            if dq is not None:
                while dq and now - dq[0] > self._window:
                    dq.popleft()
                if not dq:
                    # Fully expired -- drop the entry instead of leaving an
                    # empty deque behind, so a long-running process doesn't
                    # accumulate one dict entry per distinct key ever seen
                    # (email/IP) for as long as it stays up.
                    del self._hits[key]
                    dq = None
            if dq is None:
                dq = deque()
                self._hits[key] = dq
            if len(dq) >= self._max_hits:
                return False
            dq.append(now)
            return True

    def reset(self, key: str) -> None:
        """Forget `key`'s history (e.g. after a legitimate success), so a
        genuine user who mistyped a password a few times isn't stuck waiting
        out the rest of the window once they do get it right."""
        with self._lock:
            self._hits.pop(key, None)
