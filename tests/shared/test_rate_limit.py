"""app.shared.rate_limit.RateLimiter: fixed-window, per-key, thread-safe."""
import threading

import pytest

from app.shared.rate_limit import RateLimiter


def test_allows_up_to_max_hits_then_rejects():
    rl = RateLimiter(max_hits=3, window_seconds=60)
    assert rl.hit("a") is True
    assert rl.hit("a") is True
    assert rl.hit("a") is True
    assert rl.hit("a") is False
    assert rl.hit("a") is False  # stays rejected, doesn't "use up" further budget


def test_keys_are_independent():
    rl = RateLimiter(max_hits=1, window_seconds=60)
    assert rl.hit("a") is True
    assert rl.hit("b") is True
    assert rl.hit("a") is False
    assert rl.hit("b") is False


def test_window_expiry_lets_a_key_recover(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("app.shared.rate_limit.time.monotonic", lambda: now[0])
    rl = RateLimiter(max_hits=2, window_seconds=10)
    assert rl.hit("a") is True
    assert rl.hit("a") is True
    assert rl.hit("a") is False

    now[0] += 10.001  # first two hits are now outside the trailing window
    assert rl.hit("a") is True


def test_reset_clears_a_keys_history():
    rl = RateLimiter(max_hits=1, window_seconds=60)
    assert rl.hit("a") is True
    assert rl.hit("a") is False
    rl.reset("a")
    assert rl.hit("a") is True


def test_expired_key_is_pruned_from_internal_state(monkeypatch):
    """Not just behavior -- a fully-expired key must not linger in memory
    forever in a long-running process."""
    now = [0.0]
    monkeypatch.setattr("app.shared.rate_limit.time.monotonic", lambda: now[0])
    rl = RateLimiter(max_hits=1, window_seconds=5)
    rl.hit("a")
    assert "a" in rl._hits
    now[0] += 5.001
    rl.hit("b")  # unrelated call, just to advance/trigger cleanup on "a" too
    rl.hit("a")
    assert list(rl._hits["a"]) == [now[0]]  # only the fresh hit remains


@pytest.mark.parametrize("bad_kwargs", [{"max_hits": 0}, {"max_hits": -1}])
def test_rejects_non_positive_max_hits(bad_kwargs):
    with pytest.raises(ValueError):
        RateLimiter(window_seconds=60, **bad_kwargs)


@pytest.mark.parametrize("bad_kwargs", [{"window_seconds": 0}, {"window_seconds": -1}])
def test_rejects_non_positive_window(bad_kwargs):
    with pytest.raises(ValueError):
        RateLimiter(max_hits=1, **bad_kwargs)


def test_thread_safe_under_concurrent_hits():
    """max_hits must be a hard ceiling even under real concurrent callers,
    not just in single-threaded use."""
    rl = RateLimiter(max_hits=50, window_seconds=60)
    allowed = []
    lock = threading.Lock()

    def worker():
        ok = rl.hit("shared-key")
        with lock:
            allowed.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(200)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(allowed) == 50
