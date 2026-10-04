from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from app.shared.adapters.request_gate import RedisRequestGate
from app.shared.ports.request_gate import RequestCapacityError


def _gate(settings, **overrides):
    limits = dict(
        global_concurrency=4,
        tenant_concurrency=2,
        user_concurrency=1,
        tenant_per_minute=30,
        user_per_minute=10,
    )
    limits.update(overrides)
    return RedisRequestGate(settings.redis_url, settings.cache_index_name, **limits)


def test_admission_coordinates_independent_instances(storage_settings):
    first = _gate(storage_settings)
    second = _gate(storage_settings)
    barrier = Barrier(2)

    def acquire(gate):
        barrier.wait()
        try:
            return gate.acquire("tenant", "user")
        except RequestCapacityError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(acquire, [first, second]))
    tokens = [token for token in results if token]
    assert len(tokens) == 1
    second.renew("tenant", "user", tokens[0])
    second.release("tenant", "user", tokens[0])
    token = first.acquire("tenant", "user")
    first.release("tenant", "user", token)


def test_admission_tenant_and_global_bounds(storage_settings):
    gate = _gate(storage_settings, global_concurrency=3)
    tokens = [gate.acquire("one", "a"), gate.acquire("one", "b")]
    with pytest.raises(RequestCapacityError):
        gate.acquire("one", "c")
    other = gate.acquire("two", "a")
    with pytest.raises(RequestCapacityError):
        gate.acquire("three", "a")
    gate.release("one", "a", tokens[0])
    gate.release("one", "b", tokens[1])
    gate.release("two", "a", other)


def test_rate_quota_survives_concurrency_release(storage_settings):
    gate = _gate(storage_settings, user_per_minute=2)
    for _ in range(2):
        token = gate.acquire("tenant", "user")
        gate.release("tenant", "user", token)
    with pytest.raises(RequestCapacityError):
        gate.acquire("tenant", "user")
    token = gate.acquire("tenant", "other")
    gate.release("tenant", "other", token)
