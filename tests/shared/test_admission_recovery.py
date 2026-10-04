from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.shared.adapters.request_gate import RedisRequestGate
from app.shared.ports.request_gate import RequestCapacityError


def test_expired_holder_cannot_renew_or_release_replacement_slot(storage_settings):
    gate = RedisRequestGate(
        storage_settings.redis_url,
        storage_settings.cache_index_name,
        global_concurrency=1,
        tenant_concurrency=1,
        user_concurrency=1,
        tenant_per_minute=100,
        user_per_minute=100,
    )
    try:
        old = gate.acquire("tenant", "user")
        for key in gate._keys("tenant", "user")[:3]:
            gate._redis.zadd(key, {old: 0})
        replacement = gate.acquire("tenant", "user")
        with pytest.raises(RequestCapacityError, match="expired"):
            gate.renew("tenant", "user", old)
        gate.release("tenant", "user", old)
        with pytest.raises(RequestCapacityError):
            gate.acquire("tenant", "user")
        gate.renew("tenant", "user", replacement)
        gate.release("tenant", "user", replacement)
    finally:
        gate.close()


def test_concurrent_admission_load_never_exceeds_global_capacity(storage_settings):
    gates = [
        RedisRequestGate(
            storage_settings.redis_url,
            storage_settings.cache_index_name,
            global_concurrency=4,
            tenant_concurrency=4,
            user_concurrency=1,
            tenant_per_minute=100,
            user_per_minute=100,
        )
        for _ in range(2)
    ]
    ready = Barrier(16)
    decisions = Barrier(16)

    def request(index):
        gate = gates[index % len(gates)]
        ready.wait(timeout=10)
        token = None
        try:
            token = gate.acquire("tenant", str(index))
        except RequestCapacityError:
            pass
        decisions.wait(timeout=10)
        if token is not None:
            gate.release("tenant", str(index), token)
        return token is not None

    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            admitted = list(pool.map(request, range(16)))
        assert sum(admitted) == 4
        assert gates[0]._redis.zcard(gates[0]._keys("tenant", "0")[0]) == 0
    finally:
        for gate in gates:
            gate.close()


def test_admission_connection_failure_rejects_then_recovers(container, tenant, monkeypatch):
    gate = container.request_gate
    original = gate.acquire
    reached = Event()

    def unavailable(*args, **kwargs):
        raise ConnectionError("admission connection unavailable")

    def retrieval(*args, **kwargs):
        reached.set()
        raise AssertionError("Rejected request must not retrieve")

    headers = {"Authorization": "Bearer " + tenant["admin_token"]}
    with TestClient(create_app(container)) as client:
        with monkeypatch.context() as patch:
            patch.setattr(gate, "acquire", unavailable)
            patch.setattr(container.vectors, "search", retrieval)
            assert (
                client.post("/query", headers=headers, json={"question": "Invoice?"}).status_code
                == 503
            )
            assert not reached.is_set()
        assert gate.acquire == original
        assert (
            client.post("/query", headers=headers, json={"question": "Invoice?"}).status_code == 200
        )
