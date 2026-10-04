import asyncio
import threading
import time

import httpx
import numpy as np
import pytest

from app.shared.adapters.embedders.minilm import MiniLMEmbedder
from app.shared.adapters.postgres.db import transaction
from app.shared.execution import ExecutionBudget, RequestAborted, execution_scope, run_onnx
from app.shared.gateway.client import GatewayError, LiteLLMClient


def test_dripping_http_response_is_cancelled_at_total_deadline(monkeypatch):
    closed = threading.Event()

    class Drip(httpx.AsyncByteStream):
        async def __aiter__(self):
            while True:
                yield b" "
                await asyncio.sleep(0.01)

        async def aclose(self):
            closed.set()

    transport = httpx.MockTransport(lambda request: httpx.Response(200, stream=Drip()))
    factory = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: factory(transport=transport, **kwargs)
    )
    client = LiteLLMClient("https://gateway.test", "test", "vision", timeout=0.1)
    start = time.monotonic()
    with pytest.raises(GatewayError, match="deadline"):
        client.chat([], model="model")
    assert closed.is_set()
    assert time.monotonic() - start < 1


def test_admission_loss_cancels_inflight_gateway_request(monkeypatch):
    started = threading.Event()
    stopped = threading.Event()
    budget = ExecutionBudget(10)

    async def request(*args):
        started.set()
        try:
            await asyncio.sleep(30)
        finally:
            stopped.set()

    client = LiteLLMClient("https://gateway.test", "test", "vision")
    monkeypatch.setattr(client, "_request", request)

    def cancel():
        assert started.wait(2)
        budget.cancel("Admission lost")

    canceller = threading.Thread(target=cancel)
    canceller.start()
    with execution_scope(budget), pytest.raises(RequestAborted, match="Admission lost"):
        client.chat([], model="model")
    canceller.join(2)
    assert stopped.is_set()


def test_request_deadline_cancels_real_postgres_statement(storage_settings):

    with execution_scope(ExecutionBudget(0.15)), pytest.raises(RequestAborted):
        with transaction(storage_settings.postgres_dsn) as cur:
            cur.execute("SELECT pg_sleep(5)")
    with transaction(storage_settings.postgres_dsn) as cur:
        cur.execute("SELECT 1 AS value")
        assert cur.fetchone()["value"] == 1


def test_expired_budget_prevents_real_onnx_execution():

    embedder = MiniLMEmbedder()
    embedder._ensure_loaded()
    budget = ExecutionBudget(1)
    budget.cancel("Cancelled before inference")
    feeds = {
        "input_ids": np.ones((1, 8), dtype=np.int64),
        "attention_mask": np.ones((1, 8), dtype=np.int64),
    }
    if "token_type_ids" in embedder._input_names:
        feeds["token_type_ids"] = np.zeros((1, 8), dtype=np.int64)
    with execution_scope(budget), pytest.raises(RequestAborted):
        run_onnx(embedder._sess, feeds)
