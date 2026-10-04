import asyncio
import threading

from fastapi.testclient import TestClient

from app.api.app import create_app
from app.retrieval.rag.grounding import pack_evidence
from app.shared.gateway.client import LiteLLMClient


def test_answer_deadline_closes_gateway_work_before_releasing_slot(container, tenant, monkeypatch):
    pack_evidence("Amount?", ["Amount 007.00"], container.settings.chat_model)
    container.settings.request_timeout_seconds = 0.3
    finished = threading.Event()

    async def request(*args):
        try:
            await asyncio.sleep(5)
        finally:
            finished.set()

    monkeypatch.setattr(container.gateway, "_request", request)
    monkeypatch.setattr(container.gateway, "chat", LiteLLMClient.chat.__get__(container.gateway))
    container.embedder.embed_query("Amount?")
    with TestClient(create_app(container)) as client:
        response = client.post(
            "/answer",
            json={"question": "Amount?", "contexts": ["Amount 007.00"]},
            headers={"Authorization": "Bearer " + tenant["admin_token"]},
        )
    assert response.status_code == 504
    assert finished.is_set()
    gate = container.request_gate
    assert all(
        gate._redis.zcard(key) == 0 for key in gate._keys(tenant["id"], tenant["admin_id"])[:3]
    )
