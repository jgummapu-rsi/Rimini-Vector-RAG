import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.body_limit import BodyLimitMiddleware
from app.shared.gateway.client import GatewayError


def test_invalid_query_bounds_reject_before_retrieval(container, tenant):
    headers = {"Authorization": "Bearer " + tenant["admin_token"]}
    with TestClient(create_app(container)) as client:
        for body in (
            {"question": "q", "top_k": 0},
            {"question": "q", "top_k": 51},
            {"question": " "},
            {"question": "x" * 4097},
        ):
            assert client.post("/query", headers=headers, json=body).status_code == 422
        assert (
            client.post(
                "/answer", headers=headers, json={"question": "q", "contexts": ["x" * 256001]}
            ).status_code
            == 422
        )


def test_chunked_body_is_capped_without_content_length(container):
    container.settings.max_upload_mb = 1
    consumed = []
    sent = []
    messages = iter(
        [
            {"type": "http.request", "body": b"x" * 600000, "more_body": True},
            {"type": "http.request", "body": b"x" * 600000, "more_body": True},
            {"type": "http.request", "body": b"secret", "more_body": False},
        ]
    )

    async def receive():
        message = next(messages)
        consumed.append(len(message["body"]))
        return message

    async def send(message):
        sent.append(message)

    async def application(scope, receive, send):
        while (await receive())["more_body"]:
            pass

    scope = {"type": "http", "app": SimpleNamespace(state=SimpleNamespace(container=container))}
    asyncio.run(BodyLimitMiddleware(application)(scope, receive, send))
    assert consumed == [600000, 600000]
    assert sent[0]["status"] == 413


def test_unapproved_model_rejected_before_gateway(container, tenant, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Gateway must not be called")

    monkeypatch.setattr(container.gateway, "chat", forbidden)
    # Model discovery now augments the static allow-list; keep this boundary
    # test offline while still exercising rejection against a real catalog shape.
    monkeypatch.setattr(
        container.model_catalog,
        "get",
        lambda: {"chat_models": [{"id": container.settings.chat_model}]},
    )
    headers = {"Authorization": "Bearer " + tenant["admin_token"]}
    with TestClient(create_app(container)) as client:
        for route in ("/ask", "/answer"):
            body = {"question": "q", "model": "unapproved-expensive-model"}
            if route == "/answer":
                body["contexts"] = ["content"]
            assert client.post(route, headers=headers, json=body).status_code == 400


def test_unknown_model_fails_closed_when_catalog_is_unavailable(container, tenant, monkeypatch):
    def unavailable():
        raise GatewayError("Model catalog unavailable")

    def forbidden(*args, **kwargs):
        raise AssertionError("Generation must not run without model authorization")

    monkeypatch.setattr(container.model_catalog, "get", unavailable)
    monkeypatch.setattr(container.gateway, "chat", forbidden)
    headers = {"Authorization": "Bearer " + tenant["admin_token"]}
    with TestClient(create_app(container)) as client:
        for route in ("/ask", "/answer"):
            body = {"question": "q", "model": "unknown-model"}
            if route == "/answer":
                body["contexts"] = ["content"]
            assert client.post(route, headers=headers, json=body).status_code == 503


def test_quota_rejected_before_retrieval(container, tenant):
    container.request_gate._limits = (32, 8, 2, 300, 1)
    headers = {"Authorization": "Bearer " + tenant["admin_token"]}
    with TestClient(create_app(container)) as client:
        assert client.post("/query", headers=headers, json={"question": "first"}).status_code == 200
        response = client.post("/query", headers=headers, json={"question": "second"})
        assert response.status_code == 429 and response.headers["Retry-After"] == "60"
