"""LiteLLM gateway client: retry policy.

`_post` raises GatewayError for ANY status >= 400 and then catches its own
exception in the retry loop, so a 401 (bad key) or 400 (bad model name) used to
be resent three times with exponential backoff before failing -- seconds of
delay and triple the log noise for something that could never succeed.

No network: httpx.Client is monkeypatched to a scripted stub, and sleep is
stubbed out so the backoff schedule doesn't slow the suite.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

import app.shared.gateway.client as module
from app.shared.gateway.client import GatewayError, LiteLLMClient


class _Response:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or f"status {status_code}"

    def json(self):
        return self._payload


_OK = {"choices": [{"message": {"content": "hello"}}]}


@pytest.mark.parametrize(
    "model,temperature_supported",
    [
        ("gpt-5-nano", True),
        ("gpt-5.5", False),
        ("gpt-5.6-sol", False),
        ("gpt-6-astra", False),
        ("other-model", True),
    ],
)
def test_gateway_alias_temperature_compatibility(monkeypatch, model, temperature_supported):
    payloads = []
    gateway = LiteLLMClient("https://gateway.test", "test", model)
    monkeypatch.setattr(gateway, "_post", lambda path, payload: payloads.append(payload) or _OK)
    gateway.chat([], model)
    gateway.vision(b"image", "transcribe")

    async def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}
        )

    factory = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kw: factory(transport=httpx.MockTransport(handle), **kw)
    )
    asyncio.run(
        gateway.vision_bounded(
            b"image", "transcribe", "image/png", timeout_seconds=1, max_tokens=32
        )
    )
    assert len(payloads) == 3
    assert all(("temperature" in payload) == temperature_supported for payload in payloads)


@pytest.fixture
def client(monkeypatch):
    """A client whose HTTP layer is scripted and whose backoff doesn't sleep."""
    monkeypatch.setattr("app.shared.gateway.client.time.sleep", lambda _s: None)
    return LiteLLMClient(base_url="http://gw.test", api_key="k", vision_model="v", max_retries=3)


def _script(monkeypatch, *responses):
    """Make each POST return (or raise) the next scripted item. Returns a list
    that records one entry per attempt actually made."""
    calls: list[int] = []
    queue = list(responses)

    async def request(self, *args):
        calls.append(1)
        item = queue.pop(0) if queue else responses[-1]
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(LiteLLMClient, "_request", request)
    return calls


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_client_error_fails_immediately_without_retrying(client, monkeypatch, status):
    calls = _script(monkeypatch, _Response(status, text="nope"))

    with pytest.raises(GatewayError) as exc:
        client.chat([{"role": "user", "content": "hi"}], model="m")

    assert len(calls) == 1, f"{status} must not be retried"
    assert exc.value.status_code == status


def test_permanent_failure_keeps_the_gateways_own_message(client, monkeypatch):
    """The useful diagnostic is the gateway's reply, not 'failed after 3
    attempts' wrapped around it."""
    _script(monkeypatch, _Response(404, text="model 'gpt-5-nano' does not exist"))

    with pytest.raises(GatewayError) as exc:
        client.chat([{"role": "user", "content": "hi"}], model="gpt-5-nano")

    assert "does not exist" in str(exc.value)


@pytest.mark.parametrize("status", [429, 408, 500, 502, 503])
def test_transient_error_is_retried_to_exhaustion(client, monkeypatch, status):
    calls = _script(monkeypatch, _Response(status))

    with pytest.raises(GatewayError):
        client.chat([{"role": "user", "content": "hi"}], model="m")

    assert len(calls) == 3, f"{status} should use all max_retries attempts"


def test_transport_error_is_retried(client, monkeypatch):
    calls = _script(monkeypatch, httpx.ConnectError("connection reset"))
    with pytest.raises(GatewayError):
        client.chat([{"role": "user", "content": "hi"}], model="m")
    assert len(calls) == 3


def test_retry_succeeds_after_a_transient_blip(client, monkeypatch):
    calls = _script(monkeypatch, _Response(503), _Response(200, _OK))
    out = client.chat([{"role": "user", "content": "hi"}], model="m")
    assert out == "hello"
    assert len(calls) == 2, "should stop as soon as it succeeds"


def test_success_on_the_first_try_makes_one_call(client, monkeypatch):
    calls = _script(monkeypatch, _Response(200, _OK))
    assert client.chat([{"role": "user", "content": "hi"}], model="m") == "hello"
    assert len(calls) == 1


def test_gpt5_chat_reserves_reasoning_and_answer_budget(monkeypatch):
    client = LiteLLMClient("https://gateway.test", "test", "vision")
    seen = []

    def post(path, payload):
        seen.append((path, payload))
        return _OK

    monkeypatch.setattr(client, "_post", post)

    assert client.chat([], model="gpt-5-nano") == "hello"
    assert seen[0][1]["max_completion_tokens"] == 4096
    assert seen[0][1]["reasoning_effort"] == "low"


def test_non_gpt5_chat_does_not_send_reasoning_effort(monkeypatch):
    client = LiteLLMClient("https://gateway.test", "test", "vision")
    payloads = []
    monkeypatch.setattr(client, "_post", lambda _path, payload: payloads.append(payload) or _OK)

    assert client.chat([], model="claude-sonnet") == "hello"
    assert "reasoning_effort" not in payloads[0]


def test_a_permanent_error_after_a_transient_one_stops_early(client, monkeypatch):
    """A 503 then a 401: the 401 is permanent, so it must not keep going."""
    calls = _script(monkeypatch, _Response(503), _Response(401, text="bad key"))
    with pytest.raises(GatewayError) as exc:
        client.chat([{"role": "user", "content": "hi"}], model="m")
    assert len(calls) == 2
    assert exc.value.status_code == 401


def test_config_provider_overrides_defaults(monkeypatch):
    monkeypatch.setattr("app.shared.gateway.client.time.sleep", lambda _s: None)
    seen_urls = []

    async def request(self, url, key, payload, timeout):
        seen_urls.append((url, self._headers(key)["Authorization"]))
        return _Response(200, _OK)

    monkeypatch.setattr(LiteLLMClient, "_request", request)
    c = LiteLLMClient(
        base_url="http://env-default.test",
        api_key="env-key",
        vision_model="v",
        max_retries=3,
        config_provider=lambda: ("http://db.example", "db-key"),
    )
    out = c.chat([{"role": "user", "content": "hi"}], model="m")
    assert out == "hello"
    assert seen_urls == [("http://db.example/v1/chat/completions", "Bearer db-key")]


def test_config_provider_empty_falls_back_to_defaults(monkeypatch):
    monkeypatch.setattr("app.shared.gateway.client.time.sleep", lambda _s: None)
    seen_urls = []

    async def request(self, url, key, payload, timeout):
        seen_urls.append((url, self._headers(key)["Authorization"]))
        return _Response(200, _OK)

    monkeypatch.setattr(LiteLLMClient, "_request", request)
    c = LiteLLMClient(
        base_url="http://env-default.test",
        api_key="env-key",
        vision_model="v",
        max_retries=3,
        config_provider=lambda: ("", ""),
    )
    c.chat([{"role": "user", "content": "hi"}], model="m")
    assert seen_urls == [("http://env-default.test/v1/chat/completions", "Bearer env-key")]


def test_vision_and_embed_share_the_retry_policy(client, monkeypatch):
    """All three entry points go through _post, so the policy must not be
    accidentally chat-only."""
    calls = _script(monkeypatch, _Response(401, text="bad key"))
    with pytest.raises(GatewayError):
        client.vision(b"\x89PNG", "transcribe")
    assert len(calls) == 1

    calls2 = _script(monkeypatch, _Response(401, text="bad key"))
    with pytest.raises(GatewayError):
        client.embed(["some text"])
    assert len(calls2) == 1


def test_vision_rejects_truncated_completion(client, monkeypatch):
    payload = {
        "choices": [
            {
                "message": {"content": "partial transcription"},
                "finish_reason": "length",
            }
        ]
    }
    calls = _script(monkeypatch, _Response(200, payload))

    with pytest.raises(GatewayError, match="incomplete"):
        client.vision(b"png", "transcribe")

    assert len(calls) == 1


def test_vision_rejects_refusal(client, monkeypatch):
    payload = {
        "choices": [
            {
                "message": {"content": "", "refusal": "cannot process image"},
                "finish_reason": "stop",
            }
        ]
    }
    _script(monkeypatch, _Response(200, payload))

    with pytest.raises(GatewayError, match="refused"):
        client.vision(b"png", "transcribe")


@pytest.mark.parametrize("payload", [{}, {"choices": []}, {"choices": [{}]}])
def test_vision_rejects_invalid_completion_schema(client, monkeypatch, payload):
    _script(monkeypatch, _Response(200, payload))

    with pytest.raises(GatewayError, match="invalid completion schema"):
        client.vision(b"png", "transcribe")


def test_vision_rejects_non_text_content(client, monkeypatch):
    payload = {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]}
    _script(monkeypatch, _Response(200, payload))

    with pytest.raises(GatewayError, match="not text"):
        client.vision(b"png", "transcribe")


def test_invalid_json_response_uses_gateway_retry_policy(client, monkeypatch):
    class _InvalidJsonResponse(_Response):
        def json(self):
            raise ValueError("bad json")

    calls = _script(monkeypatch, _InvalidJsonResponse(200), _Response(200, _OK))

    assert client.vision(b"png", "transcribe") == "hello"
    assert len(calls) == 2


def test_retry_backoff_cannot_exceed_total_gateway_budget(monkeypatch):

    now = [100.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    calls = []

    async def request(self, url, key, payload, timeout):
        assert timeout <= 2
        calls.append(1)
        now[0] += 1.5
        raise httpx.ConnectError("temporary")

    monkeypatch.setattr(LiteLLMClient, "_request", request)
    client = LiteLLMClient("https://gateway.test", "test", "vision", timeout=2)
    with pytest.raises(GatewayError, match="deadline"):
        client.chat([], model="model")
    assert len(calls) == 1


@pytest.mark.parametrize(
    "choice,detail",
    [
        ({"message": {"content": "partial"}, "finish_reason": "length"}, "finish_reason=length"),
        ({"message": {"content": "", "refusal": "refused"}, "finish_reason": "stop"}, "refused"),
        ({"message": {"content": None}}, "nonempty"),
        ({}, "invalid schema"),
    ],
)
def test_invalid_or_truncated_chat_fails_loudly(monkeypatch, choice, detail):
    client = LiteLLMClient("https://gateway.test", "test", "vision")
    monkeypatch.setattr(client, "_post", lambda *args: {"choices": [choice]})
    with pytest.raises(GatewayError, match=detail):
        client.chat([], model="model")
