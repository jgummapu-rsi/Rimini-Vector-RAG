"""LiteLLM gateway client: retry policy.

`_post` raises GatewayError for ANY status >= 400 and then catches its own
exception in the retry loop, so a 401 (bad key) or 400 (bad model name) used to
be resent three times with exponential backoff before failing -- seconds of
delay and triple the log noise for something that could never succeed.

No network: httpx.Client is monkeypatched to a scripted stub, and sleep is
stubbed out so the backoff schedule doesn't slow the suite.
"""
from __future__ import annotations

import httpx
import pytest

from app.shared.gateway.client import GatewayError, LiteLLMClient


class _Response:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or f"status {status_code}"

    def json(self):
        return self._payload


_OK = {"choices": [{"message": {"content": "hello"}}]}


@pytest.fixture
def client(monkeypatch):
    """A client whose HTTP layer is scripted and whose backoff doesn't sleep."""
    monkeypatch.setattr("app.shared.gateway.client.time.sleep", lambda _s: None)
    return LiteLLMClient(base_url="http://gw.test", api_key="k",
                         vision_model="v", max_retries=3)


def _script(monkeypatch, *responses):
    """Make each POST return (or raise) the next scripted item. Returns a list
    that records one entry per attempt actually made."""
    calls: list[int] = []
    queue = list(responses)

    class _Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, *a, **k):
            calls.append(1)
            item = queue.pop(0) if queue else responses[-1]
            if isinstance(item, Exception):
                raise item
            return item

    monkeypatch.setattr(httpx, "Client", _Client)
    return calls


# ----------------------------------------------------------- no retrying --


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


# -------------------------------------------------------------- retrying --


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


# ------------------------------------------------------------- coverage --


def test_a_permanent_error_after_a_transient_one_stops_early(client, monkeypatch):
    """A 503 then a 401: the 401 is permanent, so it must not keep going."""
    calls = _script(monkeypatch, _Response(503), _Response(401, text="bad key"))
    with pytest.raises(GatewayError) as exc:
        client.chat([{"role": "user", "content": "hi"}], model="m")
    assert len(calls) == 2
    assert exc.value.status_code == 401


# ------------------------------------------------------- live config override --
# Backs POST /onboarding/gateway-config: a DB-stored base_url/api_key must win
# over the constructor's env-sourced defaults, re-read on every call, with no
# restart -- see app.shared.container.build_container()'s config_provider wiring.


def test_config_provider_overrides_defaults(monkeypatch):
    monkeypatch.setattr("app.shared.gateway.client.time.sleep", lambda _s: None)
    seen_urls = []

    class _Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, headers, json):
            seen_urls.append((url, headers["Authorization"]))
            return _Response(200, _OK)

    monkeypatch.setattr(httpx, "Client", _Client)
    c = LiteLLMClient(base_url="http://env-default.test", api_key="env-key",
                      vision_model="v", max_retries=3,
                      config_provider=lambda: ("http://db.example", "db-key"))
    out = c.chat([{"role": "user", "content": "hi"}], model="m")
    assert out == "hello"
    assert seen_urls == [("http://db.example/v1/chat/completions", "Bearer db-key")]


def test_config_provider_empty_falls_back_to_defaults(monkeypatch):
    monkeypatch.setattr("app.shared.gateway.client.time.sleep", lambda _s: None)
    seen_urls = []

    class _Client:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def post(self, url, headers, json):
            seen_urls.append((url, headers["Authorization"]))
            return _Response(200, _OK)

    monkeypatch.setattr(httpx, "Client", _Client)
    c = LiteLLMClient(base_url="http://env-default.test", api_key="env-key",
                      vision_model="v", max_retries=3,
                      config_provider=lambda: ("", ""))
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
