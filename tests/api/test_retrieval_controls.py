"""Selected-source retrieval must retain ACLs and isolate the default cache."""

import pytest

from tests.api.test_api import _auth, _drain
from tests.api.test_api import client as client
from tests.conftest import structured_answer


@pytest.fixture
def documents(client, container, tenant):
    ids = []
    for filename, text in [
        ("alpha.txt", b"Alpha quarterly revenue is 100 dollars."),
        ("beta.txt", b"Beta quarterly revenue is 200 dollars."),
    ]:
        response = client.post(
            "/ingest", files={"file": (filename, text)}, headers=_auth(tenant["member_token"])
        )
        assert response.status_code == 202
        ids.append(response.json()["document_id"])
    _drain(container)
    container.gateway.chat = lambda *args, **kwargs: structured_answer("Revenue is reported")
    return ids


def test_selection_limits_admin_results_without_granting_viewer_access(client, tenant, documents):
    request = {"question": "quarterly revenue", "document_ids": [documents[1]]}
    result = client.post("/query", json=request, headers=_auth(tenant["admin_token"]))
    assert result.status_code == 200
    assert result.json()["citations"]
    assert {c["document_id"] for c in result.json()["citations"]} == {documents[1]}
    hidden = client.post("/query", json=request, headers=_auth(tenant["viewer_token"]))
    assert hidden.json()["chunk_ids"] == []
    empty = client.post(
        "/query", json={**request, "document_ids": []}, headers=_auth(tenant["admin_token"])
    )
    assert empty.json()["chunk_ids"] == []


@pytest.mark.parametrize(
    "controls",
    [
        {"use_cache": False},
        {"enforce_min_score": False},
        {"rerank_min_score": -10},
        {"selected": True, "use_cache": True},
    ],
)
def test_scoped_tuned_and_cache_disabled_requests_neither_read_nor_write_cache(
    client, container, tenant, documents, monkeypatch, controls
):
    controls = dict(controls)
    if controls.pop("selected", False):
        controls["document_ids"] = [documents[0]]
    calls = []
    monkeypatch.setattr(container.cache, "get", lambda *a, **k: calls.append("get"))
    monkeypatch.setattr(container.cache, "put", lambda *a, **k: calls.append("put"))
    result = client.post(
        "/ask",
        json={"question": "quarterly revenue", **controls},
        headers=_auth(tenant["admin_token"]),
    )
    assert result.status_code == 200
    assert result.json()["answer_status"] == "answered"
    assert calls == []
    if "document_ids" in controls:
        assert {c["document_id"] for c in result.json()["citations"]} == {documents[0]}


def test_scoped_rerank_floor_can_be_explicitly_enabled(client, container, tenant, documents):
    class LowReranker:
        def score(self, question, texts):
            return [-11.4] * len(texts)

    container.reranker = LowReranker()
    container.settings.rerank_min_score = 0
    body = {"question": "quarterly revenue", "document_ids": [documents[0]]}
    headers = _auth(tenant["admin_token"])
    assert client.post("/query", json=body, headers=headers).json()["chunk_ids"]
    assert not client.post(
        "/query", json={**body, "enforce_min_score": True}, headers=headers
    ).json()["chunk_ids"]
    assert not client.post(
        "/query", json={**body, "rerank_min_score": -10}, headers=headers
    ).json()["chunk_ids"]


@pytest.mark.parametrize(
    "controls",
    [
        {"document_ids": ["x" * 10000]},
        {"document_ids": ["a" * 24] * 51},
        {"rerank_min_score": 21},
        {"rerank_min_score": "NaN"},
    ],
)
def test_retrieval_controls_are_bounded(client, tenant, controls):
    response = client.post(
        "/query", json={"question": "revenue", **controls}, headers=_auth(tenant["admin_token"])
    )
    assert response.status_code == 422
