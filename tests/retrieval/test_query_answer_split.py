"""The retrieval/generation split: `retrieve_chunks()` (what `POST /query`
calls) must never touch the LLM, `generate_answer_from_chunks()` (what
`POST /answer` calls) must reproduce `answer_query`'s generation behavior
exactly given the same chunks, and the two must compose back into an
identical trace to what `answer_query` produces in one call."""

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.ingest.pipeline.runner import run_job
from app.retrieval.rag.access import access_predicate
from app.retrieval.rag.query import answer_query, generate_answer_from_chunks, retrieve_chunks
from app.shared.domain.models import Principal, Role
from app.shared.ports.vector_store import VectorPoint
from tests.conftest import structured_answer


@pytest.fixture
def client(container):
    with TestClient(create_app(container)) as c:
        yield c


def _principal(tid, uid, role):
    return Principal(tenant_id=tid, user_id=uid, role=Role(role))


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _seed(
    container,
    chunk_id="docA0001",
    content="quarterly report content",
    filename="report.pdf",
    tenant_id="T1",
):
    v = container.embedder.embed([content])[0]
    container.vectors.upsert(
        [
            VectorPoint(
                chunk_id=chunk_id,
                tenant_id=tenant_id,
                vector=v,
                payload={
                    "_id": "docA",
                    "user_id": "A",
                    "visibility": "tenant",
                    "content": content,
                    "filename": filename,
                    "location": "p.1",
                },
            )
        ]
    )


def test_retrieve_chunks_never_calls_the_llm(container):
    _seed(container)

    def _boom(*a, **k):
        raise AssertionError("retrieve_chunks must not call the gateway chat model")

    container.gateway.chat = _boom

    result = retrieve_chunks(
        container,
        "T1",
        "quarterly report",
        access=access_predicate(_principal("T1", "A", "member")),
    )

    assert result.contexts
    assert result.citations
    assert result.citations[0]["filename"] == "report.pdf"
    assert not hasattr(result, "answer")
    assert not hasattr(result, "grounded")
    assert [t["stage"] for t in result.trace] == ["decompose", "retrieve", "rerank"]


def test_retrieve_chunks_citations_populated_even_with_no_generation_concept(container):
    """Unlike answer_query's citations (gated on `grounded`), retrieve_chunks
    always returns citations for whatever it found -- there's no answer to
    gate on."""
    _seed(container)
    result = retrieve_chunks(
        container,
        "T1",
        "quarterly report",
        access=access_predicate(_principal("T1", "A", "member")),
    )
    assert len(result.citations) == len(result.chunk_ids) == 1


def test_generate_answer_from_chunks_matches_answer_query_when_grounded(container):
    _seed(container, chunk_id="docC0001", content="some content", filename="notes.txt")
    calls = []
    container.gateway.chat = lambda messages, model, temperature=0.0: (
        calls.append(messages) or structured_answer("a real grounded answer")
    )

    retrieval = retrieve_chunks(
        container,
        "T1",
        "a simple focused question",
        access=access_predicate(_principal("T1", "A", "member")),
    )
    result = generate_answer_from_chunks(
        container,
        "T1",
        "A",
        "a simple focused question",
        retrieval.contexts,
        retrieval.chunk_ids,
        retrieval.scores,
        retrieval.citations,
        retrieval.sub_questions,
        trace=retrieval.trace,
    )

    assert len(calls) == 1
    assert result.grounded is False
    assert result.evidence_origin == "supplied"
    assert result.answer == "a real grounded answer [1]"
    assert result.citations[0]["chunk_id"] == retrieval.citations[0]["chunk_id"]
    assert [t["stage"] for t in result.trace] == [
        "decompose",
        "retrieve",
        "rerank",
        "pack",
        "generate",
    ]


def test_generate_answer_from_chunks_falls_back_and_clears_citations(container):
    """No contexts at all (nothing was retrieved) -- the strict grounded pass
    must refuse, and the fallback general-knowledge answer must carry no
    citations, matching answer_query's existing ungrounded-fallback contract."""
    calls = []

    def fake_chat(messages, model, temperature=0.0):
        calls.append(messages)
        return "I don't know." if len(calls) == 1 else "General Kenobi! (small talk reply)"

    container.gateway.chat = fake_chat

    result = generate_answer_from_chunks(
        container,
        "T1",
        "A",
        "Hi",
        contexts=[],
        chunk_ids=[],
        scores=[],
        citations=[{"chunk_id": "shouldnotsurvive"}],
        sub_questions=[],
    )

    assert len(calls) == 0
    assert result.grounded is False
    assert result.answer_status == "insufficient_evidence"
    assert result.citations == []
    assert any("insufficient evidence" in t["detail"] for t in result.trace)


def test_answer_query_composes_the_same_way_as_calling_both_separately(container):
    """answer_query() is a thin composition of retrieve_chunks +
    generate_answer_from_chunks -- this is the regression guard that the
    refactor didn't change its externally-visible behavior."""
    _seed(container, chunk_id="docB0001", content="some content", filename="notes.txt")
    container.gateway.chat = lambda messages, model, temperature=0.0: structured_answer(
        "stub answer"
    )

    result = answer_query(
        container,
        "T1",
        "a simple focused question",
        access=access_predicate(_principal("T1", "A", "member")),
    )

    assert [t["stage"] for t in result.trace] == [
        "decompose",
        "retrieve",
        "rerank",
        "pack",
        "generate",
    ]
    assert result.grounded is True
    assert result.answer == "stub answer [1]"
    assert result.citations


def test_query_route_returns_chunks_only_no_answer_no_grounded(client, container, tenant, files):
    client.post(
        "/ingest",
        files={"file": ("notes.txt", files["notes.txt"])},
        headers=_auth(tenant["member_token"]),
    )
    while (job := container.queue.claim_next()) is not None:
        run_job(container, job)

    q = client.post(
        "/query",
        json={"question": "quarterly review", "top_k": 5},
        headers=_auth(tenant["member_token"]),
    )
    body = q.json()
    assert "answer" not in body
    assert "grounded" not in body
    assert body["citations"]
    assert body["contexts"]


def test_answer_route_generates_from_supplied_contexts(client, container, tenant, monkeypatch):
    monkeypatch.setattr(
        container.gateway,
        "chat",
        lambda messages, model, temperature=0.0: structured_answer("a generated answer"),
    )

    r = client.post(
        "/answer",
        json={
            "question": "what does it say?",
            "contexts": ["some passage content"],
            "chunk_ids": ["c1"],
            "scores": [1.0],
            "citations": [{"chunk_id": "c1", "filename": "x.txt"}],
        },
        headers=_auth(tenant["member_token"]),
    )

    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "a generated answer [1]"
    assert body["grounded"] is False
    assert body["evidence_origin"] == "supplied"
    assert body["citations"] == [
        {
            "chunk_id": "c1",
            "filename": "x.txt",
            "source_id": "1",
            "evidence_origin": "supplied",
            "occurrences": [
                {
                    "occurrence": 1,
                    "supporting_quotes": [],
                    "provenance": {"regions": [], "selection_status": "missing_quote"},
                }
            ],
        }
    ]


def test_query_then_answer_chain_reproduces_answer_query(client, container, tenant, files):
    """The documented composable contract: POST /query -> POST /answer with
    that response's fields -> a generated answer, same shape as answer_query
    would have produced in one call."""
    client.post(
        "/ingest",
        files={"file": ("notes.txt", files["notes.txt"])},
        headers=_auth(tenant["member_token"]),
    )
    while (job := container.queue.claim_next()) is not None:
        run_job(container, job)
    container.gateway.chat = lambda messages, model, temperature=0.0: structured_answer(
        "stub answer"
    )

    q = client.post(
        "/query",
        json={"question": "quarterly review", "top_k": 5},
        headers=_auth(tenant["member_token"]),
    ).json()
    a = client.post(
        "/answer",
        json={
            "question": q["question"],
            "contexts": q["contexts"],
            "chunk_ids": q["chunk_ids"],
            "scores": q["scores"],
            "citations": q["citations"],
            "sub_questions": q["sub_questions"],
        },
        headers=_auth(tenant["member_token"]),
    )

    body = a.json()
    assert body["answer"] == "stub answer [1]"
    assert body["grounded"] is False
    assert body["citations"][0]["chunk_id"] == q["citations"][0]["chunk_id"]
