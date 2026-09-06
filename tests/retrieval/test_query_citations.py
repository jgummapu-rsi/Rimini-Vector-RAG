"""answer_query()'s UI-facing additions: `citations` (built straight off each
retrieval hit's payload -- filename/location/document_id, no extra store lookup)
and `trace` (narrates the control flow actually taken: decompose/retrieve/
rerank/generate)."""
from app.shared.domain.models import Principal, Role
from app.shared.ports.vector_store import VectorPoint
from app.retrieval.rag.access import access_predicate
from app.retrieval.rag.query import answer_query


def _principal(tid, uid, role):
    return Principal(tenant_id=tid, user_id=uid, role=Role(role))


def test_citations_carry_filename_and_location_from_payload(container):
    vs = container.vectors
    v = container.embedder.embed(["quarterly report content"])[0]
    vs.upsert([VectorPoint(
        chunk_id="docA0001", tenant_id="T1", vector=v,
        payload={"_id": "docA", "user_id": "A", "visibility": "tenant",
                 "content": "quarterly report content", "filename": "report.pdf",
                 "location": "p.4"},
    )])
    container.gateway.chat = lambda messages, model, temperature=0.0: "stub answer"

    result = answer_query(
        container, "T1", "quarterly report",
        access=access_predicate(_principal("T1", "A", "member")),
    )

    assert result.citations
    c = result.citations[0]
    assert c["chunk_id"] == "docA0001"
    assert c["document_id"] == "docA"
    assert c["filename"] == "report.pdf"
    assert c["location"] == "p.4"
    assert c["snippet"].startswith("quarterly report content")
    assert isinstance(c["score"], float)


def test_falls_back_to_general_knowledge_when_nothing_relevant_retrieved(container):
    """No documents ingested at all -- retrieval finds nothing, the strict
    grounded pass must refuse ("I don't know."), and the system should then
    fall back to a plain, ungrounded answer rather than dead-ending. Citations
    must stay empty since nothing was actually retrieved to cite."""
    calls = []

    def fake_chat(messages, model, temperature=0.0):
        calls.append(messages)
        if len(calls) == 1:
            return "I don't know."
        return "General Kenobi! (small talk reply)"

    container.gateway.chat = fake_chat

    result = answer_query(
        container, "T1", "Hi",
        access=access_predicate(_principal("T1", "A", "member")),
    )

    assert len(calls) == 2
    assert result.grounded is False
    assert result.answer == "General Kenobi! (small talk reply)"
    assert result.citations == []
    assert any("general knowledge" in t["detail"] for t in result.trace)


def test_grounded_answer_keeps_citations_and_single_chat_call(container):
    vs = container.vectors
    v = container.embedder.embed(["some content"])[0]
    vs.upsert([VectorPoint(
        chunk_id="docC0001", tenant_id="T1", vector=v,
        payload={"_id": "docC", "user_id": "A", "visibility": "tenant",
                 "content": "some content", "filename": "notes.txt", "location": ""},
    )])
    calls = []
    container.gateway.chat = lambda messages, model, temperature=0.0: (
        calls.append(messages) or "a real grounded answer"
    )

    result = answer_query(
        container, "T1", "a simple focused question",
        access=access_predicate(_principal("T1", "A", "member")),
    )

    assert len(calls) == 1  # no fallback call made
    assert result.grounded is True
    assert result.citations


def test_trace_narrates_actual_control_flow(container):
    vs = container.vectors
    v = container.embedder.embed(["some content"])[0]
    vs.upsert([VectorPoint(
        chunk_id="docB0001", tenant_id="T1", vector=v,
        payload={"_id": "docB", "user_id": "A", "visibility": "tenant",
                 "content": "some content", "filename": "notes.txt", "location": ""},
    )])
    container.gateway.chat = lambda messages, model, temperature=0.0: "stub answer"

    result = answer_query(
        container, "T1", "a simple focused question",
        access=access_predicate(_principal("T1", "A", "member")),
    )

    stages = [t["stage"] for t in result.trace]
    assert stages == ["decompose", "retrieve", "rerank", "generate"]
    # this repo's default test container has no reranker configured
    assert container.reranker is None
    assert "no reranker" in result.trace[2]["detail"]
    assert "single-pass" in result.trace[0]["detail"]
