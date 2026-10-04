import json
from types import SimpleNamespace
from unittest.mock import Mock

from app.retrieval.rag import query
from app.retrieval.rag.general import general_answer
from app.retrieval.rag.grounding import INSUFFICIENT
from app.retrieval.rag.query import QueryResult, answer_query


def fallback():
    return QueryResult(
        question="What is photosynthesis?",
        answer=INSUFFICIENT,
        contexts=[],
        chunk_ids=[],
        grounded=False,
        answer_status="insufficient_evidence",
        evidence_origin="retrieved",
    )


def test_general_answer_is_labeled_and_has_no_document_sources():
    gateway = Mock()
    gateway.chat.return_value = json.dumps(
        {
            "status": "general_answer",
            "answer": "Plants convert light into chemical energy.",
        }
    )
    result = general_answer(
        SimpleNamespace(gateway=gateway), "What is photosynthesis?", "test-model", fallback()
    )
    assert result.evidence_origin == "general"
    assert result.answer_status == "general_answer"
    assert not result.grounded and not result.citations and not result.contexts
    assert result.trace[-1]["stage"] == "general"
    messages = gateway.chat.call_args.args[0]
    assert messages[1]["content"] == "What is photosynthesis?"
    assert "private organization information" in messages[0]["content"]


def test_general_model_abstention_and_invalid_output_retain_original_result():
    gateway = Mock()
    original = fallback()
    for raw in ("not JSON", "[]", '{"status":"insufficient_evidence","answer":"unknown"}'):
        gateway.chat.return_value = raw
        assert (
            general_answer(
                SimpleNamespace(gateway=gateway), "Our contract amount?", "test-model", original
            )
            is original
        )


def test_full_query_falls_back_only_when_enabled_and_unscoped(container, monkeypatch):

    monkeypatch.setattr(container.embedder, "embed_query", lambda question: [0.0])
    monkeypatch.setattr(container.metadata, "corpus_epoch", lambda tenant: "1")
    monkeypatch.setattr(
        query,
        "retrieve_chunks",
        lambda *args, **kwargs: SimpleNamespace(
            contexts=[],
            chunk_ids=[],
            scores=[],
            citations=[],
            sub_questions=[],
            trace=[],
        ),
    )
    monkeypatch.setattr(query, "expand_evidence", lambda c, t, r, a: r)
    container.gateway.chat = Mock(
        return_value=json.dumps(
            {
                "status": "general_answer",
                "answer": "Hello! How can I help?",
            }
        )
    )
    result = answer_query(container, "tenant", "Hello", allow_general_answer=True)
    assert result.answer_status == "general_answer"
    assert container.gateway.chat.call_count == 1
    strict = answer_query(container, "tenant", "Hello", allow_general_answer=False)
    assert strict.answer_status == "insufficient_evidence"
    scoped = answer_query(
        container,
        "tenant",
        "Hello",
        allow_general_answer=True,
        access=SimpleNamespace(document_ids=["document"]),
    )
    assert scoped.answer_status == "insufficient_evidence"
    assert container.gateway.chat.call_count == 1


def test_access_change_does_not_trigger_general_generation(container, monkeypatch):

    monkeypatch.setattr(container.embedder, "embed_query", lambda question: [0.0])
    monkeypatch.setattr(container.metadata, "corpus_epoch", lambda tenant: "1")
    monkeypatch.setattr(
        query,
        "retrieve_chunks",
        lambda *args, **kwargs: SimpleNamespace(
            contexts=[],
            chunk_ids=[],
            scores=[],
            citations=[],
            sub_questions=[],
            trace=[],
        ),
    )
    monkeypatch.setattr(query, "expand_evidence", lambda c, t, r, a: r)
    original = fallback()
    original.trace = [{"stage": "generate", "detail": "source access changed before generation"}]
    monkeypatch.setattr(query, "generate_answer_from_chunks", lambda *args, **kwargs: original)
    container.gateway.chat = Mock(side_effect=AssertionError("Unexpected model call"))
    assert answer_query(container, "tenant", "Hello", allow_general_answer=True) is original
