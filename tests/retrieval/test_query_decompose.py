"""Query decomposition: the free heuristic, the LLM call's parsing/fallback
behavior, and end-to-end merging of independently-retrieved sub-questions."""

import json

from app.retrieval.rag.access import access_predicate
from app.retrieval.rag.decompose import decompose_question, listed_requests, looks_multi_part
from app.retrieval.rag.query import answer_query
from app.shared.domain.models import Principal, Role
from app.shared.ports.vector_store import VectorPoint
from tests.conftest import structured_answer


def test_looks_multi_part_detects_comparison():
    assert looks_multi_part("How does X relate to Y?")
    assert looks_multi_part("Compare VA01 and ME21N failure handling.")
    assert looks_multi_part("What is the difference between SU53 and STAUTHTRACE?")
    assert looks_multi_part("What is X? And what is Y?")


def test_looks_multi_part_false_for_single_focused_question():
    assert not looks_multi_part("What transaction shows update terminations after COMMIT WORK?")
    assert not looks_multi_part("")
    assert not looks_multi_part("Explain the pricing procedure determination process in detail.")


def test_action_lists_preserve_shared_context_without_a_model_call():
    gateway = _FakeGateway("must not be called")
    for question in [
        "A contractor wants to: export customer records, upload working files to personal storage, "
        "and reuse equipment for another engagement. Which actions violate the policy?",
        "A laboratory technician plans to:\n- freeze the test samples;\n"
        "- transport them without refrigeration;\n- discard the unused reagents. Explain each requirement.",
    ]:
        assert looks_multi_part(question)
        parts = decompose_question(gateway, "test", question)
        assert len(parts) == 3
        assert all(part.startswith(question.partition(":")[0]) for part in parts)
    assert gateway.calls == 0


def test_short_list_items_are_preserved_and_followups_use_full_planning():
    assert (
        len(
            listed_requests(
                "Compare: confidential email, large-file transfer, internal sharing, and external collaboration."
            )
        )
        == 4
    )
    question = "Compare: air, sea, and rail. Also explain why trucks and drones are excluded."
    assert listed_requests(question) == []
    gateway = _FakeGateway(
        '{"decompose":true,"sub_questions":["Compare air sea rail", "Explain trucks and drones"]}'
    )
    parts = decompose_question(gateway, "test", question)
    assert len(parts) == 3
    assert parts[0] == "Also explain why trucks and drones are excluded."
    assert gateway.calls == 1


class _FakeGateway:
    def __init__(self, response: str):
        self._response = response
        self.calls = 0

    def chat(self, messages, model, temperature=0.0):
        self.calls += 1
        return self._response


def test_decompose_question_parses_valid_decomposition():
    gw = _FakeGateway(
        json.dumps(
            {
                "decompose": True,
                "sub_questions": ["What does SU53 do?", "What does STAUTHTRACE do?"],
            }
        )
    )
    subs = decompose_question(gw, "gpt-5-nano", "difference between SU53 and STAUTHTRACE")
    assert subs == ["What does SU53 do?", "What does STAUTHTRACE do?"]
    assert gw.calls == 1


def test_decompose_question_respects_model_saying_no():
    """Safety net: even if the heuristic flagged it, the model can decide this
    is really one focused question, and callers must fall back to single-pass."""
    gw = _FakeGateway(json.dumps({"decompose": False, "sub_questions": []}))
    subs = decompose_question(gw, "gpt-5-nano", "some question")
    assert subs == []


def test_decompose_question_caps_at_four():
    gw = _FakeGateway(
        json.dumps(
            {
                "decompose": True,
                "sub_questions": [f"sub question {i}" for i in range(10)],
            }
        )
    )
    subs = decompose_question(gw, "gpt-5-nano", "some question")
    assert len(subs) == 4


def test_decompose_question_falls_back_gracefully_on_malformed_response():
    gw = _FakeGateway("not valid json at all")
    assert decompose_question(gw, "gpt-5-nano", "some question") == []


def test_decompose_question_falls_back_gracefully_on_gateway_error():
    class _BrokenGateway:
        def chat(self, messages, model, temperature=0.0):
            raise RuntimeError("gateway down")

    assert decompose_question(_BrokenGateway(), "gpt-5-nano", "some question") == []


def _pt(cid, tid, vec, **payload):
    payload.setdefault("_id", "docX")
    payload.setdefault("content", "text")
    return VectorPoint(chunk_id=cid, tenant_id=tid, vector=vec, payload=payload)


def _principal(tid, uid, role):
    return Principal(tenant_id=tid, user_id=uid, role=Role(role))


def _decompose_or_stub_gateway(decomposition_json: str, answer: str):
    """Routes based on which system prompt is sent -- the decomposition call
    uses QUERY_DECOMPOSITION_PROMPT, the generation call uses a different
    system prompt -- so one fake can stand in for both real gateway calls."""
    calls = {"decompose": 0, "generate": 0}

    def chat(messages, model, temperature=0.0):
        system = messages[0]["content"]
        if "You decide whether answering" in system:
            calls["decompose"] += 1
            return decomposition_json
        calls["generate"] += 1
        return structured_answer(answer)

    return chat, calls


def test_answer_query_merges_contexts_from_both_sub_questions(container):
    vs = container.vectors
    v_pricing = container.embedder.embed(["SAP pricing procedure determination condition records"])[
        0
    ]
    v_security = container.embedder.embed(["SAP security roles PFCG authorization objects"])[0]
    vs.upsert(
        [
            _pt(
                "cP001",
                "T1",
                v_pricing,
                _id="docP",
                user_id="A",
                visibility="tenant",
                content="Pricing procedure determination uses condition records.",
            )
        ]
    )
    vs.upsert(
        [
            _pt(
                "cS001",
                "T1",
                v_security,
                _id="docS",
                user_id="A",
                visibility="tenant",
                content="PFCG role design uses authorization objects.",
            )
        ]
    )

    decomposition = json.dumps(
        {
            "decompose": True,
            "sub_questions": [
                "How does pricing procedure determination use condition records?",
                "How does PFCG role design use authorization objects?",
            ],
        }
    )
    chat, calls = _decompose_or_stub_gateway(decomposition, "stub answer")
    container.gateway.chat = chat

    result = answer_query(
        container,
        "T1",
        "How does pricing procedure determination relate to PFCG role design?",
        top_k=5,
        access=access_predicate(_principal("T1", "A", "member")),
    )

    assert result.sub_questions == [
        "How does pricing procedure determination use condition records?",
        "How does PFCG role design use authorization objects?",
    ]
    assert calls["decompose"] == 1
    assert calls["generate"] == 2  # generation plus independent coverage review
    combined = " ".join(result.contexts)
    assert "condition records" in combined
    assert "authorization objects" in combined


def test_answer_query_skips_decomposition_for_simple_question(container):
    vs = container.vectors
    v = container.embedder.embed(["quarterly revenue figures"])[0]
    vs.upsert(
        [
            _pt(
                "cQ001",
                "T1",
                v,
                _id="docQ",
                user_id="A",
                visibility="tenant",
                content="Quarterly revenue grew across all regions.",
            )
        ]
    )

    chat, calls = _decompose_or_stub_gateway("should never be called", "stub answer")
    container.gateway.chat = chat

    result = answer_query(
        container,
        "T1",
        "What was the quarterly revenue?",
        top_k=5,
        access=access_predicate(_principal("T1", "A", "member")),
    )

    assert result.sub_questions == []
    assert calls["decompose"] == 0
    assert calls["generate"] == 1
