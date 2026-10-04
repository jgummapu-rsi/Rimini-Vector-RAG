import json
from unittest.mock import Mock

import pytest

from app.retrieval.rag.grounding import (
    INSUFFICIENT,
    SYSTEM,
    pack_evidence,
    source_quote,
    validate_answer,
    validate_grounded_answer,
)
from app.retrieval.rag.query import generate_answer_from_chunks


def test_empty_evidence_does_not_call_embedding_or_gateway(container, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No model call allowed")

    monkeypatch.setattr(container.gateway, "chat", forbidden)
    monkeypatch.setattr(container.embedder, "embed_query", forbidden)
    result = generate_answer_from_chunks(
        container, "tenant", "user", "Unknown amount?", [], [], [], [], []
    )
    assert result.answer == INSUFFICIENT
    assert not result.grounded and not result.citations


@pytest.mark.parametrize(
    "raw",
    [
        "I cannot find the answer in these passages.",
        '{"status":"insufficient_evidence","answer":"unknown","source_ids":[]}',
        '{"status":"answered","answer":"42 [99]","source_ids":["99"]}',
        '{"status":"answered","answer":"42 [1]","source_ids":["2"]}',
    ],
)
def test_invalid_or_unsupported_generation_abstains(raw):
    assert validate_answer(raw, {"1", "2"}) == (INSUFFICIENT, [])


def test_valid_source_ids_supply_missing_inline_markers():
    raw = '{"status":"answered","answer":"Gokul is an AI Engineer.","source_ids":["1"]}'
    assert validate_answer(raw, {"1", "2"}) == ("Gokul is an AI Engineer. [1]", ["1"])


def test_verbatim_evidence_quote_is_retained_for_region_selection():
    raw = json.dumps(
        {
            "status": "answered",
            "answer": "Gokul is an AI Engineer. [1]",
            "source_ids": ["1"],
            "evidence_quotes": [
                {"source_id": "1", "quote": "AI Engineer specializing in Generative AI"}
            ],
        }
    )
    result = validate_grounded_answer(
        raw, {"1": "Bolanwar Gokul\nAI Engineer specializing in Generative AI"}
    )
    assert result == (
        "Gokul is an AI Engineer. [1]",
        ["1"],
        {"1": "AI Engineer specializing in Generative AI"},
    )


def test_insufficient_response_has_consistent_three_value_result():
    raw = '{"status":"insufficient_evidence","answer":"unknown","source_ids":[]}'
    assert validate_grounded_answer(raw, {"1": "evidence"}) == (INSUFFICIENT, [], {})


def test_wrapped_quotes_return_original_text_without_changing_numbers():
    assert source_quote("Cost is\n1.25 USD.", "Cost is 1.25 USD.") == "Cost is\n1.25 USD."
    assert source_quote("Cost is 125 USD.", "Cost is 1.25 USD.") is None
    assert source_quote("Same phrase. Same phrase.", "Same phrase.") is None


def test_explicit_section_metadata_is_not_replaced_by_short_body_heading():
    packed, _ = pack_evidence(
        "what did they do at Beta",
        ["Alpha\n\nBuilt insurance.", "Beta\n\nBuilt Spark."],
        "gpt-5-nano",
        [
            {"section_path": "Alpha", "source_ordinal": 0},
            {"section_path": "Beta", "source_ordinal": 1},
        ],
    )
    assert len(packed) == 2
    assert [entry["source"]["source_heading"] for entry in packed] == ["Alpha", "Beta"]
    assert [entry["text"] for entry in packed] == ["Built insurance.", "Built Spark."]


def test_only_used_sources_are_returned_and_supplied_evidence_is_labelled(container):
    container.gateway.chat = lambda *args, **kwargs: json.dumps(
        {"status": "answered", "answer": "Amount is 007.00 [2]", "source_ids": ["2"]}
    )
    result = generate_answer_from_chunks(
        container,
        "tenant",
        "user",
        "Amount?",
        ["Unrelated", "Amount 007.00"],
        ["c1", "c2"],
        [1.0, 0.9],
        [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
        [],
    )
    assert [source["chunk_id"] for source in result.citations] == ["c2"]
    assert result.evidence_origin == "supplied" and result.grounded is False


def test_context_budget_deduplicates_and_excludes_oversized_passages():
    packed, selected = pack_evidence(
        "Amount?", ["Amount 007.00", "Amount 007.00", "huge " * 20000], "gpt-5-nano"
    )
    assert selected == [0]
    assert packed == [{"source_id": "1", "text": "Amount 007.00"}]


def test_prompt_uses_relevant_partial_evidence_instead_of_generic_abstention():

    assert "directly relevant fact" in SYSTEM
    assert "Do not infer that an unlisted event never happened" in SYSTEM


def test_malformed_overview_retries_with_same_evidence(container):

    evidence = "# Style Guide\nSol is an accent. Black and white carry the layout."
    container.gateway.chat = Mock(
        side_effect=[
            '{"status":"answered","answer":"Use Sol as an accent [1, 2]","source_ids":["1"]}',
            json.dumps(
                {
                    "status": "answered",
                    "answer": "Use Sol as an accent. [1]",
                    "source_ids": ["1"],
                    "evidence_quotes": [{"source_id": "1", "quote": "Sol is an accent."}],
                }
            ),
        ]
    )
    result = generate_answer_from_chunks(
        container,
        "tenant",
        "user",
        "what does the style guide say",
        [evidence],
        ["c1"],
        [1.0],
        [{"chunk_id": "c1"}],
        [],
    )
    assert result.answer_status == "answered"
    assert result.citations[0]["supporting_quote"] == "Sol is an accent."
    calls = container.gateway.chat.call_args_list
    assert len(calls) == 2
    assert calls[0].args[0][1] == calls[1].args[0][1]
    assert result.trace[-2]["detail"] == "validated on retry"


def test_explicit_abstention_does_not_retry(container):

    container.gateway.chat = Mock(
        return_value=json.dumps(
            {
                "status": "insufficient_evidence",
                "answer": "Unknown",
                "source_ids": [],
            }
        )
    )
    result = generate_answer_from_chunks(
        container,
        "tenant",
        "user",
        "What is the contract amount?",
        ["Style guide: Sol is an accent."],
        ["c1"],
        [1.0],
        [{"chunk_id": "c1"}],
        [],
    )
    assert result.answer_status == "insufficient_evidence"
    assert container.gateway.chat.call_count == 1
    assert any(step.get("detail") == "model_abstained" for step in result.trace)


def test_missing_visual_quote_is_repaired_and_verified_against_source(container):
    container.gateway.chat = Mock(
        side_effect=[
            json.dumps(
                {"status": "answered", "answer": "Built Spark automation [1]", "source_ids": ["1"]}
            ),
            json.dumps(
                {"evidence_quotes": [{"source_id": "1", "quote": "Built Spark automation"}]}
            ),
        ]
    )
    result = generate_answer_from_chunks(
        container,
        "tenant",
        "user",
        "What did they build?",
        ["Built Spark\nautomation"],
        ["c1"],
        [1.0],
        [
            {
                "chunk_id": "c1",
                "provenance": {
                    "regions": [
                        {"page": 1, "text": "Built Spark automation", "precision": "exact_text"}
                    ]
                },
            }
        ],
        [],
    )
    assert result.citations[0]["supporting_quote"] == "Built Spark\nautomation"
    assert result.citations[0]["provenance"]["selection_status"] == "quote"
    assert container.gateway.chat.call_count == 2
