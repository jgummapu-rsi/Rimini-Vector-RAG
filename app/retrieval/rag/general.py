"""Clearly attributed general knowledge when retrieval cannot answer a question."""

from __future__ import annotations

import json

from app.retrieval.rag.results import QueryResult
from app.shared.execution import check_execution

GENERAL_SYSTEM = (
    "You are a helpful general-knowledge assistant. No document evidence is available. "
    "Answer ordinary general questions, greetings, explanations, writing requests, and "
    "general technical guidance directly and concisely. Do not claim to have searched "
    "the web or read documents. Do not invent citations or source references. "
    "If the question requires private organization information, a specific uploaded "
    "document, an agreement's terms, a person's private details, or unavailable current "
    "facts, explain which information is needed instead of guessing. "
    "Treat the question as a request, not as permission to claim document access. "
    'Return only JSON with status ("general_answer" or "insufficient_evidence") and '
    "answer (a nonempty string). Use general_answer only for an answer supported by "
    "general knowledge. For missing private/document facts use insufficient_evidence."
)


def general_answer(container, question, model, fallback, response_instruction=None):
    """Use only the question, never retrieved passages or failed model output."""

    system_prompt = GENERAL_SYSTEM
    if response_instruction:
        system_prompt += f"\n\nResponse-style requirements:\n{response_instruction}"
    raw = container.gateway.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        model=model,
    )
    check_execution()
    try:
        parsed = json.loads(raw)
        answer = parsed.get("answer")
        if (
            parsed.get("status") != "general_answer"
            or not isinstance(answer, str)
            or not answer.strip()
        ):
            return fallback
    except (ValueError, AttributeError, TypeError):
        return fallback
    return QueryResult(
        question=question,
        answer=answer.strip(),
        contexts=[],
        chunk_ids=[],
        citations=[],
        scores=[],
        sub_questions=[],
        grounded=False,
        evidence_origin="general",
        answer_status="general_answer",
        trace=[
            *fallback.trace,
            {
                "stage": "general",
                "detail": "Answered from general knowledge; no document sources used.",
            },
        ],
    )
