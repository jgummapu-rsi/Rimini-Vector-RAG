"""Query decomposition: split a genuinely multi-part question into 2-4 focused
sub-questions so each can be retrieved independently, then merged.
An explicit follow-up instruction may add one extra bounded search.

A single retrieval pass over a combined question ("how does X relate to Y")
dilutes the ranking signal for either sub-topic -- neither X's nor Y's chunks
score as strongly as they would against a focused question about just one of
them. Decomposing first, retrieving each sub-question separately, then merging
the results fixes this without changing anything about retrieval itself.

Cost-conscious by design: a free, rule-based heuristic (`looks_multi_part`)
screens out the common case -- a single, focused question -- so most queries
never pay for an LLM call at all. Only questions with a real surface signal of
being multi-part pay for one cheap decomposition call (same model tier as
metadata extraction), which itself can also decide "actually, this is one
question" -- a safety net against the heuristic's false positives.

Best-effort only, same posture as app.ingest.pipeline.metadata_extract: on any
failure (gateway error, unparseable response), this returns no sub-questions
rather than raising, so a decomposition hiccup never breaks the answer -- it
just falls back to a normal single-pass retrieval.
"""

from __future__ import annotations

import json
import logging
import re

from app.retrieval.rag.prompts import QUERY_DECOMPOSITION_PROMPT

log = logging.getLogger("pipeline")

MAX_SUB_QUESTIONS = 4

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_MULTI_PART_RE = re.compile(
    r"\b(compare[sd]?|versus|vs\.?|difference(?:s)? between|relate[sd]?\s+to|"
    r"relationship\s+between|both\b.*\band\b|as\s+well\s+as)\b",
    re.IGNORECASE,
)


def listed_requests(question: str) -> list[str]:
    """Extract explicit lists, retaining their shared subject and constraints."""
    preamble, separator, body = question.partition(":")
    if not separator:
        return []
    body, *followup = re.split(r"[.!?]\s+(?=[A-Z])", body, maxsplit=1)
    # A second substantive instruction may introduce more subjects than the
    # explicit list. Let the bounded model planner cover the entire question.
    if followup and re.search(r"\b(also|additionally|why|compare)\b", followup[0], re.I):
        return []
    items = re.split(r"\n+|;\s*|,\s+(?:and\s+)?", body)
    items = [re.sub(r"^(?:[-*•]|\d+[.)])\s*|^and\s+", "", s.strip()).strip(" ,;") for s in items]
    items = [s for s in items if s]
    if not 2 <= len(items) <= MAX_SUB_QUESTIONS:
        return []
    return [f"{preamble.strip()}: {item}" for item in items]


def looks_multi_part(question: str) -> bool:
    """True if `question` shows a real surface signal of needing more than one
    independent search to answer well. This is a cheap pre-filter, not a
    verdict -- `decompose_question` makes the actual call and can overrule it."""
    if not question:
        return False
    if question.count("?") >= 2:
        return True
    if listed_requests(question):
        return True
    if re.search(r"\b(for each|each of|which actions|which requirements)\b", question, re.I):
        return True
    return bool(_MULTI_PART_RE.search(question))


def _parse_response(raw: str) -> list[str]:
    """Parse the model's JSON response into a capped list of sub-questions."""
    cleaned = _FENCE_RE.sub("", raw.strip()).strip()
    data = json.loads(cleaned)
    if not isinstance(data, dict) or not data.get("decompose"):
        return []
    subs = data.get("sub_questions")
    if not isinstance(subs, list):
        raise ValueError("sub_questions is not a list")
    cleaned_subs = list(
        dict.fromkeys(s.strip() for s in subs if isinstance(s, str) and 0 < len(s.strip()) <= 4096)
    )
    return cleaned_subs[:MAX_SUB_QUESTIONS] if len(cleaned_subs) >= 2 else []


def decompose_question(gateway, model: str, question: str) -> list[str]:
    """Best-effort decomposition. Returns [] if the question doesn't need it
    (either `looks_multi_part` was a false positive and the model agrees, or
    anything about the call failed) -- callers should treat [] as "retrieve
    this question normally, unchanged"."""
    explicit = listed_requests(question)
    if explicit:
        return explicit
    messages = [
        {"role": "system", "content": QUERY_DECOMPOSITION_PROMPT},
        {"role": "user", "content": question},
    ]
    try:
        raw = gateway.chat(messages, model=model, temperature=0)
        subs = _parse_response(raw)
        # Keep an explicit follow-up instruction searchable on its own. A
        # planner may dilute exclusions by bundling each with approved methods.
        followup = re.search(
            r"\b(?:Also|Additionally)\s+(?:explain|describe|compare)\b.*", question, re.I | re.S
        )
        if subs and followup:
            subs = [followup.group(0).strip(), *subs][: MAX_SUB_QUESTIONS + 1]
        return subs
    except Exception as e:  # noqa: BLE001 - deliberately broad: never fail the query
        log.warning(
            "query decomposition failed, using single-pass retrieval",
            extra={
                "event": "query_decompose_failed",
                "error": str(e)[:200],
            },
        )
        return []
