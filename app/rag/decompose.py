"""Query decomposition: split a genuinely multi-part question into 2-4 focused
sub-questions so each can be retrieved independently, then merged.

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

Best-effort only, same posture as app.pipeline.metadata_extract: on any
failure (gateway error, unparseable response), this returns no sub-questions
rather than raising, so a decomposition hiccup never breaks the answer -- it
just falls back to a normal single-pass retrieval.
"""
from __future__ import annotations

import json
import logging
import re

from app.pipeline.prompts import QUERY_DECOMPOSITION_PROMPT

log = logging.getLogger("pipeline")

MAX_SUB_QUESTIONS = 4

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

# Free, rule-based surface signals of a genuinely multi-part question: an
# explicit comparison/relation between two things, or multiple questions
# joined into one. Deliberately simple and cheap (no NLP) -- false positives
# only cost one wasted decomposition call (which can itself say "no"); false
# negatives just mean a question stays single-pass, today's existing behavior.
_MULTI_PART_RE = re.compile(
    r"\b(compare[sd]?|versus|vs\.?|difference(?:s)? between|relate[sd]?\s+to|"
    r"relationship\s+between|both\b.*\band\b|as\s+well\s+as)\b",
    re.IGNORECASE,
)


def looks_multi_part(question: str) -> bool:
    """True if `question` shows a real surface signal of needing more than one
    independent search to answer well. This is a cheap pre-filter, not a
    verdict -- `decompose_question` makes the actual call and can overrule it."""
    if not question:
        return False
    if question.count("?") >= 2:
        return True
    return bool(_MULTI_PART_RE.search(question))


def _parse_response(raw: str) -> list[str]:
    cleaned = _FENCE_RE.sub("", raw.strip()).strip()
    data = json.loads(cleaned)
    if not isinstance(data, dict) or not data.get("decompose"):
        return []
    subs = data.get("sub_questions")
    if not isinstance(subs, list):
        raise ValueError("sub_questions is not a list")
    cleaned_subs = [str(s).strip() for s in subs if isinstance(s, (str, int, float)) and str(s).strip()]
    return cleaned_subs[:MAX_SUB_QUESTIONS]


def decompose_question(gateway, model: str, question: str) -> list[str]:
    """Best-effort decomposition. Returns [] if the question doesn't need it
    (either `looks_multi_part` was a false positive and the model agrees, or
    anything about the call failed) -- callers should treat [] as "retrieve
    this question normally, unchanged"."""
    messages = [
        {"role": "system", "content": QUERY_DECOMPOSITION_PROMPT},
        {"role": "user", "content": question},
    ]
    try:
        raw = gateway.chat(messages, model=model, temperature=0)
        return _parse_response(raw)
    except Exception as e:  # noqa: BLE001 - deliberately broad: never fail the query
        log.warning("query decomposition failed, using single-pass retrieval", extra={
            "event": "query_decompose_failed", "error": str(e)[:200],
        })
        return []
