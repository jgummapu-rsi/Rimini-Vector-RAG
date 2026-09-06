"""Document-level metadata extraction (author / date / topics / entities).

An auxiliary pipeline step: best-effort only. On any failure — a gateway error,
an unparseable response, an empty document — this returns the blank-default dict
rather than raising, so a metadata-extraction hiccup never fails the ingest job
(the same graceful-degradation posture as query rewrite/decomposition elsewhere).
"""
from __future__ import annotations

import json
import logging
import re

from app.ingest.pipeline.prompts import METADATA_EXTRACTION_PROMPT
from app.ingest.pipeline.tokens import count_tokens

log = logging.getLogger("pipeline")

TEXT_BUDGET_TOKENS = 3000

_BLANK = {"author": None, "date": None, "topics": [], "entities": []}

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _truncate_to_budget(text: str, max_tokens: int) -> str:
    if count_tokens(text) <= max_tokens:
        return text
    # cheap, good-enough truncation: binary-search-free linear shrink by chars,
    # proportional to the token/char ratio already observed.
    ratio = max_tokens / max(1, count_tokens(text))
    cut = max(1, int(len(text) * ratio))
    return text[:cut]


def _parse_response(raw: str) -> dict:
    cleaned = _FENCE_RE.sub("", raw.strip()).strip()
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")

    author = data.get("author")
    date = data.get("date")
    topics = data.get("topics") or []
    entities = data.get("entities") or []
    if not isinstance(topics, list):
        raise ValueError("topics is not a list")
    if not isinstance(entities, list):
        raise ValueError("entities is not a list")

    return {
        "author": str(author) if isinstance(author, str) and author.strip() else None,
        "date": str(date) if isinstance(date, str) and date.strip() else None,
        "topics": [str(t) for t in topics if isinstance(t, (str, int, float))],
        "entities": [str(e) for e in entities if isinstance(e, (str, int, float))],
    }


def extract_metadata(gateway, model: str, text: str) -> dict:
    """Best-effort document metadata extraction. Never raises."""
    if not text or not text.strip():
        return dict(_BLANK)

    sample = _truncate_to_budget(text, TEXT_BUDGET_TOKENS)
    messages = [
        {"role": "system", "content": METADATA_EXTRACTION_PROMPT},
        {"role": "user", "content": sample},
    ]
    try:
        raw = gateway.chat(messages, model=model, temperature=0)
        return _parse_response(raw)
    except Exception as e:  # noqa: BLE001 - deliberately broad: never fail the job
        log.warning("metadata extraction failed, leaving blank", extra={
            "event": "metadata_extract_failed", "error": str(e)[:200],
        })
        return dict(_BLANK)
