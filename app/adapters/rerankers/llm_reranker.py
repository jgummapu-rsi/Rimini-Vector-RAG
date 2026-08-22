"""LLMReranker: listwise re-ranking via the gateway chat model.

Unlike CrossEncoderReranker (a dedicated cross-encoder scoring one (query,
document) pair per forward pass), this adapter sends the WHOLE candidate pool
to the chat model in a single prompt and asks it to score every passage's
relevance in one shot -- one gateway call per query instead of one per
(query, document) pair. This is the "LLM-as-reranker" pattern (cf. RankGPT):
a research/eval lever today (see eval/run_rerank_llm.py) -- there is no
dedicated rerank model on the gateway, so this repurposes the same chat model
used for RAG answer generation.

Parsing is best-effort: the model is asked for a bare JSON array of scores.
If the response can't be parsed, or its length doesn't match the input, this
falls back to all-zero scores -- `_rerank`'s sort is stable, so a fallback
preserves the incoming (fused dense+BM25) order instead of raising.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from app.ports.reranker import Reranker

log = logging.getLogger("llm_reranker")

_PROMPT_TEMPLATE = """You are scoring how relevant each passage is to a search query, for the purpose of ranking search results.

Query: {query}

Passages:
{passages}

Score each passage's relevance to the query on a 0-10 scale (10 = directly \
answers the query, 0 = completely irrelevant). Respond with ONLY a JSON \
array of {n} numbers in passage order, e.g. [7, 2, 9]. No other text, no \
markdown fences."""

_ARRAY_RE = re.compile(r"\[[^\[\]]*\]")


class LLMReranker(Reranker):
    def __init__(self, gateway, model: str):
        self._gateway = gateway
        self._model = model

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        passages = "\n\n".join(f"[{i + 1}] {doc}" for i, doc in enumerate(documents))
        prompt = _PROMPT_TEMPLATE.format(query=query, passages=passages, n=len(documents))
        messages = [{"role": "user", "content": prompt}]
        scores = None
        try:
            reply = self._gateway.chat(messages, model=self._model, temperature=0.0)
            scores = self._parse(reply, len(documents))
        except Exception as e:
            log.warning("llm rerank call failed, falling back to stable order", extra={
                "event": "llm_rerank_failed", "error": str(e)[:200],
            })
        if scores is None:
            log.warning("llm rerank response unparseable, falling back to stable order",
                        extra={"event": "llm_rerank_unparseable"})
            return [0.0] * len(documents)
        return scores

    @staticmethod
    def _parse(reply: str, expected_len: int) -> Optional[list[float]]:
        match = _ARRAY_RE.search(reply)
        if not match:
            return None
        try:
            values = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(values, list) or len(values) != expected_len:
            return None
        try:
            return [float(v) for v in values]
        except (TypeError, ValueError):
            return None
