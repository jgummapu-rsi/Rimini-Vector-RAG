"""RAG query: retrieve from `knowledgebase` and generate a grounded answer.

Query embedding = MiniLM (same as ingestion). Retrieval = tenant-scoped cosine
search over the knowledgebase. Generation = gateway chat model, instructed to
answer ONLY from the retrieved context (so faithfulness is measurable).

Questions that show a real surface signal of being multi-part (comparisons,
"how does X relate to Y", two questions joined into one) are decomposed into
2-4 focused sub-questions (app.rag.decompose), each retrieved independently,
then merged -- a single retrieval pass over a combined question dilutes the
ranking signal for either sub-topic. The final answer is still generated
against the ORIGINAL question, using the merged evidence, so the user's actual
question is what gets answered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from app.container import Container
from app.rag.decompose import decompose_question, looks_multi_part

_SYSTEM = (
    "You are a retrieval-augmented assistant. Answer the question using ONLY the "
    "provided context passages. If the answer is not in the context, say exactly "
    "\"I don't know.\" Be concise and do not invent details."
)
_REFUSAL = "i don't know"

# Fallback system prompt, used ONLY when the grounded pass above found nothing
# relevant (the model said _REFUSAL verbatim) -- e.g. small talk ("Hi") or a
# question genuinely unrelated to any ingested document. Answers from this
# pass are never grounded in retrieved content, so callers must not attach
# citations to them (see the `grounded` flag on QueryResult).
_GENERAL_SYSTEM = (
    "You are a helpful, friendly assistant. The user's question isn't answered "
    "by anything in their knowledge base, so answer directly from your own "
    "general knowledge. Be concise."
)

# Per sub-question, kept smaller than a typical single-pass top_k -- up to
# MAX_SUB_QUESTIONS (4) sub-questions x this many hits each, deduplicated,
# stays a manageable context size for generation without an artificial cap.
_SUB_QUESTION_TOP_K = 3


@dataclass
class QueryResult:
    question: str
    answer: str
    contexts: list[str]
    chunk_ids: list[str]
    scores: list[float] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)  # empty = not decomposed
    # UI-facing additions -- built from data already on each retrieval hit, no
    # extra store lookups. `trace` narrates the control flow actually taken
    # (decompose/rerank are conditional), `citations` is one entry per final hit.
    citations: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    grounded: bool = True  # False when the answer fell back to general knowledge


_SNIPPET_LEN = 300


def _fetch_k(container: Container, requested_k: int) -> int:
    """How many candidates to actually pull from the vector store. Plain
    retrieval fetches exactly what's needed; when a reranker is configured, it
    needs a WIDER pool to have something to re-order before truncating back
    down to requested_k."""
    if container.reranker is None:
        return requested_k
    return max(requested_k * container.settings.rerank_candidate_multiplier,
               container.settings.rerank_min_candidates)


def _retrieve(container: Container, tenant_id: str, question: str, top_k: int,
              access: Optional[Callable[[dict], bool]],
              qvec: Optional[list[float]] = None):
    # qvec may be precomputed by the caller (the answer cache needs the original
    # question's embedding up front); reuse it here to avoid embedding twice.
    if qvec is None:
        qvec = container.embedder.embed([question])[0]
    return container.vectors.search(tenant_id, qvec, top_k=_fetch_k(container, top_k),
                                     access=access, query_text=question)


# QueryResult fields that fully reconstruct a response body from the cache.
_CACHED_FIELDS = ("question", "answer", "contexts", "chunk_ids", "scores",
                  "sub_questions", "citations", "trace", "grounded")


def _result_to_payload(result: "QueryResult") -> dict:
    return {f: getattr(result, f) for f in _CACHED_FIELDS}


def _result_from_cache(payload: dict, similarity: float) -> "QueryResult":
    # Prepend a cache-hit marker so the trace shows this was served from cache,
    # while preserving the original trace of how the answer was first built.
    trace = [{"stage": "cache",
              "detail": f"served from semantic cache (similarity {similarity:.3f})"}]
    trace += payload.get("trace", [])
    return QueryResult(
        question=payload["question"], answer=payload["answer"],
        contexts=payload["contexts"], chunk_ids=payload["chunk_ids"],
        scores=payload["scores"], sub_questions=payload["sub_questions"],
        citations=payload["citations"], trace=trace, grounded=payload["grounded"],
    )


def _retrieve_decomposed(container: Container, tenant_id: str, sub_questions: list[str],
                          access: Optional[Callable[[dict], bool]]):
    """Retrieve each sub-question independently, then merge -- deduplicated by
    chunk_id (a chunk relevant to more than one sub-question keeps its best
    score, appears once), ranked by score across the merged pool."""
    best: dict[str, Any] = {}
    for sq in sub_questions:
        for h in _retrieve(container, tenant_id, sq, _SUB_QUESTION_TOP_K, access):
            prev = best.get(h.chunk_id)
            if prev is None or h.score > prev.score:
                best[h.chunk_id] = h
    return sorted(best.values(), key=lambda h: h.score, reverse=True)


def _rerank(container: Container, question: str, hits: list, top_k: int):
    """Re-score the candidate pool against the ORIGINAL question (not
    sub-questions -- the final answer is generated against the original
    question too) and truncate to top_k. Returns (hits, scores) so callers get
    the scores that actually determined final order, not the stale fused
    dense+BM25 scores from retrieval. No-op when no reranker is configured."""
    if container.reranker is None or not hits:
        return hits, [h.score for h in hits]
    scores = container.reranker.score(question, [h.payload.get("content", "") for h in hits])
    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [hits[i] for i in order], [scores[i] for i in order]


def answer_query(
    container: Container,
    tenant_id: str,
    question: str,
    top_k: int = 5,
    model: Optional[str] = None,
    access: Optional[Callable[[dict], bool]] = None,
    user_id: Optional[str] = None,
) -> QueryResult:
    resolved_model = model or container.settings.chat_model
    trace: list[dict] = []

    # --- semantic answer cache (optional) ---
    # Embed the ORIGINAL question once, up front: the cache is keyed on it, and
    # the non-decomposed retrieval path below reuses the same vector. A hit for
    # this (tenant, user, model) returns the stored answer without retrieval or
    # generation. Per-user scoping means a hit is grounded in evidence this user
    # already saw, so it's safe to return verbatim.
    qvec = container.embedder.embed([question])[0]
    use_cache = container.cache is not None and user_id is not None
    if use_cache:
        hit = container.cache.get(tenant_id, user_id, qvec, resolved_model)
        if hit is not None:
            container.metrics.incr("query.cache_hit")
            return _result_from_cache(hit.payload, hit.similarity)
        container.metrics.incr("query.cache_miss")

    sub_questions: list[str] = []
    if looks_multi_part(question):
        sub_questions = decompose_question(container.gateway, resolved_model, question)
    trace.append({"stage": "decompose", "detail": (
        f"split into {len(sub_questions)} sub-questions" if sub_questions
        else "single-pass retrieval (not multi-part)"
    )})

    if sub_questions:
        hits = _retrieve_decomposed(container, tenant_id, sub_questions, access)
    else:
        hits = _retrieve(container, tenant_id, question, top_k, access, qvec=qvec)
    trace.append({"stage": "retrieve", "detail":
                  f"hybrid dense+BM25 retrieval, {len(hits)} candidates"})

    hits, scores = _rerank(container, question, hits, top_k)
    trace.append({"stage": "rerank", "detail": (
        f"cross-encoder reranked to top {len(hits)}" if container.reranker is not None
        else "no reranker configured, kept fused retrieval order"
    )})

    contexts = [h.payload.get("content", "") for h in hits]
    context_block = "\n\n---\n\n".join(
        f"[{i + 1}] {c}" for i, c in enumerate(contexts)
    )
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user",
         "content": f"Context:\n{context_block}\n\nQuestion: {question}"},
    ]
    answer = container.gateway.chat(messages, model=resolved_model).strip()

    grounded = not (answer.rstrip(".").strip().lower() == _REFUSAL)
    if not grounded:
        # Nothing relevant was retrieved -- fall back to a plain, ungrounded
        # answer rather than a dead-end "I don't know." for ordinary
        # conversation (greetings, general-knowledge questions). No citations
        # are attached below: this answer did NOT come from the retrieved
        # context, so it must never be presented as if it did.
        general_messages = [
            {"role": "system", "content": _GENERAL_SYSTEM},
            {"role": "user", "content": question},
        ]
        answer = container.gateway.chat(general_messages, model=resolved_model).strip()
        trace.append({"stage": "generate", "detail":
                      "nothing relevant found in your documents -- answered from general knowledge"})
    else:
        trace.append({"stage": "generate", "detail": f"answered with {resolved_model}"})

    citations = [{
        "chunk_id": h.chunk_id,
        "document_id": h.payload.get("_id"),
        "filename": h.payload.get("filename"),
        "location": h.payload.get("location"),
        "score": score,
        "snippet": (h.payload.get("content") or "")[:_SNIPPET_LEN],
    } for h, score in zip(hits, scores)] if grounded else []

    result = QueryResult(
        question=question,
        answer=answer,
        contexts=contexts,
        chunk_ids=[h.chunk_id for h in hits],
        scores=scores,
        sub_questions=sub_questions,
        citations=citations,
        trace=trace,
        grounded=grounded,
    )

    # Store for next time (both grounded and ungrounded answers -- per-user
    # scoping makes both safe to replay to the same user).
    if use_cache:
        container.cache.put(tenant_id, user_id, question, qvec,
                            resolved_model, _result_to_payload(result))

    return result
