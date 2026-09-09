"""Retrieval-side API routes: hybrid search and grounded answer generation."""
from __future__ import annotations

import logging
from time import perf_counter
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.auth import get_container, get_principal
from app.shared.container import Container
from app.shared.domain.models import Principal
from app.shared.observability import bind
from app.retrieval.rag.access import access_predicate
from app.retrieval.rag.query import answer_query, generate_answer_from_chunks, retrieve_chunks

router = APIRouter()
log = logging.getLogger("api")


class QueryRequest(BaseModel):
    """Body for `POST /query`."""
    question: str
    top_k: int = 10


class AskRequest(BaseModel):
    """Body for `POST /ask` -- the full round trip in one call: cache-check
    -> retrieve+rerank (ONLY on a cache miss) -> generate -> cache-store.
    Unlike chaining `/query` + `/answer`, retrieval and reranking are skipped
    entirely on a cache hit instead of always running before the cache is
    ever checked. Use this endpoint when you want this system's own
    generated (and cached) answer; use `/query` + `/answer` separately to
    bring your own LLM to the retrieved chunks instead -- that path has no
    cache benefit regardless, since the cache only ever holds answers this
    system's own gateway model produced."""
    question: str
    top_k: int = 10
    model: Optional[str] = None


class AnswerRequest(BaseModel):
    """Generate a grounded answer from a set of already-retrieved context
    passages -- typically the exact fields a prior `POST /query` call just
    returned, passed straight through. Kept as a separate call from `/query`
    so a caller can take retrieval's chunks and bring their own LLM instead,
    or chain into this endpoint for generation with the same semantic answer
    cache `answer_query` always had."""
    question: str
    contexts: list[str]
    chunk_ids: list[str] = []
    scores: list[float] = []
    citations: list[dict] = []
    sub_questions: list[str] = []
    model: Optional[str] = None


@router.post("/query")
def query(
    req: QueryRequest,
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container),
) -> dict:
    """Retrieval only -- hybrid dense+BM25 search, ACL-filtered, reranked.
    No LLM call, no answer generated: this returns the chunks so ANY caller
    (our own UI included) can bring their own LLM, or chain into
    `POST /answer` with this response's fields to get one generated with the
    same model/cache this system always used."""
    bind(tenant_id=principal.tenant_id, user_id=principal.user_id)
    container.metrics.incr("query.requests")
    t0 = perf_counter()
    result = retrieve_chunks(
        container, principal.tenant_id, req.question, req.top_k,
        access=access_predicate(principal),
    )
    log.info("query answered", extra={
        "event": "query", "top_k": req.top_k, "hits": len(result.chunk_ids),
        "top_score": round(result.scores[0], 3) if result.scores else None,
        "duration_ms": round((perf_counter() - t0) * 1000, 1),
    })
    return {
        "question": result.question,
        "contexts": result.contexts,
        "chunk_ids": result.chunk_ids,
        "scores": result.scores,
        "sub_questions": result.sub_questions,
        "citations": result.citations,
        "trace": result.trace,
    }


@router.post("/ask")
def ask(
    req: AskRequest,
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container),
) -> dict:
    """Cache-check first, retrieval+rerank only on a miss, then generate --
    the fast path for a caller that just wants this system's own grounded
    answer. See `AskRequest` for how this differs from chaining `/query` +
    `/answer`."""
    bind(tenant_id=principal.tenant_id, user_id=principal.user_id)
    container.metrics.incr("ask.requests")
    t0 = perf_counter()
    result = answer_query(
        container, principal.tenant_id, req.question, req.top_k,
        model=req.model, access=access_predicate(principal),
        user_id=principal.user_id,
    )
    log.info("ask answered", extra={
        "event": "ask", "hits": len(result.chunk_ids), "grounded": result.grounded,
        "answer_len": len(result.answer),
        "duration_ms": round((perf_counter() - t0) * 1000, 1),
    })
    return {
        "question": result.question,
        "answer": result.answer,
        "contexts": result.contexts,
        "chunk_ids": result.chunk_ids,
        "scores": result.scores,
        "sub_questions": result.sub_questions,
        "citations": result.citations,
        "trace": result.trace,
        "grounded": result.grounded,
    }


@router.post("/answer")
def answer(
    req: AnswerRequest,
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container),
) -> dict:
    """Generate a grounded answer from caller-supplied context passages (no
    ACL check here -- the contexts are opaque strings the caller already has,
    typically because `POST /query` already gave them out under ACL; auth is
    just an abuse/cost guard, same as `/query`)."""
    bind(tenant_id=principal.tenant_id, user_id=principal.user_id)
    container.metrics.incr("answer.requests")
    t0 = perf_counter()
    result = generate_answer_from_chunks(
        container, principal.tenant_id, principal.user_id, req.question,
        req.contexts, req.chunk_ids, req.scores, req.citations, req.sub_questions,
        model=req.model,
    )
    log.info("answer generated", extra={
        "event": "answer", "hits": len(result.chunk_ids), "grounded": result.grounded,
        "answer_len": len(result.answer),
        "duration_ms": round((perf_counter() - t0) * 1000, 1),
    })
    return {
        "question": result.question,
        "answer": result.answer,
        "contexts": result.contexts,
        "chunk_ids": result.chunk_ids,
        "scores": result.scores,
        "sub_questions": result.sub_questions,
        "citations": result.citations,
        "trace": result.trace,
        "grounded": result.grounded,
    }
