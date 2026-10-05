"""Retrieval-side API routes: hybrid search and grounded answer generation."""

from __future__ import annotations

import logging
import re
from time import perf_counter

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.api.admission import admit_request
from app.api.auth import get_container, get_principal
from app.retrieval.rag.access import access_predicate
from app.retrieval.rag.query import answer_query, generate_answer_from_chunks, retrieve_chunks
from app.shared.container import Container
from app.shared.domain.models import Principal
from app.shared.observability import bind

router = APIRouter()
log = logging.getLogger("api")


class QueryRequest(BaseModel):
    """Body for `POST /query`."""

    question: str = Field(min_length=1, max_length=4096)
    top_k: int = Field(default=10, ge=1, le=50)
    document_ids: list[str] | None = Field(default=None, max_length=50)
    enforce_min_score: bool | None = None
    rerank_min_score: float | None = Field(default=None, ge=-20, le=20, allow_inf_nan=False)

    @field_validator("document_ids")
    @classmethod
    def valid_document_ids(cls, values):
        if values is not None:
            if any(not re.fullmatch(r"[0-9a-f]{24}", value) for value in values):
                raise ValueError("Document IDs must be 24-character lowercase hexadecimal IDs")
        return values

    @field_validator("question")
    @classmethod
    def nonempty_question(cls, value):
        if not value.strip():
            raise ValueError("Question must not be blank")
        return value


class AskRequest(QueryRequest):
    """Body for `POST /ask` -- the full round trip in one call: cache-check
    -> retrieve+rerank (ONLY on a cache miss) -> generate -> cache-store.
    Unlike chaining `/query` + `/answer`, retrieval and reranking are skipped
    entirely on a cache hit instead of always running before the cache is
    ever checked. Use this endpoint when you want this system's own
    generated (and cached) answer; use `/query` + `/answer` separately to
    bring your own LLM to the retrieved chunks instead -- that path has no
    cache benefit regardless, since the cache only ever holds answers this
    system's own gateway model produced."""

    model: str | None = Field(default=None, max_length=128)
    use_cache: bool | None = None
    allow_general_answer: bool = True
    system_prompt: str | None = Field(default=None, max_length=4096)


class BoundingRegion(BaseModel):
    page: int = Field(ge=1)
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)
    source_element: int | None = None
    precision: str = "exact_text"
    text: str = ""
    layout_id: int | None = None
    layout_label: str | None = None
    layout_confidence: float | None = None


class CitationProvenance(BaseModel):
    schema_version: int = 1
    attribution_level: str = "chunk"
    kind: str = "section"
    status: str = "unavailable"
    pages: list[int] = Field(default_factory=list, max_length=100)
    locator: dict = Field(default_factory=dict)
    regions: list[BoundingRegion] = Field(default_factory=list, max_length=4096)
    regions_truncated: bool = False
    selection_status: str | None = None


class Citation(BaseModel):
    chunk_id: str
    document_id: str | None = None
    generation_id: str | None = None
    version: int | None = None
    source_type: str | None = None
    filename: str | None = None
    location: str | None = None
    section_path: str | None = None
    source_ordinal: int | None = None
    score: float = 0.0
    snippet: str = ""
    provenance: CitationProvenance = Field(default_factory=CitationProvenance)
    source_id: str | None = None
    evidence_origin: str | None = None
    supporting_quote: str | None = None


class AnswerRequest(QueryRequest):
    """Generate a grounded answer from a set of already-retrieved context
    passages -- typically the exact fields a prior `POST /query` call just
    returned, passed straight through. Kept as a separate call from `/query`
    so a caller can take retrieval's chunks and bring their own LLM instead,
    or chain into this endpoint for generation with the same semantic answer
    cache `answer_query` always had."""
    contexts: list[str] = Field(max_length=50)
    chunk_ids: list[str] = Field(default_factory=list, max_length=50)
    scores: list[float] = Field(default_factory=list, max_length=50)
    citations: list[Citation] = Field(default_factory=list, max_length=50)
    sub_questions: list[str] = Field(default_factory=list, max_length=4)
    model: str | None = Field(default=None, max_length=128)

    @field_validator("contexts")
    @classmethod
    def bounded_contexts(cls, values):
        if sum(len(value.encode()) for value in values) > 256000:
            raise ValueError("Supplied contexts exceed the byte budget")
        return values

    system_prompt: str | None = Field(default=None, max_length=4096)


def _allowed_model(container: Container, requested: str | None) -> str:
    model = requested or container.settings.chat_model
    allowed = {container.settings.chat_model, *container.settings.allowed_chat_models}
    if model not in allowed:
        raise HTTPException(400, "Requested generation model is not allowed")
    return model


@router.post("/query", dependencies=[Depends(admit_request)])
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
        container,
        principal.tenant_id,
        req.question,
        req.top_k,
        access=access_predicate(principal, req.document_ids),
        enforce_min_score=req.enforce_min_score,
        rerank_min_score=req.rerank_min_score,
    )
    log.info(
        "query answered",
        extra={
            "event": "query",
            "top_k": req.top_k,
            "hits": len(result.chunk_ids),
            "top_score": round(result.scores[0], 3) if result.scores else None,
            "duration_ms": round((perf_counter() - t0) * 1000, 1),
        },
    )
    return {
        "question": result.question,
        "contexts": result.contexts,
        "chunk_ids": result.chunk_ids,
        "scores": result.scores,
        "sub_questions": result.sub_questions,
        "citations": result.citations,
        "trace": result.trace,
    }


@router.post("/ask", dependencies=[Depends(admit_request)])
def ask(
    req: AskRequest,
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container),
) -> dict:
    """Cache-check first, retrieval+rerank only on a miss, then generate --
    the fast path for a caller that just wants this system's own grounded
    answer. See `AskRequest` for how this differs from chaining `/query` +
    `/answer`."""
    model = _allowed_model(container, req.model)
    bind(tenant_id=principal.tenant_id, user_id=principal.user_id)
    container.metrics.incr("ask.requests")
    t0 = perf_counter()
    result = answer_query(
        container,
        principal.tenant_id,
        req.question,
        req.top_k,
        model=model,
        access=access_predicate(principal, req.document_ids),
        user_id=principal.user_id,
        enforce_min_score=req.enforce_min_score,
        rerank_min_score=req.rerank_min_score,
        use_cache_override=req.use_cache,
        allow_general_answer=req.allow_general_answer,
        response_instruction=req.system_prompt,
    )
    log.info(
        "ask answered",
        extra={
            "event": "ask",
            "hits": len(result.chunk_ids),
            "grounded": result.grounded,
            "answer_status": result.answer_status,
            "evidence_origin": result.evidence_origin,
            "answer_len": len(result.answer),
            "duration_ms": round((perf_counter() - t0) * 1000, 1),
        },
    )
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
        "evidence_origin": result.evidence_origin,
        "answer_status": result.answer_status,
    }


@router.post("/answer", dependencies=[Depends(admit_request)])
def answer(
    req: AnswerRequest,
    principal: Principal = Depends(get_principal),
    container: Container = Depends(get_container),
) -> dict:
    """Generate a grounded answer from caller-supplied context passages (no
    ACL check here -- the contexts are opaque strings the caller already has,
    typically because `POST /query` already gave them out under ACL; auth is
    just an abuse/cost guard, same as `/query`)."""
    model = _allowed_model(container, req.model)
    bind(tenant_id=principal.tenant_id, user_id=principal.user_id)
    container.metrics.incr("answer.requests")
    t0 = perf_counter()
    result = generate_answer_from_chunks(
        container,
        principal.tenant_id,
        principal.user_id,
        req.question,
        req.contexts,
        req.chunk_ids,
        req.scores,
        [
            citation.model_dump(exclude_none=True, exclude_defaults=True)
            for citation in req.citations
        ],
        req.sub_questions,
        model=model,
        response_instruction=req.system_prompt,
        cache_allowed=req.system_prompt is None,
    )
    log.info(
        "answer generated",
        extra={
            "event": "answer",
            "hits": len(result.chunk_ids),
            "grounded": result.grounded,
            "answer_len": len(result.answer),
            "duration_ms": round((perf_counter() - t0) * 1000, 1),
        },
    )
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
        "evidence_origin": result.evidence_origin,
        "answer_status": result.answer_status,
    }
