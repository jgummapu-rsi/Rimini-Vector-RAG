"""Semantic answer-cache tests (Redis Stack).

These require a running Redis with the RediSearch module (Redis Stack / Redis 8+).
They self-skip when none is reachable, so the default `pytest` run on a machine
without Redis stays green. Point them elsewhere with TEST_REDIS_URL.

Isolation: RediSearch only allows FT.CREATE on DB 0, so tests run there but use a
dedicated index name whose keys are namespaced by that name -- cleanup deletes
only the test index and its own keys, never touching a real `ans_idx` cache.
"""
from __future__ import annotations

import os

import pytest

from app.config import Settings
from app.container import build_container
from app.rag.query import answer_query
from app.ports.vector_store import VectorPoint

REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379")
_INDEX = "test_ans_idx"


def _redis_ready() -> bool:
    try:
        import redis
        r = redis.Redis.from_url(REDIS_URL)
        if not r.ping():
            return False
        modules = {m[b"name"].decode() if isinstance(m[b"name"], bytes) else m["name"]
                   for m in r.module_list()}
        return "search" in modules
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _redis_ready(), reason="Redis Stack (RediSearch module) not reachable")


def _payload(answer: str, question: str = "q") -> dict:
    return {"question": question, "answer": answer, "contexts": [], "chunk_ids": [],
            "scores": [], "sub_questions": [], "citations": [], "trace": [],
            "grounded": True}


def _clean(r) -> None:
    """Remove only this test index and its namespaced keys (keys are prefixed by
    the index name), leaving any real cache on DB 0 untouched."""
    try:
        r.ft(_INDEX).dropindex()
    except Exception:
        pass
    keys = list(r.scan_iter(match=f"{_INDEX}:*"))
    if keys:
        r.delete(*keys)


@pytest.fixture
def cache_container(tmp_path):
    import redis
    r = redis.Redis.from_url(REDIS_URL)
    _clean(r)
    s = Settings(
        data_dir=tmp_path / "data",
        metadata_backend="sqlite", blob_backend="localfs",
        vector_backend="localfile", queue_backend="sqlite",
        embedding_provider="minilm", reranker_provider="none",
        litellm_api_key="",
        redis_url=REDIS_URL, cache_index_name=_INDEX,
        cache_similarity_threshold=0.95,
    )
    c = build_container(s)
    c.gateway.chat = lambda messages, model, temperature=0.0: (
        '{"author": null, "date": null, "topics": [], "entities": []}')
    yield c
    _clean(r)


# --- adapter-level behaviour -------------------------------------------------

def test_exact_question_hits(cache_container):
    c = cache_container
    qvec = c.embedder.embed(["what is the refund policy?"])[0]
    c.cache.put("T1", "U1", "what is the refund policy?", qvec, "m",
                _payload("30 days"))
    hit = c.cache.get("T1", "U1", qvec, "m")
    assert hit is not None
    assert hit.payload["answer"] == "30 days"
    assert hit.similarity >= 0.99


def test_unrelated_question_misses(cache_container):
    c = cache_container
    stored = c.embedder.embed(["what is the refund policy?"])[0]
    c.cache.put("T1", "U1", "what is the refund policy?", stored, "m", _payload("30 days"))
    other = c.embedder.embed(["what is the capital of France?"])[0]
    assert c.cache.get("T1", "U1", other, "m") is None   # below 0.95 threshold


def test_similar_question_hits_below_threshold(cache_container):
    """A reworded question is a semantic hit once the threshold is loosened."""
    from app.adapters.cache.redis_stack import RedisStackAnswerCache
    c = cache_container
    stored = c.embedder.embed(["what is the refund policy?"])[0]
    c.cache.put("T1", "U1", "what is the refund policy?", stored, "m", _payload("30 days"))
    lenient = RedisStackAnswerCache(REDIS_URL, _INDEX, c.embedder.dim, 0.5, 60)
    para = c.embedder.embed(["tell me about your refund policy"])[0]
    hit = lenient.get("T1", "U1", para, "m")
    assert hit is not None and hit.payload["answer"] == "30 days"


def test_per_user_isolation(cache_container):
    c = cache_container
    qvec = c.embedder.embed(["what is the refund policy?"])[0]
    c.cache.put("T1", "U1", "what is the refund policy?", qvec, "m", _payload("A's answer"))
    assert c.cache.get("T1", "U2", qvec, "m") is None      # different user: no hit
    assert c.cache.get("T1", "U1", qvec, "m") is not None


def test_per_model_isolation(cache_container):
    c = cache_container
    qvec = c.embedder.embed(["what is the refund policy?"])[0]
    c.cache.put("T1", "U1", "what is the refund policy?", qvec, "model-a", _payload("x"))
    assert c.cache.get("T1", "U1", qvec, "model-b") is None
    assert c.cache.get("T1", "U1", qvec, "model-a") is not None


def test_invalidate_tenant_drops_hits(cache_container):
    c = cache_container
    qvec = c.embedder.embed(["what is the refund policy?"])[0]
    c.cache.put("T1", "U1", "what is the refund policy?", qvec, "m", _payload("30 days"))
    assert c.cache.get("T1", "U1", qvec, "m") is not None
    c.cache.invalidate_tenant("T1")
    assert c.cache.get("T1", "U1", qvec, "m") is None       # stale generation


# --- integration through answer_query ---------------------------------------

def test_answer_query_second_call_served_from_cache(cache_container):
    c = cache_container
    v = c.embedder.embed(["quarterly revenue grew in APAC"])[0]
    c.vectors.upsert([VectorPoint(chunk_id="docA0001", tenant_id="T1", vector=v,
        payload={"_id": "docA", "user_id": "U1", "visibility": "tenant",
                 "content": "quarterly revenue grew in APAC"})])

    calls = {"n": 0}

    def chat(messages, model, temperature=0.0):
        calls["n"] += 1
        return "Revenue grew in APAC."
    c.gateway.chat = chat

    q = "how did quarterly revenue do?"
    r1 = answer_query(c, "T1", q, user_id="U1")
    assert r1.answer == "Revenue grew in APAC."
    assert calls["n"] == 1

    r2 = answer_query(c, "T1", q, user_id="U1")          # identical -> cache hit
    assert r2.answer == "Revenue grew in APAC."
    assert calls["n"] == 1                                # generation NOT re-run
    assert any(t["stage"] == "cache" for t in r2.trace)

    # a different user must not see U1's cached answer
    r3 = answer_query(c, "T1", q, user_id="U2")
    assert calls["n"] == 2                                # regenerated for U2


def test_answer_query_invalidation_forces_regeneration(cache_container):
    c = cache_container
    v = c.embedder.embed(["quarterly revenue grew in APAC"])[0]
    c.vectors.upsert([VectorPoint(chunk_id="docA0001", tenant_id="T1", vector=v,
        payload={"_id": "docA", "user_id": "U1", "visibility": "tenant",
                 "content": "quarterly revenue grew in APAC"})])
    calls = {"n": 0}

    def chat(messages, model, temperature=0.0):
        calls["n"] += 1
        return "Revenue grew in APAC."
    c.gateway.chat = chat

    q = "how did quarterly revenue do?"
    answer_query(c, "T1", q, user_id="U1")
    assert calls["n"] == 1
    c.cache.invalidate_tenant("T1")                      # KB changed
    answer_query(c, "T1", q, user_id="U1")
    assert calls["n"] == 2                                # regenerated after invalidation
