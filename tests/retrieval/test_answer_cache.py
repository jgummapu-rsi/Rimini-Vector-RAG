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

from app.shared.config import Settings
from app.shared.container import build_container
from app.retrieval.rag.query import answer_query, generate_answer_from_chunks
from app.shared.ports.vector_store import VectorPoint

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
    from app.retrieval.adapters.cache.redis_stack import RedisStackAnswerCache
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


def test_invalidate_tenant_leaves_other_tenants_alone(cache_container):
    c = cache_container
    qvec = c.embedder.embed(["what is the refund policy?"])[0]
    c.cache.put("T1", "U1", "q", qvec, "m", _payload("T1 answer"))
    c.cache.put("T2", "U1", "q", qvec, "m", _payload("T2 answer"))
    c.cache.invalidate_tenant("T1")
    assert c.cache.get("T1", "U1", qvec, "m") is None
    assert c.cache.get("T2", "U1", qvec, "m") is not None    # untouched


# A scope=global document is readable from EVERY tenant, so publishing or
# deleting one changes what is retrievable for all of them. Bumping only the
# owning tenant left every other tenant answering from a knowledge base that no
# longer existed, until their TTL expired (24h by default).


def test_invalidate_all_drops_every_tenant(cache_container):
    c = cache_container
    qvec = c.embedder.embed(["what is the refund policy?"])[0]
    for tid in ("T1", "T2", "T3"):
        c.cache.put(tid, "U1", "q", qvec, "m", _payload(f"{tid} answer"))
    assert all(c.cache.get(t, "U1", qvec, "m") is not None for t in ("T1", "T2", "T3"))

    c.cache.invalidate_all()
    assert all(c.cache.get(t, "U1", qvec, "m") is None for t in ("T1", "T2", "T3"))


def test_cache_still_usable_after_a_global_invalidation(cache_container):
    """invalidate_all must advance the generation, not poison the cache: new
    answers written afterwards have to be servable."""
    c = cache_container
    qvec = c.embedder.embed(["what is the refund policy?"])[0]
    c.cache.put("T1", "U1", "q", qvec, "m", _payload("old"))
    c.cache.invalidate_all()
    c.cache.put("T1", "U1", "q", qvec, "m", _payload("new"))
    hit = c.cache.get("T1", "U1", qvec, "m")
    assert hit is not None and hit.payload["answer"] == "new"


def test_tenant_and_global_generations_compose(cache_container):
    """Both counters feed one effective generation. Interleaving them must never
    let a superseded entry become valid again."""
    c = cache_container
    qvec = c.embedder.embed(["what is the refund policy?"])[0]
    c.cache.put("T1", "U1", "q", qvec, "m", _payload("v1"))

    c.cache.invalidate_tenant("T1")
    assert c.cache.get("T1", "U1", qvec, "m") is None
    c.cache.invalidate_all()
    assert c.cache.get("T1", "U1", qvec, "m") is None, "still stale, not resurrected"

    c.cache.put("T1", "U1", "q", qvec, "m", _payload("v2"))
    assert c.cache.get("T1", "U1", qvec, "m").payload["answer"] == "v2"


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


def test_generate_answer_from_chunks_second_call_served_from_cache(cache_container):
    """Same cache behavior as answer_query, but reached the way `POST /answer`
    reaches it -- chunks supplied directly rather than retrieved inline."""
    c = cache_container
    calls = {"n": 0}

    def chat(messages, model, temperature=0.0):
        calls["n"] += 1
        return "Revenue grew in APAC."
    c.gateway.chat = chat

    q = "how did quarterly revenue do?"
    contexts = ["quarterly revenue grew in APAC"]
    r1 = generate_answer_from_chunks(c, "T1", "U1", q, contexts, ["docA0001"], [1.0], [], [])
    assert r1.answer == "Revenue grew in APAC."
    assert calls["n"] == 1

    r2 = generate_answer_from_chunks(c, "T1", "U1", q, contexts, ["docA0001"], [1.0], [], [])
    assert r2.answer == "Revenue grew in APAC."
    assert calls["n"] == 1  # generation NOT re-run
    assert any(t["stage"] == "cache" for t in r2.trace)


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


def test_publishing_a_global_document_invalidates_a_different_tenant(cache_container):
    """End-to-end version of the global-scope staleness bug: T2 caches an
    answer, the platform tenant publishes a global document, and T2's stale
    answer must not be served back."""
    from app.ingest.pipeline.runner import invalidate_cache_for

    c = cache_container
    v = c.embedder.embed(["quarterly revenue grew in APAC"])[0]
    c.vectors.upsert([VectorPoint(chunk_id="docA0001", tenant_id="T2", vector=v,
        payload={"_id": "docA", "user_id": "U1", "visibility": "tenant",
                 "content": "quarterly revenue grew in APAC"})])
    calls = {"n": 0}

    def chat(messages, model, temperature=0.0):
        calls["n"] += 1
        return "Revenue grew in APAC."
    c.gateway.chat = chat

    q = "how did quarterly revenue do?"
    answer_query(c, "T2", q, user_id="U1")
    assert calls["n"] == 1
    answer_query(c, "T2", q, user_id="U1")
    assert calls["n"] == 1, "second identical ask is served from cache"

    # T1 publishes a GLOBAL document -- readable by T2 as well
    invalidate_cache_for(c, "T1", "global")

    answer_query(c, "T2", q, user_id="U1")
    assert calls["n"] == 2, "T2's cached answer must not survive a global publish"


def test_publishing_a_tenant_document_does_not_invalidate_other_tenants(cache_container):
    """The counterpart: an ordinary tenant-scoped ingest must NOT throw away
    every other tenant's cache."""
    from app.ingest.pipeline.runner import invalidate_cache_for

    c = cache_container
    qvec = c.embedder.embed(["what is the refund policy?"])[0]
    c.cache.put("T2", "U1", "q", qvec, "m", _payload("T2 answer"))
    invalidate_cache_for(c, "T1", "tenant")
    assert c.cache.get("T2", "U1", qvec, "m") is not None
