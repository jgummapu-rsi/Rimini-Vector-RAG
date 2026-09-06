"""Reranker wiring in the query path (app.retrieval.rag.query): a configured reranker
must actually determine final hit order/scores; no reranker must leave
retrieval behavior byte-for-byte unchanged (regression guard)."""
import dataclasses

from app.shared.domain.models import Principal, Role
from app.shared.ports.vector_store import VectorPoint
from app.retrieval.rag.access import access_predicate
from app.retrieval.rag.query import answer_query


def _pt(cid, tid, vec, **payload):
    payload.setdefault("_id", "docX")
    payload.setdefault("content", "text")
    return VectorPoint(chunk_id=cid, tenant_id=tid, vector=vec, payload=payload)


def _principal(tid, uid, role):
    return Principal(tenant_id=tid, user_id=uid, role=Role(role))


class _FakeReranker:
    """Scores documents by a fixed lookup keyed on exact content match, so a
    test can force a specific final order regardless of retrieval's own
    (dense+BM25) ranking."""
    def __init__(self, score_by_content: dict[str, float]):
        self._scores = score_by_content
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query, documents):
        self.calls.append((query, list(documents)))
        return [self._scores.get(d, 0.0) for d in documents]


def _seed_three_docs(container):
    vs = container.vectors
    v = container.embedder.embed(["irrelevant filler text"])[0]
    vs.upsert([_pt("cA001", "T1", v, _id="docA", user_id="A", visibility="tenant",
                   content="Alpha content about nothing in particular.")])
    vs.upsert([_pt("cB001", "T1", v, _id="docB", user_id="A", visibility="tenant",
                   content="Bravo content about nothing in particular.")])
    vs.upsert([_pt("cC001", "T1", v, _id="docC", user_id="A", visibility="tenant",
                   content="Charlie content about nothing in particular.")])


def test_reranker_reorders_hits_by_its_own_scores(container):
    _seed_three_docs(container)
    container.gateway.chat = lambda messages, model, temperature=0.0: "stub answer"
    fake = _FakeReranker({
        "Alpha content about nothing in particular.": 0.1,
        "Bravo content about nothing in particular.": 0.9,
        "Charlie content about nothing in particular.": 0.5,
    })
    reranked_container = dataclasses.replace(container, reranker=fake)

    result = answer_query(
        reranked_container, "T1", "some query", top_k=2,
        access=access_predicate(_principal("T1", "A", "member")),
    )

    assert len(fake.calls) == 1
    assert fake.calls[0][0] == "some query"          # reranked against the ORIGINAL question
    # top_k=2: Bravo (0.9) then Charlie (0.5) -- Alpha (0.1) dropped
    assert result.chunk_ids == ["cB001", "cC001"]
    assert result.scores == [0.9, 0.5]


def test_no_reranker_leaves_retrieval_order_unchanged(container):
    """container.reranker is None by default in the test settings fixture --
    behavior must match pre-reranker retrieval exactly."""
    _seed_three_docs(container)
    container.gateway.chat = lambda messages, model, temperature=0.0: "stub answer"
    assert container.reranker is None

    result = answer_query(
        container, "T1", "some query", top_k=2,
        access=access_predicate(_principal("T1", "A", "member")),
    )
    assert len(result.chunk_ids) == 2
    assert result.scores == sorted(result.scores, reverse=True)


# ------------------------------------------------------- top_k truncation --
# `_rerank` is the single place the candidate pool narrows to what the caller
# asked for. Its no-reranker path used to return the pool untouched, which only
# showed up on the DECOMPOSED path: retrieval there runs once per sub-question
# and merges, so the pool is several times top_k.


def _force_decomposition(container, monkeypatch, subs):
    """Make answer_query take the decomposed path deterministically."""
    import app.retrieval.rag.query as q
    monkeypatch.setattr(q, "looks_multi_part", lambda _question: True)
    monkeypatch.setattr(q, "decompose_question",
                        lambda gateway, model, question: list(subs))


def test_decomposed_query_still_returns_at_most_top_k_without_a_reranker(
        container, monkeypatch):
    """The regression: 3 sub-questions x up to 3 hits each merged into one pool,
    and with no reranker configured the whole pool was returned and fed to
    generation -- for a top_k=2 request."""
    _seed_three_docs(container)
    container.gateway.chat = lambda messages, model, temperature=0.0: "stub answer"
    assert container.reranker is None
    _force_decomposition(container, monkeypatch, ["alpha?", "bravo?", "charlie?"])

    result = answer_query(
        container, "T1", "compare alpha and bravo and charlie", top_k=2,
        access=access_predicate(_principal("T1", "A", "member")),
    )

    assert result.sub_questions == ["alpha?", "bravo?", "charlie?"]
    assert len(result.chunk_ids) == 2, "must not exceed the requested top_k"
    assert len(result.contexts) == 2
    assert len(result.scores) == 2
    assert len(result.citations) == 2
    # every parallel list stays aligned after truncation
    assert len(set(result.chunk_ids)) == 2


def test_decomposed_query_truncates_with_a_reranker_too(container, monkeypatch):
    """Same bound on the reranked path, so turning the reranker on or off
    changes the ORDER of results but never how many come back."""
    _seed_three_docs(container)
    container.gateway.chat = lambda messages, model, temperature=0.0: "stub answer"
    _force_decomposition(container, monkeypatch, ["alpha?", "bravo?"])
    fake = _FakeReranker({
        "Alpha content about nothing in particular.": 0.1,
        "Bravo content about nothing in particular.": 0.9,
        "Charlie content about nothing in particular.": 0.5,
    })
    reranked = dataclasses.replace(container, reranker=fake)

    result = answer_query(
        reranked, "T1", "compare alpha and bravo", top_k=2,
        access=access_predicate(_principal("T1", "A", "member")),
    )
    assert result.chunk_ids == ["cB001", "cC001"]
    assert len(result.scores) == 2


# --------------------------------------------------------- min-score floor --
# Regression: a corpus with fewer than top_k truly relevant chunks used to pad
# the response with whatever candidates were left, however irrelevant --
# measured live against the real cross-encoder: a genuine match scored 3.4,
# an unrelated document's chunks scored ~-11.3..-11.45 in the same pool, and
# both were returned because `_rerank` only ever truncated to top_k, never
# filtered by absolute relevance.


def test_hits_below_min_score_are_dropped_even_when_top_k_not_full(container):
    _seed_three_docs(container)
    container.gateway.chat = lambda messages, model, temperature=0.0: "stub answer"
    fake = _FakeReranker({
        "Alpha content about nothing in particular.": -11.4,   # below the floor
        "Bravo content about nothing in particular.": 3.4,
        "Charlie content about nothing in particular.": -11.3,  # below the floor
    })
    reranked_container = dataclasses.replace(container, reranker=fake)

    result = answer_query(
        reranked_container, "T1", "some query", top_k=3,
        access=access_predicate(_principal("T1", "A", "member")),
    )

    # top_k=3 requested, pool has 3 candidates, but only one clears the floor.
    assert result.chunk_ids == ["cB001"]
    assert result.scores == [3.4]


def test_rerank_min_score_none_disables_the_floor(container):
    """rerank_min_score is a config knob, not a hardcoded behavior -- an admin
    who sets it to None gets the pre-existing always-pad-to-top_k behavior."""
    _seed_three_docs(container)
    container.gateway.chat = lambda messages, model, temperature=0.0: "stub answer"
    fake = _FakeReranker({
        "Alpha content about nothing in particular.": -11.4,
        "Bravo content about nothing in particular.": 3.4,
        "Charlie content about nothing in particular.": -11.3,
    })
    no_floor_settings = container.settings.model_copy(update={"rerank_min_score": None})
    reranked_container = dataclasses.replace(
        container, settings=no_floor_settings, reranker=fake)

    result = answer_query(
        reranked_container, "T1", "some query", top_k=3,
        access=access_predicate(_principal("T1", "A", "member")),
    )

    assert len(result.chunk_ids) == 3


def test_top_k_larger_than_the_pool_returns_everything_available(container):
    """Truncation must not become a floor: asking for more than exists returns
    what exists, not an error and not padding."""
    _seed_three_docs(container)
    container.gateway.chat = lambda messages, model, temperature=0.0: "stub answer"
    result = answer_query(
        container, "T1", "some query", top_k=50,
        access=access_predicate(_principal("T1", "A", "member")),
    )
    assert len(result.chunk_ids) == 3
