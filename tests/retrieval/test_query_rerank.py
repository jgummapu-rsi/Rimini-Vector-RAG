"""Reranker wiring in the query path (app.retrieval.rag.query): a configured reranker
must actually determine final hit order/scores; no reranker must leave
retrieval behavior byte-for-byte unchanged (regression guard)."""

import dataclasses

import app.retrieval.rag.query as q
from app.retrieval.rag.access import access_predicate
from app.retrieval.rag.query import answer_query
from app.shared.domain.models import Principal, Role
from app.shared.ports.vector_store import VectorPoint
from tests.conftest import structured_answer


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
    vs.upsert(
        [
            _pt(
                "cA001",
                "T1",
                v,
                _id="docA",
                user_id="A",
                visibility="tenant",
                content="Alpha content about nothing in particular.",
            )
        ]
    )
    vs.upsert(
        [
            _pt(
                "cB001",
                "T1",
                v,
                _id="docB",
                user_id="A",
                visibility="tenant",
                content="Bravo content about nothing in particular.",
            )
        ]
    )
    vs.upsert(
        [
            _pt(
                "cC001",
                "T1",
                v,
                _id="docC",
                user_id="A",
                visibility="tenant",
                content="Charlie content about nothing in particular.",
            )
        ]
    )


def test_reranker_reorders_hits_by_its_own_scores(container):
    _seed_three_docs(container)
    container.gateway.chat = lambda messages, model, temperature=0.0: "stub answer"
    fake = _FakeReranker(
        {
            "Alpha content about nothing in particular.": 0.1,
            "Bravo content about nothing in particular.": 0.9,
            "Charlie content about nothing in particular.": 0.5,
        }
    )
    reranked_container = dataclasses.replace(container, reranker=fake)

    result = answer_query(
        reranked_container,
        "T1",
        "some query",
        top_k=2,
        access=access_predicate(_principal("T1", "A", "member")),
    )

    assert len(fake.calls) == 1
    assert fake.calls[0][0] == "some query"

    assert result.chunk_ids == ["cB001", "cC001"]
    assert result.scores == [0.9, 0.5]


def test_no_reranker_leaves_retrieval_order_unchanged(container):
    """container.reranker is None by default in the test settings fixture --
    behavior must match pre-reranker retrieval exactly."""
    _seed_three_docs(container)
    container.gateway.chat = lambda messages, model, temperature=0.0: "stub answer"
    assert container.reranker is None

    result = answer_query(
        container,
        "T1",
        "some query",
        top_k=2,
        access=access_predicate(_principal("T1", "A", "member")),
    )
    assert len(result.chunk_ids) == 2
    assert result.scores == sorted(result.scores, reverse=True)


def _force_decomposition(container, monkeypatch, subs):
    """Make answer_query take the decomposed path deterministically."""
    monkeypatch.setattr(q, "looks_multi_part", lambda _question: True)
    monkeypatch.setattr(q, "decompose_question", lambda gateway, model, question: list(subs))


def test_decomposed_query_still_returns_at_most_top_k_without_a_reranker(container, monkeypatch):
    """The regression: 3 sub-questions x up to 3 hits each merged into one pool,
    and with no reranker configured the whole pool was returned and fed to
    generation -- for a top_k=2 request."""
    _seed_three_docs(container)
    container.gateway.chat = lambda messages, model, temperature=0.0: structured_answer(
        "stub answer", ("1", "2")
    )
    assert container.reranker is None
    _force_decomposition(container, monkeypatch, ["alpha?", "bravo?", "charlie?"])

    result = answer_query(
        container,
        "T1",
        "compare alpha and bravo and charlie",
        top_k=2,
        access=access_predicate(_principal("T1", "A", "member")),
    )

    assert result.sub_questions == ["alpha?", "bravo?", "charlie?"]
    assert len(result.chunk_ids) == 2, "must not exceed the requested top_k"
    assert len(result.contexts) == 2
    assert len(result.scores) == 2
    assert len(result.citations) == 2

    assert len(set(result.chunk_ids)) == 2


def test_decomposed_query_truncates_with_a_reranker_too(container, monkeypatch):
    """Same bound on the reranked path, so turning the reranker on or off
    changes the ORDER of results but never how many come back."""
    _seed_three_docs(container)
    container.gateway.chat = lambda messages, model, temperature=0.0: "stub answer"
    _force_decomposition(container, monkeypatch, ["alpha?", "bravo?"])
    fake = _FakeReranker(
        {
            "Alpha content about nothing in particular.": 0.1,
            "Bravo content about nothing in particular.": 0.9,
            "Charlie content about nothing in particular.": 0.5,
        }
    )
    reranked = dataclasses.replace(container, reranker=fake)

    result = answer_query(
        reranked,
        "T1",
        "compare alpha and bravo",
        top_k=2,
        access=access_predicate(_principal("T1", "A", "member")),
    )
    assert result.chunk_ids == ["cB001", "cC001"]
    assert len(result.scores) == 2


def test_hits_below_min_score_are_dropped_even_when_top_k_not_full(container):
    _seed_three_docs(container)
    container.gateway.chat = lambda messages, model, temperature=0.0: "stub answer"
    fake = _FakeReranker(
        {
            "Alpha content about nothing in particular.": -11.4,
            "Bravo content about nothing in particular.": 3.4,
            "Charlie content about nothing in particular.": -11.3,
        }
    )
    reranked_container = dataclasses.replace(
        container,
        reranker=fake,
        settings=container.settings.model_copy(update={"rerank_min_score": -3.0}),
    )

    result = answer_query(
        reranked_container,
        "T1",
        "some query",
        top_k=3,
        access=access_predicate(_principal("T1", "A", "member")),
    )

    assert result.chunk_ids == ["cB001"]
    assert result.scores == [3.4]


def test_rerank_min_score_none_disables_the_floor(container):
    """The default keeps negative-scored candidates in reranked order."""
    _seed_three_docs(container)
    container.gateway.chat = lambda messages, model, temperature=0.0: "stub answer"
    fake = _FakeReranker(
        {
            "Alpha content about nothing in particular.": -11.4,
            "Bravo content about nothing in particular.": 3.4,
            "Charlie content about nothing in particular.": -11.3,
        }
    )
    assert container.settings.rerank_min_score is None
    reranked_container = dataclasses.replace(container, reranker=fake)

    result = answer_query(
        reranked_container,
        "T1",
        "some query",
        top_k=3,
        access=access_predicate(_principal("T1", "A", "member")),
    )

    assert result.chunk_ids == ["cB001", "cC001", "cA001"]
    assert result.scores == [3.4, -11.3, -11.4]


def test_top_k_larger_than_the_pool_returns_everything_available(container):
    """Truncation must not become a floor: asking for more than exists returns
    what exists, not an error and not padding."""
    _seed_three_docs(container)
    container.gateway.chat = lambda messages, model, temperature=0.0: "stub answer"
    result = answer_query(
        container,
        "T1",
        "some query",
        top_k=50,
        access=access_predicate(_principal("T1", "A", "member")),
    )
    assert len(result.chunk_ids) == 3


def test_relevant_negative_logit_above_floor_is_not_dropped(container):
    _seed_three_docs(container)
    container.gateway.chat = lambda messages, model, temperature=0.0: "stub answer"
    fake = _FakeReranker(
        {
            "Alpha content about nothing in particular.": -1.4,
            "Bravo content about nothing in particular.": -8.0,
            "Charlie content about nothing in particular.": -9.0,
        }
    )
    reranked = dataclasses.replace(
        container,
        reranker=fake,
        settings=container.settings.model_copy(update={"rerank_min_score": -3.0}),
    )

    result = answer_query(
        reranked,
        "T1",
        "alpha",
        top_k=3,
        access=access_predicate(_principal("T1", "A", "member")),
    )

    assert result.chunk_ids == ["cA001"]
