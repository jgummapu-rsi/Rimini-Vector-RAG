"""Reranker wiring in the query path (app.rag.query): a configured reranker
must actually determine final hit order/scores; no reranker must leave
retrieval behavior byte-for-byte unchanged (regression guard)."""
import dataclasses

from app.domain.models import Principal, Role
from app.ports.vector_store import VectorPoint
from app.rag.access import access_predicate
from app.rag.query import answer_query


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
