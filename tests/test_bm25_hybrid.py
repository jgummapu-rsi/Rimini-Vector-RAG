"""Hybrid retrieval: BM25 lexical scoring fused with dense cosine similarity."""
import numpy as np

from app.adapters.shared.bm25 import reciprocal_rank_fusion, tokenize
from app.domain.models import Principal, Role
from app.ports.vector_store import VectorPoint
from app.rag.access import access_predicate


def _pt(cid, tid, vec, **payload):
    payload.setdefault("_id", "docX")
    payload.setdefault("content", "text")
    return VectorPoint(chunk_id=cid, tenant_id=tid, vector=vec, payload=payload)


def _principal(tid, uid, role):
    return Principal(tenant_id=tid, user_id=uid, role=Role(role))


def test_tokenize_lowercases_and_splits_on_word_boundaries():
    assert tokenize("Transaction Code MB5S, before Close!") == [
        "transaction", "code", "mb5s", "before", "close",
    ]
    assert tokenize("") == []
    assert tokenize(None) == []


def test_reciprocal_rank_fusion_known_case():
    # Candidate 0 is rank-1 in BOTH signals -> its RRF denominators are the
    # smallest possible on both terms, so it must be the unique fused winner
    # regardless of how the other candidates trade off (no ambiguity/ties here,
    # unlike a rank swap between two candidates which nets out to an exact tie).
    dense = np.array([0.9, 0.5, 0.3])
    lexical = np.array([0.8, 0.4, 0.2])
    fused = reciprocal_rank_fusion(dense, lexical, k=60)
    assert np.argmax(fused) == 0
    # a pure rank-swap between two candidates (each rank-1 in one signal, rank-2 in
    # the other) always nets an exact tie under RRF -- documenting that property so
    # it isn't mistaken for a bug when designing hybrid-ranking test cases.
    swapped = reciprocal_rank_fusion(np.array([0.9, 0.1]), np.array([0.1, 0.9]), k=60)
    assert swapped[0] == swapped[1]


def test_search_without_query_text_is_unchanged_pure_dense(container):
    """Backward compatibility: omitting query_text preserves today's pure-dense
    ranking and doesn't attach bm25_score."""
    vs = container.vectors
    v = [1.0] + [0.0] * (vs.dim - 1)
    vs.upsert([_pt("dA001", "T1", v, _id="dA", user_id="A", visibility="tenant",
                   content="anything")])
    hits = vs.search("T1", v, top_k=5)
    assert hits[0].payload["bm25_score"] is None
    assert hits[0].payload["dense_score"] == hits[0].score


def test_bm25_surfaces_exact_lexical_match(container):
    """Among several realistically-embedded, topically-unrelated chunks, only one
    contains the rare exact term being searched for. Hybrid must surface it on
    top and report a nonzero bm25_score — proving BM25 is actually contributing."""
    vs = container.vectors
    texts = {
        "dFiller1": "Quarterly business review covering revenue growth across regions.",
        "dFiller2": "Employee onboarding checklist for new hires in the finance team.",
        "dFiller3": "Office relocation announcement for the downtown campus.",
        "dFiller4": "Annual holiday schedule and public holiday calendar.",
        "dTarget": "Reconciliation must use transaction code MB5S before closing the period.",
    }
    vecs = container.embedder.embed(list(texts.values()))
    for (doc_id, content), vec in zip(texts.items(), vecs):
        vs.upsert([_pt(f"{doc_id}001", "T1", vec, _id=doc_id, user_id="A",
                       visibility="tenant", content=content)])

    qvec = container.embedder.embed(["MB5S"])[0]
    hits = vs.search("T1", qvec, top_k=1, query_text="MB5S")
    assert hits[0].payload["_id"] == "dTarget"
    assert hits[0].payload["bm25_score"] > 0


def test_bm25_matches_extracted_topics_and_entities(container):
    """LLM-extracted topics/entities (app.pipeline.metadata_extract) are real
    signal paid for at ingest time -- BM25 must be able to match a term that
    only appears there, not in the chunk's literal wording (CLAUDE.md §16)."""
    vs = container.vectors
    texts = {
        "dFiller1": "Quarterly business review covering revenue growth across regions.",
        "dFiller2": "Employee onboarding checklist for new hires in the finance team.",
        "dFiller3": "Office relocation announcement for the downtown campus.",
        "dTarget": "This section explains the renewal process and required approvals.",
    }
    vecs = container.embedder.embed(list(texts.values()))
    for (doc_id, content), vec in zip(texts.items(), vecs):
        kwargs = {"content": content}
        if doc_id == "dTarget":
            kwargs["topics"] = ["Zylantrix Contract Renewal"]
            kwargs["entities"] = ["Acme Corp"]
            kwargs["author"] = "Priyanka Vasquez"
        vs.upsert([_pt(f"{doc_id}001", "T1", vec, _id=doc_id, user_id="A",
                       visibility="tenant", **kwargs)])

    qvec = container.embedder.embed(["Zylantrix"])[0]
    hits = vs.search("T1", qvec, top_k=1, query_text="Zylantrix")
    assert hits[0].payload["_id"] == "dTarget"
    assert hits[0].payload["bm25_score"] > 0

    qvec_author = container.embedder.embed(["Vasquez"])[0]
    hits_author = vs.search("T1", qvec_author, top_k=1, query_text="Vasquez")
    assert hits_author[0].payload["_id"] == "dTarget"
    assert hits_author[0].payload["bm25_score"] > 0


def test_hybrid_respects_acl_and_global_scope(container):
    """Access filtering (visibility/ACL, tenant/global scope) happens before either
    ranking signal sees a document — hybrid must not leak anything pure-dense
    search wouldn't already show."""
    vs = container.vectors
    v = [1.0] + [0.0] * (vs.dim - 1)
    vs.upsert([_pt("dP001", "T1", v, _id="dP", user_id="A", visibility="private",
                   content="MB5S private ticket detail")])
    vs.upsert([_pt("dG001", "T1", v, _id="dG", user_id="A", visibility="private",
                   scope="global", content="MB5S global knowledge base entry")])

    # a stranger in a different tenant: sees only the global doc, even with an
    # exact lexical match against the private one's content
    hits = vs.search("T2", v, top_k=5, query_text="MB5S",
                      access=access_predicate(_principal("T2", "stranger", "viewer")))
    assert {h.payload["_id"] for h in hits} == {"dG"}
