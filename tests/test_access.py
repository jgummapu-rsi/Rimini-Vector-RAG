"""Visibility / ACL enforcement in retrieval."""
from app.domain.models import Principal, Role
from app.rag.access import access_predicate, can_view
from app.ports.vector_store import VectorPoint


def _view(viewer, role, **doc):
    return can_view(doc, viewer, role)


def test_can_view_rules():
    # owner sees own private doc
    assert _view("u1", "member", user_id="u1", visibility="private")
    # other user does NOT see a private doc
    assert not _view("u2", "member", user_id="u1", visibility="private")
    # tenant-wide is visible to anyone
    assert _view("u2", "member", user_id="u1", visibility="tenant")
    # shared: only listed users
    assert _view("u2", "member", user_id="u1", visibility="shared", acl_user_ids=["u2"])
    assert not _view("u3", "member", user_id="u1", visibility="shared", acl_user_ids=["u2"])
    # admin sees everything
    assert _view("admin", "admin", user_id="u1", visibility="private")
    # scope=global bypasses everything: non-owner, non-admin, private, no ACL match
    assert _view("stranger", "viewer", user_id="u1", visibility="private",
                  acl_user_ids=[], scope="global")


def _pt(cid, tid, vec, **payload):
    payload.setdefault("_id", "docX")
    payload.setdefault("content", "text")
    return VectorPoint(chunk_id=cid, tenant_id=tid, vector=vec, payload=payload)


def _principal(tid, uid, role):
    return Principal(tenant_id=tid, user_id=uid, role=Role(role))


def test_search_hides_private_docs_from_other_users(container):
    vs = container.vectors
    v = [1.0] + [0.0] * (vs.dim - 1)
    # A's private doc, B's tenant-wide doc
    vs.upsert([_pt("dA001", "T1", v, _id="dA", user_id="A", visibility="private")])
    vs.upsert([_pt("dB001", "T1", v, _id="dB", user_id="B", visibility="tenant")])

    # A sees both (owns dA, dB is tenant-wide)
    a_hits = {h.payload["_id"] for h in vs.search("T1", v, top_k=10,
              access=access_predicate(_principal("T1", "A", "member")))}
    assert a_hits == {"dA", "dB"}

    # B sees only dB (dA is A's private) -> the leak is closed
    b_hits = {h.payload["_id"] for h in vs.search("T1", v, top_k=10,
              access=access_predicate(_principal("T1", "B", "member")))}
    assert b_hits == {"dB"}

    # admin sees both
    adm = {h.payload["_id"] for h in vs.search("T1", v, top_k=10,
           access=access_predicate(_principal("T1", "Z", "admin")))}
    assert adm == {"dA", "dB"}


def test_search_shared_doc_visible_only_to_acl(container):
    vs = container.vectors
    v = [1.0] + [0.0] * (vs.dim - 1)
    vs.upsert([_pt("dS001", "T1", v, _id="dS", user_id="A",
                   visibility="shared", acl_user_ids=["B"])])
    assert {h.payload["_id"] for h in vs.search("T1", v, top_k=5,
            access=access_predicate(_principal("T1", "B", "member")))} == {"dS"}
    assert vs.search("T1", v, top_k=5,
            access=access_predicate(_principal("T1", "C", "member"))) == []


def test_search_global_doc_visible_cross_tenant(container):
    """A scope=global document upserted under tenant T1 shows up when a completely
    different tenant (T2) searches; a normal tenant-scoped doc under T1 does not."""
    vs = container.vectors
    v = [1.0] + [0.0] * (vs.dim - 1)
    vs.upsert([_pt("dG001", "T1", v, _id="dG", user_id="A",
                   visibility="private", scope="global")])
    vs.upsert([_pt("dT001", "T1", v, _id="dT", user_id="A",
                   visibility="tenant", scope="tenant")])

    t2_hits = {h.payload["_id"] for h in vs.search("T2", v, top_k=10,
               access=access_predicate(_principal("T2", "stranger", "viewer")))}
    assert t2_hits == {"dG"}   # only the global doc crosses the tenant boundary

    t1_hits = {h.payload["_id"] for h in vs.search("T1", v, top_k=10,
               access=access_predicate(_principal("T1", "A", "member")))}
    assert t1_hits == {"dG", "dT"}   # T1 still sees both its own doc and the global one


def test_query_endpoint_respects_acl(container, monkeypatch):
    """End-to-end via answer_query: a non-owner can't retrieve a private doc's content."""
    from app.rag.query import answer_query
    # stub generation so the test doesn't hit the gateway
    monkeypatch.setattr(container.gateway, "chat",
                        lambda messages, model, temperature=0.0: "stub answer")
    vs = container.vectors
    v = container.embedder.embed(["secret quarterly revenue figures"])[0]
    vs.upsert([_pt("dP001", "T1", v, _id="dP", user_id="A", visibility="private",
                   content="secret quarterly revenue figures")])
    # owner A retrieves it; other user B gets nothing
    a = answer_query(container, "T1", "quarterly revenue", top_k=5,
                     access=access_predicate(_principal("T1", "A", "member")))
    b = answer_query(container, "T1", "quarterly revenue", top_k=5,
                     access=access_predicate(_principal("T1", "B", "member")))
    assert a.contexts and "secret" in " ".join(a.contexts)
    assert b.contexts == []
