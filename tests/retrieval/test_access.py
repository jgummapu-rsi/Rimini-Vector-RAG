"""Visibility / ACL enforcement in retrieval."""

import pytest

from app.retrieval.rag.access import access_predicate, can_view
from app.retrieval.rag.query import answer_query
from app.shared.adapters.pgvector.vector_store import acl_pushdown
from app.shared.domain.models import Principal, Role
from app.shared.ports.vector_store import VectorPoint


def _view(viewer, role, **doc):
    return can_view(doc, viewer, role)


def test_can_view_rules():

    assert _view("u1", "member", user_id="u1", visibility="private")

    assert not _view("u2", "member", user_id="u1", visibility="private")

    assert _view("u2", "member", user_id="u1", visibility="tenant")

    assert _view("u2", "member", user_id="u1", visibility="shared", acl_user_ids=["u2"])
    assert not _view("u3", "member", user_id="u1", visibility="shared", acl_user_ids=["u2"])

    assert _view("admin", "admin", user_id="u1", visibility="private")

    assert _view(
        "stranger", "viewer", user_id="u1", visibility="private", acl_user_ids=[], scope="global"
    )


def _pt(cid, tid, vec, **payload):
    payload.setdefault("_id", "docX")
    payload.setdefault("content", "text")
    return VectorPoint(chunk_id=cid, tenant_id=tid, vector=vec, payload=payload)


def _principal(tid, uid, role):
    return Principal(tenant_id=tid, user_id=uid, role=Role(role))


def test_search_hides_private_docs_from_other_users(container):
    vs = container.vectors
    v = [1.0] + [0.0] * (vs.dim - 1)

    vs.upsert([_pt("dA001", "T1", v, _id="dA", user_id="A", visibility="private")])
    vs.upsert([_pt("dB001", "T1", v, _id="dB", user_id="B", visibility="tenant")])

    a_hits = {
        h.payload["_id"]
        for h in vs.search(
            "T1", v, top_k=10, access=access_predicate(_principal("T1", "A", "member"))
        )
    }
    assert a_hits == {"dA", "dB"}

    b_hits = {
        h.payload["_id"]
        for h in vs.search(
            "T1", v, top_k=10, access=access_predicate(_principal("T1", "B", "member"))
        )
    }
    assert b_hits == {"dB"}

    adm = {
        h.payload["_id"]
        for h in vs.search(
            "T1", v, top_k=10, access=access_predicate(_principal("T1", "Z", "admin"))
        )
    }
    assert adm == {"dA", "dB"}


def test_search_shared_doc_visible_only_to_acl(container):
    vs = container.vectors
    v = [1.0] + [0.0] * (vs.dim - 1)
    vs.upsert(
        [_pt("dS001", "T1", v, _id="dS", user_id="A", visibility="shared", acl_user_ids=["B"])]
    )
    assert {
        h.payload["_id"]
        for h in vs.search(
            "T1", v, top_k=5, access=access_predicate(_principal("T1", "B", "member"))
        )
    } == {"dS"}
    assert (
        vs.search("T1", v, top_k=5, access=access_predicate(_principal("T1", "C", "member"))) == []
    )


def test_search_global_doc_visible_cross_tenant(container):
    """A scope=global document upserted under tenant T1 shows up when a completely
    different tenant (T2) searches; a normal tenant-scoped doc under T1 does not."""
    vs = container.vectors
    v = [1.0] + [0.0] * (vs.dim - 1)
    vs.upsert([_pt("dG001", "T1", v, _id="dG", user_id="A", visibility="private", scope="global")])
    vs.upsert([_pt("dT001", "T1", v, _id="dT", user_id="A", visibility="tenant", scope="tenant")])

    t2_hits = {
        h.payload["_id"]
        for h in vs.search(
            "T2", v, top_k=10, access=access_predicate(_principal("T2", "stranger", "viewer"))
        )
    }
    assert t2_hits == {"dG"}

    t1_hits = {
        h.payload["_id"]
        for h in vs.search(
            "T1", v, top_k=10, access=access_predicate(_principal("T1", "A", "member"))
        )
    }
    assert t1_hits == {"dG", "dT"}


def test_query_endpoint_respects_acl(container, monkeypatch):
    """End-to-end via answer_query: a non-owner can't retrieve a private doc's content."""

    monkeypatch.setattr(
        container.gateway, "chat", lambda messages, model, temperature=0.0: "stub answer"
    )
    vs = container.vectors
    v = container.embedder.embed(["secret quarterly revenue figures"])[0]
    vs.upsert(
        [
            _pt(
                "dP001",
                "T1",
                v,
                _id="dP",
                user_id="A",
                visibility="private",
                content="secret quarterly revenue figures",
            )
        ]
    )

    a = answer_query(
        container,
        "T1",
        "quarterly revenue",
        top_k=5,
        access=access_predicate(_principal("T1", "A", "member")),
    )
    b = answer_query(
        container,
        "T1",
        "quarterly revenue",
        top_k=5,
        access=access_predicate(_principal("T1", "B", "member")),
    )
    assert a.contexts and "secret" in " ".join(a.contexts)
    assert b.contexts == []


def test_access_predicate_is_still_a_plain_predicate():
    """Regression guard: AccessFilter gained attributes, but every existing
    consumer calls it as `predicate(payload)` and must keep working."""
    f = access_predicate(_principal("T1", "A", "member"))
    assert callable(f)
    assert f({"user_id": "A", "visibility": "private"}) is True
    assert f({"user_id": "B", "visibility": "private"}) is False

    payload = {"user_id": "B", "visibility": "shared", "acl_user_ids": ["A"]}
    assert f(payload) == can_view(payload, "A", "member")


def test_acl_pushdown_emits_nothing_when_there_is_nothing_to_filter():

    assert acl_pushdown(None) == ("", ())

    assert acl_pushdown(access_predicate(_principal("T1", "Z", "admin"))) == ("", ())


def test_acl_pushdown_for_a_normal_user_binds_the_user_id():
    sql, params = acl_pushdown(access_predicate(_principal("T1", "A", "member")))
    assert sql, "a non-admin must produce a filter"
    assert params == ("A", "A")

    assert "scope='global'" in sql
    assert "payload->>'user_id'=%s" in sql
    assert "payload->>'visibility'='tenant'" in sql
    assert "payload->>'visibility'='shared'" in sql
    assert "jsonb_exists(payload->'acl_user_ids', %s)" in sql

    assert sql.count("%s") == len(params)


def test_acl_pushdown_degrades_safely_for_an_opaque_predicate():
    """A caller passing a bare lambda (as tests and eval scripts do) can't be
    translated. That must fall back to no SQL clause -- never to a clause that
    accidentally admits everything -- with the Python filter still applied."""
    sql, params = acl_pushdown(lambda payload: True)
    assert (sql, params) == ("", ())


@pytest.mark.parametrize(
    "role,user_id,visibility,acl,scope,expected",
    [
        ("member", "A", "private", [], "tenant", True),
        ("member", "B", "private", [], "tenant", False),
        ("member", "B", "tenant", [], "tenant", True),
        ("member", "B", "shared", ["A"], "tenant", True),
        ("member", "B", "shared", ["C"], "tenant", False),
        ("viewer", "B", "private", [], "global", True),
    ],
)
def test_sql_filter_mirrors_can_view(role, user_id, visibility, acl, scope, expected):
    """The SQL is a translation of `can_view`; simulate it row-by-row and assert
    the two agree. If someone edits one rule and not the other, this fails."""
    payload = {"user_id": user_id, "visibility": visibility, "acl_user_ids": acl, "scope": scope}
    me = "A"
    assert can_view(payload, me, role) is expected

    f = access_predicate(_principal("T1", me, role))
    sql, params = acl_pushdown(f)
    if not sql:
        sql_admits = True
    else:
        sql_admits = (
            scope == "global"
            or payload["user_id"] == params[0]
            or visibility == "tenant"
            or (visibility == "shared" and params[1] in acl)
        )
    assert sql_admits is expected, f"SQL and can_view disagree for {payload}"
