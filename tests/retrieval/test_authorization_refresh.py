from app.ingest.pipeline.runner import run_job
from app.retrieval.rag.access import access_predicate
from app.retrieval.rag.query import answer_query
from app.shared.adapters.postgres.db import transaction
from app.shared.domain.models import Principal, Role
from tests.conftest import structured_answer
from tests.ingest.test_generation_publication import _queue


def test_revoked_source_is_hidden_from_search_and_cached_answer(container, tenant):
    doc, job = _queue(container, tenant, "Invoice amount 007.00 USD")
    run_job(container, job)
    viewer = Principal(tenant["id"], tenant["member_id"], Role.MEMBER)
    access = access_predicate(viewer)
    container.gateway.chat = lambda *args, **kwargs: structured_answer("Amount is 007.00")
    first = answer_query(
        container, tenant["id"], "Invoice amount?", access=access, user_id=viewer.user_id
    )
    assert first.grounded
    with transaction(container.settings.postgres_dsn) as cur:
        cur.execute("UPDATE documents SET visibility='private' WHERE id=%s", (doc.id,))
    second = answer_query(
        container, tenant["id"], "Invoice amount?", access=access, user_id=viewer.user_id
    )
    assert not second.grounded and not second.contexts and not second.citations


def test_global_scope_reduction_immediately_hides_old_vectors(container, tenant, other_tenant):
    doc, job = _queue(container, tenant, "Global policy amount 007.00")
    with transaction(container.settings.postgres_dsn) as cur:
        cur.execute("UPDATE documents SET scope='global' WHERE id=%s", (doc.id,))
    run_job(container, job)
    access = access_predicate(Principal(other_tenant["id"], "reader", Role.MEMBER))
    vector = container.embedder.embed_query("Global policy")
    assert container.vectors.search(other_tenant["id"], vector, access=access)
    with transaction(container.settings.postgres_dsn) as cur:
        cur.execute("UPDATE documents SET scope='tenant' WHERE id=%s", (doc.id,))
    assert container.vectors.search(other_tenant["id"], vector, access=access) == []


def test_cache_outage_does_not_prevent_fresh_answer(container, tenant, monkeypatch):
    _, job = _queue(container, tenant, "Invoice amount 007.00 USD")
    run_job(container, job)
    container.gateway.chat = lambda *args, **kwargs: structured_answer("Amount is 007.00")

    def unavailable(*args, **kwargs):
        raise ConnectionError("Redis unavailable")

    monkeypatch.setattr(container.cache, "get", unavailable)
    monkeypatch.setattr(container.cache, "put", unavailable)
    result = answer_query(container, tenant["id"], "Invoice amount?", user_id=tenant["admin_id"])
    assert result.answer == "Amount is 007.00 [1]"


def test_suspended_tenant_token_is_rejected(container, tenant):
    with transaction(container.settings.postgres_dsn) as cur:
        cur.execute("UPDATE tenants SET status='suspended' WHERE id=%s", (tenant["id"],))
    assert container.metadata.get_principal_by_token(tenant["admin_token"]) is None
