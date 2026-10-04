import pytest

from app.ingest.pipeline.runner import run_job
from app.retrieval.rag.access import access_predicate
from app.retrieval.rag.query import answer_query
from app.shared.adapters.postgres.db import transaction
from app.shared.domain.models import Document, Principal, Role
from app.shared.ids import new_object_id
from app.shared.ports.vector_store import VectorPoint
from tests.conftest import structured_answer
from tests.ingest.test_generation_publication import _queue


@pytest.mark.parametrize("used_source", ["1", "2"])
def test_revocation_during_generation_clears_all_evidence(container, tenant, used_source):
    first, job = _queue(container, tenant, "Invoice private amount 007.00")
    run_job(container, job)
    second, job = _queue(container, tenant, "Invoice public amount 008.00")
    run_job(container, job)
    principal = Principal(tenant["id"], tenant["member_id"], Role.MEMBER)
    before = container.metadata.corpus_epoch(tenant["id"])

    def revoke(messages, **kwargs):
        with transaction(container.settings.postgres_dsn) as cur:
            cur.execute("UPDATE documents SET visibility='private' WHERE id=%s", (first.id,))
        return structured_answer("Invoice amount", (used_source,))

    container.gateway.chat = revoke
    result = answer_query(
        container,
        tenant["id"],
        "Invoice amount",
        top_k=2,
        access=access_predicate(principal),
        user_id=principal.user_id,
    )
    assert container.metadata.corpus_epoch(tenant["id"]) != before
    assert result.contexts == result.chunk_ids == result.citations == result.scores == []
    assert result.answer_status == "insufficient_evidence"


def test_global_revocation_changes_other_tenants_epoch(container, tenant, other_tenant):
    document, job = _queue(container, tenant, "Global policy")
    with transaction(container.settings.postgres_dsn) as cur:
        cur.execute("UPDATE documents SET scope='global' WHERE id=%s", (document.id,))
    run_job(container, job)
    before = container.metadata.corpus_epoch(other_tenant["id"])
    with transaction(container.settings.postgres_dsn) as cur:
        cur.execute("UPDATE documents SET scope='tenant' WHERE id=%s", (document.id,))
    assert container.metadata.corpus_epoch(other_tenant["id"]) != before


def test_legacy_vectors_follow_current_acl_and_deletion_marker(container, tenant):

    document = Document(
        new_object_id(),
        tenant["id"],
        tenant["admin_id"],
        "text",
        "unused",
        "legacy",
        "text/plain",
        "legacy.txt",
        "tenant",
        [],
    )
    container.metadata.create_document(document)
    vector = container.embedder.embed_query("legacy invoice")
    point = VectorPoint(
        "legacy-chunk",
        tenant["id"],
        vector,
        {
            "_id": document.id,
            "content": "legacy invoice",
            "visibility": "tenant",
            "user_id": tenant["admin_id"],
        },
    )
    container.vectors.upsert([point])
    access = access_predicate(Principal(tenant["id"], tenant["member_id"], Role.MEMBER))
    assert container.vectors.search(tenant["id"], vector, access=access)
    container.metadata.update_document_access(tenant["id"], document.id, "private", "tenant")
    assert container.vectors.search(tenant["id"], vector, access=access) == []
    container.metadata.delete_document(tenant["id"], document.id)
    assert container.vectors.search(tenant["id"], vector) == []
    with pytest.raises(ValueError, match="Deleted document"):
        container.vectors.upsert([point])
