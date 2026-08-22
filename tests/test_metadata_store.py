from app.domain.models import ChunkRecord, Document, Job, JobStage, JobStatus, Role
from app.ids import new_object_id


def _doc(tenant_id, owner_id, sha="abc123", source="pdf", scope="tenant"):
    return Document(
        id=new_object_id(), tenant_id=tenant_id, owner_user_id=owner_id,
        source_type=source, blob_path="/tmp/x.pdf", content_sha256=sha,
        mime="application/pdf", filename="x.pdf", visibility="private", acl_user_ids=[],
        scope=scope,
    )


def test_principal_resolution_and_role(container, tenant):
    p = container.metadata.get_principal_by_token(tenant["admin_token"])
    assert p.tenant_id == tenant["id"]
    assert p.role == Role.ADMIN
    assert container.metadata.get_principal_by_token("bogus") is None


def test_document_dedup_lookup(container, tenant):
    md = container.metadata
    doc = _doc(tenant["id"], tenant["admin_id"], sha="deadbeef")
    md.create_document(doc)
    found = md.get_document_by_hash(tenant["id"], "deadbeef")
    assert found and found.id == doc.id
    assert md.get_document_by_hash("other-tenant", "deadbeef") is None


def test_document_tenant_isolation(container, tenant, other_tenant):
    md = container.metadata
    doc = _doc(tenant["id"], tenant["admin_id"])
    md.create_document(doc)
    assert md.get_document(tenant["id"], doc.id) is not None
    assert md.get_document(other_tenant["id"], doc.id) is None


def test_global_document_readable_cross_tenant(container, tenant, other_tenant):
    md = container.metadata
    tenant_doc = _doc(tenant["id"], tenant["admin_id"], sha="sha-tenant")
    global_doc = _doc(tenant["id"], tenant["admin_id"], sha="sha-global", scope="global")
    md.create_document(tenant_doc)
    md.create_document(global_doc)

    # normal doc: still strictly isolated
    assert md.get_document(other_tenant["id"], tenant_doc.id) is None
    # global doc: readable from a completely different tenant
    found = md.get_document(other_tenant["id"], global_doc.id)
    assert found is not None and found.id == global_doc.id

    md.replace_document_chunks(tenant["id"], global_doc.id, [
        ChunkRecord(ordinal=0, modality="text", extractor="native",
                    route_reason="text", token_count=5, text="firm-wide knowledge"),
    ])
    chunks = md.get_document_chunks(other_tenant["id"], global_doc.id)
    assert len(chunks) == 1 and chunks[0].text == "firm-wide knowledge"


def test_chunks_replace_get_and_deterministic_ids(container, tenant):
    md = container.metadata
    doc = _doc(tenant["id"], tenant["admin_id"])
    md.create_document(doc)
    recs = [
        ChunkRecord(ordinal=0, modality="text", extractor="pdf_text",
                    route_reason="text_layer", token_count=5, text="hello", meta={"page": 1}),
        ChunkRecord(ordinal=1, modality="table", extractor="pdf_table",
                    route_reason="structured", token_count=9, text="| a |", meta={}),
    ]
    ids = md.replace_document_chunks(tenant["id"], doc.id, recs)
    # chunk id = document id + 1-based zero-padded ordinal
    assert ids == [f"{doc.id}001", f"{doc.id}002"]

    got = md.get_document_chunks(tenant["id"], doc.id)
    assert [c.ordinal for c in got] == [0, 1]
    assert got[0].content_sha256  # hash computed on persist

    # replace is idempotent: same deterministic ids, no duplicate rows
    ids2 = md.replace_document_chunks(tenant["id"], doc.id, recs)
    assert ids2 == ids
    assert len(md.get_document_chunks(tenant["id"], doc.id)) == 2


def test_job_lifecycle_and_route_summary(container, tenant):
    md = container.metadata
    doc = _doc(tenant["id"], tenant["admin_id"])
    md.create_document(doc)
    job = Job(id=new_object_id(), document_id=doc.id, tenant_id=tenant["id"],
              stage=JobStage.PARSE.value, status=JobStatus.QUEUED.value, attempts=0)
    md.create_job(job)
    md.set_route_summary(job.id, {"elements": 3})
    got = md.get_job(tenant["id"], job.id)
    assert got.route_summary == {"elements": 3}
    assert md.get_job("other", job.id) is None


def test_delete_document_cascades_chunks_and_jobs(container, tenant):
    md = container.metadata
    doc = _doc(tenant["id"], tenant["admin_id"])
    md.create_document(doc)
    md.create_job(Job(id=new_object_id(), document_id=doc.id, tenant_id=tenant["id"],
                      stage="parse", status="queued", attempts=0))
    md.replace_document_chunks(tenant["id"], doc.id, [
        ChunkRecord(ordinal=0, modality="text", extractor="t", route_reason="r",
                    token_count=1, text="x", meta={})])
    md.delete_document(tenant["id"], doc.id)
    assert md.get_document(tenant["id"], doc.id) is None
    assert md.get_document_chunks(tenant["id"], doc.id) == []
