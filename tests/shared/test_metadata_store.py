import sqlite3

import pytest

from app.shared.domain.models import (
    ChunkRecord,
    Document,
    Job,
    JobStage,
    JobStatus,
    Role,
    finalize_chunks,
)
from app.shared.ids import new_object_id
from app.shared.ports.metadata_store import EmailAlreadyRegistered
from app.shared.security import hash_token


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


# ------------------------------------------------------- chunk identity --
# The chunk id is the SHARED key between the metadata store (chunks.id) and the
# vector store (point id). It used to be derived privately inside each store as
# a side effect of persisting, and the pipeline then read it back off the
# objects it had passed in -- so a store that didn't mutate in place would have
# sent chunk_id=None to the vector store, silently.


def _recs(n=3):
    return [ChunkRecord(ordinal=i, modality="text", extractor="x",
                        route_reason="r", token_count=1, text=f"body {i}")
            for i in range(n)]


def test_finalize_chunks_stamps_ids_and_hashes():
    recs = _recs(2)
    assert all(r.id is None and r.content_sha256 is None for r in recs)
    ids = finalize_chunks("DOC", recs)
    assert ids == ["DOC001", "DOC002"]
    assert [r.id for r in recs] == ids
    assert all(r.content_sha256 for r in recs)


def test_finalize_chunks_is_idempotent():
    """The pipeline stamps chunks and the store stamps them again; running it
    twice must not renumber or rehash anything."""
    recs = _recs(3)
    first = finalize_chunks("DOC", recs)
    hashes = [r.content_sha256 for r in recs]
    assert finalize_chunks("DOC", recs) == first
    assert [r.content_sha256 for r in recs] == hashes


def test_finalize_chunks_hash_tracks_content():
    a, b = _recs(1), _recs(1)
    finalize_chunks("DOC", a)
    b[0].text = "different body"
    finalize_chunks("DOC", b)
    assert a[0].id == b[0].id                       # same position
    assert a[0].content_sha256 != b[0].content_sha256   # different content


def test_store_returns_ids_matching_what_it_persisted(container, tenant):
    """The store's return value and the ids actually written must agree -- the
    pipeline uses the stamped records, the API reads the rows back."""
    md = container.metadata
    doc = _doc(tenant["id"], tenant["admin_id"], sha="ids-match")
    md.create_document(doc)
    recs = _recs(4)
    returned = md.replace_document_chunks(tenant["id"], doc.id, recs)
    persisted = [c.id for c in md.get_document_chunks(tenant["id"], doc.id)]
    assert returned == persisted == [r.id for r in recs]


# --------------------------------------------------- password-based onboarding --
# Backs app.api.onboarding_routes: email/password signup+login as an
# alternative to scripts/seed.py.


def test_create_user_with_password_then_found_by_email(container, tenant):
    md = container.metadata
    uid = md.create_user_with_password(
        tenant["id"], "new.dev@acme.test", Role.ADMIN.value, "sk-new", "pbkdf2_sha256$1$aa$bb")
    found = md.get_user_by_email("new.dev@acme.test")
    assert found is not None
    assert found.id == uid
    assert found.tenant_id == tenant["id"]
    assert not hasattr(found, "api_token")  # the raw token is never recoverable
    assert found.password_hash == "pbkdf2_sha256$1$aa$bb"
    # the raw token still resolves to a Principal, via hashed lookup
    principal = md.get_principal_by_token("sk-new")
    assert principal is not None
    assert principal.user_id == uid


def test_api_token_is_not_stored_in_plaintext(container, tenant):
    uid = container.metadata.create_user(
        tenant["id"], "hashed.dev@acme.test", Role.MEMBER.value, "sk-plaintext-check")
    with sqlite3.connect(container.settings.sqlite_path) as conn:
        stored = conn.execute("SELECT api_token FROM users WHERE id = ?", (uid,)).fetchone()[0]
    assert stored != "sk-plaintext-check"
    assert stored == hash_token("sk-plaintext-check")


def test_rotate_api_token_invalidates_the_previous_token(container, tenant):
    md = container.metadata
    uid = md.create_user(tenant["id"], "rotate.dev@acme.test", Role.MEMBER.value, "sk-old")
    assert md.get_principal_by_token("sk-old") is not None
    md.rotate_api_token(uid, "sk-new-rotated")
    assert md.get_principal_by_token("sk-old") is None
    principal = md.get_principal_by_token("sk-new-rotated")
    assert principal is not None
    assert principal.user_id == uid


def test_get_user_by_email_unknown_returns_none(container):
    assert container.metadata.get_user_by_email("nobody@nowhere.test") is None


def test_get_user_by_email_prefers_password_holder_on_collision(container, tenant, other_tenant):
    """A plain (no-password) user in one tenant must never shadow a real
    onboarding account with the same email in another tenant."""
    md = container.metadata
    md.create_user(other_tenant["id"], "shared@example.test", Role.MEMBER.value, "sk-other-shared")
    uid = md.create_user_with_password(
        tenant["id"], "shared@example.test", Role.ADMIN.value, "sk-shared", "pbkdf2_sha256$1$aa$bb")
    found = md.get_user_by_email("shared@example.test")
    assert found.id == uid
    assert found.password_hash is not None


def test_create_user_with_password_rejects_a_second_account_for_the_same_email(
    container, tenant, other_tenant,
):
    """Two password-holding accounts for the same email must never coexist,
    even across DIFFERENT tenants (onboarding's get_user_by_email pre-check
    can't fully close this on its own -- see idx_users_email_password_unique
    in schema.sql and app.shared.ports.metadata_store.EmailAlreadyRegistered).
    This is the store-level guard the pre-check races against."""
    md = container.metadata
    md.create_user_with_password(
        tenant["id"], "race@example.test", Role.ADMIN.value, "sk-race-1", "pbkdf2_sha256$1$aa$bb")
    with pytest.raises(EmailAlreadyRegistered):
        md.create_user_with_password(
            other_tenant["id"], "race@example.test", Role.ADMIN.value,
            "sk-race-2", "pbkdf2_sha256$1$cc$dd")
    # the first account is unaffected, and there is still exactly one match
    found = md.get_user_by_email("race@example.test")
    assert found.tenant_id == tenant["id"]


def test_create_user_with_password_allows_same_email_when_prior_account_has_no_password(
    container, tenant, other_tenant,
):
    """The partial index only guards password-HOLDING rows: a plain
    create_user (seed-script style, no password) sharing an email must not
    block a later onboarding signup for that same email."""
    md = container.metadata
    md.create_user(other_tenant["id"], "seed-only@example.test", Role.MEMBER.value, "sk-seed-only")
    uid = md.create_user_with_password(
        tenant["id"], "seed-only@example.test", Role.ADMIN.value,
        "sk-onboard", "pbkdf2_sha256$1$aa$bb")
    assert md.get_user_by_email("seed-only@example.test").id == uid


def test_system_config_roundtrip_and_overwrite(container):
    md = container.metadata
    assert md.get_system_config("litellm_base_url") is None
    md.set_system_config("litellm_base_url", "https://gw.example.com")
    assert md.get_system_config("litellm_base_url") == "https://gw.example.com"
    md.set_system_config("litellm_base_url", "https://other.example.com")
    assert md.get_system_config("litellm_base_url") == "https://other.example.com"


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
