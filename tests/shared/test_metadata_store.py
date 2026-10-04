from datetime import UTC, datetime

import pytest

from app.shared.adapters.postgres.db import transaction
from app.shared.adapters.postgres.metadata_store import PostgresMetadataStore
from app.shared.domain.models import (
    ChunkRecord,
    Document,
    DocumentVersion,
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
        id=new_object_id(),
        tenant_id=tenant_id,
        owner_user_id=owner_id,
        source_type=source,
        blob_path="/tmp/x.pdf",
        content_sha256=sha,
        mime="application/pdf",
        filename="x.pdf",
        visibility="private",
        acl_user_ids=[],
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
    found = md.get_document_by_hash(tenant["id"], "deadbeef", tenant["admin_id"])
    assert found and found.id == doc.id
    assert md.get_document_by_hash("other-tenant", "deadbeef", tenant["admin_id"]) is None


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

    assert md.get_document(other_tenant["id"], tenant_doc.id) is None

    found = md.get_document(other_tenant["id"], global_doc.id)
    assert found is not None and found.id == global_doc.id

    md.replace_document_chunks(
        tenant["id"],
        global_doc.id,
        [
            ChunkRecord(
                ordinal=0,
                modality="text",
                extractor="native",
                route_reason="text",
                token_count=5,
                text="firm-wide knowledge",
            ),
        ],
    )
    chunks = md.get_document_chunks(other_tenant["id"], global_doc.id)
    assert len(chunks) == 1 and chunks[0].text == "firm-wide knowledge"


def test_chunks_replace_get_and_deterministic_ids(container, tenant):
    md = container.metadata
    doc = _doc(tenant["id"], tenant["admin_id"])
    md.create_document(doc)
    recs = [
        ChunkRecord(
            ordinal=0,
            modality="text",
            extractor="pdf_text",
            route_reason="text_layer",
            token_count=5,
            text="hello",
            meta={"page": 1},
        ),
        ChunkRecord(
            ordinal=1,
            modality="table",
            extractor="pdf_table",
            route_reason="structured",
            token_count=9,
            text="| a |",
            meta={},
        ),
    ]
    ids = md.replace_document_chunks(tenant["id"], doc.id, recs)

    assert ids == [f"{doc.id}001", f"{doc.id}002"]

    got = md.get_document_chunks(tenant["id"], doc.id)
    assert [c.ordinal for c in got] == [0, 1]
    assert got[0].content_sha256

    ids2 = md.replace_document_chunks(tenant["id"], doc.id, recs)
    assert ids2 == ids
    assert len(md.get_document_chunks(tenant["id"], doc.id)) == 2


def test_job_lifecycle_and_route_summary(container, tenant):
    md = container.metadata
    doc = _doc(tenant["id"], tenant["admin_id"])
    md.create_document(doc)
    job = Job(
        id=new_object_id(),
        document_id=doc.id,
        tenant_id=tenant["id"],
        stage=JobStage.PARSE.value,
        status=JobStatus.QUEUED.value,
        attempts=0,
    )
    md.create_job(job)
    md.set_route_summary(job.id, {"elements": 3})
    got = md.get_job(tenant["id"], job.id)
    assert got.route_summary == {"elements": 3}
    assert md.get_job("other", job.id) is None


def _recs(n=3):
    return [
        ChunkRecord(
            ordinal=i,
            modality="text",
            extractor="x",
            route_reason="r",
            token_count=1,
            text=f"body {i}",
        )
        for i in range(n)
    ]


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
    assert a[0].id == b[0].id
    assert a[0].content_sha256 != b[0].content_sha256


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


def test_create_user_with_password_then_found_by_email(container, tenant):
    md = container.metadata
    uid = md.create_user_with_password(
        tenant["id"], "new.dev@acme.test", Role.ADMIN.value, "sk-new", "pbkdf2_sha256$1$aa$bb"
    )
    found = md.get_user_by_email("new.dev@acme.test")
    assert found is not None
    assert found.id == uid
    assert found.tenant_id == tenant["id"]
    assert not hasattr(found, "api_token")
    assert found.password_hash == "pbkdf2_sha256$1$aa$bb"

    principal = md.get_principal_by_token("sk-new")
    assert principal is not None
    assert principal.user_id == uid


def test_api_token_is_not_stored_in_plaintext(container, tenant):
    uid = container.metadata.create_user(
        tenant["id"], "hashed.dev@acme.test", Role.MEMBER.value, "sk-plaintext-check"
    )
    with transaction(container.settings.postgres_dsn) as cur:
        cur.execute("SELECT api_token FROM users WHERE id = %s", (uid,))
        stored = cur.fetchone()["api_token"]
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
        tenant["id"], "shared@example.test", Role.ADMIN.value, "sk-shared", "pbkdf2_sha256$1$aa$bb"
    )
    found = md.get_user_by_email("shared@example.test")
    assert found.id == uid
    assert found.password_hash is not None


def test_create_user_with_password_rejects_a_second_account_for_the_same_email(
    container,
    tenant,
    other_tenant,
):
    """Two password-holding accounts for the same email must never coexist,
    even across DIFFERENT tenants (onboarding's get_user_by_email pre-check
    can't fully close this on its own -- see idx_users_email_password_unique
    in schema.sql and app.shared.ports.metadata_store.EmailAlreadyRegistered).
    This is the store-level guard the pre-check races against."""
    md = container.metadata
    md.create_user_with_password(
        tenant["id"], "race@example.test", Role.ADMIN.value, "sk-race-1", "pbkdf2_sha256$1$aa$bb"
    )
    with pytest.raises(EmailAlreadyRegistered):
        md.create_user_with_password(
            other_tenant["id"],
            "race@example.test",
            Role.ADMIN.value,
            "sk-race-2",
            "pbkdf2_sha256$1$cc$dd",
        )

    found = md.get_user_by_email("race@example.test")
    assert found.tenant_id == tenant["id"]


def test_create_user_with_password_allows_same_email_when_prior_account_has_no_password(
    container,
    tenant,
    other_tenant,
):
    """The partial index only guards password-HOLDING rows: a plain
    create_user (seed-script style, no password) sharing an email must not
    block a later onboarding signup for that same email."""
    md = container.metadata
    md.create_user(other_tenant["id"], "seed-only@example.test", Role.MEMBER.value, "sk-seed-only")
    uid = md.create_user_with_password(
        tenant["id"],
        "seed-only@example.test",
        Role.ADMIN.value,
        "sk-onboard",
        "pbkdf2_sha256$1$aa$bb",
    )
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
    md.create_job(
        Job(
            id=new_object_id(),
            document_id=doc.id,
            tenant_id=tenant["id"],
            stage="parse",
            status="queued",
            attempts=0,
        )
    )
    md.replace_document_chunks(
        tenant["id"],
        doc.id,
        [
            ChunkRecord(
                ordinal=0,
                modality="text",
                extractor="t",
                route_reason="r",
                token_count=1,
                text="x",
                meta={},
            )
        ],
    )
    md.delete_document(tenant["id"], doc.id)
    assert md.get_document(tenant["id"], doc.id) is None
    assert md.get_document_chunks(tenant["id"], doc.id) == []


def test_get_document_by_filename_is_scoped_to_the_owner(container, tenant):
    """A shared filename must not let one member overwrite another's document.

    `report.pdf` is an entirely unremarkable name. If the re-ingest target were
    resolved tenant-wide, the second member to upload one would silently
    replace the first member's private document -- and merely returning it
    would leak that the document exists.
    """
    md = container.metadata
    mine = _doc(tenant["id"], tenant["member_id"], sha="sha-mine")
    theirs = _doc(tenant["id"], tenant["viewer_id"], sha="sha-theirs")
    md.create_document(mine)
    md.create_document(theirs)

    found = md.get_document_by_filename(tenant["id"], tenant["member_id"], "x.pdf")
    assert found is not None and found.id == mine.id

    other = md.get_document_by_filename(tenant["id"], tenant["viewer_id"], "x.pdf")
    assert other is not None and other.id == theirs.id

    assert md.get_document_by_filename(tenant["id"], tenant["admin_id"], "x.pdf") is None
    assert md.get_document_by_filename(tenant["id"], tenant["member_id"], "nope.pdf") is None
    assert md.get_document_by_filename(tenant["id"], tenant["member_id"], "") is None


def test_get_document_by_filename_does_not_cross_tenants(container, tenant, other_tenant):
    md = container.metadata
    doc = _doc(tenant["id"], tenant["admin_id"], sha="sha-one")
    md.create_document(doc)
    assert md.get_document_by_filename(other_tenant["id"], tenant["admin_id"], "x.pdf") is None


def test_new_document_starts_at_version_one_with_updated_at_set(container, tenant):
    md = container.metadata
    doc = _doc(tenant["id"], tenant["admin_id"], sha="sha-v1")
    md.create_document(doc)

    stored = md.get_document(tenant["id"], doc.id)
    assert stored.version == 1
    assert stored.updated_at is not None
    assert stored.created_at is not None


def test_update_document_content_bumps_version_and_repoints_storage(container, tenant):
    md = container.metadata
    doc = _doc(tenant["id"], tenant["admin_id"], sha="sha-v1")
    md.create_document(doc)

    version = md.update_document_content(
        tenant["id"],
        doc.id,
        blob_path="/tmp/v2.pdf",
        content_sha256="sha-v2",
        mime="application/pdf",
        source_type="pdf",
        filename="x.pdf",
        visibility="tenant",
        scope="tenant",
    )
    assert version == 2

    stored = md.get_document(tenant["id"], doc.id)
    assert stored.version == 2
    assert stored.content_sha256 == "sha-v2"
    assert stored.blob_path == "/tmp/v2.pdf"

    assert stored.visibility == "tenant"

    assert md.get_document_by_hash(tenant["id"], "sha-v1", tenant["admin_id"]) is None
    assert md.get_document_by_hash(tenant["id"], "sha-v2", tenant["admin_id"]).id == doc.id

    assert (
        md.update_document_content(
            tenant["id"],
            doc.id,
            blob_path="/tmp/v3.pdf",
            content_sha256="sha-v3",
            mime="application/pdf",
            source_type="pdf",
            filename="x.pdf",
            visibility="tenant",
            scope="tenant",
        )
        == 3
    )


def test_update_document_content_may_change_the_source_type(container, tenant):
    """A replacement can legitimately arrive in a different format."""
    md = container.metadata
    doc = _doc(tenant["id"], tenant["admin_id"], sha="sha-md", source="docx")
    md.create_document(doc)

    md.update_document_content(
        tenant["id"],
        doc.id,
        blob_path="/tmp/x.pdf",
        content_sha256="sha-pdf",
        mime="application/pdf",
        source_type="pdf",
        filename="x.pdf",
        visibility="private",
        scope="tenant",
    )
    assert md.get_document(tenant["id"], doc.id).source_type == "pdf"


def test_version_history_is_listed_newest_first(container, tenant):
    md = container.metadata
    doc = _doc(tenant["id"], tenant["admin_id"], sha="sha-v1")
    md.create_document(doc)
    for n, sha in ((1, "sha-v1"), (2, "sha-v2"), (3, "sha-v3")):
        md.add_document_version(
            DocumentVersion(
                id=new_object_id(),
                document_id=doc.id,
                tenant_id=tenant["id"],
                version=n,
                content_sha256=sha,
                blob_path=f"/tmp/{sha}.pdf",
                filename="x.pdf",
                byte_size=100 * n,
                uploaded_by=tenant["admin_id"],
                job_id=f"job{n}",
            )
        )

    versions = md.list_document_versions(tenant["id"], doc.id)
    assert [v.version for v in versions] == [3, 2, 1]
    assert versions[0].byte_size == 300
    assert md.get_document_version(tenant["id"], doc.id, 2).content_sha256 == "sha-v2"
    assert md.get_document_version(tenant["id"], doc.id, 99) is None

    assert versions[0].chunks_added is None
    assert versions[0].delta == {}


def test_set_version_delta_matches_on_job_id(container, tenant):
    md = container.metadata
    doc = _doc(tenant["id"], tenant["admin_id"], sha="sha-v1")
    md.create_document(doc)
    md.add_document_version(
        DocumentVersion(
            id=new_object_id(),
            document_id=doc.id,
            tenant_id=tenant["id"],
            version=1,
            content_sha256="sha-v1",
            blob_path="/tmp/a.pdf",
            filename="x.pdf",
            byte_size=10,
            uploaded_by=tenant["admin_id"],
            job_id="job-1",
        )
    )

    md.set_version_delta(
        "job-1",
        {
            "added": 5,
            "removed": 3,
            "unchanged": 41,
            "added_preview": ["new section"],
            "removed_preview": ["old section"],
        },
    )

    v = md.list_document_versions(tenant["id"], doc.id)[0]
    assert (v.chunks_added, v.chunks_removed, v.chunks_unchanged) == (5, 3, 41)
    assert v.delta["added_preview"] == ["new section"]


def test_set_version_delta_is_a_no_op_for_an_unknown_job(container, tenant):
    """A /reprocess job creates no version row, but still produces a delta.

    The runner calls this unconditionally after every successful job; it must
    not raise when there is nothing to attach the delta to.
    """
    md = container.metadata
    md.set_version_delta("job-that-created-no-version", {"added": 1, "removed": 0})


def test_delete_document_cascades_version_history(container, tenant):
    md = container.metadata
    doc = _doc(tenant["id"], tenant["admin_id"], sha="sha-v1")
    md.create_document(doc)
    md.add_document_version(
        DocumentVersion(
            id=new_object_id(),
            document_id=doc.id,
            tenant_id=tenant["id"],
            version=1,
            content_sha256="sha-v1",
            blob_path="/tmp/a.pdf",
            filename="x.pdf",
            byte_size=10,
            uploaded_by=tenant["admin_id"],
            job_id="job-1",
        )
    )

    md.delete_document(tenant["id"], doc.id)

    assert md.get_document(tenant["id"], doc.id) is None
    assert md.list_document_versions(tenant["id"], doc.id) == []


def test_count_blob_references_sees_live_rows_and_history(container, tenant):
    """Guards the delete path from unlinking a blob someone else still uses.

    Content-addressed storage means identical bytes are one file. Once a
    document's v1 hash is no longer its CURRENT hash, ingest dedup stops
    matching it, so those same bytes can come back as a separate document while
    the first document's history row still names the file.
    """
    md = container.metadata
    shared = "/tmp/shared.pdf"
    mine = _doc(tenant["id"], tenant["admin_id"], sha="sha-mine")
    mine.blob_path = shared
    other = _doc(tenant["id"], tenant["admin_id"], sha="sha-other")
    other.blob_path = shared
    md.create_document(mine)
    md.create_document(other)

    assert md.count_blob_references(shared, mine.id) == 1

    assert md.count_blob_references("/tmp/lonely.pdf", mine.id) == 0

    md.add_document_version(
        DocumentVersion(
            id=new_object_id(),
            document_id=other.id,
            tenant_id=tenant["id"],
            version=1,
            content_sha256="sha-other",
            blob_path=shared,
            filename="x.pdf",
            byte_size=1,
            uploaded_by=tenant["admin_id"],
            job_id="job-x",
        )
    )

    assert md.count_blob_references(shared, mine.id) == 2

    assert md.count_blob_references(shared, other.id) == 1


def test_init_schema_migrates_a_database_that_predates_versioning(storage_settings):

    with transaction(storage_settings.postgres_dsn) as con:
        con.execute("""
        CREATE TABLE documents (
            id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL,
            owner_user_id TEXT NOT NULL, source_type TEXT NOT NULL,
            blob_path TEXT NOT NULL, content_sha256 TEXT NOT NULL,
            mime TEXT, filename TEXT,
            visibility TEXT NOT NULL DEFAULT 'private',
            acl_user_ids JSONB NOT NULL DEFAULT '[]',
            scope TEXT NOT NULL DEFAULT 'tenant',
            extracted_metadata JSONB NOT NULL DEFAULT '{}',
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (tenant_id, content_sha256)
        )""")
        con.execute(
            "INSERT INTO documents (id, tenant_id, owner_user_id, source_type, "
            "blob_path, content_sha256, mime, filename, created_at) "
            "VALUES ('d1','t1','u1','pdf','/tmp/a.pdf','sha-legacy','x','old.pdf',"
            "'2020-01-01 00:00:00')"
        )
    store = PostgresMetadataStore(storage_settings.postgres_dsn)
    store.init_schema()

    doc = store.get_document("t1", "d1")
    assert doc.version == 1
    assert datetime.fromisoformat(doc.updated_at) == datetime(2020, 1, 1, tzinfo=UTC)
    assert datetime.fromisoformat(doc.created_at) == datetime(2020, 1, 1, tzinfo=UTC)

    store.update_document_content(
        "t1",
        "d1",
        blob_path="/tmp/b.pdf",
        content_sha256="sha-new",
        mime="x",
        source_type="pdf",
        filename="old.pdf",
        visibility="private",
        scope="tenant",
    )
    touched = store.get_document("t1", "d1").updated_at
    store.init_schema()
    assert store.get_document("t1", "d1").updated_at == touched
    assert store.get_document("t1", "d1").version == 2
    with transaction(storage_settings.postgres_dsn) as cur:
        cur.execute(
            "INSERT INTO documents (id,tenant_id,owner_user_id,source_type,blob_path,content_sha256) "
            "VALUES ('d2','t1','u2','pdf','/tmp/b.pdf','sha-new')"
        )
    assert store.get_document_by_hash("t1", "sha-new", "u1").id == "d1"
    assert store.get_document_by_hash("t1", "sha-new", "u2").id == "d2"


def test_has_pending_job_is_false_with_no_job_at_all(container, tenant):
    md = container.metadata
    doc = _doc(tenant["id"], tenant["admin_id"], sha="sha-no-job")
    md.create_document(doc)
    assert md.has_pending_job(tenant["id"], doc.id) is False


def test_has_pending_job_true_for_queued_or_running_false_once_terminal(container, tenant):
    md = container.metadata
    doc = _doc(tenant["id"], tenant["admin_id"], sha="sha-pending")
    md.create_document(doc)
    job = Job(
        id=new_object_id(),
        document_id=doc.id,
        tenant_id=tenant["id"],
        stage=JobStage.PARSE.value,
        status=JobStatus.QUEUED.value,
        attempts=0,
    )
    md.create_job(job)
    assert md.has_pending_job(tenant["id"], doc.id) is True

    md.set_route_summary(job.id, {"elements": 1})
    assert md.has_pending_job(tenant["id"], doc.id) is True

    other = _doc(tenant["id"], tenant["admin_id"], sha="sha-other-doc")
    md.create_document(other)
    assert md.has_pending_job(tenant["id"], other.id) is False


def test_create_document_with_job_inserts_both_atomically(container, tenant):
    md = container.metadata
    doc = _doc(tenant["id"], tenant["admin_id"], sha="sha-atomic-create")
    job = Job(
        id=new_object_id(),
        document_id=doc.id,
        tenant_id=tenant["id"],
        stage=JobStage.PARSE.value,
        status=JobStatus.QUEUED.value,
        attempts=0,
    )

    md.create_document_with_job(doc, job)

    stored_doc = md.get_document(tenant["id"], doc.id)
    assert stored_doc is not None and stored_doc.content_sha256 == "sha-atomic-create"

    stored_job = md.get_job(tenant["id"], job.id)
    assert stored_job is not None and stored_job.status == JobStatus.QUEUED.value

    assert stored_job.blob_path == doc.blob_path
    assert stored_job.content_sha256 == doc.content_sha256
    assert stored_job.version == doc.version


def test_update_document_content_and_queue_is_atomic_and_pins_the_new_job(container, tenant):
    md = container.metadata
    doc = _doc(tenant["id"], tenant["admin_id"], sha="sha-v1-atomic")
    md.create_document(doc)

    job = Job(
        id=new_object_id(),
        document_id=doc.id,
        tenant_id=tenant["id"],
        stage=JobStage.PARSE.value,
        status=JobStatus.QUEUED.value,
        attempts=0,
    )

    version = md.update_document_content_and_queue(
        tenant["id"],
        doc.id,
        blob_path="/tmp/v2.pdf",
        content_sha256="sha-v2-atomic",
        mime="application/pdf",
        source_type="pdf",
        filename="x.pdf",
        visibility="tenant",
        scope="tenant",
        job=job,
        uploaded_by=tenant["admin_id"],
        byte_size=1234,
    )
    assert version == 2

    stored_doc = md.get_document(tenant["id"], doc.id)
    assert stored_doc.version == 2
    assert stored_doc.content_sha256 == "sha-v2-atomic"

    stored_job = md.get_job(tenant["id"], job.id)
    assert stored_job.blob_path == "/tmp/v2.pdf"
    assert stored_job.content_sha256 == "sha-v2-atomic"
    assert stored_job.version == 2

    versions = md.list_document_versions(tenant["id"], doc.id)
    assert versions[0].version == 2
    assert versions[0].content_sha256 == "sha-v2-atomic"
    assert versions[0].byte_size == 1234
    assert versions[0].job_id == job.id
