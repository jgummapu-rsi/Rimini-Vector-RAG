"""End-to-end tests for re-ingesting an updated document.

Before this feature, `POST /ingest` deduped purely on the SHA-256 of the
uploaded bytes, so an EDITED file had a different hash, fell straight through
the dedup check, and became a second, unrelated document. That is a correctness
bug rather than clutter: both versions stayed retrievable, so a query could be
answered from the superseded document and cite it as authoritative.

The tests below pin the three outcomes `/ingest` now distinguishes (created /
updated / deduplicated), the ACL rules that decide which document an upload is
an update OF, and -- most importantly -- that superseded content actually stops
being retrievable rather than merely being marked old.

Idiom follows tests/api/test_api.py: a TestClient over the hermetic container,
with the queue drained by hand because no worker runs in-process.
"""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.ingest.pipeline.runner import run_job
from app.ingest.pipeline.safety import UnsafeContentError
from app.shared.domain.models import Document
from app.shared.ids import new_object_id

_V1 = b"""# Employee Handbook

## Travel Policy
Employees must book travel through the corporate portal at least fourteen days
in advance. Reimbursement requires original receipts submitted within a month.

## Expenses
Expense reports are filed monthly and approved by the reporting manager.
"""

_V2 = b"""# Employee Handbook

## Remote Work
Employees may work remotely up to three days each week with manager approval.
Home office equipment is provided on request through the facilities desk.

## Expenses
Expense reports are filed monthly and approved by the reporting manager.
"""


@pytest.fixture
def client(container):
    with TestClient(create_app(container)) as c:
        yield c


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _drain(container):
    while (job := container.queue.claim_next()) is not None:
        run_job(container, job)


def _ingest(client, container, token, data, filename="handbook.md", **form):
    """Upload and fully process one file; return the /ingest response body."""
    r = client.post("/ingest", files={"file": (filename, data)}, data=form, headers=_auth(token))
    assert r.status_code in (200, 202), r.text
    _drain(container)
    return r.json()


def _chunk_texts(client, token, doc_id):
    r = client.get(f"/documents/{doc_id}/chunks?preview=2000", headers=_auth(token))
    assert r.status_code == 200, r.text
    return " ".join(c["preview"] for c in r.json()["chunks"])


def test_same_filename_new_bytes_updates_in_place(client, container, tenant):
    """The headline case: edit a file, re-upload it, get ONE document at v2."""
    first = _ingest(client, container, tenant["member_token"], _V1)
    assert first["version"] == 1
    assert first.get("created") is True

    second = _ingest(client, container, tenant["member_token"], _V2)

    assert second["document_id"] == first["document_id"]
    assert second["updated"] is True
    assert second["version"] == 2
    assert second["previous_version"] == 1

    listing = client.get("/documents", headers=_auth(tenant["member_token"])).json()
    handbooks = [d for d in listing["documents"] if d["filename"] == "handbook.md"]
    assert len(handbooks) == 1
    assert handbooks[0]["version"] == 2


def test_update_stamps_updated_at_without_moving_created_at(client, container, tenant):
    first = _ingest(client, container, tenant["member_token"], _V1)
    doc_id = first["document_id"]
    before = client.get(f"/documents/{doc_id}", headers=_auth(tenant["member_token"])).json()

    _ingest(client, container, tenant["member_token"], _V2)

    after = client.get(f"/documents/{doc_id}", headers=_auth(tenant["member_token"])).json()
    assert after["created_at"] == before["created_at"]
    assert after["updated_at"] >= before["updated_at"]
    assert after["version"] == 2


def test_identical_bytes_are_deduplicated_without_a_job(client, container, tenant):
    """Re-uploading an unchanged file must not bump the version or re-run anything."""
    first = _ingest(client, container, tenant["member_token"], _V1)

    r = client.post(
        "/ingest", files={"file": ("handbook.md", _V1)}, headers=_auth(tenant["member_token"])
    )
    body = r.json()

    assert body["deduplicated"] is True
    assert body["document_id"] == first["document_id"]
    assert body["version"] == 1
    assert "job_id" not in body
    assert container.queue.claim_next() is None


def test_explicit_document_id_updates_across_a_rename(client, container, tenant):
    """A caller that knows the document id isn't bound by the filename match."""
    first = _ingest(client, container, tenant["member_token"], _V1, filename="handbook.md")

    second = _ingest(
        client,
        container,
        tenant["member_token"],
        _V2,
        filename="handbook-2026.md",
        document_id=first["document_id"],
    )

    assert second["document_id"] == first["document_id"]
    assert second["version"] == 2
    detail = client.get(
        f"/documents/{first['document_id']}", headers=_auth(tenant["member_token"])
    ).json()
    assert detail["filename"] == "handbook-2026.md"


def test_a_different_filename_creates_a_separate_document(client, container, tenant):
    first = _ingest(client, container, tenant["member_token"], _V1, filename="handbook.md")
    second = _ingest(client, container, tenant["member_token"], _V2, filename="policies.md")

    assert second["document_id"] != first["document_id"]
    assert second["version"] == 1


def test_one_members_upload_never_overwrites_anothers(client, container, tenant):
    """A shared filename is unremarkable; it must not be a write capability.

    Filename matching is owner-scoped precisely so that the second person in a
    tenant to upload a `handbook.md` gets their own document rather than
    silently replacing a colleague's.
    """
    theirs = _ingest(client, container, tenant["member_token"], _V1)
    mine = _ingest(client, container, tenant["admin_token"], _V2)

    assert mine["document_id"] != theirs["document_id"]
    assert mine["version"] == 1
    assert theirs["version"] == 1

    assert "Travel Policy" in _chunk_texts(client, tenant["member_token"], theirs["document_id"])


def test_non_owner_cannot_update_via_an_explicit_document_id(client, container, tenant):
    """Read access to a document is not permission to replace its contents."""
    theirs = _ingest(client, container, tenant["member_token"], _V1, visibility="tenant")

    r = client.post(
        "/ingest",
        files={"file": ("other.md", _V2)},
        data={"document_id": theirs["document_id"]},
        headers=_auth(tenant["viewer_token"]),
    )
    assert r.status_code == 403

    unknown = client.post(
        "/ingest",
        files={"file": ("other.md", _V2)},
        data={"document_id": "0" * 24},
        headers=_auth(tenant["member_token"]),
    )
    assert unknown.status_code == 404


def test_a_member_cannot_replace_a_tenant_shared_document_they_do_not_own(
    client, container, tenant
):
    """Replacing CONTENT is a stronger act than /reprocess, and needs stronger auth.

    `/reprocess` only needs visible+own-tenant, which is fine because it re-runs
    the pipeline over bytes that do not change. Rewriting a document shared with
    the whole tenant, under its original author's name, is not the same thing --
    so the update path additionally requires ownership (or tenant admin).
    """
    theirs = _ingest(
        client, container, tenant["admin_token"], _V1, filename="shared.md", visibility="tenant"
    )
    doc_id = theirs["document_id"]

    assert (
        client.get(f"/documents/{doc_id}", headers=_auth(tenant["member_token"])).status_code == 200
    )

    r = client.post(
        "/ingest",
        files={"file": ("shared.md", _V2)},
        data={"document_id": doc_id},
        headers=_auth(tenant["member_token"]),
    )
    assert r.status_code == 404

    assert "Travel Policy" in _chunk_texts(client, tenant["admin_token"], doc_id)

    ok = client.post(
        "/ingest",
        files={"file": ("shared.md", _V2)},
        data={"document_id": doc_id},
        headers=_auth(tenant["admin_token"]),
    )
    assert ok.status_code == 202
    assert ok.json()["version"] == 2


def test_update_does_not_reach_across_tenants(client, container, tenant, other_tenant):
    mine = _ingest(client, container, tenant["member_token"], _V1)

    r = client.post(
        "/ingest",
        files={"file": ("handbook.md", _V2)},
        data={"document_id": mine["document_id"]},
        headers=_auth(other_tenant["admin_token"]),
    )
    assert r.status_code == 404


def test_being_able_to_READ_a_global_document_does_not_allow_replacing_it(
    client, container, tenant, other_tenant
):
    """`_owned_or_404` on the update path, exercised where it actually bites.

    A scope=global document is readable from every tenant, so `_visible_or_404`
    passes for an outsider. Only the second check stops them overwriting the
    platform's shared knowledge base with their own file.
    """

    blob = container.blob.put(tenant["id"], hashlib.sha256(_V1).hexdigest(), ".md", _V1)
    doc = Document(
        id=new_object_id(),
        tenant_id=tenant["id"],
        owner_user_id=tenant["admin_id"],
        source_type="docx",
        blob_path=blob,
        content_sha256="sha-global",
        mime="text/markdown",
        filename="global-handbook.md",
        visibility="tenant",
        acl_user_ids=[],
        scope="global",
    )
    container.metadata.create_document(doc)

    assert (
        client.get(f"/documents/{doc.id}", headers=_auth(other_tenant["admin_token"])).status_code
        == 200
    )

    r = client.post(
        "/ingest",
        files={"file": ("global-handbook.md", _V2)},
        data={"document_id": doc.id},
        headers=_auth(other_tenant["admin_token"]),
    )
    assert r.status_code == 404

    theirs = _ingest(
        client, container, other_tenant["admin_token"], _V2, filename="global-handbook.md"
    )
    assert theirs["document_id"] != doc.id
    assert theirs["version"] == 1


def test_colliding_bytes_are_rejected_with_a_conflict(client, container, tenant):
    """`documents` carries UNIQUE (tenant_id, content_sha256).

    Pointing one document at bytes another already owns would otherwise surface
    as a raw IntegrityError from inside the adapter.
    """
    _ingest(client, container, tenant["member_token"], _V1, filename="handbook.md")
    _ingest(client, container, tenant["member_token"], _V2, filename="policies.md")

    r = client.post(
        "/ingest", files={"file": ("handbook.md", _V2)}, headers=_auth(tenant["member_token"])
    )
    assert r.status_code == 409
    assert "already ingested" in r.json()["detail"]


def test_removed_content_disappears_and_new_content_appears(client, container, tenant):
    first = _ingest(client, container, tenant["member_token"], _V1)
    doc_id = first["document_id"]

    assert "Travel Policy" in _chunk_texts(client, tenant["member_token"], doc_id)
    assert "Remote Work" not in _chunk_texts(client, tenant["member_token"], doc_id)

    _ingest(client, container, tenant["member_token"], _V2)

    text = _chunk_texts(client, tenant["member_token"], doc_id)
    assert "Remote Work" in text
    assert "Travel Policy" not in text
    assert "Expense reports" in text


def test_superseded_content_is_no_longer_retrievable(client, container, tenant):
    """The correctness claim. Chunks are one thing; the SEARCH INDEX is the thing
    that decides what an answer can be grounded in and cite."""
    _ingest(client, container, tenant["member_token"], _V1)
    _ingest(client, container, tenant["member_token"], _V2)

    r = client.post(
        "/query",
        json={"question": "travel booking portal receipts", "top_k": 10},
        headers=_auth(tenant["member_token"]),
    )
    assert r.status_code == 200, r.text
    assert "corporate portal" not in " ".join(r.json()["contexts"])

    r2 = client.post(
        "/query",
        json={"question": "working remotely from home", "top_k": 10},
        headers=_auth(tenant["member_token"]),
    )
    assert "remotely" in " ".join(r2.json()["contexts"]).lower()


def test_vector_count_reflects_only_the_current_version(client, container, tenant):
    """Tombstoned vectors from the old version must not still be counted."""
    first = _ingest(client, container, tenant["member_token"], _V1)
    doc_id = first["document_id"]

    _ingest(client, container, tenant["member_token"], _V2)

    chunks = client.get(f"/documents/{doc_id}/chunks", headers=_auth(tenant["member_token"])).json()
    assert container.vectors.count(tenant["id"]) == chunks["chunk_count"]


def test_empty_replacement_preserves_the_last_good_generation(client, container, tenant):
    """The empty-replacement case that `_stage_upsert` used to get wrong.

    It returned early on an empty chunk list BEFORE dropping prior vectors, so a
    corrupt or empty replacement left v1 searchable while the document row
    already reported v2 -- answering from content it claimed to have replaced.
    """
    first = _ingest(client, container, tenant["member_token"], _V1)
    doc_id = first["document_id"]
    assert container.vectors.count(tenant["id"]) > 0

    before = container.metadata.get_document(tenant["id"], doc_id)
    with pytest.raises(UnsafeContentError, match="Empty extraction"):
        _ingest(client, container, tenant["member_token"], b"   \n\n   \n")
    after = container.metadata.get_document(tenant["id"], doc_id)
    assert after.version == 2 and after.indexed_version == 1
    assert after.active_generation_id == before.active_generation_id
    assert container.vectors.count(tenant["id"]) > 0
    r = client.post(
        "/query",
        json={"question": "travel booking portal", "top_k": 5},
        headers=_auth(tenant["member_token"]),
    )
    assert r.json()["contexts"]


def test_update_preserves_visibility_when_the_field_is_omitted(client, container, tenant):
    """Re-uploading a tenant-shared document must not silently re-privatise it."""
    first = _ingest(client, container, tenant["member_token"], _V1, visibility="tenant")
    doc_id = first["document_id"]
    assert first["visibility"] == "tenant"

    second = _ingest(client, container, tenant["member_token"], _V2)

    assert second["visibility"] == "tenant"
    detail = client.get(f"/documents/{doc_id}", headers=_auth(tenant["member_token"])).json()
    assert detail["visibility"] == "tenant"


def test_update_applies_visibility_when_it_is_supplied(client, container, tenant):
    first = _ingest(client, container, tenant["member_token"], _V1, visibility="tenant")
    second = _ingest(client, container, tenant["member_token"], _V2, visibility="private")

    assert second["visibility"] == "private"
    assert second["document_id"] == first["document_id"]


def test_a_new_document_still_defaults_to_private(client, container, tenant):
    """The create path's default is unchanged by any of the above."""
    body = _ingest(client, container, tenant["member_token"], _V1)
    assert body["visibility"] == "private"


def test_versions_endpoint_lists_history_with_the_chunk_delta(client, container, tenant):
    first = _ingest(client, container, tenant["member_token"], _V1)
    doc_id = first["document_id"]
    _ingest(client, container, tenant["member_token"], _V2)

    r = client.get(f"/documents/{doc_id}/versions", headers=_auth(tenant["member_token"]))
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["version"] == 2
    assert body["count"] == 2
    assert [v["version"] for v in body["versions"]] == [2, 1]

    latest, original = body["versions"]
    assert latest["current"] is True
    assert original["current"] is False
    assert latest["byte_size"] == len(_V2)
    assert latest["uploaded_by"] == tenant["member_id"]

    assert original["chunks_removed"] == 0
    assert original["chunks_added"] > 0
    assert original["chunks_unchanged"] == 0

    assert latest["chunks_added"] > 0
    assert latest["chunks_removed"] > 0
    assert latest["chunks_unchanged"] > 0
    assert any("Remote Work" in p for p in latest["added_preview"])
    assert any("Travel Policy" in p for p in latest["removed_preview"])


def test_chunk_delta_is_recorded_in_the_job_trace(client, container, tenant):
    """The delta lands on the chunk stage's trace detail as well as the version
    row, so a /reprocess -- which creates no version -- still reports one."""
    _ingest(client, container, tenant["member_token"], _V1)
    second = _ingest(client, container, tenant["member_token"], _V2)

    trace = client.get(
        f"/jobs/{second['job_id']}/trace", headers=_auth(tenant["member_token"])
    ).json()
    chunk_stage = next(s for s in trace["stages"] if s["stage"] == "chunk")

    assert chunk_stage["detail"]["added"] > 0
    assert chunk_stage["detail"]["removed"] > 0
    assert chunk_stage["detail"]["unchanged"] > 0
    assert trace["document"]["version"] == 2
    assert trace["document"]["updated_at"] is not None


def test_reprocess_reports_an_all_unchanged_delta(client, container, tenant):
    """Re-running the pipeline over the same bytes should change nothing.

    A free regression signal on chunker changes -- and it must not create a
    spurious second version.
    """
    first = _ingest(client, container, tenant["member_token"], _V1)
    doc_id = first["document_id"]

    r = client.post(f"/documents/{doc_id}/reprocess", headers=_auth(tenant["member_token"]))
    assert r.status_code == 202
    _drain(container)

    trace = client.get(
        f"/jobs/{r.json()['job_id']}/trace", headers=_auth(tenant["member_token"])
    ).json()
    detail = next(s for s in trace["stages"] if s["stage"] == "chunk")["detail"]
    assert detail["added"] == 0
    assert detail["removed"] == 0
    assert detail["unchanged"] == detail["chunks"]

    versions = client.get(
        f"/documents/{doc_id}/versions", headers=_auth(tenant["member_token"])
    ).json()
    assert versions["count"] == 1
    assert versions["version"] == 1


def test_chunks_endpoint_exposes_the_diff_key(client, container, tenant):
    body = _ingest(client, container, tenant["member_token"], _V1)
    chunks = client.get(
        f"/documents/{body['document_id']}/chunks", headers=_auth(tenant["member_token"])
    ).json()["chunks"]

    assert all(c["chunk_id"] for c in chunks)
    assert all(c["content_sha256"] for c in chunks)


def test_versions_endpoint_respects_document_visibility(client, container, tenant):
    private = _ingest(client, container, tenant["member_token"], _V1)

    r = client.get(
        f"/documents/{private['document_id']}/versions", headers=_auth(tenant["viewer_token"])
    )
    assert r.status_code == 404


def test_source_endpoint_serves_exact_versions_and_ranges(client, container, tenant):
    first = _ingest(client, container, tenant["member_token"], _V1)
    _ingest(client, container, tenant["member_token"], _V2)
    path = f"/documents/{first['document_id']}/versions"

    old = client.get(f"{path}/1/content", headers=_auth(tenant["member_token"]))
    assert old.status_code == 200
    assert old.content == _V1
    assert old.headers["accept-ranges"] == "bytes"
    assert old.headers["x-content-type-options"] == "nosniff"

    current = client.get(
        f"{path}/2/content",
        headers={
            **_auth(tenant["member_token"]),
            "Range": "bytes=2-8",
        },
    )
    assert current.status_code == 206
    assert current.content == _V2[2:9]
    assert current.headers["content-range"] == f"bytes 2-8/{len(_V2)}"

    suffix = client.get(
        f"{path}/1/content",
        headers={
            **_auth(tenant["member_token"]),
            "Range": "bytes=-8",
        },
    )
    assert suffix.content == _V1[-8:]


def test_source_endpoint_hides_private_documents(client, container, tenant):
    first = _ingest(client, container, tenant["member_token"], _V1)
    path = f"/documents/{first['document_id']}/versions/1/content"
    assert client.get(path, headers=_auth(tenant["viewer_token"])).status_code == 404


def test_source_endpoint_rejects_invalid_ranges(client, container, tenant):
    first = _ingest(client, container, tenant["member_token"], _V1)
    path = f"/documents/{first['document_id']}/versions/1/content"
    response = client.get(
        path,
        headers={
            **_auth(tenant["member_token"]),
            "Range": "bytes=999999-",
        },
    )
    assert response.status_code == 416
    assert response.headers["content-range"] == f"bytes */{len(_V1)}"


def test_delete_removes_version_history(client, container, tenant):
    first = _ingest(client, container, tenant["member_token"], _V1)
    doc_id = first["document_id"]
    _ingest(client, container, tenant["member_token"], _V2)

    r = client.delete(f"/documents/{doc_id}", headers=_auth(tenant["admin_token"]))
    assert r.status_code == 200

    assert (
        client.get(
            f"/documents/{doc_id}/versions", headers=_auth(tenant["admin_token"])
        ).status_code
        == 404
    )
    assert container.metadata.list_document_versions(tenant["id"], doc_id) == []


def test_delete_keeps_a_blob_another_document_still_references(client, container, tenant):
    """Content-addressed blobs are shared, and versioning makes sharing reachable.

    Once handbook.md's v1 bytes are no longer its CURRENT hash, ingest dedup
    stops matching them -- so the same bytes can legitimately come back as a
    separate document while the first document's v1 history row still names the
    file. Deleting the first document must not empty out the second.
    """
    handbook = _ingest(client, container, tenant["member_token"], _V1, filename="handbook.md")
    _ingest(client, container, tenant["member_token"], _V2, filename="handbook.md")

    twin = _ingest(client, container, tenant["member_token"], _V1, filename="archive.md")
    assert twin["document_id"] != handbook["document_id"]

    twin_doc = container.metadata.get_document(tenant["id"], twin["document_id"])
    shared_blob = twin_doc.blob_path

    r = client.delete(f"/documents/{handbook['document_id']}", headers=_auth(tenant["admin_token"]))
    assert r.status_code == 200

    assert container.blob.get(shared_blob)
    assert "Travel Policy" in _chunk_texts(client, tenant["member_token"], twin["document_id"])
