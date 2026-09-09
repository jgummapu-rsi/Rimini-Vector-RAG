"""API tests via FastAPI TestClient with an injected hermetic container.

Since no worker runs in-process, tests drain the queue manually (claim + run_job)
to exercise the async half deterministically.
"""
import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.shared.domain.models import Role
from app.ingest.pipeline.runner import run_job


@pytest.fixture
def client(container):
    with TestClient(create_app(container)) as c:
        yield c


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _drain(container):
    while (job := container.queue.claim_next()) is not None:
        run_job(container, job)


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ingest_requires_auth(client, files):
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])})
    assert r.status_code == 401
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    headers=_auth("bogus"))
    assert r.status_code == 401


def test_viewer_cannot_ingest(client, tenant, files):
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    headers=_auth(tenant["viewer_token"]))
    assert r.status_code == 403


def test_unsupported_type_rejected(client, tenant):
    r = client.post("/ingest", files={"file": ("bad.zzz", b"x")},
                    headers=_auth(tenant["member_token"]))
    assert r.status_code == 415


def test_ingest_flow_and_chunks(client, container, tenant, files):
    r = client.post("/ingest", files={"file": ("report.docx", files["report.docx"])},
                    headers=_auth(tenant["member_token"]))
    assert r.status_code == 202
    body = r.json()
    doc_id, job_id = body["document_id"], body["job_id"]

    _drain(container)

    j = client.get(f"/jobs/{job_id}", headers=_auth(tenant["member_token"]))
    assert j.json()["status"] == "done"

    ch = client.get(f"/documents/{doc_id}/chunks", headers=_auth(tenant["member_token"]))
    data = ch.json()
    assert data["chunk_count"] >= 1
    assert {c["modality"] for c in data["chunks"]} & {"text", "table"}


def test_job_status_includes_route_summary_and_timestamps(client, container, tenant, files):
    r = client.post("/ingest", files={"file": ("report.docx", files["report.docx"])},
                    headers=_auth(tenant["member_token"]))
    job_id = r.json()["job_id"]
    _drain(container)

    j = client.get(f"/jobs/{job_id}", headers=_auth(tenant["member_token"])).json()
    assert j["status"] == "done"
    assert j["route_summary"] is not None
    assert j["created_at"] is not None
    assert j["updated_at"] is not None


def test_query_response_includes_citations_and_trace(client, container, tenant, files, monkeypatch):
    monkeypatch.setattr(container.gateway, "chat",
                         lambda messages, model, temperature=0.0: "stub answer")
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    headers=_auth(tenant["member_token"]))
    _drain(container)

    q = client.post("/query", json={"question": "quarterly review", "top_k": 5},
                     headers=_auth(tenant["member_token"]))
    body = q.json()
    assert body["trace"] and body["trace"][0]["stage"] == "decompose"
    assert body["citations"]
    citation = body["citations"][0]
    assert citation["filename"] == "notes.txt"
    assert "chunk_id" in citation and "score" in citation


def test_ask_returns_generated_answer_with_citations_and_trace(
        client, container, tenant, files, monkeypatch):
    monkeypatch.setattr(container.gateway, "chat",
                         lambda messages, model, temperature=0.0: "stub answer")
    client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                headers=_auth(tenant["member_token"]))
    _drain(container)

    r = client.post("/ask", json={"question": "quarterly review", "top_k": 5},
                     headers=_auth(tenant["member_token"]))
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "stub answer"
    assert body["grounded"] is True
    assert body["citations"] and body["citations"][0]["filename"] == "notes.txt"
    assert any(t["stage"] == "rerank" for t in body["trace"])


def test_document_metadata_extraction_surfaced_on_get(client, container, tenant, files, monkeypatch):
    monkeypatch.setattr(container.gateway, "chat", lambda messages, model, temperature=0.0: (
        '{"author": "Priya", "date": "2024-03-31", "topics": ["close checklist"], "entities": ["FI"]}'
    ))
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    headers=_auth(tenant["member_token"]))
    doc_id = r.json()["document_id"]
    _drain(container)

    d = client.get(f"/documents/{doc_id}", headers=_auth(tenant["member_token"]))
    assert d.json()["extracted_metadata"] == {
        "author": "Priya", "date": "2024-03-31",
        "topics": ["close checklist"], "entities": ["FI"],
    }


def test_dedup_returns_same_document(client, tenant, files):
    h = _auth(tenant["member_token"])
    r1 = client.post("/ingest", files={"file": ("data.csv", files["data.csv"])}, headers=h)
    r2 = client.post("/ingest", files={"file": ("data.csv", files["data.csv"])}, headers=h)
    assert r2.json().get("deduplicated") is True
    assert r2.json()["document_id"] == r1.json()["document_id"]


def test_tenant_isolation_on_read(client, container, tenant, other_tenant, files):
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    headers=_auth(tenant["member_token"]))
    doc_id = r.json()["document_id"]
    r2 = client.get(f"/documents/{doc_id}", headers=_auth(other_tenant["admin_token"]))
    assert r2.status_code == 404


def test_delete_cascade(client, container, tenant, files):
    r = client.post("/ingest", files={"file": ("finance.xlsx", files["finance.xlsx"])},
                    headers=_auth(tenant["admin_token"]))
    doc_id = r.json()["document_id"]
    _drain(container)
    assert container.vectors.count(tenant["id"]) >= 1

    d = client.delete(f"/documents/{doc_id}", headers=_auth(tenant["admin_token"]))
    assert d.status_code == 200 and d.json()["deleted"] is True
    assert d.json()["vectors_removed"] >= 1
    assert container.vectors.count(tenant["id"]) == 0
    assert client.get(f"/documents/{doc_id}",
                      headers=_auth(tenant["admin_token"])).status_code == 404


def test_private_document_hidden_from_other_user(client, tenant, files):
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    headers=_auth(tenant["member_token"]))
    doc_id = r.json()["document_id"]

    # owner can read it
    assert client.get(f"/documents/{doc_id}",
                       headers=_auth(tenant["member_token"])).status_code == 200
    # admin (sees all in tenant) can read it
    assert client.get(f"/documents/{doc_id}",
                       headers=_auth(tenant["admin_token"])).status_code == 200
    # a different, non-owning user in the same tenant cannot
    r2 = client.get(f"/documents/{doc_id}", headers=_auth(tenant["viewer_token"]))
    assert r2.status_code == 404


def test_private_document_chunks_hidden_from_other_user(client, container, tenant, files):
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    headers=_auth(tenant["member_token"]))
    doc_id = r.json()["document_id"]
    _drain(container)

    assert client.get(f"/documents/{doc_id}/chunks",
                       headers=_auth(tenant["member_token"])).status_code == 200
    r2 = client.get(f"/documents/{doc_id}/chunks", headers=_auth(tenant["viewer_token"]))
    assert r2.status_code == 404


def test_global_scope_publish_and_cross_tenant_read(client, container, tenant, other_tenant,
                                                     files, monkeypatch):
    container.settings.platform_tenant_id = tenant["id"]
    monkeypatch.setattr(container.gateway, "chat",
                        lambda messages, model, temperature=0.0: "stub answer")

    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    data={"scope": "global"}, headers=_auth(tenant["admin_token"]))
    assert r.status_code == 202
    assert r.json()["scope"] == "global"
    doc_id = r.json()["document_id"]
    _drain(container)

    # a user in a completely different tenant can read it and its chunks
    other = _auth(other_tenant["admin_token"])
    d = client.get(f"/documents/{doc_id}", headers=other)
    assert d.status_code == 200 and d.json()["scope"] == "global"
    ch = client.get(f"/documents/{doc_id}/chunks", headers=other)
    assert ch.status_code == 200 and ch.json()["chunk_count"] >= 1

    # and can retrieve it via /query (gateway.chat stubbed; retrieval itself is real)
    q = client.post("/query", json={"question": "quarterly review", "top_k": 5},
                     headers=other)
    assert q.status_code == 200
    assert q.json()["chunk_ids"] and doc_id in q.json()["chunk_ids"][0]


def test_non_platform_tenant_cannot_publish_global(client, tenant, files):
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    data={"scope": "global"}, headers=_auth(tenant["admin_token"]))
    assert r.status_code == 403  # platform_tenant_id unset -> nobody may publish global


def test_other_tenant_cannot_delete_or_reprocess_global_doc(client, container, tenant, other_tenant, files):
    container.settings.platform_tenant_id = tenant["id"]
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    data={"scope": "global"}, headers=_auth(tenant["admin_token"]))
    doc_id = r.json()["document_id"]
    _drain(container)

    other = _auth(other_tenant["admin_token"])
    assert client.post(f"/documents/{doc_id}/reprocess", headers=other).status_code == 404
    assert client.delete(f"/documents/{doc_id}", headers=other).status_code == 404

    # the owning tenant still can
    own = _auth(tenant["admin_token"])
    assert client.delete(f"/documents/{doc_id}", headers=own).status_code == 200


def test_member_cannot_delete(client, tenant, files):
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    headers=_auth(tenant["member_token"]))
    doc_id = r.json()["document_id"]
    d = client.delete(f"/documents/{doc_id}", headers=_auth(tenant["member_token"]))
    assert d.status_code == 403


# --------------------------------------------------------- trace / listing --
# These back the Open WebUI "Document Trace" sidebar section.


def test_job_trace_returns_full_stage_timeline(client, container, tenant, files):
    r = client.post("/ingest", files={"file": ("report.docx", files["report.docx"])},
                    headers=_auth(tenant["member_token"]))
    job_id = r.json()["job_id"]
    _drain(container)

    t = client.get(f"/jobs/{job_id}/trace", headers=_auth(tenant["member_token"]))
    assert t.status_code == 200
    body = t.json()

    assert body["job"]["status"] == "done"
    assert body["job"]["route_summary"] is not None
    assert body["document"]["filename"] == "report.docx"
    assert [s["stage"] for s in body["stages"]] == [
        "parse", "route", "extract", "chunk", "metadata", "embed", "binarize", "upsert"]
    assert all(s["status"] == "ok" for s in body["stages"])
    assert body["chunk_count"] >= 1
    assert body["token_total"] > 0


def test_job_trace_is_available_while_still_queued(client, container, tenant, files):
    """The UI polls from the moment of upload - before any worker has run."""
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    headers=_auth(tenant["member_token"]))
    job_id = r.json()["job_id"]

    body = client.get(f"/jobs/{job_id}/trace",
                      headers=_auth(tenant["member_token"])).json()
    assert body["job"]["status"] == "queued"
    assert body["stages"] == []
    assert body["chunk_count"] == 0


def test_job_trace_hidden_from_other_tenant(client, container, tenant, other_tenant, files):
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    headers=_auth(tenant["member_token"]))
    job_id = r.json()["job_id"]
    _drain(container)

    r2 = client.get(f"/jobs/{job_id}/trace", headers=_auth(other_tenant["admin_token"]))
    assert r2.status_code == 404


def test_job_trace_respects_document_visibility(client, container, tenant, files):
    """A private document's trace must not leak to a non-owner in the same tenant."""
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    headers=_auth(tenant["member_token"]))
    job_id = r.json()["job_id"]
    _drain(container)

    assert client.get(f"/jobs/{job_id}/trace",
                      headers=_auth(tenant["member_token"])).status_code == 200
    r2 = client.get(f"/jobs/{job_id}/trace", headers=_auth(tenant["viewer_token"]))
    assert r2.status_code == 404


def test_list_documents_newest_first_with_job_status(client, container, tenant, files):
    h = _auth(tenant["member_token"])
    client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])}, headers=h)
    client.post("/ingest", files={"file": ("data.csv", files["data.csv"])}, headers=h)
    _drain(container)

    body = client.get("/documents", headers=h).json()
    names = [d["filename"] for d in body["documents"]]
    assert set(names) == {"notes.txt", "data.csv"}
    assert all(d["status"] == "done" for d in body["documents"])
    assert all(d["job_id"] for d in body["documents"])
    assert body["count"] == 2


def test_list_documents_applies_acl(client, container, tenant, files):
    """member ingests privately; viewer must not see it, admin must."""
    client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                headers=_auth(tenant["member_token"]))
    _drain(container)

    viewer = client.get("/documents", headers=_auth(tenant["viewer_token"])).json()
    assert viewer["documents"] == []

    admin = client.get("/documents", headers=_auth(tenant["admin_token"])).json()
    assert [d["filename"] for d in admin["documents"]] == ["notes.txt"]


def test_list_documents_isolated_across_tenants(client, container, tenant, other_tenant, files):
    client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                headers=_auth(tenant["member_token"]))
    _drain(container)

    body = client.get("/documents", headers=_auth(other_tenant["admin_token"])).json()
    assert body["documents"] == []


def test_trace_ui_is_served(client):
    r = client.get("/ui/trace/")
    assert r.status_code == 200
    assert "Document Trace" in r.text


def test_delete_removes_document_from_listing_and_trace(client, container, tenant, files):
    """The trace panel's delete button claims it removes the chunks, the vectors
    and the ingestion trace. Hold the API to that."""
    h = _auth(tenant["admin_token"])
    r = client.post("/ingest", files={"file": ("report.docx", files["report.docx"])}, headers=h)
    doc_id, job_id = r.json()["document_id"], r.json()["job_id"]
    _drain(container)

    assert client.get(f"/jobs/{job_id}/trace", headers=h).json()["chunk_count"] >= 1
    assert container.metadata.get_job_events(tenant["id"], job_id)

    d = client.delete(f"/documents/{doc_id}", headers=h)
    assert d.status_code == 200 and d.json()["vectors_removed"] >= 1

    assert client.get(f"/jobs/{job_id}/trace", headers=h).status_code == 404
    assert client.get(f"/documents/{doc_id}/chunks", headers=h).status_code == 404
    assert container.metadata.get_job_events(tenant["id"], job_id) == []
    assert [x["document_id"] for x in client.get("/documents", headers=h).json()["documents"]] == []


def test_non_admin_cannot_delete(client, container, tenant, files):
    """The panel surfaces a specific message for this; make sure the API is the
    thing actually enforcing it."""
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    headers=_auth(tenant["member_token"]))
    doc_id = r.json()["document_id"]
    _drain(container)

    d = client.delete(f"/documents/{doc_id}", headers=_auth(tenant["member_token"]))
    assert d.status_code == 403
    assert "cannot delete" in d.json()["detail"]



# ------------------------------------------------------------ authorization --
# Three endpoints were under-guarded: /metrics took no credentials at all, and
# /jobs/{id} and /reprocess checked tenant membership without checking whether
# the caller was allowed to see the DOCUMENT behind the job.


def test_metrics_requires_an_admin_token(client, tenant):
    """/metrics counters are deployment-wide, not tenant-scoped. It used to be
    readable with no credentials whatsoever."""
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers=_auth("bogus")).status_code == 401
    assert client.get("/metrics", headers=_auth(tenant["member_token"])).status_code == 403
    assert client.get("/metrics", headers=_auth(tenant["viewer_token"])).status_code == 403

    ok = client.get("/metrics", headers=_auth(tenant["admin_token"]))
    assert ok.status_code == 200 and isinstance(ok.json(), dict)


def test_healthz_stays_open_for_load_balancer_probes(client):
    """Deliberately unauthenticated -- it reports liveness, not data."""
    assert client.get("/healthz").status_code == 200


def test_job_status_respects_document_visibility(client, container, tenant, files):
    """A job carries the document id, its pipeline stage and any error text, so
    it is exactly as sensitive as the document behind it. /jobs/{id}/trace
    already enforced this; /jobs/{id} did not."""
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    headers=_auth(tenant["member_token"]))
    job_id = r.json()["job_id"]
    _drain(container)

    assert client.get(f"/jobs/{job_id}", headers=_auth(tenant["member_token"])).status_code == 200
    assert client.get(f"/jobs/{job_id}", headers=_auth(tenant["admin_token"])).status_code == 200
    # a non-owning member of the same tenant must not be able to watch it
    assert client.get(f"/jobs/{job_id}", headers=_auth(tenant["viewer_token"])).status_code == 404


def test_job_status_still_hidden_from_another_tenant(client, container, tenant,
                                                      other_tenant, files):
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    headers=_auth(tenant["member_token"]))
    job_id = r.json()["job_id"]
    assert client.get(f"/jobs/{job_id}",
                      headers=_auth(other_tenant["admin_token"])).status_code == 404


def test_reprocess_respects_document_visibility(client, container, tenant, files):
    """Reprocessing burns real work (re-parse, re-embed, LLM calls). A member
    who cannot even see the document must not be able to trigger it."""
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    headers=_auth(tenant["member_token"]))
    doc_id = r.json()["document_id"]
    _drain(container)

    # the owner can
    assert client.post(f"/documents/{doc_id}/reprocess",
                       headers=_auth(tenant["member_token"])).status_code == 202
    # a viewer can't ingest at all, so use a second member who doesn't own this
    # private document
    container.metadata.create_user(tenant["id"], "stranger@acme.test", Role.MEMBER.value, "sk-stranger")
    assert client.post(f"/documents/{doc_id}/reprocess",
                       headers=_auth("sk-stranger")).status_code == 404


# --------------------------------------------------------- ingest visibility --
# /ingest used to hardcode visibility=private, so a document could only be made
# tenant-readable by editing the database. With per-user identity on (the Open
# WebUI pipe), that meant a shared knowledge base was unreachable by everyone
# except whoever uploaded it.


def test_ingest_defaults_to_private(client, container, tenant, files):
    """The default must not change: a personal upload stays personal unless
    tenant-wide is explicitly asked for."""
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    headers=_auth(tenant["member_token"]))
    assert r.json()["visibility"] == "private"
    doc = container.metadata.get_document(tenant["id"], r.json()["document_id"])
    assert doc.visibility == "private"


def test_ingest_can_publish_tenant_wide(client, container, tenant, files):
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    data={"visibility": "tenant"},
                    headers=_auth(tenant["member_token"]))
    assert r.status_code == 202
    assert r.json()["visibility"] == "tenant"
    doc = container.metadata.get_document(tenant["id"], r.json()["document_id"])
    assert doc.visibility == "tenant"


def test_tenant_wide_document_is_readable_by_another_user(client, container, tenant, files):
    """The point of the option: someone who did NOT upload it can retrieve it."""
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    data={"visibility": "tenant"},
                    headers=_auth(tenant["member_token"]))
    doc_id = r.json()["document_id"]
    _drain(container)

    assert client.get(f"/documents/{doc_id}",
                      headers=_auth(tenant["viewer_token"])).status_code == 200
    assert client.get(f"/documents/{doc_id}/chunks",
                      headers=_auth(tenant["viewer_token"])).status_code == 200
    listing = client.get("/documents", headers=_auth(tenant["viewer_token"])).json()
    assert [d["filename"] for d in listing["documents"]] == ["notes.txt"]


def test_tenant_wide_document_is_retrievable_by_another_user(
        client, container, tenant, files, monkeypatch):
    """Retrieval reads the visibility copy denormalised into the vector payload,
    not the documents row -- so this asserts the ingest path writes both."""
    monkeypatch.setattr(container.gateway, "chat",
                        lambda messages, model, temperature=0.0: "stub answer")
    client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                data={"visibility": "tenant"}, headers=_auth(tenant["member_token"]))
    _drain(container)

    body = client.post("/query", json={"question": "quarterly review"},
                       headers=_auth(tenant["viewer_token"])).json()
    assert body["contexts"], "a tenant-wide document must be retrievable by a non-owner"


def test_tenant_wide_document_still_does_not_cross_tenants(
        client, container, tenant, other_tenant, files):
    """`visibility` is an INTRA-tenant control; `scope` is the cross-tenant one."""
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    data={"visibility": "tenant"},
                    headers=_auth(tenant["member_token"]))
    doc_id = r.json()["document_id"]
    _drain(container)
    assert client.get(f"/documents/{doc_id}",
                      headers=_auth(other_tenant["admin_token"])).status_code == 404


@pytest.mark.parametrize("bad", ["public", "shared", "TENANT", "", "everyone"])
def test_ingest_rejects_an_invalid_visibility(client, tenant, files, bad):
    """`shared` is rejected too: it is only meaningful with an acl_user_ids list,
    which this endpoint has no way to supply, so accepting it would silently
    create a document shared with nobody."""
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    data={"visibility": bad}, headers=_auth(tenant["member_token"]))
    assert r.status_code == 400
    assert "visibility" in r.json()["detail"]


def test_visibility_and_scope_are_independent(client, container, tenant, files):
    container.settings.platform_tenant_id = tenant["id"]
    r = client.post("/ingest", files={"file": ("notes.txt", files["notes.txt"])},
                    data={"scope": "global", "visibility": "tenant"},
                    headers=_auth(tenant["admin_token"]))
    assert r.status_code == 202
    body = r.json()
    assert body["scope"] == "global" and body["visibility"] == "tenant"


# ------------------------------------------------------------ upload limits --


def test_oversized_upload_is_rejected(client, container, tenant):
    """MAX_UPLOAD_MB used to be checked only AFTER the whole body had been read
    into memory, so the limit did not actually bound anything."""
    container.settings.max_upload_mb = 1
    big = b"x" * (2 * 1024 * 1024)
    r = client.post("/ingest", files={"file": ("big.txt", big)},
                    headers=_auth(tenant["member_token"]))
    assert r.status_code == 413


def test_upload_at_the_limit_is_accepted(client, container, tenant):
    """The cap must be a boundary, not an off-by-one rejection of valid files."""
    container.settings.max_upload_mb = 1
    just_under = b"a" * (1024 * 1024 - 5000)
    r = client.post("/ingest", files={"file": ("ok.txt", just_under)},
                    headers=_auth(tenant["member_token"]))
    assert r.status_code == 202


def test_empty_upload_is_still_rejected(client, tenant):
    r = client.post("/ingest", files={"file": ("empty.txt", b"")},
                    headers=_auth(tenant["member_token"]))
    assert r.status_code == 400


