"""API tests via FastAPI TestClient with an injected hermetic container.

Since no worker runs in-process, tests drain the queue manually (claim + run_job)
to exercise the async half deterministically.
"""
import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.pipeline.runner import run_job


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
