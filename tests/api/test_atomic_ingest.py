import hashlib
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.shared.adapters.postgres.db import transaction
from app.shared.domain.models import Document, DocumentVersion, Job
from app.shared.ids import new_object_id
from app.shared.ports.metadata_store import IngestionConflict


def _intent(tenant, content, filename="invoice.txt"):
    document = Document(
        new_object_id(),
        tenant["id"],
        tenant["member_id"],
        "text",
        "blob",
        hashlib.sha256(content).hexdigest(),
        "text/plain",
        filename,
        "private",
        [],
    )
    job = Job(new_object_id(), document.id, tenant["id"], "parse", "queued", 0)
    version = DocumentVersion(
        new_object_id(),
        document.id,
        tenant["id"],
        1,
        document.content_sha256,
        document.blob_path,
        filename,
        len(content),
        tenant["member_id"],
        job.id,
    )
    return document, job, version


def test_initial_history_failure_rolls_back_document_and_job(container, tenant):
    with transaction(container.settings.postgres_dsn) as cur:
        cur.execute(
            "CREATE FUNCTION reject_initial_version() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'history unavailable'; END $$"
        )
        cur.execute(
            "CREATE TRIGGER reject_initial_version BEFORE INSERT ON document_versions FOR EACH ROW EXECUTE FUNCTION reject_initial_version()"
        )
    document, job, version = _intent(tenant, b"invoice 00123")
    with pytest.raises(Exception, match="history unavailable"):
        container.metadata.create_document_with_job(document, job, version)
    assert container.metadata.get_document(tenant["id"], document.id) is None
    assert container.metadata.get_job(tenant["id"], job.id) is None
    assert container.queue.claim_next() is None


@pytest.mark.parametrize("same_filename,same_content", [(True, False), (False, True), (True, True)])
def test_concurrent_first_uploads_have_one_complete_winner(
    container, tenant, same_filename, same_content
):
    barrier = Barrier(2)

    def create(index):
        intent = _intent(
            tenant,
            b"same" if same_content else str(index).encode(),
            "same.txt" if same_filename else f"{index}.txt",
        )
        barrier.wait()
        try:
            container.metadata.create_document_with_job(*intent)
            return intent[0].id
        except IngestionConflict:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(create, (0, 1)))
    winners = [document_id for document_id in results if document_id is not None]
    assert len(winners) == 1
    versions = container.metadata.list_document_versions(tenant["id"], winners[0])
    assert len(versions) == 1 and versions[0].version == 1
    assert container.metadata.get_job(tenant["id"], versions[0].job_id) is not None
    with transaction(container.settings.postgres_dsn) as cur:
        for table in ("documents", "ingestion_jobs", "document_versions"):
            cur.execute(f"SELECT count(*) AS n FROM {table}")
            assert cur.fetchone()["n"] == 1


def test_identical_private_bytes_create_owner_scoped_documents(container, tenant):
    def headers(token):
        return {"Authorization": "Bearer " + token}

    with TestClient(create_app(container)) as client:
        first = client.post(
            "/ingest",
            files={"file": ("secret.txt", b"private invoice 00123")},
            headers=headers(tenant["admin_token"]),
        )
        second = client.post(
            "/ingest",
            files={"file": ("mine.txt", b"private invoice 00123")},
            headers=headers(tenant["member_token"]),
        )
        assert first.status_code == second.status_code == 202
        assert first.json()["document_id"] != second.json()["document_id"]
        assert second.json()["created"] is True
        assert (
            client.get(
                "/documents/" + first.json()["document_id"], headers=headers(tenant["member_token"])
            ).status_code
            == 404
        )
    with transaction(container.settings.postgres_dsn) as cur:
        cur.execute("SELECT blob_path FROM documents")
        assert len({row["blob_path"] for row in cur.fetchall()}) == 1


def test_stale_update_does_not_mutate_document_or_history(container, tenant):
    document, job, version = _intent(tenant, b"first")
    container.metadata.create_document_with_job(document, job, version)
    claimed = container.queue.claim_next()
    container.queue.complete(claimed.id, claimed.lease_token)
    container.metadata.update_document_content(
        tenant["id"],
        document.id,
        blob_path="new",
        content_sha256="new",
        mime="text/plain",
        source_type="text",
        filename="invoice.txt",
        visibility="private",
        scope="tenant",
    )
    replacement = Job(new_object_id(), document.id, tenant["id"], "parse", "queued", 0)
    with pytest.raises(IngestionConflict, match="version changed"):
        container.metadata.update_document_content_and_queue(
            tenant["id"],
            document.id,
            blob_path="stale",
            content_sha256="stale",
            mime="text/plain",
            source_type="text",
            filename="invoice.txt",
            visibility="private",
            scope="tenant",
            job=replacement,
            uploaded_by=tenant["member_id"],
            byte_size=5,
            expected_version=1,
        )
    current = container.metadata.get_document(tenant["id"], document.id)
    assert current.version == 2 and current.content_sha256 == "new"
    assert container.metadata.get_job(tenant["id"], replacement.id) is None
    assert len(container.metadata.list_document_versions(tenant["id"], document.id)) == 1
