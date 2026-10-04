import pytest

from app.shared.adapters.postgres.db import transaction
from app.shared.ports.vector_store import VectorPoint


def _pt(cid, tid, vec, **payload):
    payload.setdefault("_id", "docX")
    payload.setdefault("content", "some text")
    return VectorPoint(chunk_id=cid, tenant_id=tid, vector=vec, payload=payload)


def test_ensure_collection_writes_meta(container):
    vs = container.vectors
    with transaction(container.settings.postgres_dsn) as cur:
        cur.execute(
            "SELECT atttypmod FROM pg_attribute WHERE attrelid='vector_chunks'::regclass AND attname='embedding'"
        )
        assert cur.fetchone()["atttypmod"] == vs.dim


def test_upsert_count_and_search(container):
    vs = container.vectors
    d = vs.dim
    a = [1.0] + [0.0] * (d - 1)
    b = [0.0, 1.0] + [0.0] * (d - 2)
    vs.upsert(
        [
            _pt("d1001", "T1", a, _id="d1", modality="text", source_type="pdf"),
            _pt("d1002", "T1", b, _id="d1", modality="table", source_type="csv"),
        ]
    )
    assert vs.count("T1") == 2

    hits = vs.search("T1", a, top_k=2)
    assert hits[0].chunk_id == "d1001"
    assert hits[0].score > hits[1].score

    assert hits[0].payload["_id"] == "d1"
    assert hits[0].payload["content"]
    assert "user_id" in hits[0].payload
    assert "embeddings" not in hits[0].payload


def test_tenant_isolation_in_count_and_search(container):
    vs = container.vectors
    v = [1.0] + [0.0] * (vs.dim - 1)
    vs.upsert([_pt("p1", "T1", v)])
    assert vs.count("T2") == 0
    assert vs.search("T2", v, top_k=5) == []


def test_delete_by_document_tombstones(container):
    vs = container.vectors
    v = [1.0] + [0.0] * (vs.dim - 1)
    vs.upsert(
        [
            _pt("d1001", "T1", v, _id="d1"),
            _pt("d2001", "T1", v, _id="d2"),
        ]
    )
    removed = vs.delete_by_document("T1", "d1")
    assert removed == 1
    assert vs.count("T1") == 1
    assert all(h.payload["_id"] == "d2" for h in vs.search("T1", v, top_k=5))


def test_each_chunk_persists_its_document_evidence(container):
    vs = container.vectors
    d = vs.dim
    vs.upsert(
        [
            _pt(
                "d1001",
                "T1",
                [1.0] + [0.0] * (d - 1),
                _id="d1",
                user_id="u1",
                modality="text",
                content="chunk one",
            ),
            _pt(
                "d1002",
                "T1",
                [0.0, 1.0] + [0.0] * (d - 2),
                _id="d1",
                user_id="u1",
                modality="table",
                content="chunk two",
            ),
        ]
    )
    with transaction(container.settings.postgres_dsn) as cur:
        cur.execute(
            "SELECT chunk_id, document_id, payload, vector_dims(embedding) AS dim FROM vector_chunks ORDER BY chunk_id"
        )
        rows = cur.fetchall()
    assert [row["chunk_id"] for row in rows] == ["d1001", "d1002"]
    assert [row["payload"]["modality"] for row in rows] == ["text", "table"]
    assert all(row["document_id"] == "d1" and row["dim"] == d for row in rows)
    assert all(row["payload"]["content"] and row["payload"]["user_id"] == "u1" for row in rows)
    assert vs.count("T1") == 2


def test_dim_mismatch_raises(container):
    with pytest.raises(ValueError):
        container.vectors.upsert([_pt("p", "T1", [0.1, 0.2])])
