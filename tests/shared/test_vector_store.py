import pytest

from app.shared.ports.vector_store import VectorPoint


def _pt(cid, tid, vec, **payload):
    payload.setdefault("_id", "docX")
    payload.setdefault("content", "some text")
    return VectorPoint(chunk_id=cid, tenant_id=tid, vector=vec, payload=payload)


def test_ensure_collection_writes_meta(container):
    vs = container.vectors
    assert vs._meta_path.exists()


def test_upsert_count_and_search(container):
    vs = container.vectors
    d = vs.dim
    a = [1.0] + [0.0] * (d - 1)
    b = [0.0, 1.0] + [0.0] * (d - 2)
    vs.upsert([
        _pt("d1001", "T1", a, _id="d1", modality="text", source_type="pdf"),
        _pt("d1002", "T1", b, _id="d1", modality="table", source_type="csv"),
    ])
    assert vs.count("T1") == 2

    hits = vs.search("T1", a, top_k=2)
    assert hits[0].chunk_id == "d1001"       # closest to itself
    assert hits[0].score > hits[1].score
    # hit carries the chunk id + its document/content; no embeddings echoed back
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
    vs.upsert([
        _pt("d1001", "T1", v, _id="d1"),
        _pt("d2001", "T1", v, _id="d2"),
    ])
    removed = vs.delete_by_document("T1", "d1")
    assert removed == 1
    assert vs.count("T1") == 1
    assert all(h.payload["_id"] == "d2" for h in vs.search("T1", v, top_k=5))


def test_one_record_per_document_with_embedded_chunks(container):
    vs = container.vectors
    d = vs.dim
    # two chunks of ONE document d1 -> a single record with chunks[] of length 2
    vs.upsert([
        _pt("d1001", "T1", [1.0] + [0.0] * (d - 1), _id="d1", user_id="u1",
            modality="text", content="chunk one"),
        _pt("d1002", "T1", [0.0, 1.0] + [0.0] * (d - 2), _id="d1", user_id="u1",
            modality="table", content="chunk two"),
    ])
    import json
    from app.shared.adapters.sqlite.db import transaction
    with transaction(container.settings.sqlite_path) as c:
        n_docs = c.execute("SELECT COUNT(*) n FROM vector_documents").fetchone()["n"]
        rec = json.loads(c.execute(
            "SELECT record FROM vector_documents WHERE _id='d1'").fetchone()["record"])

    assert n_docs == 1
    assert rec["_id"] == "d1" and rec["user_id"] == "u1"
    assert isinstance(rec["created_at"], int) and isinstance(rec["updated_at"], int)
    assert "point_id" not in rec
    # chunks embedded inside, each self-contained
    assert [c["chunk_id"] for c in rec["chunks"]] == ["d1001", "d1002"]
    assert [c["modality"] for c in rec["chunks"]] == ["text", "table"]
    for ch in rec["chunks"]:
        assert ch["content"] and len(ch["embeddings"]) == d
        assert isinstance(ch["created_at"], int) and isinstance(ch["updated_at"], int)
    # count is total chunk-vectors, not documents
    assert vs.count("T1") == 2


def test_dim_mismatch_raises(container):
    with pytest.raises(ValueError):
        container.vectors.upsert([_pt("p", "T1", [0.1, 0.2])])
