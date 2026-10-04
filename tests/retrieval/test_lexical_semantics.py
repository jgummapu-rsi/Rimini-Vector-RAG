import numpy as np

from app.shared.adapters.bm25 import reciprocal_rank_fusion
from app.shared.adapters.pgvector.db import transaction
from app.shared.adapters.pgvector.vector_store import lexical_query
from app.shared.ports.vector_store import VectorPoint


def test_no_lexical_matches_cannot_change_dense_order():
    dense = np.array([0.1, 0.2, 0.3, 0.4])
    fused = reciprocal_rank_fusion(dense, np.zeros(4))
    assert np.argsort(-fused).tolist() == [3, 2, 1, 0]


def test_prose_is_or_but_quotes_negation_and_identifiers_are_preserved():
    assert lexical_query("invoice approval workflow") == "invoice OR approval OR workflow"
    assert lexical_query('"invoice approval" workflow') == '"invoice approval" workflow'
    assert lexical_query("invoice -cancelled") == "invoice -cancelled"
    assert lexical_query("INV-00123") == "INV-00123"


def test_partial_prose_and_exact_identifier_retrieve_real_fts(container):

    vector = [1.0] + [0.0] * (container.embedder.dim - 1)
    container.vectors.upsert(
        [
            VectorPoint(
                "approval",
                "tenant",
                vector,
                {"_id": "a", "content": "invoice approval policy INV-00123"},
            ),
            VectorPoint(
                "workflow", "tenant", vector, {"_id": "b", "content": "workflow execution"}
            ),
        ]
    )
    with transaction(container.settings.postgres_dsn) as cur:
        cur.execute(
            "SELECT count(*) AS n FROM vector_chunks WHERE tsv @@ websearch_to_tsquery('english','invoice approval workflow')"
        )
        assert cur.fetchone()["n"] == 0
        rows = container.vectors._lexical_candidates(cur, "tenant", "invoice approval workflow", 10)
        assert {row["chunk_id"] for row in rows} == {"approval", "workflow"}
        rows = container.vectors._lexical_candidates(cur, "tenant", "INV-00123", 10)
        assert [row["chunk_id"] for row in rows] == ["approval"]
