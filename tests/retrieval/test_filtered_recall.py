import math
import time

from app.retrieval.rag.access import access_predicate
from app.shared.domain.models import Principal, Role
from app.shared.ports.vector_store import VectorPoint


def test_selective_acl_dense_results_match_exact_search(container):
    dimension = container.embedder.dim
    points = []
    for index in range(2500):
        angle = index / 2500
        vector = [math.cos(angle), math.sin(angle)] + [0.0] * (dimension - 2)
        allowed = index % 250 == 0
        points.append(
            VectorPoint(
                f"chunk{index:05d}",
                "tenant",
                vector,
                {
                    "_id": f"doc{index}",
                    "content": "evidence",
                    "user_id": "reader" if allowed else "other",
                    "visibility": "private",
                },
            )
        )
    container.vectors.upsert(points)
    access = access_predicate(Principal("tenant", "reader", Role.MEMBER))
    query = [1.0] + [0.0] * (dimension - 1)
    durations = []
    for _ in range(10):
        started = time.perf_counter()
        hits = container.vectors.search("tenant", query, top_k=5, access=access)
        durations.append((time.perf_counter() - started) * 1000)
        assert [hit.chunk_id for hit in hits] == [
            f"chunk{index:05d}" for index in range(0, 1250, 250)
        ]
    assert sorted(durations)[-1] < 2000
