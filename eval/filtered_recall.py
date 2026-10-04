from __future__ import annotations

import json
import logging
import math
import time

import numpy as np
from psycopg2.extras import execute_values

from app.retrieval.rag.access import access_predicate
from app.shared.adapters.pgvector.db import transaction
from app.shared.adapters.pgvector.vector_store import PgVectorStore, acl_pushdown
from app.shared.adapters.postgres.metadata_store import PostgresMetadataStore
from app.shared.domain.embedding import EmbeddingProfile
from app.shared.domain.models import Principal, Role
from scripts.disposable_storage import evaluation_settings

log = logging.getLogger(__name__)


def main() -> None:
    with evaluation_settings() as cfg:
        profile = EmbeddingProfile("benchmark", "circle", "v1", "none", 3, 256, "none", True)
        PostgresMetadataStore(cfg.postgres_dsn).init_schema()
        store = PgVectorStore(cfg.postgres_dsn, 3, profile=profile)
        store.ensure_collection(3)
        rows = []
        for index in range(20000):
            angle = index * 2 * math.pi / 20000
            owner = "rare" if index % 1000 == 0 else "medium" if index % 10 == 0 else "other"
            tenant = "tenant" if index < 10000 else f"unrelated{index % 100}"
            rows.append(
                (
                    f"c{index:05d}",
                    tenant,
                    f"d{index}",
                    [math.cos(angle), math.sin(angle), 0.05],
                    json.dumps(
                        {
                            "_id": f"d{index}",
                            "user_id": owner,
                            "visibility": "private",
                            "content": "synthetic",
                        }
                    ),
                )
            )
        with transaction(cfg.postgres_dsn) as cur:
            execute_values(
                cur,
                "INSERT INTO vector_chunks(chunk_id,tenant_id,document_id,embedding,payload) VALUES %s",
                rows,
                page_size=500,
            )
            cur.execute(
                "UPDATE vector_chunks SET deleted=true WHERE tenant_id<>'tenant' AND chunk_id LIKE '%5'"
            )
            cur.execute("ANALYZE vector_chunks")
        measurements = []
        for user, eligible in (("rare", 10), ("medium", 990), ("other", 9000)):
            access = access_predicate(Principal("tenant", user, Role.MEMBER))
            acl, params = acl_pushdown(access)
            found = total = 0
            timings = []
            for index in range(30):
                angle = index * 0.2 + 0.001
                query = [math.cos(angle), math.sin(angle), 0.05]
                with transaction(cfg.postgres_dsn) as cur:
                    cur.execute(
                        "WITH eligible AS MATERIALIZED (SELECT chunk_id,embedding FROM vector_chunks "
                        "WHERE tenant_id=%s AND deleted=false "
                        + acl
                        + ") SELECT chunk_id FROM eligible ORDER BY embedding <=> %s::vector LIMIT 10",
                        ("tenant", *params, query),
                    )
                    exact = {row["chunk_id"] for row in cur.fetchall()}
                started = time.perf_counter()
                hits = store.search("tenant", query, top_k=10, access=access)
                timings.append((time.perf_counter() - started) * 1000)
                found += len(exact & {hit.chunk_id for hit in hits})
                total += len(exact)
            with transaction(cfg.postgres_dsn) as cur:
                cur.execute("SET LOCAL enable_seqscan=off")
                cur.execute("SET LOCAL hnsw.iterative_scan='strict_order'")
                cur.execute("SET LOCAL hnsw.ef_search=400")
                cur.execute("SET LOCAL hnsw.max_scan_tuples=20000")
                cur.execute(
                    "EXPLAIN (FORMAT JSON) SELECT chunk_id FROM vector_chunks "
                    "WHERE tenant_id=%s AND deleted=false "
                    + acl
                    + " ORDER BY embedding <=> %s::vector LIMIT 10",
                    ("tenant", *params, [1.0, 0.0, 0.05]),
                )
                plan = cur.fetchone()["QUERY PLAN"]
            measurements.append(
                {
                    "slice": user,
                    "eligible": eligible,
                    "corpus": len(rows),
                    "recalled": found,
                    "relevant": total,
                    "queries": len(timings),
                    "p50_ms": float(np.percentile(timings, 50)),
                    "p95_ms": float(np.percentile(timings, 95)),
                    "explain": plan,
                }
            )
        print(json.dumps(measurements, indent=2))
        if any(
            item["recalled"] / item["relevant"] < 0.95 or item["p95_ms"] > 250
            for item in measurements
        ):
            raise RuntimeError("Filtered retrieval synthetic recall/latency gate failed")


if __name__ == "__main__":
    main()
