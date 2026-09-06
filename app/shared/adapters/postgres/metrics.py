"""Postgres-backed metrics (counters + accumulated timings).

Stored in the shared DB so the API and worker processes both contribute and a
single `/metrics` read returns the whole picture. Each metric row is a name with
a count and total milliseconds (for timings); avg is derived on read.
"""
from __future__ import annotations

from app.shared.adapters.postgres.db import transaction


class PostgresMetrics:
    """Postgres-backed counter/timing store."""

    def __init__(self, dsn: str):
        """Bind to the Postgres database at `dsn`."""
        self.dsn = dsn

    def incr(self, name: str, count: int = 1, ms: float = 0.0) -> None:
        """Add `count` occurrences and `ms` milliseconds to a named metric."""
        with transaction(self.dsn) as cur:
            cur.execute(
                "INSERT INTO metrics (name, count, total_ms) VALUES (%s, %s, %s) "
                "ON CONFLICT(name) DO UPDATE SET "
                "count = metrics.count + excluded.count, "
                "total_ms = metrics.total_ms + excluded.total_ms",
                (name, count, ms),
            )

    def snapshot(self) -> dict:
        """Return every metric as {name: {count, total_ms, avg_ms}}."""
        with transaction(self.dsn) as cur:
            cur.execute("SELECT name, count, total_ms FROM metrics ORDER BY name")
            rows = cur.fetchall()
        return {
            r["name"]: {
                "count": r["count"],
                "total_ms": round(r["total_ms"], 2),
                "avg_ms": round(r["total_ms"] / r["count"], 2) if r["count"] else 0.0,
            }
            for r in rows
        }
