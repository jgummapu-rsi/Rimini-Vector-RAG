"""SQLite-backed metrics (counters + accumulated timings).

Stored in the shared DB so the API and worker processes both contribute and a
single `/metrics` read returns the whole picture. Each metric row is a name with
a count and total milliseconds (for timings); avg is derived on read.
"""
from __future__ import annotations

from pathlib import Path

from app.adapters.sqlite.db import connect


class SqliteMetrics:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def incr(self, name: str, count: int = 1, ms: float = 0.0) -> None:
        conn = connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO metrics (name, count, total_ms) VALUES (?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET "
                "count = count + excluded.count, total_ms = total_ms + excluded.total_ms",
                (name, count, ms),
            )
            conn.commit()
        finally:
            conn.close()

    def snapshot(self) -> dict:
        conn = connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT name, count, total_ms FROM metrics ORDER BY name"
            ).fetchall()
        finally:
            conn.close()
        return {
            r["name"]: {
                "count": r["count"],
                "total_ms": round(r["total_ms"], 2),
                "avg_ms": round(r["total_ms"] / r["count"], 2) if r["count"] else 0.0,
            }
            for r in rows
        }
