"""Shared Postgres connection pool + transaction helper.

Unlike SQLite (cheap open/close per call, local file), Postgres is networked, so
we keep one small connection pool per process -- app/api/app.py builds one
Container for the process lifetime, and app/worker.py is a single long-lived
loop, so a couple of connections is enough for one process.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg2
import psycopg2.extras
import psycopg2.pool

_pools: dict[str, psycopg2.pool.ThreadedConnectionPool] = {}


def get_pool(dsn: str, minconn: int = 1, maxconn: int = 10) -> psycopg2.pool.ThreadedConnectionPool:
    """One pool per distinct DSN per process (so a test DSN never collides with
    the app's own pool)."""
    if dsn not in _pools:
        _pools[dsn] = psycopg2.pool.ThreadedConnectionPool(minconn, maxconn, dsn)
    return _pools[dsn]


@contextmanager
def transaction(dsn: str) -> Iterator[psycopg2.extras.RealDictCursor]:
    """Yield a RealDictCursor (dict-like rows: row["col"] AND row.keys() both
    work, matching how sqlite3.Row is used throughout the metadata store).
    Commits on success, rolls back on exception, always returns the connection
    to the pool."""
    pool = get_pool(dsn)
    conn = pool.getconn()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)
