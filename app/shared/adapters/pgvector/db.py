"""Connection helper for the pgvector adapter.

Shares the same pool as app/adapters/postgres/db.py (same physical database),
but additionally registers pgvector's psycopg2 type adapter on every checked-out
connection so a Python list[float]/np.ndarray adapts to/from the SQL `vector`
type transparently. register_vector() is cheap and idempotent to call on every
checkout (psycopg2 pools have no native "new connection" hook to call it once).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import psycopg2.extras
from pgvector.psycopg2 import register_vector

from app.shared.adapters.postgres.db import get_pool


@contextmanager
def transaction(dsn: str) -> Iterator[psycopg2.extras.RealDictCursor]:
    """Yield a RealDictCursor with pgvector's type adapter registered, from
    the shared Postgres pool. Commits on success, rolls back on exception."""
    pool = get_pool(dsn)
    conn = pool.getconn()
    try:
        try:
            register_vector(conn)
        except psycopg2.ProgrammingError:
            # On a brand-new database the `vector` extension doesn't exist yet --
            # PgVectorStore.ensure_collection()'s very first transaction IS what
            # runs `CREATE EXTENSION IF NOT EXISTS vector`. register_vector's
            # lookup is a plain SELECT (raises its own ProgrammingError in
            # Python after finding no rows, not a failed statement), so the
            # transaction is not left aborted -- just proceed unregistered for
            # this one checkout. Every later checkout re-registers successfully
            # once the extension exists.
            pass
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)
