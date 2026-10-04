"""Connection helper for the pgvector adapter.

Shares the same pool as app/adapters/postgres/db.py (same physical database),
but additionally registers pgvector's psycopg2 type adapter on every checked-out
connection so a Python list[float]/np.ndarray adapts to/from the SQL `vector`
type transparently. register_vector() is cheap and idempotent to call on every
checkout (psycopg2 pools have no native "new connection" hook to call it once).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg2.extras
from pgvector.psycopg2 import register_vector

from app.shared.adapters.postgres.db import get_pool
from app.shared.execution import cancel_on_budget, check_execution


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
            pass
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            with cancel_on_budget(conn.cancel):
                yield cur
        check_execution()
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)
