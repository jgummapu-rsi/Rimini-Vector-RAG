"""Shared Postgres connection pool + transaction helper.

we keep one small connection pool per process -- app/api/app.py builds one
Container for the process lifetime, and app/worker.py is a single long-lived
loop, so a couple of connections is enough for one process.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from threading import BoundedSemaphore, Lock

import psycopg2
import psycopg2.extras
import psycopg2.pool

from app.shared.execution import cancel_on_budget, check_execution, remaining_seconds

_pools: dict[str, psycopg2.pool.ThreadedConnectionPool] = {}
_pool_lock = Lock()


class BoundedConnectionPool(psycopg2.pool.ThreadedConnectionPool):
    def __init__(self, minconn, maxconn, dsn):
        super().__init__(
            minconn,
            maxconn,
            dsn,
            connect_timeout=5,
            keepalives=1,
            keepalives_idle=10,
            keepalives_interval=5,
            keepalives_count=3,
            tcp_user_timeout=15000,
        )
        self._slots = BoundedSemaphore(maxconn)

    def getconn(self, key=None):
        check_execution()
        if not self._slots.acquire(timeout=remaining_seconds(3)):
            raise psycopg2.pool.PoolError("Database connection capacity exhausted after 3 seconds")
        try:
            connection = super().getconn(key)
        except BaseException:
            self._slots.release()
            raise
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT set_config('statement_timeout',%s,false)",
                    (str(max(1, int(remaining_seconds(60) * 1000))),),
                )
                cursor.execute("SET lock_timeout='5000ms'")
                cursor.execute("SET idle_in_transaction_session_timeout='60000ms'")
            connection.commit()
            return connection
        except BaseException:
            super().putconn(connection, key, close=True)
            self._slots.release()
            raise

    def putconn(self, conn, key=None, close=False):
        try:
            super().putconn(conn, key, close=close)
        finally:
            self._slots.release()


def get_pool(dsn: str, minconn: int = 1, maxconn: int = 10) -> psycopg2.pool.ThreadedConnectionPool:
    """One pool per distinct DSN per process (so a test DSN never collides with
    the app's own pool)."""
    with _pool_lock:
        if dsn not in _pools:
            _pools[dsn] = BoundedConnectionPool(minconn, maxconn, dsn)
        return _pools[dsn]


def close_pool(dsn: str) -> None:
    with _pool_lock:
        pool = _pools.pop(dsn, None)
    if pool is not None:
        pool.closeall()


@contextmanager
def transaction(dsn: str) -> Iterator[psycopg2.extras.RealDictCursor]:
    """Yield a RealDictCursor.
    Commits on success, rolls back on exception, always returns the connection
    to the pool."""
    pool = get_pool(dsn)
    conn = pool.getconn()
    try:
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
