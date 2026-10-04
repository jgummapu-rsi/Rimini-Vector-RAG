import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import psycopg2
import pytest

from app.shared.adapters.postgres.db import get_pool, transaction
from app.shared.container import build_container


def test_configured_pool_limits_and_release_wake_waiter(storage_settings):
    settings = storage_settings.model_copy(
        update={"postgres_pool_min": 1, "postgres_pool_max": 1, "reranker_provider": "none"}
    )
    container = build_container(settings)
    pool = get_pool(settings.postgres_dsn)
    assert pool.maxconn == 1
    held = pool.getconn()
    waiting = Event()

    def checkout():
        waiting.set()
        with transaction(settings.postgres_dsn) as cur:
            cur.execute("SELECT 1 AS value")
            return cur.fetchone()["value"]

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(checkout)
        assert waiting.wait(1)
        time.sleep(0.05)
        assert not future.done()
        pool.putconn(held)
        assert future.result(timeout=2) == 1
    container.close()
    assert pool.closed


def test_pool_has_real_database_statement_deadlines(storage_settings):
    with transaction(storage_settings.postgres_dsn) as cur:
        cur.execute("SHOW statement_timeout")
        assert cur.fetchone()["statement_timeout"] == "1min"
    with pytest.raises(psycopg2.errors.QueryCanceled):
        with transaction(storage_settings.postgres_dsn) as cur:
            cur.execute("SET LOCAL statement_timeout='20ms'")
            cur.execute("SELECT pg_sleep(1)")
    with transaction(storage_settings.postgres_dsn) as cur:
        cur.execute("SELECT 1 AS value")
        assert cur.fetchone()["value"] == 1
