from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from uuid import uuid4

import psycopg2
import redis
from psycopg2 import sql
from psycopg2.extensions import make_dsn

from app.shared.adapters.postgres.db import close_pool
from app.shared.config import Settings

log = logging.getLogger(__name__)


@contextmanager
def disposable_settings(database_url: str, redis_url: str):
    namespace = "rag_test_" + uuid4().hex
    admin = psycopg2.connect(database_url, connect_timeout=5)
    admin.autocommit = True
    cache = redis.Redis.from_url(redis_url, socket_connect_timeout=5, socket_timeout=5)
    try:
        cache.execute_command("FT._LIST")
        with admin.cursor() as cur:
            cur.execute(
                sql.SQL("CREATE DATABASE {} TEMPLATE template0").format(sql.Identifier(namespace))
            )
        dsn = make_dsn(database_url, dbname=namespace)
        try:
            with psycopg2.connect(dsn) as conn, conn.cursor() as cur:
                cur.execute("CREATE EXTENSION vector WITH SCHEMA public")
                cur.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(namespace)))
            dsn = make_dsn(dsn, options=f"-csearch_path={namespace},public")
            with TemporaryDirectory(prefix=namespace) as data_dir:
                yield Settings(
                    _env_file=None,
                    database_url=dsn,
                    data_dir=data_dir,
                    redis_url=redis_url,
                    cache_index_name=namespace,
                    litellm_base_url="",
                    litellm_api_key="",
                    embedding_provider="minilm",
                    embedding_dim=384,
                )
        finally:
            close_pool(dsn)
            try:
                indexes = cache.execute_command("FT._LIST")
                for index in indexes:
                    name = index.decode() if isinstance(index, bytes) else index
                    if name == namespace or name.startswith(namespace + "_"):
                        cache.execute_command("FT.DROPINDEX", name, "DD")
                keys = list(cache.scan_iter(match=f"{namespace}:*"))
                if keys:
                    cache.delete(*keys)
            finally:
                with admin.cursor() as cur:
                    cur.execute(
                        sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(namespace))
                    )
    finally:
        admin.close()
        cache.close()


@contextmanager
def evaluation_settings():
    database_url = os.environ["EVAL_DATABASE_URL"]
    redis_url = os.environ["EVAL_REDIS_URL"]
    with disposable_settings(database_url, redis_url) as settings:
        yield settings.model_copy(
            update={
                "embedding_provider": os.environ.get("EVAL_EMBEDDING_PROVIDER", "gateway"),
                "embedding_model": os.environ.get("EVAL_EMBEDDING_MODEL", "text-embedding-3-large"),
                "embedding_dim": int(os.environ.get("EVAL_EMBEDDING_DIM", "1536")),
                "embedding_revision": os.environ.get(
                    "EVAL_EMBEDDING_REVISION", "text-embedding-3-large"
                ),
                "embed_batch_size": int(os.environ.get("EVAL_EMBED_BATCH_SIZE", "64")),
                "litellm_base_url": os.environ.get("EVAL_LITELLM_BASE_URL", ""),
                "litellm_api_key": os.environ.get("EVAL_LITELLM_API_KEY", ""),
            }
        )
