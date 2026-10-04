import secrets
from uuid import uuid4

import psycopg2
import pytest
from psycopg2 import sql
from psycopg2.extensions import make_dsn

from app.ingest.pipeline.runner import run_job
from app.shared.adapters.pgvector.migration import activate_shadow, build_shadow
from app.shared.adapters.postgres.db import close_pool
from app.shared.container import build_container
from scripts.grant_runtime import grant_runtime
from tests.ingest.test_generation_publication import _queue


def test_dml_only_runtime_can_ingest_but_cannot_modify_schema_or_profile(container, tenant):
    dsn = container.settings.postgres_dsn
    role = "runtime_" + uuid4().hex
    password = secrets.token_urlsafe(32)
    runtime = None
    runtime_dsn = make_dsn(dsn, user=role, password=password)
    with psycopg2.connect(dsn) as connection, connection.cursor() as cursor:
        cursor.execute("SELECT current_schema()")
        schema = cursor.fetchone()[0]
        cursor.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD %s NOSUPERUSER NOCREATEDB NOCREATEROLE").format(
                sql.Identifier(role)
            ),
            (password,),
        )
    try:
        grant_runtime(dsn, schema, role)
        runtime = build_container(
            container.settings.model_copy(
                update={"database_url": runtime_dsn, "initialize_schema": False}
            )
        )
        runtime.gateway.chat = lambda *args, **kwargs: (
            '{"topics":[],"entities":[],"author":null,"date":null}'
        )
        document, job = _queue(runtime, tenant, "Invoice 00123 amount 007.00")
        run_job(runtime, job)
        assert runtime.metadata.get_document(tenant["id"], document.id).indexed_version == 1
        runtime.readiness_check()
        shadow = build_shadow(dsn, container.embedder)
        grant_runtime(dsn, schema, role)
        for table in (shadow, "embedding_migrations", "runtime_database_roles"):
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                with psycopg2.connect(runtime_dsn) as connection, connection.cursor() as cursor:
                    cursor.execute(sql.SQL("DELETE FROM {}").format(sql.Identifier(table)))
        previous = activate_shadow(dsn, shadow)
        vector = runtime.embedder.embed_query("Invoice 00123")
        assert runtime.vectors.search(tenant["id"], vector)
        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            with psycopg2.connect(runtime_dsn) as connection, connection.cursor() as cursor:
                cursor.execute(sql.SQL("DELETE FROM {}").format(sql.Identifier(previous)))
        activate_shadow(dsn, previous)
        assert runtime.vectors.search(tenant["id"], vector)
        for command in (
            "CREATE TABLE forbidden (id int)",
            "UPDATE embedding_profile SET profile_id='forged'",
        ):
            with pytest.raises(psycopg2.errors.InsufficientPrivilege):
                with psycopg2.connect(runtime_dsn) as connection, connection.cursor() as cursor:
                    cursor.execute(command)
    finally:
        if runtime is not None:
            runtime.close()
        close_pool(runtime_dsn)
        with psycopg2.connect(dsn) as connection, connection.cursor() as cursor:
            cursor.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
            cursor.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))
