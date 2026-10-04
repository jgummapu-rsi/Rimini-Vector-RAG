import ast
import os
from pathlib import Path

import psycopg2
import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.shared.adapters.postgres.db import transaction
from app.shared.config import Settings
from app.shared.container import build_container
from scripts.disposable_storage import disposable_settings, evaluation_settings


@pytest.mark.parametrize(
    "field", ["metadata_backend", "vector_backend", "queue_backend", "blob_backend"]
)
def test_backend_selection_is_rejected(field):
    with pytest.raises(ValueError, match="Backend selection"):
        Settings(_env_file=None, **{field: "unsupported"})


def test_storage_is_isolated_and_removed():
    database_url = os.environ["TEST_DATABASE_URL"]
    with disposable_settings(database_url, os.environ["TEST_REDIS_URL"]) as cfg:
        with transaction(cfg.postgres_dsn) as cur:
            cur.execute("SELECT current_database() AS db, current_schema() AS schema")
            identity = cur.fetchone()
            assert identity["db"].startswith("rag_test_")
            assert identity["schema"] == identity["db"]
            cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
            assert tuple(map(int, cur.fetchone()["extversion"].split("."))) >= (0, 8, 0)
        c = build_container(cfg.model_copy(update={"reranker_provider": "none"}))
        assert c.cache is not None
        assert c.queue.claim_next() is None
        with TestClient(create_app(c)) as client:
            assert client.get("/healthz").json()["backends"]["metadata"] == "postgres"
    with psycopg2.connect(database_url) as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (identity["db"],))
        assert cur.fetchone() is None


def test_missing_vector_extension_rejects_startup(storage_settings):
    with transaction(storage_settings.postgres_dsn) as cur:
        cur.execute("DROP EXTENSION vector")
    with pytest.raises(RuntimeError, match="pgvector >= 0.8.0"):
        build_container(storage_settings)


def test_missing_redis_rejects_startup(storage_settings):
    with pytest.raises(ValueError, match="REDIS_URL is required"):
        build_container(storage_settings.model_copy(update={"redis_url": ""}))


def test_eval_entry_points_require_isolated_settings():
    root = Path(__file__).resolve().parents[2] / "eval"
    checked = []
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_container"
        ]
        if not calls:
            continue
        if path.name == "network_load.py":
            exercise = next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "exercise"
            )
            contexts = [
                item.context_expr
                for node in ast.walk(exercise)
                if isinstance(node, ast.With)
                for item in node.items
            ]
            assert any(
                isinstance(context, ast.Call)
                and isinstance(context.func, ast.Name)
                and context.func.id == "evaluation_settings"
                for context in contexts
            )
            assert all(call.args for call in calls)
            continue
        main = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        if path.name == "compare_subset_reranking.py":
            contexts = [
                item.context_expr
                for node in ast.walk(main)
                if isinstance(node, ast.With)
                for item in node.items
            ]
            assert any(
                isinstance(context, ast.Call)
                and isinstance(context.func, ast.Name)
                and context.func.id == "evaluation_settings"
                for context in contexts
            )
            assert all(call.args for call in calls)
            continue
        if path.name == "restore_drill.py":
            contexts = [
                item.context_expr
                for node in ast.walk(main)
                if isinstance(node, ast.With)
                for item in node.items
            ]
            factories = {
                context.func.id
                for context in contexts
                if isinstance(context, ast.Call) and isinstance(context.func, ast.Name)
            }
            assert {"evaluation_settings", "disposable_settings"} <= factories
            assert all(call.args for call in calls)
            assert not any(
                isinstance(node, ast.ImportFrom)
                and node.module == "app.shared.config"
                and any(alias.name == "settings" for alias in node.names)
                for node in ast.walk(tree)
            )
            continue
        assert any(
            isinstance(d, ast.Name) and d.id == "isolated_evaluation" for d in main.decorator_list
        ), path
        assert all(
            call.args and isinstance(call.args[0], ast.Name) and call.args[0].id == "settings"
            for call in calls
        ), path
        assert not any(
            isinstance(node, ast.ImportFrom)
            and node.module == "app.shared.config"
            and any(alias.name == "settings" for alias in node.names)
            for node in ast.walk(tree)
        ), path
        checked.append(path)
    assert checked, "No isolated evaluation entry points were checked"


def test_evaluation_storage_ignores_application_environment(monkeypatch):

    monkeypatch.setenv("EVAL_DATABASE_URL", os.environ["TEST_DATABASE_URL"])
    monkeypatch.setenv("EVAL_REDIS_URL", os.environ["TEST_REDIS_URL"])
    monkeypatch.setenv("EVAL_EMBEDDING_PROVIDER", "minilm")
    monkeypatch.setenv("EVAL_EMBEDDING_DIM", "384")
    monkeypatch.setenv("DATABASE_URL", "postgresql://invalid.invalid/application")
    monkeypatch.setenv("REDIS_URL", "redis://invalid.invalid")
    monkeypatch.setenv("CACHE_INDEX_NAME", "application_index")
    with evaluation_settings() as cfg:
        assert "invalid.invalid" not in cfg.database_url
        assert cfg.cache_index_name.startswith("rag_test_")
        assert cfg.blob_dir.parent.name.startswith("rag_test_")
        container = build_container(cfg.model_copy(update={"reranker_provider": "none"}))
        assert container.queue.claim_next() is None
