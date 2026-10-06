import json
from concurrent.futures import Future
from dataclasses import asdict, replace
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from psycopg2.extras import Json

from app.api.app import create_app
from app.ingest.adapters.publication import PostgresPublicationStore
from app.ingest.pipeline.runner import run_job
from app.shared.adapters.pgvector.vector_store import PgVectorStore
from app.shared.adapters.postgres.db import transaction
from app.shared.domain.embedding import EmbeddingProfile
from app.shared.execution import ExecutionBudget, RequestAborted, execution_scope
from app.shared.model_catalog import ModelCatalog
from app.shared.ports.vector_store import VectorPoint
from app.shared.workspace_runtime import WorkspaceRuntime
from tests.ingest.test_generation_publication import _queue


def test_catalog_excludes_claude_and_non_chat_endpoints_without_loading_models(monkeypatch):
    calls = []
    entries = [
        ("gpt-6-sol", "chat", "azure/gpt-6-sol"),
        ("claude-sonnet-5", "chat", "anthropic/claude-sonnet-5"),
        ("hidden-alias", "chat", "anthropic/claude-opus-5"),
        ("gpt-6-astra", None, "azure/responses/gpt-6-astra"),
        ("image-model", "image_generation", "azure/image"),
        ("text-embedding-3-small", "embedding", "azure/text-embedding-3-small"),
    ]

    def get(url, **kwargs):
        calls.append(url)
        data = [
            {
                "model_name": name,
                "model_info": {"mode": mode},
                "litellm_params": {"model": upstream},
            }
            for name, mode, upstream in entries
        ]
        return httpx.Response(200, request=httpx.Request("GET", url), json={"data": data})

    monkeypatch.setattr("app.shared.model_catalog.httpx.get", get)
    gateway = SimpleNamespace(_resolve_config=lambda: ("https://gateway.test", "secret"))
    catalog = ModelCatalog(gateway)
    result = catalog.get()
    assert [item["id"] for item in result["chat_models"]] == ["gpt-6-sol"]
    assert [item["id"] for item in result["embedding_models"]] == [
        "text-embedding-3-small",
        "local:minilm",
        "local:bge-small",
    ]
    assert "secret" not in json.dumps(result)
    assert catalog.get() == result and len(calls) == 1


def test_workspace_loader_is_lazy_and_does_not_load_unused_models(monkeypatch):
    models = {"chat_model": "gpt-6-sol", "embedding_model": "local:bge-small"}
    root = SimpleNamespace(
        metadata=SimpleNamespace(get_workspace_models=lambda _: models),
        settings=SimpleNamespace(
            model_copy=lambda **_: None,
            embedding_provider="gateway",
            embedding_model="text-embedding-3-large",
        ),
    )
    runtime = WorkspaceRuntime(root)
    submitted = []
    pending = Future()
    monkeypatch.setattr(
        runtime._executor, "submit", lambda function, model: submitted.append(model) or pending
    )
    assert submitted == []
    with execution_scope(ExecutionBudget(5)), pytest.raises(RequestAborted, match="loading"):
        runtime.for_tenant("workspace")
    assert submitted == ["local:bge-small"]
    assert runtime.prepare("local:bge-small") is pending
    assert submitted == ["local:bge-small"]
    pending.cancel()
    runtime.close()


def test_signup_persists_models_and_team_inherits_them(container, monkeypatch):
    catalog = {
        "chat_models": [{"id": "gpt-6-sol", "label": "GPT"}],
        "embedding_models": [{"id": "local:bge-small", "label": "BGE", "provider": "local"}],
    }
    monkeypatch.setattr(container.model_catalog, "get", lambda: catalog)
    prepared = []
    monkeypatch.setattr(container.workspace_runtime, "prepare", prepared.append)
    with TestClient(create_app(container)) as client:
        signup = {
            "email": "models@example.test",
            "password": "password-for-tests",
            "chat_model": "gpt-6-sol",
            "embedding_model": "local:bge-small",
        }
        response = client.post("/onboarding/register", json=signup)
        assert response.status_code == 201
        auth = {"Authorization": "Bearer " + response.json()["api_token"]}
        identity = client.get("/onboarding/me", headers=auth).json()
        assert identity["embedding_model"] == "local:bge-small"
        assert identity["chat_model"] == "gpt-6-sol"
        assert prepared == ["local:bge-small"]
        credentials = {"email": "member-models@example.test", "password": "password-for-tests"}
        assert client.post("/onboarding/members", headers=auth, json=credentials).status_code == 201
        login = client.post("/onboarding/login", json=credentials).json()
        member = client.get(
            "/onboarding/me", headers={"Authorization": "Bearer " + login["api_token"]}
        ).json()
        assert member["embedding_model"] == identity["embedding_model"]
        rejected = client.post(
            "/onboarding/register",
            json=dict(signup, email="reject@example.test", chat_model="claude-sonnet-5"),
        )
        assert rejected.status_code == 400
        assert container.metadata.get_user_by_email("reject@example.test") is None


def test_workspace_vectors_isolate_same_dimension_models_and_other_dimensions(container, tenant):
    stores = []
    for name, dim in [("model-a", 384), ("model-b", 384), ("model-c", 1536)]:
        profile = EmbeddingProfile("test", name, "1", "test", dim, 512, "mean", True)
        with transaction(container.settings.postgres_dsn) as cur:
            cur.execute(
                "INSERT INTO workspace_embedding_profiles VALUES(%s,%s)",
                (profile.id, Json(asdict(profile))),
            )
        store = PgVectorStore(container.settings.postgres_dsn, dim, profile=profile, workspace=True)
        payload = {
            "_id": name,
            "scope": "tenant",
            "visibility": "tenant",
            "content": "Invoice amount 007",
        }
        store.upsert([VectorPoint(name, tenant["id"], [1.0] + [0.0] * (dim - 1), payload)])
        stores.append(store)
    for store in stores:
        hits = store.search(tenant["id"], [1.0] + [0.0] * (store.dim - 1), query_text="Invoice")
        assert [hit.chunk_id for hit in hits] == [store.profile.model]
        assert store.validate_sources(tenant["id"], [store.profile.model])
        other = "model-b" if store.profile.model != "model-b" else "model-a"
        assert not store.validate_sources(tenant["id"], [other])
    container.vectors.delete_by_document(tenant["id"], "model-a")
    assert stores[0].search(tenant["id"], [1.0] + [0.0] * 383) == []


def test_workspace_publication_uses_selected_index(container, tenant):
    profile = container.embedder.profile
    with transaction(container.settings.postgres_dsn) as cur:
        cur.execute(
            "INSERT INTO workspace_embedding_profiles VALUES(%s,%s)",
            (profile.id, Json(asdict(profile))),
        )
    vectors = PgVectorStore(
        container.settings.postgres_dsn, profile.dimensions, profile=profile, workspace=True
    )
    publication = PostgresPublicationStore(container.settings.postgres_dsn, profile, workspace=True)
    scoped = replace(container, workspace_runtime=None, vectors=vectors, publication=publication)
    document, job = _queue(scoped, tenant, "Invoice 00123 has amount 007.00 USD.")
    run_job(scoped, job)
    assert scoped.vectors.search(tenant["id"], scoped.embedder.embed_query("Invoice 00123"))
    assert not container.vectors.search(
        tenant["id"], container.embedder.embed_query("Invoice 00123")
    )
    assert container.metadata.get_document(tenant["id"], document.id).indexed_version == 1


def test_request_model_is_forwarded_and_claude_is_rejected(container, tenant, monkeypatch):
    monkeypatch.setattr(
        container.model_catalog,
        "get",
        lambda: {
            "chat_models": [{"id": "gpt-6-sol", "label": "GPT"}],
            "embedding_models": [],
        },
    )
    calls = []

    def chat(messages, model, **kwargs):
        calls.append(model)
        return json.dumps(
            {
                "status": "answered",
                "answer": "The amount is 007 USD [1].",
                "source_ids": ["1"],
                "evidence_quotes": [],
            }
        )

    monkeypatch.setattr(container.gateway, "chat", chat)
    with TestClient(create_app(container)) as client:
        auth = {"Authorization": "Bearer " + tenant["admin_token"]}
        body = {
            "question": "What is the invoice amount?",
            "model": "gpt-6-sol",
            "contexts": ["The invoice amount is 007 USD."],
            "chunk_ids": ["invoice"],
        }
        result = client.post("/answer", headers=auth, json=body)
        assert result.status_code == 200
        assert result.json()["answer_status"] == "answered"
        assert calls == ["gpt-6-sol"]
        rejected = client.post("/answer", headers=auth, json=dict(body, model="claude-opus-5"))
        assert rejected.status_code == 400
        assert calls == ["gpt-6-sol"]


def test_selected_local_model_is_used_for_worker_and_queries(container, tenant):
    container.metadata.set_workspace_models(tenant["id"], "gpt-6-sol", "local:minilm")
    selected = container.for_workspace(tenant["id"])
    assert selected.settings.chat_model == "gpt-6-sol"
    assert selected.embedder.profile.model == "Xenova/all-MiniLM-L6-v2"
    assert selected.embedder.profile.revision == "751bff37182d3f1213fa05d7196b954e230abad9"
    assert selected.cache._index != container.cache._index
    assert set(container.workspace_runtime._futures) == {"local:minilm"}
    document, job = _queue(container, tenant, "Invoice 00123 has amount 007.00 USD.")
    run_job(container, job)
    hits = selected.vectors.search(tenant["id"], selected.embedder.embed_query("Invoice 00123"))
    assert hits and hits[0].payload["_id"] == document.id
    assert container.vectors.count(tenant["id"]) == len(hits)
    assert container.for_workspace(tenant["id"]).embedder is selected.embedder
    container.workspace_runtime.close()


def test_bge_selection_uses_its_pinned_pooling_and_query_recipe(container, monkeypatch):
    configured = []

    def selected_embedder(repo, dimensions, **kwargs):
        configured.append((repo, dimensions, kwargs))
        return container.embedder

    monkeypatch.setattr("app.shared.workspace_runtime.OnnxEmbedder", selected_embedder)
    runtime = container.workspace_runtime.prepare("local:bge-small").result(timeout=10)
    repo, dimensions, options = configured[0]
    assert repo == "Xenova/bge-small-en-v1.5" and dimensions == 384
    assert options["pooling"] == "cls"
    assert options["max_length"] == 512
    assert options["revision"] == "ea104dacec62c0de699686887e3f920caeb4f3e3"
    assert options["query_instruction"].startswith("Represent this sentence")
    assert runtime[1].workspace
    container.workspace_runtime.close()
