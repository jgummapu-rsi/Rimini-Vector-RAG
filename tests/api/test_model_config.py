"""Governed runtime model configuration API."""
import pytest
from fastapi.testclient import TestClient

from app.api.app import create_app


@pytest.fixture
def client(container):
    with TestClient(create_app(container)) as test_client:
        yield test_client


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_model_catalog_requires_admin(client, tenant):
    assert client.get("/admin/models").status_code == 401
    response = client.get("/admin/models", headers=_auth(tenant["viewer_token"]))
    assert response.status_code == 403


def test_admin_can_change_non_embedding_models(client, container, tenant):
    container.settings.platform_tenant_id = tenant["id"]
    response = client.put(
        "/admin/models",
        headers=_auth(tenant["admin_token"]),
        json={
            "chat_model": "gpt-5.5",
            "vision_model": "gpt-5.6-sol",
            "embedding_preset": "minilm",
            "reranker_preset": "none",
        },
    )
    assert response.status_code == 200
    assert response.json()["selected"]["chat_model"] == "gpt-5.5"
    assert response.json()["reset"] is None
    assert container.metadata.get_system_config("rag_vision_model") == "gpt-5.6-sol"
    assert container.reranker is None


def test_embedding_change_requires_confirmation(client, tenant):
    client.app.state.container.settings.platform_tenant_id = tenant["id"]
    response = client.put(
        "/admin/models",
        headers=_auth(tenant["admin_token"]),
        json={
            "chat_model": "gpt-5-nano",
            "vision_model": "gpt-5.6-sol",
            "embedding_preset": "bge-large",
            "reranker_preset": "ms-marco-minilm",
        },
    )
    assert response.status_code == 409
    assert "clears all indexed documents" in response.json()["detail"]


def test_embedding_change_rejects_non_pgvector_backend(client, tenant):
    client.app.state.container.settings.platform_tenant_id = tenant["id"]
    response = client.put(
        "/admin/models",
        headers=_auth(tenant["admin_token"]),
        json={
            "chat_model": "gpt-5-nano",
            "vision_model": "gpt-5.6-sol",
            "embedding_preset": "bge-large",
            "reranker_preset": "ms-marco-minilm",
            "confirm_embedding_reset": True,
        },
    )
    assert response.status_code == 409
    assert "pgvector" in response.json()["detail"]
