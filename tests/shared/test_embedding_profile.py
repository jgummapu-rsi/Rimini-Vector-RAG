from dataclasses import replace

import pytest
from psycopg2 import sql

from app.shared.adapters.embedders.onnx_embedder import OnnxEmbedder
from app.shared.adapters.pgvector.migration import activate_shadow, build_shadow
from app.shared.adapters.pgvector.vector_store import PgVectorStore
from app.shared.adapters.postgres.db import transaction
from app.shared.domain.embedding import EmbeddingProfile, validate_vectors
from app.shared.gateway.client import GatewayError, LiteLLMClient
from app.shared.ports.embedder import Embedder
from app.shared.ports.vector_store import VectorPoint


def test_gateway_restores_input_order_and_sends_dimensions(monkeypatch):
    client = LiteLLMClient("https://gateway.test", "test", "vision", "text-embedding-3-small")
    seen = []

    def post(path, payload):
        seen.append(payload)
        return {
            "data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]
        }

    monkeypatch.setattr(client, "_post", post)
    assert client.embed(["first", "second"], dimensions=2) == [[1.0, 0.0], [0.0, 1.0]]
    assert seen[0]["dimensions"] == 2


@pytest.mark.parametrize(
    "items",
    [
        [],
        [{"index": 0, "embedding": [1.0, 0.0]}],
        [{"index": 0, "embedding": [1.0, 0.0]}, {"index": 0, "embedding": [1.0, 0.0]}],
        [{"index": -1, "embedding": [1.0, 0.0]}, {"index": 1, "embedding": [1.0, 0.0]}],
        [{"index": True, "embedding": [1.0, 0.0]}, {"index": 0, "embedding": [1.0, 0.0]}],
        [{"index": 0, "embedding": [float("nan"), 0.0]}, {"index": 1, "embedding": [1.0, 0.0]}],
        [{"index": 0, "embedding": [float("inf"), 0.0]}, {"index": 1, "embedding": [1.0, 0.0]}],
        [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [1.0, 0.0]}],
    ],
)
def test_malformed_gateway_embeddings_fail(items, monkeypatch):
    client = LiteLLMClient("https://gateway.test", "test", "vision", "embedding")
    monkeypatch.setattr(client, "_post", lambda *args: {"data": items})
    with pytest.raises(GatewayError, match="Invalid embedding response"):
        client.embed(["first", "second"], dimensions=2)


def test_same_dimension_different_profile_is_rejected(container):

    profile = replace(container.embedder.profile, model="different-model")
    store = PgVectorStore(container.settings.postgres_dsn, profile.dimensions, profile=profile)
    with pytest.raises(ValueError, match="profile mismatch"):
        store.ensure_collection(profile.dimensions)


def test_unprofiled_existing_vectors_cannot_be_silently_adopted(container):

    container.vectors.upsert(
        [VectorPoint("legacy", "tenant", [1.0] + [0.0] * 383, {"_id": "legacy"})]
    )
    with transaction(container.settings.postgres_dsn) as cur:
        cur.execute("DELETE FROM embedding_profile")
    with pytest.raises(ValueError, match="no verified embedding profile"):
        container.vectors.ensure_collection(384)


def test_query_instruction_is_applied_through_public_contract(monkeypatch):

    embedder = OnnxEmbedder("test-model", 2, query_instruction="query: ")
    seen = []
    monkeypatch.setattr(embedder, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(embedder, "count_tokens", lambda text: len(text.split()))
    monkeypatch.setattr(
        embedder, "_embed_batch", lambda texts: seen.extend(texts) or [[1.0, 0.0] for _ in texts]
    )
    embedder.embed_query("invoice")
    embedder.embed_documents(["invoice"])
    assert seen == ["query: invoice", "invoice"]


def test_profile_is_immutable_and_identity_includes_instructions():
    profile = EmbeddingProfile("onnx", "model", "revision", "tokenizer", 384, 256, "mean", True)
    assert profile.id != replace(profile, query_instruction="query: ").id
    assert profile.id != replace(profile, revision="new-revision").id
    with pytest.raises(AttributeError):
        profile.model = "changed"


@pytest.mark.parametrize("vector", [[0.0, 0.0], [True, 1.0], ["1", 1.0], [1e100, 1.0]])
def test_invalid_cosine_vectors_are_rejected(vector):
    with pytest.raises(ValueError):
        validate_vectors([vector], 1, 2)


def test_shadow_migration_switches_atomically_and_fences_old_process(container):

    class Challenger(Embedder):
        dim = 2
        profile = EmbeddingProfile("test", "challenger", "v1", "words", 2, 256, "test", True)

        def count_tokens(self, text):
            return len(text.split())

        def embed(self, texts):
            return [[1.0, 0.0] for text in texts]

    container.vectors.upsert(
        [
            VectorPoint(
                "source",
                "tenant",
                [1.0] + [0.0] * 383,
                {"_id": "document", "content": "Invoice 00123"},
            )
        ]
    )
    challenger = Challenger()
    name = build_shadow(container.settings.postgres_dsn, challenger)
    assert container.vectors.search("tenant", [1.0] + [0.0] * 383)[0].chunk_id == "source"
    previous = activate_shadow(container.settings.postgres_dsn, name)
    assert previous.startswith("embedding_previous_")
    with pytest.raises(ValueError, match="differs"):
        container.vectors.search("tenant", [1.0] + [0.0] * 383)
    current = PgVectorStore(container.settings.postgres_dsn, 2, profile=challenger.profile)
    current.ensure_collection(2)
    assert current.search("tenant", [1.0, 0.0])[0].payload["content"] == "Invoice 00123"
    activate_shadow(container.settings.postgres_dsn, previous)
    assert container.vectors.search("tenant", [1.0] + [0.0] * 383)[0].chunk_id == "source"
    with pytest.raises(ValueError, match="differs"):
        current.search("tenant", [1.0, 0.0])


def test_shadow_migration_rejects_changed_corpus(container):

    point = VectorPoint(
        "source", "tenant", [1.0] + [0.0] * 383, {"_id": "document", "content": "Invoice 00123"}
    )
    container.vectors.upsert([point])
    name = build_shadow(container.settings.postgres_dsn, container.embedder)
    point.payload["content"] = "Invoice 00456"
    container.vectors.upsert([point])
    with pytest.raises(ValueError, match="Corpus changed"):
        activate_shadow(container.settings.postgres_dsn, name)


def test_shadow_migration_rejects_same_count_tampering(container):

    container.vectors.upsert(
        [
            VectorPoint(
                "source",
                "tenant",
                [1.0] + [0.0] * 383,
                {"_id": "document", "content": "Invoice 00123"},
            )
        ]
    )
    name = build_shadow(container.settings.postgres_dsn, container.embedder)
    with transaction(container.settings.postgres_dsn) as cursor:
        cursor.execute(
            sql.SQL("UPDATE {} SET payload=jsonb_set(payload,'{{content}}','\"forged\"')").format(
                sql.Identifier(name)
            )
        )
    with pytest.raises(ValueError, match="integrity mismatch"):
        activate_shadow(container.settings.postgres_dsn, name)
