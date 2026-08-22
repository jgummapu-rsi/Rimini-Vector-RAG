"""Embedder tests.

- GatewayEmbedder batching is verified with a stub client (no network).
- The real ONNX MiniLM embedder runs for real (model is cached after first load).
"""
from app.adapters.embedders.gateway import GatewayEmbedder


class _StubClient:
    def __init__(self):
        self.batch_sizes = []

    def embed(self, texts):
        self.batch_sizes.append(len(texts))
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def test_gateway_embedder_batches():
    stub = _StubClient()
    emb = GatewayEmbedder(stub, dim=4, batch_size=2)
    out = emb.embed(["a", "b", "c", "d", "e"])
    assert len(out) == 5
    assert emb.dim == 4
    assert stub.batch_sizes == [2, 2, 1]


def test_gateway_embedder_empty():
    assert GatewayEmbedder(_StubClient(), dim=4).embed([]) == []


def test_minilm_real_embeddings():
    from app.adapters.embedders.minilm import MiniLMEmbedder
    emb = MiniLMEmbedder()
    assert emb.dim == 384
    v = emb.embed(["quarterly revenue", "annual sales table", "a cat on a sofa"])
    assert all(len(x) == 384 for x in v)
    cos = lambda a, b: sum(i * j for i, j in zip(a, b))
    assert cos(v[0], v[1]) > cos(v[0], v[2])  # related > unrelated
