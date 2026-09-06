"""Embedder tests.

- GatewayEmbedder batching is verified with a stub client (no network).
- The real ONNX MiniLM embedder runs for real (model is cached after first load).
"""
from app.shared.adapters.embedders.gateway import GatewayEmbedder


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
    from app.shared.adapters.embedders.minilm import MiniLMEmbedder
    emb = MiniLMEmbedder()
    assert emb.dim == 384
    v = emb.embed(["quarterly revenue", "annual sales table", "a cat on a sofa"])
    assert all(len(x) == 384 for x in v)
    cos = lambda a, b: sum(i * j for i, j in zip(a, b))
    assert cos(v[0], v[1]) > cos(v[0], v[2])  # related > unrelated


# --------------------------------------------------------- local batching --
# The ingest path embeds a whole document's chunks in ONE embed() call, so an
# unbatched forward pass allocated a (n_chunks, seq, dim) tensor -- gigabytes,
# and an OOM worker, for a large PDF. Batching bounds that regardless of input
# size; these tests pin both the bound and that batching changes nothing about
# the OUTPUT.


def test_minilm_batches_large_inputs():
    from app.shared.adapters.embedders.minilm import MiniLMEmbedder
    emb = MiniLMEmbedder(batch_size=4)
    seen = []
    real = emb._embed_batch
    emb._embed_batch = lambda texts: seen.append(len(texts)) or real(texts)

    out = emb.embed([f"chunk number {i}" for i in range(10)])
    assert len(out) == 10
    assert seen == [4, 4, 2], "must split into bounded forward passes"
    assert max(seen) <= 4


def test_minilm_batching_does_not_change_the_vectors():
    """Batch boundaries must not perturb results: pooling is per-row and
    normalisation is per-row, so a batch of 1 and a batch of 8 must agree."""
    from app.shared.adapters.embedders.minilm import MiniLMEmbedder
    texts = [f"passage about topic {i}" for i in range(8)]
    one_shot = MiniLMEmbedder(batch_size=64).embed(texts)
    batched = MiniLMEmbedder(batch_size=3).embed(texts)

    assert len(one_shot) == len(batched) == 8
    for a, b in zip(one_shot, batched):
        assert max(abs(x - y) for x, y in zip(a, b)) < 1e-5


def test_minilm_preserves_input_order_across_batches():
    from app.shared.adapters.embedders.minilm import MiniLMEmbedder
    emb = MiniLMEmbedder(batch_size=2)
    texts = ["quarterly revenue", "a cat on a sofa", "annual sales table",
             "the weather today", "profit margins"]
    out = emb.embed(texts)
    # re-embedding each text alone must match its position in the batched run
    for i, t in enumerate(texts):
        alone = emb.embed([t])[0]
        assert max(abs(x - y) for x, y in zip(out[i], alone)) < 1e-5


def test_minilm_rejects_a_nonsense_batch_size():
    from app.shared.adapters.embedders.minilm import MiniLMEmbedder
    # clamped, not crashed -- a 0/negative batch would loop forever
    assert MiniLMEmbedder(batch_size=0)._batch == 1
    assert MiniLMEmbedder(batch_size=-5)._batch == 1


def test_cross_encoder_batches_large_candidate_pools():
    from app.retrieval.adapters.rerankers.cross_encoder import CrossEncoderReranker
    rr = CrossEncoderReranker(batch_size=3)
    seen = []
    real = rr._score_batch
    rr._score_batch = lambda q, docs: seen.append(len(docs)) or real(q, docs)

    scores = rr.score("what is SM13?", [f"passage {i}" for i in range(7)])
    assert len(scores) == 7
    assert seen == [3, 3, 1]


def test_cross_encoder_batching_preserves_score_alignment():
    """Scores must stay aligned with input order across batch boundaries --
    misalignment here would silently reorder search results."""
    from app.retrieval.adapters.rerankers.cross_encoder import CrossEncoderReranker
    docs = [
        "SM13 shows update terminations that occurred after COMMIT WORK.",
        "A cat sat on a sofa in the afternoon sun.",
        "Update terminations are inspected with transaction SM13.",
        "Bananas are a popular yellow fruit.",
    ]
    q = "What transaction code shows update terminations?"
    one_shot = CrossEncoderReranker(batch_size=64).score(q, docs)
    batched = CrossEncoderReranker(batch_size=1).score(q, docs)

    assert len(one_shot) == len(batched) == 4
    for a, b in zip(one_shot, batched):
        assert abs(a - b) < 1e-4
    # the two relevant passages still outrank the two irrelevant ones
    assert min(batched[0], batched[2]) > max(batched[1], batched[3])
