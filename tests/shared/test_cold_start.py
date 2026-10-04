import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.retrieval.adapters.rerankers import cross_encoder
from app.shared.adapters.embedders import minilm
from app.shared.adapters.embedders.minilm import MiniLMEmbedder
from app.shared.domain.embedding import EmbeddingProfile
from app.shared.execution import ExecutionBudget, RequestAborted, execution_scope
from app.shared.model_loading import model_load_lock


def test_cold_model_is_not_loaded_inside_request(monkeypatch):
    embedder = MiniLMEmbedder(repo="unloaded/test-model")
    with (
        execution_scope(ExecutionBudget(1)),
        pytest.raises(RequestAborted, match="not initialized"),
    ):
        embedder.embed_query("invoice")


def test_cancelled_waiter_does_not_wait_for_cold_loader():
    lock = threading.Lock()
    lock.acquire()
    start = time.monotonic()
    try:
        with execution_scope(ExecutionBudget(0.1)), pytest.raises(RequestAborted):
            with model_load_lock(lock):
                raise AssertionError("must not acquire held lock")
    finally:
        lock.release()
    assert time.monotonic() - start < 1


def test_simultaneous_cold_instances_share_one_load(monkeypatch):
    cache = {}
    monkeypatch.setattr(minilm, "_MODEL_CACHE", cache)
    calls = []
    profile = EmbeddingProfile("onnx", "test", "revision", "tokenizer", 384, 256, "mean", True)

    def load(self, key):
        calls.append(key)
        time.sleep(0.05)
        return object(), object(), {"input_ids"}

    monkeypatch.setattr(minilm.MiniLMEmbedder, "_load", load)
    embedders = [minilm.MiniLMEmbedder() for _ in range(8)]
    for embedder in embedders:
        embedder._profile = profile
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda embedder: embedder._ensure_loaded(), embedders))
    assert len(calls) == 1
    assert all(embedder._sess is embedders[0]._sess for embedder in embedders)


def test_reranker_cache_separates_token_limits(monkeypatch):
    monkeypatch.setattr(cross_encoder, "_MODEL_CACHE", {})
    monkeypatch.setattr(cross_encoder, "tokenizer_identity", lambda repo: ("revision", "hash"))
    monkeypatch.setattr(
        cross_encoder.CrossEncoderReranker, "_load", lambda self: (object(), object(), set())
    )
    first = cross_encoder.CrossEncoderReranker(max_length=128)
    second = cross_encoder.CrossEncoderReranker(max_length=512)
    first._ensure_loaded()
    second._ensure_loaded()
    assert first._sess is not second._sess


def test_container_warms_models_before_readiness(container):
    assert container.embedder._sess is not None
    container.readiness_check()
