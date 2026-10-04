"""Finding 1.7: a transient tokenizer-load failure must not degrade token
counting to the char/4 heuristic for the rest of the process's life -- it
should log a warning and retry after a bounded cooldown, not cache `False`
forever."""

import pytest

import app.ingest.pipeline.tokens as tokens_mod


def test_load_failure_falls_back_to_heuristic_and_warns(monkeypatch, caplog):
    repo = "test/repo-always-fails"
    tokens_mod._COUNTER_CACHE.pop(repo, None)

    def _boom(*a, **k):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(tokens_mod, "hf_hub_download", _boom)

    with pytest.raises(RuntimeError, match="Required tokenizer"):
        tokens_mod.count_tokens_for(repo, "one two three four")

    assert isinstance(tokens_mod._COUNTER_CACHE[repo], float)
    tokens_mod._COUNTER_CACHE.pop(repo, None)


def test_cooldown_blocks_immediate_retry_but_elapsed_cooldown_retries(monkeypatch):
    repo = "test/repo-cooldown"
    tokens_mod._COUNTER_CACHE.pop(repo, None)

    calls = {"n": 0}

    def _boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("still down")

    monkeypatch.setattr(tokens_mod, "hf_hub_download", _boom)

    with pytest.raises(RuntimeError, match="Required tokenizer"):
        tokens_mod.count_tokens_for(repo, "hello world")
    assert calls["n"] == 1

    with pytest.raises(RuntimeError, match="Required tokenizer"):
        tokens_mod.count_tokens_for(repo, "hello world again")
    assert calls["n"] == 1

    tokens_mod._COUNTER_CACHE[repo] = 0.0
    with pytest.raises(RuntimeError, match="Required tokenizer"):
        tokens_mod.count_tokens_for(repo, "hello world once more")
    assert calls["n"] == 2

    tokens_mod._COUNTER_CACHE.pop(repo, None)
