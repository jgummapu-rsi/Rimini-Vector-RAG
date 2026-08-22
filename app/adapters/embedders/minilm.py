"""MiniLMEmbedder: real local semantic embeddings via ONNX.

Runs sentence-transformers/all-MiniLM-L6-v2 (384-dim) through onnxruntime with a
HuggingFace tokenizer — no torch, no compiler. Model + tokenizer are downloaded
once and cached by huggingface_hub. Embeddings are mean-pooled over tokens
(attention-masked) and L2-normalized, matching the sentence-transformers recipe.

Loading is lazy: `dim` is known without downloading, so container startup and the
vector-store dimension don't require the model until the first embed call.
"""
from __future__ import annotations

from app.pipeline.tokens import EMBED_MAX_TOKENS
from app.ports.embedder import Embedder

_REPO = "Xenova/all-MiniLM-L6-v2"
_DIM = 384
_MAX_LEN = EMBED_MAX_TOKENS  # single source of truth, shared with chunk sizing (app.pipeline.tokens)

# process-level cache: (repo, max_length) -> (session, tokenizer, input_names)
# so multiple embedder instances (and every test) share one loaded model.
_MODEL_CACHE: dict = {}


class MiniLMEmbedder(Embedder):
    def __init__(self, repo: str = _REPO, max_length: int = _MAX_LEN):
        self._repo = repo
        self._max_length = max_length
        self._sess = None
        self._tok = None
        self._input_names: set[str] = set()

    @property
    def dim(self) -> int:
        return _DIM

    @property
    def max_tokens(self) -> int:
        return self._max_length

    # count_tokens is inherited from Embedder: its default counts in the MiniLM
    # tokenizer, which is exactly this model's tokenizer.

    def _ensure_loaded(self) -> None:
        if self._sess is not None:
            return
        key = (self._repo, self._max_length)
        cached = _MODEL_CACHE.get(key)
        if cached is None:
            cached = self._load(key)
            _MODEL_CACHE[key] = cached
        self._sess, self._tok, self._input_names = cached

    def _load(self, key):
        from app.runtime import ensure_native_runtime
        ensure_native_runtime()

        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        model_path = hf_hub_download(self._repo, "onnx/model.onnx")
        tok_path = hf_hub_download(self._repo, "tokenizer.json")

        tok = Tokenizer.from_file(tok_path)
        tok.enable_truncation(max_length=self._max_length)
        tok.enable_padding()
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        return sess, tok, {i.name for i in sess.get_inputs()}

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        import numpy as np

        encs = self._tok.encode_batch(texts)
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)

        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)

        hidden = self._sess.run(None, feeds)[0]                    # (b, seq, 384)
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (pooled / norms).astype(np.float32).tolist()
