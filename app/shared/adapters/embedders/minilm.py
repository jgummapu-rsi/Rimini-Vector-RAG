"""MiniLMEmbedder: real local semantic embeddings via ONNX.

Runs sentence-transformers/all-MiniLM-L6-v2 (384-dim) through onnxruntime with a
HuggingFace tokenizer — no torch, no compiler. Model + tokenizer are downloaded
once and cached by huggingface_hub. Embeddings are mean-pooled over tokens
(attention-masked) and L2-normalized, matching the sentence-transformers recipe.

Loading is lazy: `dim` is known without downloading, so container startup and the
vector-store dimension don't require the model until the first embed call.
"""

from __future__ import annotations

from threading import Lock

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from app.ingest.pipeline.tokens import EMBED_MAX_TOKENS, count_tokens_for
from app.shared.adapters.embedders.profile import tokenizer_identity
from app.shared.domain.embedding import EmbeddingProfile
from app.shared.execution import check_execution, run_onnx
from app.shared.model_loading import model_load_lock, require_startup_loading
from app.shared.ports.embedder import Embedder
from app.shared.runtime import ensure_native_runtime

_REPO = "Xenova/all-MiniLM-L6-v2"
_DIM = 384
_MAX_LEN = EMBED_MAX_TOKENS

_BATCH = 64

_MODEL_CACHE: dict = {}
_LOAD_LOCK = Lock()


class MiniLMEmbedder(Embedder):
    """Local ONNX all-MiniLM-L6-v2 embedder: 384-dim, mean-pooled, L2-normalized.

    `count_tokens` is inherited from `Embedder` unchanged: its default counts in
    the MiniLM tokenizer, which is exactly this model's tokenizer.
    """

    def __init__(self, repo: str = _REPO, max_length: int = _MAX_LEN, batch_size: int = _BATCH):
        """Configure the HF `repo`, truncation `max_length`, and embed `batch_size`."""
        self._repo = repo
        self._max_length = max_length
        self._batch = max(1, batch_size)
        self._sess = None
        self._tok = None
        self._input_names: set[str] = set()
        self._profile = None

    @property
    def profile(self) -> EmbeddingProfile:
        if self._profile is None:
            require_startup_loading()
            revision, tokenizer = tokenizer_identity(self._repo)
            self._profile = EmbeddingProfile(
                "onnx", self._repo, revision, tokenizer, self.dim, self.max_tokens, "mean", True
            )
        return self._profile

    def count_tokens(self, text: str) -> int:
        return count_tokens_for(self._repo, text, revision=self.profile.revision)

    @property
    def dim(self) -> int:
        """Dimensionality of vectors this embedder produces (384)."""
        return _DIM

    @property
    def max_tokens(self) -> int:
        """This model's hard truncation limit."""
        return self._max_length

    def _ensure_loaded(self) -> None:
        """Load the ONNX session and tokenizer once, sharing the process-level cache."""
        check_execution()
        if self._sess is not None:
            return
        with model_load_lock(_LOAD_LOCK):
            key = (self._repo, self.profile.revision, self._max_length)
            cached = _MODEL_CACHE.get(key)
            if cached is None:
                require_startup_loading()
                cached = self._load(key)
                _MODEL_CACHE[key] = cached
            self._sess, self._tok, self._input_names = cached

    def prepare(self) -> None:
        self._ensure_loaded()
        self.embed_query("startup validation")

    def _load(self, key):
        """Download (if needed) and construct the ONNX session + tokenizer for `key`."""
        ensure_native_runtime()

        # Native DLL discovery must happen before importing ONNX Runtime.
        import onnxruntime as ort  # noqa: PLC0415

        model_path = hf_hub_download(self._repo, "onnx/model.onnx", revision=self.profile.revision)
        tok_path = hf_hub_download(self._repo, "tokenizer.json", revision=self.profile.revision)

        tok = Tokenizer.from_file(tok_path)
        tok.enable_truncation(max_length=self._max_length)
        tok.enable_padding()
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        return sess, tok, {i.name for i in sess.get_inputs()}

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one 384-dim embedding per text in `texts`, batched."""
        if not texts:
            return []
        self.validate_inputs(texts)
        self._ensure_loaded()

        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            check_execution()
            out.extend(self._embed_batch(texts[i : i + self._batch]))
        return out

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """One forward pass. Padding is per-batch (to the longest text in THIS
        batch), so batching also avoids padding every short chunk out to the
        length of the single longest one in the document."""

        encs = self._tok.encode_batch(texts)
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)

        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)

        hidden = run_onnx(self._sess, feeds)[0]
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (pooled / norms).astype(np.float32).tolist()
