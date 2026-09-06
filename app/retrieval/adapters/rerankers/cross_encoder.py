"""CrossEncoderReranker: real local re-ranking via ONNX.

Runs cross-encoder/ms-marco-MiniLM-L-6-v2 (Xenova's ONNX conversion) through
onnxruntime with a HuggingFace tokenizer -- no torch, no compiler, same recipe
as MiniLMEmbedder (app.shared.adapters.embedders.minilm). Unlike the embedder, this
model scores a (query, document) PAIR jointly in one forward pass -- that
joint attention is what makes a cross-encoder more precise than independent
embedding similarity, and also why it only runs over an already-narrowed
candidate pool rather than the whole corpus.

Loading is lazy, same as the embedder: the model isn't downloaded/loaded until
the first `score()` call.
"""
from __future__ import annotations

from app.retrieval.ports.reranker import Reranker

_REPO = "Xenova/ms-marco-MiniLM-L-6-v2"
_MAX_LEN = 512  # this model's own limit -- unrelated to EMBED_MAX_TOKENS (a different model)

# Pairs per forward pass. Same memory reasoning as the embedders, but the pool
# size here is driven by RERANK_MIN_CANDIDATES (50) and multiplies again on the
# decomposed query path (one retrieval per sub-question, merged), so the input
# is not as small as "just the top k" suggests. 32 pairs at 512 tokens keeps
# peak allocation flat regardless of how wide the candidate pool gets.
_BATCH = 32

# process-level cache: repo -> (session, tokenizer, input_names), shared across
# every CrossEncoderReranker instance (and every test) so the model loads once.
_MODEL_CACHE: dict = {}


class CrossEncoderReranker(Reranker):
    """Local ONNX cross-encoder reranker (query, document pairs scored jointly)."""

    def __init__(self, repo: str = _REPO, max_length: int = _MAX_LEN,
                 batch_size: int = _BATCH):
        """Configure the HF repo, max token length, and per-forward-pass batch size."""
        self._repo = repo
        self._max_length = max_length
        self._batch = max(1, batch_size)
        self._sess = None
        self._tok = None
        self._input_names: set[str] = set()

    def _ensure_loaded(self) -> None:
        """Load the model/tokenizer on first use, from the process-level cache if present."""
        if self._sess is not None:
            return
        cached = _MODEL_CACHE.get(self._repo)
        if cached is None:
            cached = self._load()
            _MODEL_CACHE[self._repo] = cached
        self._sess, self._tok, self._input_names = cached

    def _load(self):
        """Download and initialize the ONNX session and tokenizer for `self._repo`."""
        from app.shared.runtime import ensure_native_runtime
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

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Score every document against `query`, batching internally."""
        if not documents:
            return []
        self._ensure_loaded()

        out: list[float] = []
        for i in range(0, len(documents), self._batch):
            out.extend(self._score_batch(query, documents[i:i + self._batch]))
        return out

    def _score_batch(self, query: str, documents: list[str]) -> list[float]:
        """One forward pass over at most `_batch` (query, document) pairs."""
        import numpy as np

        encs = self._tok.encode_batch([(query, doc) for doc in documents])
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)

        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.array([e.type_ids for e in encs], dtype=np.int64)

        logits = self._sess.run(None, feeds)[0]
        return np.asarray(logits, dtype=np.float64).reshape(-1).tolist()
