"""OnnxEmbedder: configurable local ONNX sentence embedder (no torch).

Generalizes MiniLMEmbedder to any HuggingFace model that ships an ONNX export
(onnx/model.onnx + tokenizer.json), so a stronger embedder (e.g. BAAI bge) can be
swapped in without adding torch/sentence-transformers. The two things that must
match each model's training recipe -- otherwise scores DROP rather than improve:

  - pooling: 'cls' (bge and most retrieval BERT/-RoBERTa models take the [CLS]/
    position-0 hidden state) vs 'mean' (all-MiniLM / e5 mean-pool over tokens).
  - query instruction: bge-en-v1.5 and e5 prepend a short instruction to the
    QUERY side only (asymmetric encoding). `embed(..., is_query=True)` applies
    `query_instruction`; passages are embedded with no prefix.

Everything is L2-normalized when `normalize=True` (required for the pgvector
HNSW `vector_cosine_ops` index to behave like cosine similarity).

Loading is lazy and process-cached, exactly like MiniLMEmbedder.
"""
from __future__ import annotations

from app.ports.embedder import Embedder

# process-level cache: (repo, max_length) -> (session, tokenizer, input_names, output_names)
_MODEL_CACHE: dict = {}


class OnnxEmbedder(Embedder):
    def __init__(
        self, repo: str, dim: int, *,
        pooling: str = "cls", query_instruction: str = "",
        max_length: int = 512, normalize: bool = True,
    ):
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")
        self._repo = repo
        self._dim = dim
        self._pooling = pooling
        self._query_instruction = query_instruction
        self._max_length = max_length
        self._normalize = normalize
        self._sess = None
        self._tok = None
        self._input_names: set[str] = set()
        self._output_names: list[str] = []

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def max_tokens(self) -> int:
        # The embedding tokenizer truncates to this; chunks must stay within it.
        return self._max_length

    def count_tokens(self, text: str) -> int:
        # True length in THIS model's own tokenizer (e.g. bge's WordPiece vocab),
        # not MiniLM's -- different vocab, different counts. Routed through the
        # shared per-repo counting cache in app.pipeline.tokens.
        from app.pipeline.tokens import count_tokens_for
        return count_tokens_for(self._repo, text)

    def _ensure_loaded(self) -> None:
        if self._sess is not None:
            return
        key = (self._repo, self._max_length)
        cached = _MODEL_CACHE.get(key)
        if cached is None:
            cached = self._load()
            _MODEL_CACHE[key] = cached
        self._sess, self._tok, self._input_names, self._output_names = cached

    def _load(self):
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
        return (sess, tok, {i.name for i in sess.get_inputs()},
                [o.name for o in sess.get_outputs()])

    def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        self._ensure_loaded()
        import numpy as np

        if is_query and self._query_instruction:
            texts = [self._query_instruction + t for t in texts]

        encs = self._tok.encode_batch(texts)
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)

        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)

        # last_hidden_state: (batch, seq, dim). Prefer the named output; fall back
        # to the first output (Xenova feature-extraction ONNX emits token embeddings,
        # pooling/normalization are done here, not inside the graph).
        outs = self._sess.run(None, feeds)
        if "last_hidden_state" in self._output_names:
            hidden = outs[self._output_names.index("last_hidden_state")]
        else:
            hidden = outs[0]

        if self._pooling == "cls":
            pooled = hidden[:, 0, :]                                  # [CLS] / position 0
        else:
            m = mask[..., None].astype(np.float32)
            pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)

        pooled = pooled.astype(np.float32)
        if self._normalize:
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            pooled = pooled / norms
        return pooled.tolist()
