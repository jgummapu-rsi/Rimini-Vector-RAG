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

from threading import Lock

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from app.retrieval.ports.reranker import Reranker
from app.shared.adapters.embedders.profile import tokenizer_identity
from app.shared.execution import check_execution, run_onnx
from app.shared.model_loading import model_load_lock, require_startup_loading
from app.shared.runtime import ensure_native_runtime

_REPO = "Xenova/ms-marco-MiniLM-L-6-v2"
_MAX_LEN = 512

_BATCH = 32

_MODEL_CACHE: dict = {}
_LOAD_LOCK = Lock()


class CrossEncoderReranker(Reranker):
    """Local ONNX cross-encoder reranker (query, document pairs scored jointly)."""

    def __init__(self, repo: str = _REPO, max_length: int = _MAX_LEN, batch_size: int = _BATCH):
        """Configure the HF repo, max token length, and per-forward-pass batch size."""
        self._repo = repo
        self._max_length = max_length
        self._batch = max(1, batch_size)
        self._sess = None
        self._tok = None
        self._input_names: set[str] = set()
        self._revision = None

    def _ensure_loaded(self) -> None:
        """Load the model/tokenizer on first use, from the process-level cache if present."""
        check_execution()
        if self._sess is not None:
            return
        with model_load_lock(_LOAD_LOCK):
            if self._revision is None:
                require_startup_loading()
                self._revision = tokenizer_identity(self._repo)[0]
            key = (self._repo, self._revision, self._max_length)
            cached = _MODEL_CACHE.get(key)
            if cached is None:
                require_startup_loading()
                cached = self._load()
                _MODEL_CACHE[key] = cached
            self._sess, self._tok, self._input_names = cached

    def prepare(self) -> None:
        self._ensure_loaded()
        self.score("startup validation", ["startup validation"])

    def _load(self):
        """Download and initialize the ONNX session and tokenizer for `self._repo`."""
        ensure_native_runtime()

        # Native DLL discovery must happen before importing ONNX Runtime.
        import onnxruntime as ort  # noqa: PLC0415

        model_path = hf_hub_download(self._repo, "onnx/model.onnx", revision=self._revision)
        tok_path = hf_hub_download(self._repo, "tokenizer.json", revision=self._revision)

        tok = Tokenizer.from_file(tok_path)
        tok.enable_truncation(
            max_length=self._max_length,
            strategy="only_second",
            stride=min(64, self._max_length // 4),
        )
        tok.enable_padding()
        options = ort.SessionOptions()
        # Compose allocates two CPUs. Host-wide default thread pools cause
        # oversubscription and throttling, especially during repeated reranks.
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        sess = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        return sess, tok, {i.name for i in sess.get_inputs()}

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Score every document against `query`, batching internally."""
        if not documents:
            return []
        self._ensure_loaded()

        out: list[float] = []
        for i in range(0, len(documents), self._batch):
            check_execution()
            out.extend(self._score_batch(query, documents[i : i + self._batch]))
        return out

    def _score_batch(self, query: str, documents: list[str]) -> list[float]:
        """Score all passage windows, returning the best score per document.

        Ingest and rerank tokenizers differ. Even a valid 476-token embedding
        chunk can exceed this model's pair budget; silently truncating its tail
        discards exactly the evidence the first-stage search found.
        """

        # Bound the query independently so only_second always has passage room.
        query_encoding = self._tok.encode(query, add_special_tokens=False)
        query_limit = max(1, self._max_length // 4)
        if len(query_encoding.ids) > query_limit:
            query = self._tok.decode(query_encoding.ids[:query_limit])
        counter = Tokenizer.from_str(self._tok.to_str())
        counter.no_truncation()
        counter.no_padding()
        pairs, owners = [], []
        for index, document in enumerate(documents):
            heading, separator, body = document.partition("\n\n")
            prefix = heading + "\n\n" if separator and len(heading) <= 128 else ""
            body = body if prefix else document
            encoding = counter.encode(body, add_special_tokens=False)
            overhead = len(counter.encode(query, prefix).ids)
            # Passage-sized windows reduce topic dilution as well as avoiding
            # truncation. Repeat the source heading on every window so a tail
            # fact retains its subject when ranked independently.
            capacity = max(1, min(192, self._max_length - overhead - 4))
            step = max(1, capacity - min(64, capacity // 4))
            starts = range(0, max(1, len(encoding.ids)), step)
            for start in starts:
                end = min(start + capacity, len(encoding.ids))
                text = (
                    body[encoding.offsets[start][0] : encoding.offsets[end - 1][1]]
                    if encoding.ids
                    else body
                )
                pairs.append((query, prefix + text))
                owners.append(index)
                if end == len(encoding.ids):
                    break
        encs = self._tok.encode_batch(pairs)
        windows = [
            (index, window)
            for index, encoding in zip(owners, encs, strict=False)
            for window in [encoding, *encoding.overflowing]
        ]
        scores = [float("-inf")] * len(documents)
        for start in range(0, len(windows), self._batch):
            check_execution()
            batch = windows[start : start + self._batch]
            width = max(len(window.ids) for _, window in batch)
            pad_id = self._tok.padding["pad_id"]
            feeds = {
                "input_ids": np.array(
                    [e.ids + [pad_id] * (width - len(e.ids)) for _, e in batch], dtype=np.int64
                ),
                "attention_mask": np.array(
                    [e.attention_mask + [0] * (width - len(e.ids)) for _, e in batch],
                    dtype=np.int64,
                ),
            }
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.array(
                    [e.type_ids + [0] * (width - len(e.ids)) for _, e in batch], dtype=np.int64
                )
            logits = np.asarray(run_onnx(self._sess, feeds)[0], dtype=np.float64).reshape(-1)
            if len(logits) != len(batch) or not np.isfinite(logits).all():
                raise ValueError("Reranker must return one finite score per passage window")
            for (index, _), score in zip(batch, logits, strict=False):
                scores[index] = max(scores[index], float(score))
        return scores
