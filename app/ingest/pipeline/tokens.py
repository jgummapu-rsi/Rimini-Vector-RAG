"""Token counting for chunk sizing.

Must measure against the tokenizer the embedding model actually uses, not a
generic proxy. `all-MiniLM-L6-v2` (app/adapters/embedders/minilm.py) hard-
truncates every input to `EMBED_MAX_TOKENS` WordPiece tokens. A generic BPE
tokenizer (e.g. tiktoken's cl100k_base) does NOT measure the same thing --
verified on our own corpus, cl100k undercounts real WordPiece length by a wide
enough margin that chunks sized to "512 cl100k tokens" were routinely 500+
real tokens, more than double the model's actual limit. That silently drops
roughly the back half of most chunks from ever being embedded, with no error,
no log, nothing -- dense/semantic search was only ever seeing the first ~256
tokens of most chunks. See KHUB_COMPARISON_REPORT.md for the measured before/
after. `count_tokens` reports the TRUE (untruncated) length in the same
tokenizer the embedder uses, so sizing decisions and the embedder's real limit
are the same ruler.

The embedder is the SOURCE OF TRUTH for which tokenizer + limit apply: a
different embedder (e.g. bge-base, 512 tokens, a different WordPiece vocab)
needs a different ruler. `count_tokens_for(repo, text)` counts in any model's
tokenizer; `count_tokens(text)` is the MiniLM default kept for the many callers
(and tests) that predate the multi-embedder path. Embedders expose their own
counting via app.shared.ports.embedder.Embedder.count_tokens, which routes here.
"""

from __future__ import annotations

import logging
import time

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

log = logging.getLogger(__name__)

EMBED_MAX_TOKENS = 256

_DEFAULT_REPO = "Xenova/all-MiniLM-L6-v2"

_RETRY_COOLDOWN_SECONDS = 60.0

_COUNTER_CACHE: dict[str, object] = {}


def _counting_tokenizer(repo: str, revision: str | None = None):
    key = (repo, revision) if revision else repo
    cached = _COUNTER_CACHE.get(key)
    if cached is not None:
        if not isinstance(cached, float):
            return cached
        if time.monotonic() < cached:
            raise RuntimeError(f"Required tokenizer {repo} is unavailable; retry after cooldown")

    try:
        tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json", revision=revision))

        tok.no_truncation()
        tok.no_padding()
        _COUNTER_CACHE[key] = tok
        return tok
    except Exception as exc:
        _COUNTER_CACHE[key] = time.monotonic() + _RETRY_COOLDOWN_SECONDS
        raise RuntimeError(f"Required tokenizer {repo} could not be loaded") from exc


def count_tokens_for(repo: str, text: str, revision: str | None = None) -> int:
    if not text:
        return 0
    tok = _counting_tokenizer(repo, revision)
    return len(tok.encode(text).ids)


def count_tokens(text: str) -> int:
    """MiniLM-tokenizer token count (the default embedder). Prefer an embedder's
    own `count_tokens` when a specific embedder is in play; this stays as the
    process-wide default for callers that don't have one to hand."""
    return count_tokens_for(_DEFAULT_REPO, text)
