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

# Default embedding model's real hard limit -- the MiniLM baseline. Concrete
# embedders report their own via Embedder.max_tokens; this remains the fallback
# for the default (MiniLM) counting path and for offline size heuristics.
EMBED_MAX_TOKENS = 256

_DEFAULT_REPO = "Xenova/all-MiniLM-L6-v2"

# Per-repo cache of counting tokenizers (truncation + padding DISABLED, so we
# measure true length). A `False` value means that repo's tokenizer is
# permanently unavailable this process (offline / uncached first run) and we
# fall back to the char heuristic without retrying the download every call.
_COUNTER_CACHE: dict[str, object] = {}


def _counting_tokenizer(repo: str):
    """Load (once, process-cached) a no-truncation/no-padding tokenizer for
    `repo`. Returns the tokenizer, or False if it can't be loaded."""
    cached = _COUNTER_CACHE.get(repo)
    if cached is not None:
        return cached
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        # A model's tokenizer.json often bakes in truncation/padding defaults
        # (e.g. MiniLM's is 128, not even the embedder's real 256) -- both must
        # be disabled to measure true length; leaving padding on silently
        # inflates the count for any text shorter than the pad length.
        tok.no_truncation()
        tok.no_padding()
        _COUNTER_CACHE[repo] = tok
        return tok
    except Exception:  # pragma: no cover - offline / first-run-without-cache fallback
        _COUNTER_CACHE[repo] = False
        return False


def count_tokens_for(repo: str, text: str) -> int:
    """True (untruncated) token length of `text` in `repo`'s tokenizer. Falls
    back to a ~4-chars-per-token heuristic only if the tokenizer is
    unavailable."""
    if not text:
        return 0
    tok = _counting_tokenizer(repo)
    if tok:
        return len(tok.encode(text).ids)
    return max(1, (len(text) + 3) // 4)


def count_tokens(text: str) -> int:
    """MiniLM-tokenizer token count (the default embedder). Prefer an embedder's
    own `count_tokens` when a specific embedder is in play; this stays as the
    process-wide default for callers that don't have one to hand."""
    return count_tokens_for(_DEFAULT_REPO, text)
