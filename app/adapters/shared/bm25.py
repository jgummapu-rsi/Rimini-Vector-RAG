"""BM25 lexical scoring + reciprocal rank fusion, shared by any vector-store
adapter that does Python-side ranking over a SQL/file-filtered candidate pool
(today: `localfs`). A flip to Qdrant would use Qdrant's own native sparse+dense
fusion instead of this; `pgvector` uses Postgres-native full-text search
(`ts_rank_cd`) as its lexical signal, not this module's BM25 — see that adapter.

Two correctness properties this module is careful about:

- **Corpus-relative IDF.** BM25 IDF depends on how *rare* a term is across the
  whole corpus. Deriving it from a small, already-similar candidate pool makes
  it degenerate (a term common to the pool looks worthless). `bm25_scores` takes
  explicit `CorpusStats` so the caller controls what "the corpus" is — the
  localfs adapter passes stats over the full tenant corpus it scans anyway.
- **Query-adaptive fusion.** Identifier-heavy queries (part numbers, codes,
  ticket ids) are better served by lexical match than by a modest embedder;
  `classify_query_weights` shifts the fusion weight toward BM25 for those while
  leaving natural-language prose balanced 50/50.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

_WORD_RE = re.compile(r"\w+")

RRF_K = 60  # standard constant (same default Elasticsearch/Qdrant hybrid fusion uses)

# Okapi BM25 free parameters (the conventional defaults; same as rank_bm25).
_BM25_K1 = 1.5
_BM25_B = 0.75

# A deliberately tiny stopword set — just enough to tell "natural-language
# question" (many function words) from "bag of identifiers" (few/none). Not a
# linguistic resource; a query-shape signal for classify_query_weights.
_STOPWORDS = frozenset(
    "a an the of to in on for and or is are was were be been being do does did "
    "how what why when where which who whom this that these those with without "
    "as at by from into about over under between within".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, \\w+ word split. Deterministic, no NLP dependency."""
    return _WORD_RE.findall((text or "").lower())


def searchable_text(content: str, topics: list[str] | None = None,
                    entities: list[str] | None = None, author: str | None = None) -> str:
    """Chunk text + the document's LLM-extracted topics/entities/author, concatenated
    for BM25 tokenization. This is real signal already paid for at ingest time
    (app.pipeline.metadata_extract) but otherwise never touches ranking — folding it
    in here lets keyword search match a normalized entity/topic/author name even
    when the chunk's own wording doesn't. `date` is deliberately excluded: it's a
    structured-filter field, not a lexical one -- a bare year token would false-
    positive-match any chunk that happens to mention that year in passing.
    """
    extra = " ".join((topics or []) + (entities or []) + ([author] if author else []))
    return f"{content} {extra}".strip() if extra else (content or "")


def _is_identifier_token(raw_token: str) -> bool:
    """A token that reads like a code/id rather than a word: contains a digit
    (MB5S, ORA-00600, v2), or is an ALL-CAPS acronym of length >= 2 (SAP, HTTP).
    `raw_token` is case-preserving (pre-lowercasing) so the caps test works."""
    if any(c.isdigit() for c in raw_token):
        return True
    return len(raw_token) >= 2 and raw_token.isupper() and raw_token.isalpha()


def classify_query_weights(query_text: str) -> tuple[float, float]:
    """Return (w_dense, w_lex) fusion weights for `query_text`.

    Natural-language prose stays balanced (1.0, 1.0) — identical to plain RRF, so
    existing behavior is unchanged for ordinary questions. Identifier-heavy
    queries shift weight toward BM25, because an exact lexical hit on a part
    number/code beats a modest embedder's fuzzy neighbourhood (CLAUDE.md §15).
    Pure heuristic, no NLP dependency — same posture as decompose.looks_multi_part.
    """
    raw = _WORD_RE.findall(query_text or "")
    if not raw:
        return 1.0, 1.0
    id_ratio = sum(_is_identifier_token(t) for t in raw) / len(raw)
    has_stopwords = any(t.lower() in _STOPWORDS for t in raw)

    if id_ratio >= 0.5:
        # mostly identifiers ("MB5S", "ORA-00600 fix") — lean hard on lexical
        return 1.0, 2.0
    if id_ratio > 0 and (len(raw) <= 3 or not has_stopwords):
        # a short, keyword-shaped query containing an id — mild lexical favor
        return 1.0, 1.5
    return 1.0, 1.0


@dataclass
class CorpusStats:
    """Corpus-wide statistics BM25 needs, decoupled from the candidate pool so
    IDF reflects term rarity across the whole corpus, not the retrieved subset."""
    n_docs: int
    df: dict[str, int]           # document frequency per term
    avgdl: float                 # average document length (in tokens)


def corpus_stats(corpus_tokens: list[list[str]]) -> CorpusStats:
    """Build CorpusStats from a fully-materialized tokenized corpus. Used by the
    localfs adapter, whose candidate pool IS the whole tenant corpus, so these
    stats are genuinely corpus-wide."""
    n_docs = len(corpus_tokens)
    df: dict[str, int] = {}
    total_len = 0
    for doc in corpus_tokens:
        total_len += len(doc)
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    avgdl = (total_len / n_docs) if n_docs else 0.0
    return CorpusStats(n_docs=n_docs, df=df, avgdl=avgdl)


def _idf(term: str, stats: CorpusStats) -> float:
    """Nonnegative Okapi IDF variant: ln(1 + (N - df + 0.5)/(df + 0.5)). Always
    >= 0 (no epsilon-floor hack for the pathological df > N/2 case that plain
    Okapi produces), and larger for rarer terms."""
    df = stats.df.get(term, 0)
    return math.log(1.0 + (stats.n_docs - df + 0.5) / (df + 0.5))


def bm25_scores(query_tokens: list[str], docs_tokens: list[list[str]],
                stats: CorpusStats, k1: float = _BM25_K1, b: float = _BM25_B) -> np.ndarray:
    """One BM25 score per document in `docs_tokens`, using corpus-wide `stats`
    for IDF and average length. `docs_tokens` is the pool being ranked (may be a
    subset of the corpus `stats` was built from); order aligns with the input.
    """
    if not docs_tokens:
        return np.zeros(0, dtype=np.float64)
    q_terms = set(query_tokens)
    idf = {t: _idf(t, stats) for t in q_terms}
    avgdl = stats.avgdl or 1.0

    out = np.zeros(len(docs_tokens), dtype=np.float64)
    for i, doc in enumerate(docs_tokens):
        if not doc:
            continue
        dl = len(doc)
        denom_len = k1 * (1.0 - b + b * dl / avgdl)
        score = 0.0
        counts: dict[str, int] = {}
        for tok in doc:
            if tok in q_terms:
                counts[tok] = counts.get(tok, 0) + 1
        for term, f in counts.items():
            score += idf[term] * (f * (k1 + 1.0)) / (f + denom_len)
        out[i] = score
    return out


def weighted_rrf(dense_scores: np.ndarray, lexical_scores: np.ndarray,
                 w_dense: float = 1.0, w_lex: float = 1.0, k: int = RRF_K) -> np.ndarray:
    """Weighted reciprocal rank fusion of two rankings (higher score = better)
    into one score per candidate, aligned with the input arrays' shared indexing:

        score = w_dense/(k + dense_rank) + w_lex/(k + lex_rank)   (1-indexed ranks)

    RRF sidesteps calibrating cosine similarity (bounded, ~[-1, 1]) against BM25
    (unbounded, corpus-dependent scale) by fusing on RANK POSITION rather than
    raw score magnitude. The weights let a caller favor one signal — see
    classify_query_weights — without changing that rank-based robustness.
    """
    n = len(dense_scores)
    dense_rank = np.empty(n, dtype=np.int64)
    dense_rank[np.argsort(-dense_scores)] = np.arange(1, n + 1)
    lex_rank = np.empty(n, dtype=np.int64)
    lex_rank[np.argsort(-lexical_scores)] = np.arange(1, n + 1)

    return w_dense / (k + dense_rank) + w_lex / (k + lex_rank)


def reciprocal_rank_fusion(dense_scores: np.ndarray, bm25_scores_: np.ndarray,
                           k: int = RRF_K) -> np.ndarray:
    """Unweighted RRF (equal 50/50 weighting) — the balanced special case of
    `weighted_rrf`. Kept as the stable, parameter-free entry point."""
    return weighted_rrf(dense_scores, bm25_scores_, 1.0, 1.0, k)
