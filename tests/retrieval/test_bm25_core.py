"""Unit tests for the shared BM25 core: query-adaptive fusion weighting,
corpus-relative IDF, and the weighted/unweighted RRF relationship."""
import numpy as np

from app.shared.adapters import bm25


def test_prose_query_stays_balanced():
    """An ordinary natural-language question must fuse 50/50 (weights equal), so
    existing behavior is unchanged for the common case."""
    w_dense, w_lex = bm25.classify_query_weights(
        "how does the reconciliation process affect the closing period")
    assert w_dense == w_lex == 1.0


def test_identifier_query_favors_lexical():
    """A query that is mostly identifiers/codes must weight BM25 above dense —
    an exact lexical hit beats the modest embedder's fuzzy neighbourhood."""
    w_dense, w_lex = bm25.classify_query_weights("MB5S")
    assert w_lex > w_dense

    w_dense2, w_lex2 = bm25.classify_query_weights("ORA-00600 ORA-01555")
    assert w_lex2 > w_dense2


def test_short_keyword_query_with_id_gets_mild_lexical_favor():
    w_dense, w_lex = bm25.classify_query_weights("reset code v2")
    assert w_lex > w_dense
    # milder than the pure-identifier case
    assert w_lex < bm25.classify_query_weights("MB5S")[1]


def test_empty_query_is_balanced():
    assert bm25.classify_query_weights("") == (1.0, 1.0)
    assert bm25.classify_query_weights(None) == (1.0, 1.0)


def test_idf_is_corpus_relative_not_pool_relative():
    """The whole point of Fix 3: IDF must come from corpus-wide document
    frequency. A rare term must score a higher IDF than a common one, and that
    ordering must be driven by the corpus stats, not the local pool."""
    # 'zylantrix' appears in 1 of 100 docs; 'the' appears in 90 of 100.
    df = {"zylantrix": 1, "the": 90}
    stats = bm25.CorpusStats(n_docs=100, df=df, avgdl=20.0)
    assert bm25._idf("zylantrix", stats) > bm25._idf("the", stats)
    # a term the corpus has never seen is treated as maximally rare, not an error
    assert bm25._idf("neverseen", stats) > bm25._idf("zylantrix", stats)
    # nonnegative even for a term in a majority of docs (no epsilon-floor hack)
    assert bm25._idf("the", stats) >= 0.0


def test_pool_only_idf_would_invert_signal():
    """Demonstrate the bug being fixed: if IDF were derived from a narrow pool in
    which every doc contains the term (df == N), that term looks worthless —
    exactly the degeneracy the corpus-stats decoupling prevents."""
    pool_stats = bm25.corpus_stats([["mb5s", "foo"], ["mb5s", "bar"], ["mb5s", "baz"]])
    # df('mb5s') == N == 3 in this pool -> near-zero IDF
    corpus_stats = bm25.CorpusStats(n_docs=1000, df={"mb5s": 3}, avgdl=2.0)
    assert bm25._idf("mb5s", pool_stats) < 0.35            # degenerate in-pool
    assert bm25._idf("mb5s", corpus_stats) > 4.0           # correct corpus-wide


def test_bm25_scores_rank_rare_term_match_first():
    corpus = [
        bm25.tokenize("the annual revenue report for the finance team"),
        bm25.tokenize("employee onboarding checklist for new hires"),
        bm25.tokenize("reconciliation must use transaction code mb5s before closing"),
    ]
    stats = bm25.corpus_stats(corpus)
    scores = bm25.bm25_scores(bm25.tokenize("mb5s"), corpus, stats)
    assert int(np.argmax(scores)) == 2
    assert scores[2] > 0 and scores[0] == 0 and scores[1] == 0


def test_bm25_scores_empty_docs():
    stats = bm25.CorpusStats(n_docs=0, df={}, avgdl=0.0)
    assert bm25.bm25_scores(["x"], [], stats).shape == (0,)


def test_weighted_rrf_equal_weights_equals_plain_rrf():
    dense = np.array([0.9, 0.5, 0.3, 0.1])
    lexical = np.array([0.2, 0.8, 0.4, 0.6])
    plain = bm25.reciprocal_rank_fusion(dense, lexical)
    weighted = bm25.weighted_rrf(dense, lexical, 1.0, 1.0)
    assert np.allclose(plain, weighted)


def test_weighted_rrf_lexical_favor_can_flip_the_winner():
    """With lexical favored, the candidate that wins the lexical ranking should
    be able to overtake the candidate that only wins dense."""
    dense = np.array([0.9, 0.1])       # cand 0 wins dense
    lexical = np.array([0.1, 0.9])     # cand 1 wins lexical
    balanced = bm25.weighted_rrf(dense, lexical, 1.0, 1.0)
    assert balanced[0] == balanced[1]  # exact tie under equal weights
    favored = bm25.weighted_rrf(dense, lexical, 1.0, 2.0)
    assert np.argmax(favored) == 1     # lexical winner now on top
