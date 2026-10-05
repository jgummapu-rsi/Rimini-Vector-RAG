from types import SimpleNamespace

import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, processors

import app.retrieval.rag.query as query
from app.retrieval.adapters.rerankers import cross_encoder
from app.retrieval.rag.grounding import pack_evidence
from app.retrieval.rag.query import retrieve_chunks
from app.shared.ports.vector_store import SearchHit, VectorPoint


def test_decomposition_keeps_original_constraints_and_requested_depth(monkeypatch):
    monkeypatch.setattr(query, "looks_multi_part", lambda _: True)
    monkeypatch.setattr(query, "decompose_question", lambda *a: ["alpha", "beta"])
    calls = []

    def search(tenant, vector, top_k, access, query_text):
        calls.append((query_text, top_k))
        return [
            SearchHit(query_text + str(i), 1.0, {"content": query_text + str(i)})
            for i in range(top_k)
        ]

    container = SimpleNamespace(
        settings=SimpleNamespace(chat_model="test"),
        gateway=None,
        reranker=None,
        embedder=SimpleNamespace(embed_query=lambda _: [1.0]),
        vectors=SimpleNamespace(search=search),
    )
    result = retrieve_chunks(container, "tenant", "compare alpha beta in 2026", top_k=10)
    assert calls == [("compare alpha beta in 2026", 10), ("alpha", 10), ("beta", 10)]
    assert len(result.chunk_ids) == 10
    assert "compare alpha beta in 20260" in result.chunk_ids


def test_identical_attributes_from_different_subjects_are_not_deduplicated():
    packed, selected = pack_evidence(
        "Who studied here?",
        ["University X", "University X"],
        "gpt-5-nano",
        [
            {"document_id": "bundle", "section_path": "Person A", "location": "p.2"},
            {"document_id": "bundle", "section_path": "Person B", "location": "p.7"},
        ],
    )
    assert selected == [0, 1]
    assert packed[0]["source"]["section_path"] == "Person A"
    assert packed[1]["source"]["section_path"] == "Person B"


def test_unscoped_scenario_does_not_replace_policy_with_unrelated_tables(monkeypatch):
    monkeypatch.setattr(query, "decompose_question", lambda *args: [])
    policy = SearchHit(
        "policy", 1, {"content": "Personal computers cannot access company applications."}
    )
    table = SearchHit(
        "style",
        0.1,
        {"content": "| Token | Use |\n| --- | --- |\n| red | Error |\n| green | Success |"},
    )

    def collection(*args):
        raise AssertionError("Per-device questions are not inventory requests")

    container = SimpleNamespace(
        settings=SimpleNamespace(chat_model="test"),
        gateway=None,
        reranker=None,
        embedder=SimpleNamespace(embed_query=lambda _: [1.0]),
        vectors=SimpleNamespace(
            search=lambda *a, **k: [policy, table], collection_candidates=collection
        ),
    )
    result = retrieve_chunks(
        container,
        "tenant",
        "What business activities may they perform on each personal device?",
        top_k=1,
    )
    assert result.chunk_ids == ["policy"]


def test_focused_evidence_survives_whole_question_ranking(monkeypatch):
    monkeypatch.setattr(query, "looks_multi_part", lambda _: True)
    monkeypatch.setattr(query, "decompose_question", lambda *a: ["transport", "storage"])
    passages = [
        SearchHit("transport", 1, {"content": "transport requirements"}),
        SearchHit("storage", 0.5, {"content": "storage requirements"}),
        SearchHit("noise", 0.9, {"content": "general overview"}),
    ]

    def score(question, texts):
        if question == "storage":
            return [9 if "storage" in t else -10 for t in texts]
        return [9 if "transport" in t else (8 if "overview" in t else -10) for t in texts]

    container = SimpleNamespace(
        settings=SimpleNamespace(
            chat_model="test",
            rerank_candidate_multiplier=2,
            rerank_min_candidates=3,
            rerank_min_score=0,
        ),
        gateway=None,
        embedder=SimpleNamespace(embed_query=lambda _: [1.0]),
        reranker=SimpleNamespace(score=score),
        vectors=SimpleNamespace(search=lambda *a, **k: passages),
    )
    result = retrieve_chunks(container, "t", "compare transport with storage", top_k=2)
    assert set(result.chunk_ids) == {"transport", "storage"}
    # An explicit relevance floor is applied to the aspect channels as well.
    empty = retrieve_chunks(
        container, "t", "compare transport with storage", top_k=2, rerank_min_score=20
    )
    assert empty.chunk_ids == []


def test_hybrid_fetches_beyond_final_cutoff_and_recovers_lexical_error(container, monkeypatch):
    query = [1.0] + [0.0] * (container.embedder.dim - 1)
    container.vectors.upsert([VectorPoint("a", "tenant", query, {"content": "alpha"})])
    depths = []
    original = container.vectors._dense_candidates

    def dense(cur, tenant, query, pool_size, *args):
        depths.append(pool_size)
        return original(cur, tenant, query, pool_size, *args)

    def broken_lexical(cur, *args):
        cur.execute("SELECT 1 / 0")

    monkeypatch.setattr(container.vectors, "_dense_candidates", dense)
    monkeypatch.setattr(container.vectors, "_lexical_candidates", broken_lexical)
    hits = container.vectors.search("tenant", query, top_k=1, query_text="alpha")
    assert depths == [50]
    assert [hit.chunk_id for hit in hits] == ["a"]


def test_reranker_scores_evidence_at_end_of_long_passage(monkeypatch):

    vocab = {
        word: index
        for index, word in enumerate(
            ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "query", "filler", "answer"]
        )
    }
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    tokenizer.enable_padding(pad_id=1, pad_token="[PAD]")
    tokenizer.enable_truncation(max_length=32, strategy="only_second", stride=8)
    reranker = cross_encoder.CrossEncoderReranker(max_length=32, batch_size=2)
    reranker._tok, reranker._sess = tokenizer, object()
    monkeypatch.setattr(
        cross_encoder,
        "run_onnx",
        lambda sess, feeds: [np.any(feeds["input_ids"] == vocab["answer"], axis=1).astype(float)],
    )
    assert reranker.score("query", ["filler " * 100 + "answer", "filler " * 60]) == [1.0, 0.0]

    # Both the section identity and the tail fact must occur in the same pair.
    monkeypatch.setattr(
        cross_encoder,
        "run_onnx",
        lambda sess, feeds: [
            (
                np.any(feeds["input_ids"] == vocab["answer"], axis=1)
                & (np.sum(feeds["input_ids"] == vocab["query"], axis=1) >= 2)
            ).astype(float)
        ],
    )
    assert reranker.score("query", ["query\n\n" + "filler " * 100 + "answer"]) == [1.0]
