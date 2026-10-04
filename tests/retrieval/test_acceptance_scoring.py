from types import SimpleNamespace

import pandas as pd
import pytest

from eval import run_retrieval_postgres as benchmark
from eval.answer_acceptance import score_case
from eval.compare_subset_reranking import rank_variants


def test_reranking_comparison_uses_one_pool_and_applies_floor_after_top_k():

    hits = ["first", "second", "third"]
    variants = rank_variants(hits, [-5.0, -4.0, 2.0], keep=2, floor=-3.0)
    assert variants == {
        "hybrid": ["first", "second"],
        "rerank_no_floor": ["third", "second"],
        "rerank_floor": ["third"],
    }
    assert hits == ["first", "second", "third"]
    with pytest.raises(ValueError, match="align"):
        rank_variants(hits, [1.0])


def test_acceptance_reports_missing_evidence_false_citations_and_invented_amounts():
    case = {
        "id": "amount",
        "answerable": True,
        "evidence": ["Invoice 00123 amount 007.00"],
        "required_values": ["007.00"],
        "forbidden_values": ["999.00"],
    }
    retrieval = SimpleNamespace(contexts=["unrelated"])
    answer = SimpleNamespace(
        contexts=["unrelated"],
        citations=[{"snippet": "unrelated"}],
        answer="Amount 999.00",
        answer_status="answered",
    )
    scored = score_case(case, retrieval, answer)
    assert scored["retrieved_spans_50"] == scored["supported_citations"] == 0
    assert scored["critical_invented_value"] is True
    assert scored["answered_correct_values"] is False


@pytest.mark.parametrize("rerank,pipeline", [(False, True), (True, True), (True, False)])
def test_postgres_benchmark_modes_use_isolated_corpus_and_write_results(
    storage_settings, tmp_path, monkeypatch, rerank, pipeline
):

    docs = [
        SimpleNamespace(
            doc_id="d1", title="Cooling system", text="Inspect the cooling system every month."
        ),
        SimpleNamespace(doc_id="d2", title="Payroll", text="Payroll is processed every Friday."),
    ]
    dataset = SimpleNamespace(
        docs_iter=lambda: iter(docs),
        queries_iter=lambda: iter(
            [SimpleNamespace(query_id="q1", text="cooling system inspection")]
        ),
        qrels_iter=lambda: iter([SimpleNamespace(query_id="q1", doc_id="d1", relevance=1)]),
    )
    monkeypatch.setattr(benchmark.ir_datasets, "load", lambda name: dataset)
    output = tmp_path / "results.xlsx"
    args = benchmark._parse_args(
        ["1", "--workers", "1", "--output", str(output)]
        + (["--pipeline"] if pipeline else [])
        + (["--rerank"] if rerank else [])
    )
    cfg = storage_settings.model_copy(update={"reranker_provider": "cross_encoder"})
    benchmark.main.__wrapped__(cfg, args)
    summary = pd.read_excel(output, sheet_name="summary").set_index("metric")["value"]
    assert summary["corpus_docs"] == summary["num_queries"] == 1
    assert summary["recall@1"] == 1
    assert bool(summary["rerank"]) is rerank
    if pipeline:
        assert summary["documents_published"] == summary["jobs_done"] == 1
    assert summary["generation_calls"] == 0
    assert output.with_suffix(".json").exists()
    assert len(pd.read_excel(output, sheet_name="per_query")) == 1
