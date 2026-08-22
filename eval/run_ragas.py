"""End-to-end RAGAS evaluation of the RAG pipeline.

Pipeline under test: ingest the corpus -> MiniLM embed -> knowledgebase retrieve
-> gateway LLM answer. RAGAS then scores each answer with an LLM judge.

- Judge LLM  : gateway chat model via langchain-openai (OpenAI-compatible).
- Embeddings : our local MiniLM (the gateway has no embedding model).
- Metrics    : faithfulness, answer relevancy, context precision, context recall.

Run:  python -m eval.run_ragas  [num_questions]
Env:  RAGAS_JUDGE_MODEL (default claude-sonnet-5)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, RunConfig, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from app.config import settings
from app.container import build_container
from app.domain.models import Document, Job, JobStage, JobStatus, Role
from app.ids import new_object_id
from app.pipeline.runner import run_job
from app.rag.query import answer_query

GOLDEN = Path(__file__).with_name("golden.json")


class _MiniLMLangchain(Embeddings):
    """Adapt our MiniLM embedder to the langchain Embeddings interface."""
    def __init__(self, embedder):
        self._e = embedder

    def embed_documents(self, texts):
        return self._e.embed(list(texts))

    def embed_query(self, text):
        return self._e.embed([text])[0]


def _ingest_corpus(container, tenant_id, user_id, corpus_path: Path) -> None:
    data = corpus_path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    blob = container.blob.put(tenant_id, sha, corpus_path.suffix, data)
    doc = Document(id=new_object_id(), tenant_id=tenant_id, owner_user_id=user_id,
                   source_type="docx", blob_path=blob, content_sha256=sha,
                   mime="text/markdown", filename=corpus_path.name,
                   visibility="private", acl_user_ids=[])
    container.metadata.create_document(doc)
    container.metadata.create_job(Job(id=new_object_id(), document_id=doc.id,
        tenant_id=tenant_id, stage=JobStage.PARSE.value,
        status=JobStatus.QUEUED.value, attempts=0))
    run_job(container, container.queue.claim_next())


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    judge_model = os.environ.get("RAGAS_JUDGE_MODEL", "claude-sonnet-5")

    spec = json.loads(GOLDEN.read_text())
    qa = spec["qa"][:limit] if limit else spec["qa"]
    corpus = Path(__file__).resolve().parents[2] / spec["corpus_file"]
    if not corpus.exists():
        corpus = Path.home() / spec["corpus_file"]
    print(f"corpus: {corpus}\nquestions: {len(qa)}\njudge: {judge_model}\n")

    container = build_container()
    tid = container.metadata.create_tenant("RagasEval")
    uid = container.metadata.create_user(tid, "eval@x.test", Role.ADMIN.value,
                                         "sk-" + new_object_id())
    _ingest_corpus(container, tid, uid, corpus)
    print(f"ingested corpus -> {container.vectors.count(tid)} vectors\n")

    samples = []
    for i, row in enumerate(qa, 1):
        r = answer_query(container, tid, row["question"], top_k=5)
        samples.append({
            "user_input": row["question"],
            "retrieved_contexts": r.contexts,
            "response": r.answer,
            "reference": row["ground_truth"],
        })
        print(f"[{i}/{len(qa)}] {row['question'][:60]}")
    dataset = EvaluationDataset.from_list(samples)

    judge = LangchainLLMWrapper(ChatOpenAI(
        model=judge_model,
        api_key=settings.litellm_api_key,
        base_url=settings.litellm_base_url.rstrip("/") + "/v1",
        temperature=0,
    ))
    emb = LangchainEmbeddingsWrapper(_MiniLMLangchain(container.embedder))

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]

    print("\nrunning RAGAS...\n")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge,
        embeddings=emb,
        run_config=RunConfig(max_workers=3, timeout=180),
    )

    print("\n================ RAGAS SCORES ================")
    print(result)
    df = result.to_pandas()
    out = Path(__file__).with_name("ragas_results.csv")
    df.to_csv(out, index=False)
    print(f"\nper-question results -> {out}")


if __name__ == "__main__":
    main()
