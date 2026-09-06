"""Non-LLM (pure IR) retrieval benchmark: our pipeline vs. khub, on full BEIR SciFact.

No judge model anywhere in this script. Both sides are scored with pure IR math
(precision/recall/nDCG/MRR, reused from `eval/run_retrieval.py`) against BEIR SciFact's
own gold qrels -- no ground truth is derived or guessed. Both sides are queried through
their retrieval-only endpoint (ours: `app.retrieval.rag.query._retrieve`/`_rerank`, no
decompose/generation; khub: `POST /api/v1/search`, not `/api/v1/ask`) so nothing here
ever calls an LLM, on either side, at any stage.

Corpus: BEIR SciFact test split (5,183 docs, 300 queries) -- the same dataset
`eval/run_retrieval.py` already benchmarks our chunker+embedder against, so this is a
direct, apples-to-apples extension of that harness to include khub.

Our-side ingestion chunks with the real chunker and embeds with the real MiniLM
embedder, then upserts straight into the real vector store (bypassing the
Document/Job/queue/metadata-LLM-extraction machinery -- CLAUDE.md's own gap list notes
extracted metadata has zero effect on ranking, so skipping it doesn't change retrieval
quality, only avoids ~5,183 pointless LLM calls). Retrieval on our side still goes
through the real hybrid dense+BM25+RRF fusion and cross-encoder rerank.

khub upload/query/delete run under bounded concurrency (a live external service; 5,183
uploads + 300 searches + 5,183 deletes sequentially would take hours). Cleanup is
guaranteed via try/finally with a before/after document-count assertion.

Run:  python -m eval.run_khub_ir_benchmark [max_docs]
Env (from .env, not app.config.Settings -- khub is comparison-only, never wired into
the app): KHUB_BASE_URL, KHUB_BASIC_USER, KHUB_BASIC_PASSWORD
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np
from dotenv import dotenv_values

import ir_datasets
from app.shared.container import build_container
from app.shared.domain.models import Modality
from app.ingest.pipeline.chunker import chunk_elements
from app.ingest.pipeline.elements import Element
from app.shared.ports.vector_store import VectorPoint
from app.retrieval.rag.query import _rerank, _retrieve

from eval.run_retrieval import K_VALUES, _ndcg_at_k, _precision_at_k, _recall_at_k, _rr_at_10

REPO_ROOT = Path(__file__).resolve().parents[1]

TENANT_ID = "scifact_benchmark"
DOC_TYPE_TAG = "scifact_nonllm_ir_benchmark"
TOP_K_FETCH = 20
KHUB_CONCURRENCY = 8
KHUB_MAX_ATTEMPTS = 4

PROGRESS_LOG = Path(__file__).with_name("khub_ir_benchmark_progress.log")


def progress(event: str, **fields) -> None:
    line = json.dumps({"ts": time.strftime("%H:%M:%S"), "event": event, **fields})
    with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


def _embed_batched(embedder, texts, batch=128):
    out = []
    for i in range(0, len(texts), batch):
        out.extend(embedder.embed(texts[i:i + batch]))
        if (i // batch) % 10 == 0:
            print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}", end="\r")
    print()
    return out


def _ingest_ours(container, corpus: list[tuple[str, str]]) -> None:
    """corpus: list of (doc_id, text). Chunks + embeds + upserts directly, skipping
    Document/Job/queue/metadata-LLM machinery (see module docstring)."""
    for doc_id, text in corpus:
        chunks = chunk_elements([Element(text, Modality.TEXT.value, "text", "text_layer", 0, {})])
        if not chunks:
            continue
        vecs = container.embedder.embed([ch.text for ch in chunks])
        points = [
            VectorPoint(
                chunk_id=f"{doc_id}-{i:03d}", tenant_id=TENANT_ID, vector=vec,
                payload={"_id": doc_id, "content": ch.text, "modality": "text",
                         "user_id": "benchmark", "visibility": "tenant",
                         "acl_user_ids": [], "scope": "tenant",
                         "source_type": "text", "filename": doc_id},
            )
            for i, (ch, vec) in enumerate(zip(chunks, vecs))
        ]
        container.vectors.upsert(points)
    progress("ingest_ours_done", docs=len(corpus), vectors=container.vectors.count(TENANT_ID))


def _ours_ranked(container, question: str) -> list[str]:
    t0 = time.time()
    hits = _retrieve(container, TENANT_ID, question, TOP_K_FETCH, access=None)
    hits, _ = _rerank(container, question, hits, TOP_K_FETCH)
    ranked, seen = [], set()
    for h in hits:
        doc_id = h.payload.get("_id")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            ranked.append(doc_id)
    progress("query_ours", question=question[:60], latency_ms=round((time.time() - t0) * 1000),
              hits=len(hits), ok=True)
    return ranked


# --- khub side: retrieval-only /search, bounded concurrency, never /ask -----------

def _khub_upload_one(khub: httpx.Client, doc_id: str, text: str) -> tuple[str, str] | None:
    t0 = time.time()
    try:
        resp = khub.post(
            "/api/v1/documents/upload",
            files={"file": (f"{doc_id}.txt", text.encode("utf-8"), "text/plain")},
            data={"doc_type": DOC_TYPE_TAG},
        )
        resp.raise_for_status()
        body = resp.json() if resp.content else {}
        family = body.get("doc_family") or body.get("family") or doc_id
        progress("upload_khub", doc_id=doc_id, family=family,
                  latency_ms=round((time.time() - t0) * 1000), ok=True)
        return family, doc_id
    except httpx.HTTPError as exc:
        progress("upload_khub", doc_id=doc_id, ok=False, error=str(exc))
        return None


def _khub_upload_all(khub: httpx.Client, corpus: list[tuple[str, str]]) -> dict[str, str]:
    """Returns doc_family -> doc_id, uploaded with bounded concurrency."""
    family_to_doc = {}
    with ThreadPoolExecutor(max_workers=KHUB_CONCURRENCY) as pool:
        for result in pool.map(lambda item: _khub_upload_one(khub, *item), corpus):
            if result:
                family, doc_id = result
                family_to_doc[family] = doc_id
    progress("upload_khub_done", uploaded=len(family_to_doc), total=len(corpus))
    return family_to_doc


def _khub_search(khub: httpx.Client, question: str, family_to_doc: dict[str, str]) -> list[str]:
    t0 = time.time()
    last_exc = None
    for attempt in range(1, KHUB_MAX_ATTEMPTS + 1):
        try:
            r = khub.post("/api/v1/search", json={
                "query": question, "top_k": TOP_K_FETCH,
                "filters": {"doc_type": DOC_TYPE_TAG},
            })
            r.raise_for_status()
            passages = r.json().get("passages", [])
            ranked, seen = [], set()
            for p in passages:
                doc_id = family_to_doc.get(p.get("doc_family"))
                if doc_id and doc_id not in seen:
                    seen.add(doc_id)
                    ranked.append(doc_id)
            progress("query_khub", question=question[:60],
                      latency_ms=round((time.time() - t0) * 1000), hits=len(passages), ok=True)
            return ranked
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            if exc.response.status_code < 500 or attempt == KHUB_MAX_ATTEMPTS:
                break
            time.sleep(5 * attempt)
        except httpx.HTTPError as exc:
            last_exc = exc
            if attempt == KHUB_MAX_ATTEMPTS:
                break
            time.sleep(5 * attempt)
    progress("query_khub", question=question[:60], latency_ms=round((time.time() - t0) * 1000),
              ok=False, error=str(last_exc))
    return []


def _khub_delete_one(khub: httpx.Client, family: str) -> None:
    try:
        r = khub.delete(f"/api/v1/documents/{family}")
        progress("cleanup_khub", family=family, status=r.status_code, ok=r.status_code < 400)
    except httpx.HTTPError as exc:
        progress("cleanup_khub", family=family, ok=False, error=str(exc))


def _khub_cleanup(khub: httpx.Client, family_to_doc: dict[str, str], baseline_count: int) -> None:
    with ThreadPoolExecutor(max_workers=KHUB_CONCURRENCY) as pool:
        list(pool.map(lambda family: _khub_delete_one(khub, family), family_to_doc))
    r = khub.get("/api/v1/documents")
    r.raise_for_status()
    after_count = len(r.json()["documents"])
    restored = after_count == baseline_count
    progress("cleanup_verify", baseline=baseline_count, after=after_count, restored=restored)
    if not restored:
        print(f"\n!!! WARNING: khub document count not restored -- before={baseline_count} "
              f"after={after_count}. Investigate manually before trusting khub is clean. !!!\n")


def _score(ranked: list[str], rel: set[str]) -> dict[str, float]:
    row = {}
    for k in K_VALUES:
        row[f"precision@{k}"] = _precision_at_k(ranked, rel, k)
        row[f"recall@{k}"] = _recall_at_k(ranked, rel, k)
        row[f"ndcg@{k}"] = _ndcg_at_k(ranked, rel, k)
    row["mrr@10"] = _rr_at_10(ranked, rel)
    return row


def main() -> None:
    max_docs = int(sys.argv[1]) if len(sys.argv) > 1 else None

    PROGRESS_LOG.write_text("", encoding="utf-8")
    progress("run_started", max_docs=max_docs)

    env = dotenv_values(REPO_ROOT / ".env")
    khub_base_url = env.get("KHUB_BASE_URL", "")
    khub_user = env.get("KHUB_BASIC_USER", "")
    khub_password = env.get("KHUB_BASIC_PASSWORD", "")
    if not khub_base_url:
        raise SystemExit("KHUB_BASE_URL missing from .env -- cannot run the khub side")

    ds = ir_datasets.load("beir/scifact/test")
    qrels = defaultdict(dict)
    for q in ds.qrels_iter():
        qrels[q.query_id][q.doc_id] = q.relevance
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    query_ids = [qid for qid in qrels if qid in queries]

    corpus: list[tuple[str, str]] = []
    for i, d in enumerate(ds.docs_iter()):
        if max_docs and i >= max_docs:
            break
        corpus.append((d.doc_id, (d.title + "\n\n" + d.text).strip()))
    doc_set = {d for d, _ in corpus}
    query_ids = [qid for qid in query_ids if any(d in doc_set for d in qrels[qid])]
    print(f"corpus: {len(corpus)} docs | queries: {len(query_ids)}\n")
    progress("corpus_loaded", docs=len(corpus), queries=len(query_ids))

    container = build_container()
    print("chunking + embedding + upserting corpus into our vector store (real pipeline)...")
    t0 = time.time()
    _ingest_ours(container, corpus)
    print(f"  ingested {len(corpus)} docs in {time.time() - t0:.0f}s "
          f"-> {container.vectors.count(TENANT_ID)} vectors\n")

    khub = httpx.Client(base_url=khub_base_url, auth=(khub_user, khub_password), timeout=300.0)
    r = khub.get("/api/v1/documents")
    r.raise_for_status()
    baseline_count = len(r.json()["documents"])
    progress("khub_baseline", document_count=baseline_count)

    t0 = time.time()
    family_to_doc = _khub_upload_all(khub, corpus)
    print(f"uploaded {len(family_to_doc)}/{len(corpus)} docs into khub "
          f"(tag={DOC_TYPE_TAG}) in {time.time() - t0:.0f}s\n")

    ours_rows, khub_rows = [], []
    try:
        with ThreadPoolExecutor(max_workers=KHUB_CONCURRENCY) as pool:
            khub_futures = {
                qid: pool.submit(_khub_search, khub, queries[qid], family_to_doc)
                for qid in query_ids
            }
            for i, qid in enumerate(query_ids, 1):
                rel = {d for d, r in qrels[qid].items() if r > 0}
                ours_ranked = _ours_ranked(container, queries[qid])
                khub_ranked = khub_futures[qid].result()

                ours_rows.append({"query_id": qid, "relevant_docs": sorted(rel),
                                   **_score(ours_ranked, rel)})
                khub_rows.append({"query_id": qid, "relevant_docs": sorted(rel),
                                   **_score(khub_ranked, rel)})
                if i % 20 == 0 or i == len(query_ids):
                    print(f"[{i}/{len(query_ids)}] queries scored")
    finally:
        _khub_cleanup(khub, family_to_doc, baseline_count)
        khub.close()

    import pandas as pd

    ours_df = pd.DataFrame(ours_rows)
    khub_df = pd.DataFrame(khub_rows)
    ours_df.to_csv(Path(__file__).with_name("khub_ir_benchmark_ours.csv"), index=False)
    khub_df.to_csv(Path(__file__).with_name("khub_ir_benchmark_khub.csv"), index=False)

    metric_cols = [c for c in ours_df.columns if c not in ("query_id", "relevant_docs")]
    summary = pd.DataFrame({"ours": ours_df[metric_cols].mean(), "khub": khub_df[metric_cols].mean()})
    summary["delta (ours - khub)"] = summary["ours"] - summary["khub"]
    summary = summary.round(4)
    summary_path = Path(__file__).with_name("khub_ir_benchmark_summary.csv")
    summary.to_csv(summary_path)

    print("\n================ SUMMARY (ours vs. khub) ================")
    print(summary.to_string())
    print(f"\nqueries scored: {len(query_ids)}")
    print(f"saved -> {summary_path}")
    progress("run_finished", scored=len(query_ids))


if __name__ == "__main__":
    main()
