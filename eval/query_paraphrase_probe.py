"""Query-paraphrase retrieval probe — demonstrates that hybrid retrieval is a
non-deterministic function of PHRASING, not just of meaning.

Same underlying intent, four different phrasings of the same question, run
through the REAL retrieval path (`app.retrieval.rag.query._retrieve` -> real MiniLM
embed + real hybrid search: dense cosine + BM25, RRF-fused) against the real
gateway-docs corpus already used by `eval/run_ragas.py`. For each phrasing we
show the top-3 retrieved chunks (with their dense/BM25/fused scores) side by
side against the ground truth, plus a similarity score between phrasings'
retrieved chunk sets, so the drift is visible and quantified instead of
asserted.

This is a demo/diagnostic script, not a benchmark: no qrels, one question at a
time. Point of it: pick a real question, watch what "weighs" the ranking, then
reorder/mangle the same words and watch the ranking move.

Run:  python -m eval.query_paraphrase_probe
"""
from __future__ import annotations

import hashlib
from itertools import combinations
from pathlib import Path

from app.shared.container import build_container
from app.shared.domain.models import Role
from app.shared.ids import new_object_id
from app.retrieval.rag.query import _retrieve
from eval.run_ragas import GOLDEN, _ingest_corpus

TOP_K = 3

# Base question pulled from the real golden set (eval/golden.json), plus three
# paraphrases that reorder/compress the exact same words -- mirrors the
# "invoice details" / "details invoice" / "invoice me details" pattern.
BASE_QUESTION = "How do you ingest documents into the built-in RAG store?"
GROUND_TRUTH = "POST /v1/rag/ingest ingests documents into the built-in RAG store."
VARIANTS = [
    ("original", BASE_QUESTION),
    ("compressed", "RAG store ingest documents"),
    ("reordered", "Documents ingest RAG store"),
    ("mangled", "Ingest me documents RAG store"),
]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main() -> None:
    corpus = Path(__file__).resolve().parents[2] / "litellm-gateway-api-docs.md"
    if not corpus.exists():
        corpus = Path.home() / "litellm-gateway-api-docs.md"
    if not corpus.exists():
        spec_corpus = GOLDEN.read_text()  # surfaces a clear error if missing
        raise FileNotFoundError(f"corpus file not found near {corpus}; check golden.json: {spec_corpus[:120]}")

    container = build_container()
    tid = container.metadata.create_tenant("paraphrase-probe-" + hashlib.sha1(str(corpus).encode()).hexdigest()[:8])
    uid = container.metadata.create_user(tid, "probe@x.test", Role.ADMIN.value, "sk-" + new_object_id())
    _ingest_corpus(container, tid, uid, corpus)
    print(f"ingested {corpus.name} -> {container.vectors.count(tid)} vectors\n")

    print(f"GROUND TRUTH: {GROUND_TRUTH}\n")
    print("=" * 100)

    results: dict[str, list] = {}
    for label, qtext in VARIANTS:
        hits = _retrieve(container, tid, qtext, TOP_K, access=None)
        results[label] = hits
        print(f'\n[{label}]  "{qtext}"')
        for rank, h in enumerate(hits, 1):
            content = h.payload.get("content", "").replace("\n", " ")[:110]
            dense = h.payload.get("dense_score")
            bm25 = h.payload.get("bm25_score")
            print(f"  #{rank}  fused={h.score:.4f}  dense={dense:.4f}  "
                  f"bm25={bm25 if bm25 is None else round(bm25, 2)}  "
                  f"chunk={h.chunk_id}")
            print(f"       {content}...")

    print("\n" + "=" * 100)
    print("PAIRWISE OVERLAP OF TOP-3 RETRIEVED CHUNKS (Jaccard similarity, 1.0 = identical set)\n")
    labels = [l for l, _ in VARIANTS]
    for a, b in combinations(labels, 2):
        set_a = {h.chunk_id for h in results[a]}
        set_b = {h.chunk_id for h in results[b]}
        print(f"  {a:12} vs {b:12}  {_jaccard(set_a, set_b):.2f}   "
              f"(top result same: {results[a][0].chunk_id == results[b][0].chunk_id if results[a] and results[b] else 'n/a'})")

    print("\nTakeaway: identical intent, reworded -> the retrieved evidence set and its "
          "ranking shift. Natural-language retrieval is an indeterministic weighing "
          "over surface wording, not a lookup on meaning.")


if __name__ == "__main__":
    main()
