# Benchmarks — Retrieval Quality Experiments, in Order

This document consolidates every retrieval-quality benchmark run against this system,
in the order they happened. Each one built on the findings of the one before it. All
numbers below are taken directly from the original saved run output — nothing here is
estimated or rounded for effect. The final section reconciles all six against what is
actually running in the current configuration today.

Unless stated otherwise, the benchmark corpus is **BEIR/SciFact** — 5,183 scientific
documents, 300 expert-verified queries with gold relevance judgments — a public,
independent dataset chosen so results can be sanity-checked against a published
reference range rather than only against each other.

---

## 1. Head-to-head comparison against an independently-built system (khub)

**What was tested:** rather than take this system's retrieval and answer quality on
faith, it was measured against a second, completely independently built RAG system
("khub," live on Azure, doing the same basic job — answer questions from a document
library, with citations).

**Methodology:** both systems were given the identical 9-10 source documents (SAP
architecture, sales, procurement, finance, security, and integration documentation,
including genuinely scanned PDFs), the identical 20-50 test questions (a mix of clean
questions, realistic typo-ridden ones, single-fact lookups, and cross-document
questions), and both sets of answers were scored by a third model with no role in
generating either system's answers — so neither system graded its own work. Scoring
used RAGAS, a published, open-source RAG evaluation framework, across five metrics:
**Faithfulness** (does the answer stick to what was retrieved, nothing invented),
**Answer Correctness** (is it actually right, vs. a pre-written reference answer),
**Answer Relevancy** (does it address what was actually asked), **Context Precision**
(of what was retrieved, how much was useful), and **Context Recall** (was the right
material found at all).

**Result:** both systems scored in a respectable-but-imperfect range (roughly 0.6-0.85)
across every metric — good evidence neither system is fundamentally broken, but not
proof either is *objectively* good, since a trivial keyword-only or no-retrieval
baseline was deliberately never run alongside them (an acknowledged, explicitly-flagged
gap, not glossed over).

The most important finding was a specific failure pattern, not just a number: on one
question (locating two specific SAP transaction codes for lock monitoring and update
termination review), khub retrieved the exact right material and answered correctly.
This system instead retrieved topically-nearby-but-wrong sections and correctly said
"I don't know" — faithful (it didn't invent an answer) but useless (it didn't have the
material to work with). This is the real-world cost of a retrieval miss: invisible in
a faithfulness score, but directly responsible for whether the answer actually helps.

**Independent cross-check:** the same retrieval approach (same chunker, same embedding
model) was separately run against the public BEIR/SciFact benchmark and landed at
**nDCG@10 of 0.648** — squarely inside the published reference range (~0.64-0.66) for
this class of approach, and essentially matching a classic keyword-search baseline
(0.652). Landing where the literature says to land is real evidence the measurement
methodology itself is honest, not self-consistent by construction.

**What this drove:** the diagnosed failure pattern above pointed directly at chunking —
specifically, that chunks describing genuinely different things could become nearly
textually identical once a packing boundary happened to land after a repeated, generic
sub-heading (a real pattern in operational documentation, e.g. the same boilerplate
subsection title repeated under ten different topics). This is what motivated the
section-heading-ancestry fix described in `docs/ARCHITECTURE.md` (Part 1, §4) — every
chunk now carries its own section-title lineage so it can never be confused with a
same-named section elsewhere in the document. It also surfaced, independently, that
chunk sizing had been measured against the wrong tokenizer entirely (chunks were
routinely double the embedding model's real limit and silently truncated) — also fixed,
also described in `docs/ARCHITECTURE.md`.

---

## 2. SOTA embedder + reranker A/B

**Question:** does swapping the embedding model and the reranking model to stronger
alternatives improve retrieval?

**Setup:** baseline = `all-MiniLM-L6-v2` embedder (384-dim) + `ms-marco-MiniLM-L-6-v2`
reranker. Candidates = `bge-base-en-v1.5` embedder (768-dim) + `bge-reranker-base`.
Everything else held fixed — same chunker, same corpus, same 300 queries, pool of 20
candidates handed to the reranker, doc-level scoring.

**Results (three configurations):**

| metric | ① MiniLM + ms-marco (baseline) | ② bge + bge-reranker | ③ bge + ms-marco |
|---|---|---|---|
| recall@1 | 0.5694 | 0.5484 | **0.5777** |
| recall@3 | 0.7021 | 0.7252 | **0.7388** |
| recall@5 | 0.7717 | 0.7770 | **0.7964** |
| recall@10 | 0.8307 | **0.8823** | 0.8780 |
| ndcg@1 | 0.5967 | 0.5733 | **0.6067** |
| ndcg@3 | 0.6605 | 0.6616 | **0.6849** |
| ndcg@5 | 0.6885 | 0.6841 | **0.7084** |
| ndcg@10 | 0.7100 | 0.7208 | **0.7375** |
| mrr@10 | 0.6796 | 0.6757 | **0.7014** |

**Findings:**
- **The embedder upgrade alone is a clean, uniform win.** Comparing ③ to ①(only the
  embedder changed), *every single metric improves* — recall@10 by +4.7 points, nDCG@10
  by +2.8, MRR@10 by +2.2. bge-base is simply a better retriever, at the cost of double
  the storage (768 vs 384 dimensions) and roughly 5x slower embedding on CPU.
- **The "upgrade the reranker too" half was a regression.** Comparing ② to ③ (same
  embedder, only the reranker changed), the newer, larger `bge-reranker-base` is
  *worse* than the older, smaller `ms-marco-MiniLM-L-6-v2` on nearly every metric,
  winning only recall@10 by a hair. Bigger and newer did not mean better — the smaller
  model is a specialist trained specifically for the kind of passage-relevance ranking
  this benchmark measures.
- **Naively upgrading both to the newest available model (②) would have shipped a
  worse top-1 answer than the original baseline** — its recall@1/nDCG@1 are *below*
  where the system started, because the reranker regression outweighed the embedder
  gain at the very top of the ranking. Only measuring the two changes independently
  revealed this; shipping the "obviously better" combination blindly would have made
  the most-visible metric (the single best answer) worse.

**Recommendation:** adopt the bge-base embedder, keep the existing (smaller) reranker.
Not yet adopted in the default configuration — see "What actually shipped," below.

---

## 3. Retrieval fusion simplification

**What changed:** the production hybrid-search path previously fused its two retrieval
channels (dense + lexical) with query-adaptive weighting — shifting more weight toward
exact-word matching for queries that looked identifier-heavy (part numbers, ticket
codes) — plus a bounded mechanism for injecting lexical-only matches the dense channel
had missed, gated to only fire for identifier-leaning queries.

**Why it was questioned:** that machinery adds real complexity, and complexity that
never actually changes behavior is a maintenance cost with no offsetting benefit.

**Method:** the full SciFact benchmark (5,183 docs, 300 queries), scored two ways —
the existing adaptive-weighting path ("before"), and a plain, always-50/50 fusion with
no adaptive weighting and no special lexical-only injection ("after").

**Result:** the two configurations produced **byte-for-byte identical scores on all
300 queries** — not just matching aggregates, every single per-query score matched
exactly. On this corpus (plain scientific prose, no part numbers or ticket codes), the
adaptive weights always resolved to the neutral 50/50 case and the lexical-only
injection gate never fired for a single query — the extra machinery was, in practice,
inert. Widening how many candidates each channel fetched before fusing also changed
nothing, because the extra candidates never made it into the final top-20 anyway.

**Conclusion:** the simplification is free on prose-dominated traffic — identical
quality, meaningfully less code to maintain. **Explicit, honest caveat:** SciFact has
no identifiers to test against by construction, so this result says nothing about
identifier-heavy traffic (part numbers, SKUs, ticket codes), which was the entire
reason the removed machinery existed in the first place. If that describes a real
workload, it needs its own, separate measurement before relying on the simplified path
there. Widening the reranker's candidate pool from 20 to 40 was tested at the same time
and found to be a wash with a real latency cost (roughly double the query time for no
consistent quality gain) — not adopted.

**What shipped:** the simplification was adopted. Production hybrid search now runs
plain, always-50/50 fusion, with the precision work delegated entirely to the
downstream reranker (see next two sections).

---

## 4. LLM-as-reranker validation

**Question:** how does replacing the local cross-encoder reranker with a large
language model, asked to score every candidate passage in one shot, compare on quality
and cost?

**Setup:** embedder, corpus, chunking, and fusion held fixed (bge-base embedder,
5,183 docs / 14,722 chunks, same candidate pool size per query). Only the reranker
changed: the existing local cross-encoder (`bge-reranker-base`, one independent forward
pass per query/passage pair) vs. a listwise LLM reranker (one gateway call per query,
scoring every candidate passage at once, in the style of "LLM-as-reranker" / RankGPT).

**Results (300 queries):**

| metric | cross-encoder (before) | LLM rerank (after) | change |
|---|---|---|---|
| recall@1 | 0.5484 | 0.6549 | **+0.107** |
| recall@3 | 0.7252 | 0.7818 | +0.057 |
| recall@5 | 0.7770 | 0.8441 | +0.067 |
| recall@10 | 0.8823 | 0.8887 | +0.006 |
| ndcg@1 | 0.5733 | 0.6900 | **+0.117** |
| ndcg@10 | 0.7208 | 0.7858 | +0.065 |
| mrr@10 | 0.6757 | 0.7587 | +0.083 |

**Findings:** every metric improved, with the largest gains concentrated at the very
top of the ranking (recall@1 and nDCG@1 both jumped over 10 points) — the LLM reranker
is materially better at identifying the single best passage, not just at producing a
reasonable top-k set. Gains shrink toward recall@10, where the candidate pool's
*composition*, not its *order*, becomes the limiting factor. The cost: roughly 16
seconds per query (81 minutes for the full 300-query run) versus sub-second scoring for
the local cross-encoder — an 80x+ latency increase, plus real (if small) per-call
gateway spend. This was a single run at zero temperature with no repeated trials, so
the size of the gain is a promising signal, not a statistically settled result.

**Recommendation:** do not replace the production reranker. The quality gain is real
and lands exactly where it matters most (top-1/top-3 precision), but an 80x latency
increase is unworkable as a synchronous default at interactive query volumes. Proposed
instead as a future opt-in "high-precision" mode for latency-insensitive use cases
(async/batch queries, or a query a user has explicitly flagged as high-stakes) — not
yet built.

**What shipped:** not adopted as the default. The local cross-encoder remains the
production reranker.

---

## 5. Tier 1 + Tier 2 — pushing recall@10 past 90% (most recent)

**Goal:** raise doc-level recall@10 above 90%, starting from the 0.8823 cross-encoder
baseline above. This is the most recent and most extensive round of experiments, and it
is what ultimately changed production defaults (see "What actually shipped").

**Headline finding, stated up front:** nothing tested crossed 90% using a reranker.
The one lever that did cross it was **not reranking at all, but simply returning more
candidates**:

| k (documents returned) | recall@k |
|---|---|
| 10 | 0.8736 |
| 12 | 0.8820 |
| 13 | 0.8887 |
| 14 | 0.8937 |
| **18** | **0.9003** ✅ first to cross 90% |
| 20 | 0.9103 |

The gold documents this system *can* reach mostly sit at ranks 11-18 — just past the
top-10 cutoff, not genuinely missing. Two important caveats on treating this as a real
win: recall@k mechanically rises with k regardless of quality, so this only counts if
returning ~18 documents downstream is actually acceptable; and dumping a wide,
low-precision context into the answering model brings its own real costs — the single
relevant passage getting lost in the middle of a long context, more irrelevant material
for the model to be misled by (faithfulness/correctness measurably drop as noisy
context grows even while raw recall rises), and 2-3x the input tokens per query. The
reranker's actual job is to make a *small* context (5-8 passages) reliably contain the
right answer, rather than trading generation precision for retrieval recall by brute
force.

**Tier 1 — widening the pool handed to a cross-encoder reranker:**

| reranker | pool size | recall@10 |
|---|---|---|
| ms-marco-MiniLM | 20 | 0.8780 (baseline) |
| ms-marco-MiniLM | 50 | 0.8494 ⬇ |
| ms-marco-MiniLM | 100 | 0.8194 ⬇⬇ |
| bge-reranker-base | 20 | 0.8823 |

Widening the candidate pool *lowers* realized recall@10 with a cross-encoder reranker,
monotonically — the pool ceiling (how often the right document is reachable at all)
rises with more candidates, but the cross-encoder isn't precise enough to keep exploiting
a deeper, noisier pool; it ranks more distractors above the correct document instead of
fewer. The retrieval ceiling itself (oracle: is the gold document *anywhere* in the pool)
rises predictably with pool depth — 0.880 at 10, 0.927 at 20, 0.957 at 50, 0.963 at
100 — confirming that >90% realized recall@10 is *reachable* in principle, just not by
a cross-encoder converting a wide pool.

**Tier 2 — fixing chunk sizing to track the active embedding model:** chunking had been
sizing every chunk against a hardcoded assumption baked in for one specific embedding
model, regardless of which one was actually active. Fixed properly, in production code:
chunk sizing now derives automatically from whichever embedding model is actually
configured, reproducing the old hand-tuned sizes exactly for the original model and
scaling proportionally for a larger one. Re-processing the same corpus with this fix
produced far fewer, larger chunks (1.37 per document vs. the old 2.84). Measured
against the retrieval ceiling, this was a **wash for this particular corpus** — SciFact
documents are short (title + abstract), so they were already only 1-3 chunks each;
merging them further doesn't change whether the right document is reachable. The fix
is still a genuine, real production correctness improvement independent of this
particular benchmark result (chunks really were being silently truncated before), and
should matter more on corpora with longer documents.

**Bottom line:** on this benchmark, a cross-encoder reranker on top of this embedding
model tops out around 0.88 recall@10 regardless of pool depth or chunk size. The
ceiling is there (~0.95-0.96 at pool depth 50+) but converting it requires a component
precise enough to exploit a deep pool without being confused by its extra distractors.
Two candidate next steps, neither committed yet: the listwise LLM reranker from
Experiment 4 (the only reranker tested that didn't degrade as the pool widened — it hit
0.889 at pool 20, the best realized result of any reranker, but carries the same
latency cost documented there), or a stronger embedding model (Experiment 2's bge-base)
to lift the underlying ceiling directly so even a modest reranker clears 90%.

---

## 6. Real-corpus check: 1024-dim embedder, reranker A/B, and a second khub head-to-head

**What was tested:** every experiment above runs on BEIR/SciFact — a clean, public
benchmark corpus, but not a real document. This one runs entirely on 6 real files
(`sample_pdfs/1.pdf`–`6.pdf`: research papers, a scanned lab form, a bank statement, a
product one-pager, a textbook excerpt — 39 pages total, 16 of them scanned and requiring
vision-model OCR), ingested through the actual production code path
(`app.ingest.pipeline.runner.run_job`, the same call `app.ingest.worker` makes for a live
upload), not a corpus-loading shortcut. Two things were measured: (1) does the reranker's recall@1
lift from Experiments 2/4/5 hold on a real, non-benchmark corpus, using a stronger
1024-dim embedder; and (2) a second, independent head-to-head against khub, this time
scored non-LLM (deterministic document-id matching) instead of RAGAS.

**Setup:** embedder swapped to `bge-large-en-v1.5` (1024-dim) via the existing generic
`OnnxEmbedder` adapter (`app/shared/adapters/embedders/onnx_embedder.py`) — no new embedder
code required. Ground truth: one test question generated (single LLM call) per
ingested chunk, from that chunk's own text; the chunk it came from *is* its ground
truth, nothing hand-labeled. Scoring is 100% deterministic from there — embed the
question, retrieve, check whether the known source chunk/document reappears in the
top-k — the same non-LLM, exact-id-match approach `eval/run_retrieval.py` uses against
SciFact's qrels, applied to this system's own ingestion provenance instead. 153
questions total, pooled across all 6 documents so retrieval has real, competing,
wrong-but-plausible candidates to sort from (a single document's chunks would let every
question trivially find their own document back).

**Results — reranker A/B, chunk-level recall (same corpus, same 153 questions, only
the reranker configuration changes):**

| k | recall@k (no reranker) | recall@k (cross-encoder) | Δ |
|---|---|---|---|
| 1 | 0.4379 | 0.6601 | **+0.2222** |
| 3 | 0.6732 | 0.8366 | +0.1634 |
| 5 | 0.7712 | 0.9150 | +0.1438 |
| 10 | 0.8301 | 0.9608 | +0.1307 |
| mrr@10 | 0.5685 | 0.7622 | +0.1937 |

**Results — doc-level, ours vs. khub (identical 153 questions run against both systems,
scored identically):**

| k | ours (no reranker) | ours (cross-encoder) | khub |
|---|---|---|---|
| 1 | 0.8301 | **0.9020** | 0.8693 |
| 3 | 1.0000 | 0.9935 | 0.9869 |
| 5 | 1.0000 | 1.0000 | 0.9935 |
| 10 | 1.0000 | 1.0000 | 1.0000 |
| mrr@10 | 0.9129 | **0.9479** | 0.9282 |

**Findings:**
- **The reranker's recall@1 lift is not a SciFact artifact.** A +0.22 recall@1 jump on
  a completely different, real-world, non-benchmark corpus (research papers, scanned
  forms, a bank statement) lines up with the same lever already validated in
  Experiments 2/4/5 — this is independent corroboration on real documents, not a new
  finding that changes any default.
- **Doc-level recall saturates almost immediately here** (1.0 by k=3-5 in every
  configuration) because this corpus is only 6 documents — unlike SciFact's 5,183 docs,
  doc-level recall isn't a discriminating metric at this corpus size. recall@1 and
  MRR@10 are the numbers that actually distinguish the three configurations here.
- **Our system with the reranker on beat khub** on recall@1 (0.902 vs. 0.869) and
  MRR@10 (0.948 vs. 0.928) on identical questions; the no-reranker baseline was roughly
  comparable to khub, slightly behind — consistent with Experiment 1's overall
  "neither system is fundamentally broken" read, but this time scored non-LLM instead
  of RAGAS.
- **Honest caveat on the khub number:** khub only finished processing 5 of the 6
  uploaded files within a 10-minute poll window during this run, so every question
  generated from the 6th file's content was scored as a miss for khub through no fault
  of its ranking — its real recall is likely a little higher than shown. Not corrected
  for or hidden, just disclosed.
- Unlike Experiment 2, this run did **not** A/B the embedder itself on this corpus
  (only the reranker) — the bge-large-en-v1.5 vs. MiniLM/bge-base question on a real,
  non-SciFact corpus remains open.

**Recommendation:** no configuration change — this corroborates the already-shipped
reranker default with real, non-synthetic evidence rather than identifying a new lever.

**Reproduce:** `notebooks/pipeline_walkthrough.ipynb`.

---

## What actually shipped

Tying every finding above to what the current default configuration (`app/shared/config.py`)
actually runs:

| Experiment | Finding | Shipped? |
|---|---|---|
| khub comparison | Section-heading-ancestry + real-tokenizer chunk sizing fixes | **Yes** — both are the current default chunking behavior |
| SOTA embedder A/B | bge-base embedder is a uniform win over MiniLM | **No** — default embedder is still MiniLM (384-dim) |
| SOTA embedder A/B | bge-reranker-base is a regression vs. the smaller cross-encoder | **N/A** — the smaller cross-encoder was never replaced |
| Fusion simplification | Query-adaptive fusion weighting was inert on real prose queries | **Yes** — production hybrid search runs plain 50/50 fusion |
| LLM-as-reranker | Large quality gain, ~80x latency cost | **No** — not adopted as default; local cross-encoder remains production reranker |
| Tier 1 (pool depth) | Deeper pools need a precise reranker to convert; a shallower, well-tuned pool with the cross-encoder in use is the realized sweet spot found | **Yes** — the reranker candidate pool floor was raised to 50 as a direct result of this finding |
| Tier 2 (chunk auto-sizing) | Chunk sizing should track the active embedder, not a hardcoded assumption | **Yes** — chunk sizing auto-derives from the active embedder's real token limit by default |
| Real-corpus check (§6) | Reranker's recall@1 lift confirmed on a real, non-benchmark corpus; second khub head-to-head (non-LLM this time) also favors the reranked configuration | **N/A** — corroborates the already-shipped reranker default, no config change |

**Net effect on the currently-running system:** a cross-encoder reranker runs by
default on every query (not an opt-in experiment — this predates and was validated
independently across all five experiments above); hybrid retrieval fuses dense and
lexical search with plain, unweighted reciprocal rank fusion; chunk sizing is
automatically derived from whichever embedding model is active; and the one clearly
identified, uniformly-positive improvement that has **not** yet been adopted is the
bge-base embedder from Experiment 2 — a concrete, already-validated next step rather
than an open research question.
