# RKB-EPIC-03: RAG Retrieval & Performance

**Status: ⚠️ Partially complete — retrieval-quality work is done, benchmarked,
and in some cases deliberately *not* changed after being tested; scalability
work (distributed workers, load testing) has not been done.**

**DRI:** John Paul Gummapu (AI Data Engineer) · **Sign-off:** Amith K A
**Depends on:** RAG Platform Foundation (Epic 01)

---

## What this epic is, in plain terms

Epic 01 built a system that finds and answers correctly. This epic is about
two different things: (1) making the *quality* of what it finds as good as
possible, and (2) making sure it can handle *more* documents and *more*
simultaneous users without falling over. These are genuinely separate
concerns — a system can give great answers and still not scale, or scale
fine and give mediocre answers.

---

## Story RKB-STORY-06: Retrieval quality

**Status: ✅ Done — including two items that were deliberately tested and
then intentionally left as-is, which is a good outcome, not a gap.**

### Cross-encoder re-ranking — ✅ Done

**In plain terms:** the first search pass (the hybrid vector+keyword search
from Epic 01) is fast but approximate. A second, slower but much more
precise model — a "cross-encoder" — re-reads the top candidates and
re-orders them by how well each one actually answers the question, before
anything is shown to the user. This second pass was benchmarked against a
stronger alternative cross-encoder and against a completely different
approach (using the answering AI model itself as the re-ranker) before being
kept as the default — the choice was validated, not assumed.

### Metadata-aware retrieval — ✅ Done

**In plain terms:** the author, topics, and named entities pulled out during
ingestion (Epic 01) now actually influence search ranking — a document whose
extracted topics match the question ranks higher. Previously this metadata
was extracted but had zero effect on what got returned; that's fixed.

### Adaptive retrieval fusion — ✅ Evaluated, deliberately not implemented

**In plain terms:** the idea was to make the system automatically shift more
weight toward keyword-search versus meaning-search depending on what kind of
question was asked (e.g. leaning on keyword search for something with a
part number or ticket ID in it). This was actually built and tested in an
A/B comparison against the simpler fixed 50/50 blend. The result: on real,
natural-language questions, the adaptive version never actually triggered
differently in a way that helped — so it was removed in favor of the
simpler, equally-effective approach, with the re-ranking step (above) doing
the fine-grained precision work instead. **This is recorded as a deliberate,
evidence-based decision**, not an unfinished feature — see
`docs/BENCHMARKS.md` for the actual comparison data. It would be revisited
only if real search traffic full of identifiers (part numbers, ticket codes)
is measured to need it.

### Higher-quality embedding models — ✅ Validated, not yet the default

**In plain terms:** "embeddings" are how the system turns text into
something it can compare mathematically to find similar meaning. The current
default model (MiniLM) was A/B tested against a stronger, larger model
(`bge-base-en-v1.5`), which won cleanly on every retrieval-quality metric
measured. It has **not** been switched to the new default yet — not because
the improvement is in doubt, but because switching requires **re-processing
every document already ingested** (the new model produces different-sized,
incompatible representations), which is a real migration project, not a
config flip. This is a validated, ready next step whenever that migration is
scheduled.

---

## Story RKB-STORY-07: Scalability & performance

**Status: ⚠️ Partially done.**

### ANN vector indexing (HNSW/IVFFlat) — ✅ Done

**In plain terms:** when there are millions of document chunks, comparing a
question against every single one directly would be far too slow. An "ANN"
(Approximate Nearest Neighbor) index lets the database jump straight to the
likely-relevant chunks instead of scanning everything. This is built and
active today — the production storage backend (PostgreSQL + pgvector) uses
an **HNSW** index specifically, which is the modern standard choice for this
kind of search and is already configured with tunable performance
parameters.

### Distributed workers and parallel processing — ⚠️ Partially done, unverified

**In plain terms:** "distributed workers" means multiple processing engines
working through the upload queue at once, so a spike in uploads doesn't back
up behind a single worker.

- The production storage/queue backend (PostgreSQL) is already built to
  safely support **more than one worker process** pulling from the same
  queue at the same time without two workers ever grabbing the same job —
  this is a real, tested capability, not aspirational.
- What's **not** built: an actual orchestrated distributed-worker system
  (e.g. Celery + Redis) with auto-scaling, backpressure, or job priority.
  Today, running multiple workers means literally starting more than one
  worker process by hand against the same database — it works, but it
  hasn't been exercised under real concurrent load to know how it behaves
  at scale.
- **Flagging this as unverified, not "done."**

### Load and performance testing — ⏳ Not started

**In plain terms:** nobody has yet thrown a realistic volume of simultaneous
uploads or questions at the system to see where it slows down or breaks.
Worker throughput under concurrent multi-tenant load, how the system behaves
if the AI gateway starts rate-limiting under sustained traffic, and how
Postgres/pgvector holds up under contention are all currently **unmeasured**.
This is flagged as a real gap ahead of any go-live decision, not something
quietly assumed to be fine.

---

## Bottom line for this epic

Retrieval *quality* work is genuinely finished — built, benchmarked, and in
two cases deliberately left alone after being proven not worth the added
complexity, which reflects real engineering rigor rather than lack of
effort. Retrieval *scale* work is the opposite story: the foundational piece
(a production-grade ANN index) is in place, but real multi-worker
orchestration and any actual load testing have not happened yet and should
be treated as an open item before this platform carries production traffic
at volume.
