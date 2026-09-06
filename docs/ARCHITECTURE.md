# Architecture — What We Have Built, End to End

This document explains how the system works, in plain language, with no code. It is
the authoritative narrative account of everything that exists today: the two runtime
flows (**getting a document in**, and **getting an answer out**), the surfaces built
on top of them (the HTTP API, the onboarding UI, the trace UI), the cross-cutting
machinery (identity, permissions, observability), and the infrastructure choices that
sit behind a switch.

```
INGESTION:  upload ──▶ queue ──▶ parse/route/extract ──▶ chunk ──▶ tag ──▶ embed ──▶ store
RETRIEVAL:  question ──▶ [decompose?] ──▶ embed ──▶ hybrid search (dense + lexical)
            ──▶ ACL filter ──▶ rerank ──▶ [cache?] ──▶ grounded answer + citations ──▶ cache
```

Both flows are built on the same **ports-and-adapters** discipline: every piece of
infrastructure (where metadata lives, where files live, where vectors live, how work
is queued, which model embeds, which model reranks, whether answers are cached) sits
behind an interface. Each interface has at least one zero-setup local implementation
and, where it matters, a production implementation — both real, both exercised,
selected purely by configuration. Nothing in the ingestion or retrieval logic below
knows or cares which one is active.

Scale, for orientation: ~6,750 lines of Python across 75 application modules, plus
~290 lines of SQL schema, ~4,000 lines of tests (28 test modules, 248 `def test_`
functions — pytest's collected-case count runs higher once parametrized variants are
counted), a benchmark/evaluation harness, and two self-contained static UIs
(onboarding, and the document trace view).

> **Reading order.** `CLAUDE.md` at the repo root is the fast index (directory map,
> config switches, known gaps and roadmap). This file is the depth. `docs/BENCHMARKS.md`
> is the measured evidence behind the retrieval-quality decisions described here.

---

# PART 0 — THE SYSTEM AT A GLANCE

## The moving parts

Four kinds of thing are running, and it helps to keep them straight:

- **The API process.** A web service holding every HTTP surface: signup/login, upload,
  retrieve, generate an answer, inspect a job, list documents, delete a document,
  health, metrics, and the two static UIs. It does fast, bounded work only — it never
  parses a document.
- **The worker process.** A separate, long-running loop that claims queued ingestion
  jobs and runs them through the whole pipeline. This is where all the slow,
  unpredictable work happens. It can be stopped and restarted freely; job state lives
  in the database, not in the process.
- **The stores.** Three logically distinct places: the **metadata store** (tenants,
  users, documents, jobs, per-stage traces, chunk text, audit log, metric counters,
  and — new — a small live-config table, see Part 4), the **blob store** (the
  original uploaded bytes, addressed by content hash), and the **vector store** (one
  searchable point per chunk, carrying enough permission data to be filtered before
  ranking). Optionally, a fourth: the **answer cache**.
- **The models.** An external AI gateway (chat + vision, OpenAI-compatible, spoken to
  via LiteLLM) used for transcription of unreadable content, document tagging,
  question decomposition, and final answer generation. Alongside it, two models that
  run **locally, on this machine, with no per-call cost**: the embedding model and the
  reranking model.

The API and the worker share the same composition root and the same stores; they are
two entry points into one system, not two systems.

## The single most important design idea

**Structure first, AI only when structure runs out.**

If a piece of content is already structured or already typed, ordinary, free software
reads it perfectly and no AI model is ever involved. Real tables are extracted as
tables. Real typed paragraphs are read directly. Word documents keep their heading
hierarchy and their tables. Spreadsheets keep every sheet as its own clean table.

AI is invoked for exactly one case ordinary software genuinely cannot handle: content
that is a **picture of information** rather than encoded information — a scanned page,
a photographed form, an embedded chart or screenshot.

This is not a stylistic preference; it is what makes onboarding a client's full
document history financially realistic. If every page cost a paid model call just to
be read, the price of ingesting years of accumulated knowledge would scale with how
much history exists. Here, the bill tracks only the amount of genuinely unreadable
content.

---

# PART 1 — IDENTITY, TENANCY, AND PERMISSION

Everything else in this document assumes this layer, so it comes first.

## Getting in the door

Two ways to acquire an account exist today, and both end up with the same shape of
credential:

- **Self-service.** `POST /onboarding/register` takes an email, a password, and an
  optional display name. It creates a brand-new tenant, creates an **admin**-role user
  in it, and returns a bearer API token. Every self-registered user is an admin of
  their own new tenant — there is no self-serve member/viewer signup, only "start a new
  tenant." `POST /onboarding/login` verifies the password (hashed with
  PBKDF2-HMAC-SHA256, 390,000 iterations — the password is hashed, but note below that
  the bearer token itself is not) and returns that same stored token again; tokens are
  not regenerated on login. `GET /onboarding/status` is unauthenticated and just
  reports whether an LLM gateway is already configured (env or DB — see Part 4).
- **Seed script.** `python -m scripts.seed` still works and provisions a tenant + admin
  the same way, printing the token once. Its one remaining distinctive job is
  `--platform`: seeding the one tenant that will become `PLATFORM_TENANT_ID`, so it can
  publish firm-wide documents. Ordinary tenant bootstrap is otherwise redundant with
  self-service registration now.

Either way, once a tenant has an admin, that admin can add teammates only through
whatever the tenant's own process is for sharing the token — there is still no
in-product "invite a user" or tenant/user/token-management API. That remains a real
gap (Part 8), just narrower than "no self-service at all."

Every request carries a bearer token. That token resolves to a **principal**: a
tenant (the client organisation), a user, and a role. Three roles exist:

- **admin** — may read everything belonging to their own tenant, may ingest, may
  delete, and (if their tenant is the designated platform tenant) may publish
  firm-wide. Only an admin may call the gateway-config endpoint (Part 4).
- **member** — may ingest, and may read what they own or what has been shared with
  them or their whole tenant.
- **viewer** — read-only; may not ingest and may not delete.

## Two independent axes of reach

A document's reach is governed by two orthogonal properties, both stored on the
document and both carried onto every one of its vectors:

- **Scope** — how far the document reaches *across* tenants. `tenant` (the default)
  means it never leaves its own client. `global` means it is firm-wide institutional
  knowledge, readable by any authenticated user at any client. Publishing to global
  scope is tightly gated: only an admin belonging to one specifically designated
  platform tenant may do it, and if no platform tenant is configured, nobody can.
- **Visibility** — how far it reaches *within* its tenant. `private` (owner only),
  `tenant` (everyone at that client), or `shared` (a named list of users).

Documents are currently created **private** to their uploader. The visibility and
ACL machinery is fully built and enforced everywhere, but there is no endpoint yet
to change a document's visibility after upload — so in practice, today, a document
is visible to its uploader, to admins of its tenant, and (if published globally) to
everyone.

## The rule, in one place

A single permission predicate answers "may this person see this?", checked in a fixed
order, first match wins:

1. Is it firm-wide (`scope=global`)? → visible, full stop.
2. Is the asker an admin of their own tenant? → visible.
3. Did the asker upload it? → visible.
4. Is it shared with the whole tenant? → visible.
5. Is it shared with a named list that includes the asker? → visible.
6. Otherwise → not visible.

That rule is enforced in exactly one place and reused by every caller: retrieval,
document listing, document detail, chunk listing, and the trace view. (An admin's
"sees everything in their tenant" case is exposed as a cheap `sees_everything` flag,
so a store can skip the predicate entirely for that caller rather than evaluating it
per document — a performance detail, not a rule change.) **Reading** firm-wide content
never implies the right to **modify or delete** it — writes and deletes additionally
require that the document belong to the caller's own tenant.

Every ingest and every delete is written to an audit trail: who did what, to what,
when, with a small structured detail record.

---

# PART 2 — INGESTION

Ingestion has exactly one entry point and two distinct speeds: a fast, synchronous
"accept this file" step, and a slower, asynchronous "actually process it" step that
runs in a different process entirely.

## 1. The front door — accepting a file

There is one way a document enters the system: an authenticated upload. No batch
importer, no special-case path, no second door — a one-page text file and a fifty-page
scanned archive box both go through the same handler.

What happens the moment that request arrives is deliberately small and fast:

1. **Who is this, and may they add documents?** Admins and members may; viewers may
   not.
2. **Is this meant to be client-private or firm-wide?** Firm-wide publication is
   checked here and rejected unless the caller is the platform tenant's admin.
3. **Is this a file type we understand?** Recognised by file extension only — PDF,
   Word, Markdown, plain text, RTF, CSV, HTML, Excel (both modern and legacy), and a
   handful of image formats. Anything else is rejected immediately with a clear list
   of what is accepted.
4. **Is it empty, or too large?** Both rejected before any real work happens; the
   size ceiling is configurable.
5. **Have we already seen these exact bytes, for this exact client?** The content is
   fingerprinted with a cryptographic hash. If that fingerprint already exists for
   this tenant, the upload is a no-op — the caller gets back the existing document's
   identity and an explicit "deduplicated" marker rather than a duplicate.
6. **Store the raw bytes, untouched**, addressed by that same fingerprint. Identical
   bytes are stored once.
7. **Write down two facts**: a document exists, and a job exists to process it,
   sitting in a "queued" state. The ingest is recorded in the audit trail.
8. **Respond immediately** with the document's identity and the job's identity. No
   parsing, chunking, or model call has happened yet.

**Why split it this way?** Processing time is wildly unpredictable — a one-page
invoice might take under a second; a fifty-page scan with no typed text at all needs
dozens of individual vision calls and takes minutes. If the upload request waited for
that, every upload would tie up a web request for as long as the worst-case document
takes. The front door does only the cheap checks above; the real work is handed off.

## 2. The worker — claiming and running jobs, safely

A separate, continuously running process asks "is there a queued job?" and, whenever
one exists, claims it (flipping it to "running" so no other worker takes the same job)
and runs it through every stage in order.

Claiming is atomic, and *how* atomic depends on which queue backend is active. On the
local embedded database it uses an immediate write transaction — correct, and fine for
one worker. On PostgreSQL it uses a row-level skip-locked claim, which lets many
worker processes each grab a different job with no contention at all. That is the
same code path from the worker's point of view; only the adapter differs.

If a stage raises, the job is not lost: attempts are incremented and the job goes back
into the queue with a **delayed availability** — not immediately reclaimable. Both
queue adapters share one exponential-backoff schedule (2s, 4s, 8s, 16s, 32s, capped at
60s) that is written as an "available again at" timestamp on retry and checked on every
claim, so a flaky stage gets breathing room before it's retried rather than
hot-looping. Only after a configurable number of failed attempts (5 by default) is a
job marked permanently **dead**. A dead job is a first-class, inspectable end state
with its error text retained, not a crash and not a silent disappearance. The *gateway
client* separately backs off exponentially between its own retries of a failed model
call, on its own schedule — so a transient network failure gets absorbed twice over:
once inside the call, once at the queue level if it still fails every attempt.

This used to be an honest limitation — a worker killed mid-job left that job "running"
forever — but it is fixed: every claimed job gets a lease (`JOB_LEASE_SECONDS`, default
300s), and a reaper checked once per worker poll loop reclaims (or dead-letters, past
`MAX_ATTEMPTS`) any job whose lease expired while still "running". See CLAUDE.md roadmap
item #1 and `app/ingest/adapters/queue/{sqlite,postgres}.py`.

## 3. The stages

Eight named stages run in a fixed order: **parse → route → extract → chunk → metadata
→ embed → binarize → upsert**. Each one is timed, each one has its outcome recorded
both as a structured log line and as a durable row in a per-stage trace table, and
each one bumps a metric counter.

Three of those names describe one indivisible act. Parsing, routing, and extraction
are performed together by the per-format loader, because routing decisions are
inherently format-specific (a PDF's "is this page a scan?" question has no meaning for
a spreadsheet). The route and extract stages therefore exist as explicit, visible
steps in the trace (each a literal no-op in code, with a comment pointing back at
parse), so the lifecycle stays legible to anyone watching a document flow through,
without pretending the pipeline has a generic routing phase it doesn't have.

## 4. Reading the document — per format

Every loader records, on every piece of content it produces, three things: **what the
content is** (prose, table, or image-derived), **which extractor produced it**, and
**why that path was chosen**. Nothing is read anonymously. That provenance rides all
the way onto the stored chunk and into the trace UI, where it becomes the "read for
free vs. needed the vision model" breakdown.

**PDF — decided page by page, not file by file.** For each page:

- *Are there real, extractable tables?* Pulled out deterministically and converted
  straight to clean markdown — no AI, no cost. The area those tables occupy is then
  excluded from the page's plain-text extraction, so the same content never appears
  twice.
- *Is there a real typed text layer?* Read directly — free.
- *Is the page overwhelmingly a picture?* The specific, narrow signal is **almost no
  extractable text (under ~20 characters) combined with an embedded image present**.
  Only then is the page rendered to an image at roughly 200 dpi and sent to the vision
  model under a strict transcription instruction. If the transcription itself contains
  something shaped like a table, it is re-detected and split back out as a proper
  table.
- *Is there a real photo or figure on an otherwise ordinary typed page?* Only images
  above a minimum area (so logos, icons and decorative rules are skipped) are
  individually cropped out and sent to the vision model — not the whole page, just
  the picture. The rest of that page is still read for free.

**Word documents.** Paragraphs and tables are emitted in true document order, so their
relationship survives. Heading styles become real markdown headings, and a level-aware
**heading stack** is tracked while walking the document — not just the nearest heading,
but the whole ancestry. Tables become deterministic markdown with the nearest heading
as a caption. Embedded images go to the vision model. (A gateway failure there fails
the job so it can be retried; an unreadable embedded object is simply skipped.)

**Markdown, plain text, and RTF.** All three go through one shared block splitter,
which understands markdown table syntax, tab-separated tables (pasted spreadsheet
data), and fenced code blocks — code is kept atomic and never split. RTF is
de-formatted to plain text first (via `striprtf`), then handled by the exact same
splitter, so a markdown-sourced chunk and an RTF-sourced chunk get identical
heading-stack ancestry treatment.

**Spreadsheets.** Each non-empty sheet becomes its own deterministic markdown table,
captioned with the sheet name. Charts and pictures embedded in modern workbooks are
best-effort routed to the vision model.

**CSV and HTML.** Parsed as real tables and converted to markdown. A CSV that will not
parse as a table falls back to being treated as plain text rather than failing.

**Images.** Sent straight to the vision model under the strict transcription prompt,
and table-aware on the way back — an image that is a table becomes a table.

**The transcription instruction itself** is deliberately severe: transcribe only what
is literally visible, in reading order; never summarise, translate, rephrase, explain,
or guess; never add a word or number that isn't clearly present; mark unreadable
regions explicitly rather than inventing a plausible replacement; emit any table as a
markdown table; and mark visually obvious section titles with markdown heading syntax
— but only where the visual prominence is unambiguous. That last instruction matters:
without it, a scanned document would get none of the section-ancestry benefit
described below, even though scanned documents are the harder, higher-value ingestion
path this system is built around.

## 5. Chunking — pieces that are individually meaningful

A whole document is never searched as one unit. It is broken into right-sized pieces,
because a search needs to compare a question against something small and specific
enough that a good match means something, and because the final answer must point back
to an exact passage.

**Pieces are sized against the search model's real, measured limit.** The embedding
model has a hard ceiling on how much text it reads; anything past it is silently
dropped before the fingerprint is even computed. Chunk sizes are therefore measured
with that specific model's own tokenizer — truncation and padding explicitly disabled,
so what is measured is true length rather than the already-truncated length. Sizing
**derives itself from whichever embedding model is active** (`CHUNK_AUTO_SIZE`,
default on): the default 256-token model yields a target of 180 tokens, a hard cap of
220, and 20 tokens of overlap; a 512-token model yields roughly 390/476/43 instead of
being needlessly fragmented at the smaller model's limit. Headroom is reserved beneath
the cap for the section-ancestry prefix described below, so the finished chunk fits
rather than discovering afterwards that the prefix pushed it over.

**A piece is never cut mid-thought.** Splitting respects paragraph boundaries first,
then sentence boundaries, and only hard-splits mid-sentence for the rare oversized
sentence that cannot otherwise fit. Sentence splitting deliberately refuses to split
on a period that is actually a numbered-list marker — without that exclusion, a
numbered list gets shredded at every item marker, tearing items out of their own list.

**Overlap is a whole trailing sentence, never a raw slice.** Where a chunk carries a
little context across a boundary, it carries complete sentences, and it never reaches
back past a heading line — a heading belongs to the chunk it introduces, not to an
overlap seed dragged into the next one.

**Tables get their own treatment.** A table too large for one chunk is split **by
rows**, and the caption, header row and separator are repeated at the top of every
resulting piece, so a fragment is never missing the column meaning that makes it
readable. A row sitting right at a split boundary is carried into both neighbours, so
it is findable from either side. A single row too large even for that is hard-split,
still under the header.

**Every piece remembers what it is part of.** Each chunk carries its section
ancestry, and the ancestor path (everything above its own nearest heading) is
prepended to the chunk text itself. The chunk's own nearest heading is already visible
in its text, so repeating it would be waste; the ancestors are what was otherwise
missing. This matters most in operational documentation that repeats the same
boilerplate structure under many different topic headings, where two chunks about
completely different topics would otherwise be nearly identical text with no way for
a search to tell them apart.

Two smaller refinements exist for the same reason: a heading with no body of its own
(a pure container for its subsections) is not emitted as a content-free chunk, since
its text already appears in every child's ancestry prefix; and a tiny trailing chunk is
merged back into its predecessor when it fits.

None of this was designed in the abstract. The section-ancestry fix in particular was
made in direct response to a measured retrieval-quality problem found during a
head-to-head comparison against an independently built system (`docs/BENCHMARKS.md`),
not guessed at ahead of time.

## 6. Automatic tagging — metadata for free

A single call to a small, fast chat model reads a sample of the document's text (capped
to a token budget) and returns structured facts about it: who likely wrote it, its own
date, the topics it covers, and the named things it mentions — systems, products,
standards, organisations, identifiers. The model is explicitly instructed not to guess:
anything undeterminable comes back blank rather than confidently wrong.

This step is **auxiliary by design**. If the call fails, or the response can't be
parsed, the document ends up with blank tags rather than the whole job failing.

It is not a dead-end nicety either: those tags are folded into what keyword search
matches against (Part 3), so a topic or named entity mentioned in a document can be
found even when the searcher's wording doesn't match the chunk's own wording.

## 7. Turning text into searchable meaning

Every chunk's text is converted into a numeric fingerprint of meaning — a vector — by a
model that runs **entirely on local infrastructure**, at no incremental cost per
document, rather than a paid external call.

The default is a compact 384-dimension model run through a portable inference runtime
with no heavy machine-learning framework required; outputs are mean-pooled over tokens
and length-normalised, matching the model's own training recipe. Two alternatives are
built: a configurable local runner for any model that ships a comparable export
(handling the details that must match a model's recipe or scores drop rather than
improve — which token to pool from, and whether the model expects a short instruction
prefixed to queries but not to passages), and a gateway-backed embedder for the case
where an embedding model is provisioned externally. The configurable local runner is
what the benchmark harness uses to evaluate stronger embedders; the shipped
configuration switch currently offers the default local model or the gateway one.

A **binarize** stage exists between embedding and storage as an explicit placeholder
for eventually compressing fingerprints into a smaller, faster-to-search form. Today it
passes vectors through untouched; keeping it as a named stage means that change lands
in one obvious place rather than being threaded through the pipeline later.

## 8. What gets stored, and how it stays consistent

Once a document is read, chunked, tagged and embedded, two things are written together
and kept in lockstep:

- **The record of ownership, permission and content** — which client this belongs to,
  who uploaded it, who else may see it under which sharing rule, the extracted tags,
  and the exact text of every chunk with its own stable, predictable identity (the
  document's identity plus a zero-padded ordinal, so chunk identities are deterministic
  and sort correctly).
- **The searchable index** — every chunk's fingerprint, carrying alongside it enough of
  that same ownership and visibility information that a search can be narrowed to what
  a given person may see **before** anything is ranked, never after the fact. The
  payload also carries the chunk's own text, its human-readable location (a page
  number, a page range, a sheet name, a section title), the filename, and the
  document's extracted topics, entities and author.

Two full implementations exist side by side, selected by configuration:

- A **local, zero-setup implementation**: one embedded database file plus local
  directories, requiring no external infrastructure at all. The vector side stores one
  JSON record per document with its chunks nested inside.
- A **production implementation**: PostgreSQL plus its native vector extension. Here
  the vector side stores **one row per chunk**, with the fingerprint in a native
  vector column under an approximate-nearest-neighbour index, and a second, full-text
  index alongside it. Finding the best matches becomes an index-accelerated lookup
  rather than pulling every stored fingerprint into memory on every question.

Re-processing an existing document (say, after a chunking improvement) is safe to run
repeatedly: prior chunks and prior vectors are dropped first, so re-running never
leaves stale, duplicate or orphaned data behind. Deletion of vectors is a soft delete
(tombstoning) in both adapters, which is fast and safe but means storage grows
monotonically with churn until a compaction step exists.

Finally, when a job completes successfully, the tenant's cached answers are
invalidated (Part 3), because the knowledge base just changed. That happens at one
choke point that every successful ingest and every reprocess funnels through.

## 9. The trace, kept as data

The runner logs every stage, but logs are not queryable. So the same information is
also persisted: one row per stage per attempt, with its duration, its status, and a
structured detail record of what it actually produced. Failures are recorded too, with
their error text, before the exception is allowed to propagate to the worker's retry
logic.

This is what lets a document's journey be replayed long after the job finished, and it
is the entire data source for the trace UI in Part 4. Trace-keeping is treated as
observability, never correctness: if the store cannot take the write, it is logged and
swallowed rather than failing an otherwise-good ingest.

---

# PART 3 — RETRIEVAL

Retrieval answers a question against everything already ingested. Unlike ingestion it
runs synchronously, because the work here is small and bounded — a handful of lookups
and one or two model calls — rather than open-ended.

**Retrieval and generation are two separately callable steps, not one.** The query
logic exposes three entry points: **retrieve** (decompose → embed → hybrid search → ACL
filter → rerank — no cache, no generation), **generate from given chunks** (cache
check → LLM answer → cache store, given someone else's already-fetched chunks), and a
**composed round trip** that chains both together end to end. The HTTP API exposes the
first two separately (`POST /query`, then `POST /answer` — Part 4) so a caller can
inspect or re-rank candidates before paying for generation, or bring their own chunks
from elsewhere; the composed round trip exists for eval scripts and notebooks that want
one call. The stage-by-stage description below covers the full logical pipeline
regardless of which entry point a caller uses; where a step only applies to the
composed/generate path, it's called out.

## 1. Have we already answered this? *(generate step only)*

If the optional answer cache is enabled, the question is embedded and used to look for
a previously generated answer that is the same or **semantically similar** — similarity
above a configurable threshold (0.95 by default, where 1.0 would mean exact match
only). A hit returns the stored answer instantly, skipping generation entirely, with a
marker prepended to the response's trace so it is visible that the answer came from
cache and at what similarity. Because retrieval and generation are separate HTTP calls,
this embedding is computed independently from the one retrieval used for search — a
composed single call (the eval-only round trip) can reuse one embedding for both; the
split `/query` + `/answer` path cannot, since they're independent requests.

The cache is scoped **per user** (not per tenant) and keyed by the answering model —
both are correctness properties, not tuning knobs: answers are grounded in chunks
already filtered by that specific user's permissions, so replaying one to the same user
is safe by construction and can never leak between colleagues; keying by model means
switching models doesn't serve answers from the old one.

Freshness has two mechanisms. The primary one is a per-tenant **generation counter**
(plus a global counter covering both a per-tenant bump and an "invalidate everything"
bump): when a tenant's knowledge base changes (any successful ingest, reprocess, or
delete), the counter is bumped, and every entry built under the old generation
instantly stops matching — a constant-time check, not a scan. The backstop is a
per-entry time-to-live, 24 hours by default.

The whole cache is inert unless configured; with no cache backend the query path
behaves exactly as if it never existed. One operational caveat: its index is created
with the active embedding model's dimensionality, so changing embedding models requires
dropping and recreating it.

## 2. Is this really one question, or several? *(retrieve step)*

Before anything is searched, a **free, rule-based check** asks whether the question
shows a real surface signal of being two or more distinct questions glued together —
two or more question marks, or a comparison/relation phrase ("compare", "versus",
"difference between", "relationship between", "as well as", and similar). This costs
nothing and runs on every question.

Only if that free check fires does the system spend one cheap model call to actually
decide — and that call can overrule the heuristic and say "no, this is genuinely one
focused question." The instruction is explicit that a long, detailed question about one
topic is *not* multi-part, and that any sub-questions must preserve the exact codes,
names and numbers from the original rather than paraphrasing them away.

When a question genuinely is multi-part, it is split into at most four focused
sub-questions, each searched independently, and the results merged and de-duplicated:
a chunk relevant to more than one sub-question keeps its best score and counts once.
Critically, **the final answer is still generated against the original question**,
using the combined evidence — the person asking never sees their question mechanically
dismantled.

Like tagging, this is best-effort: any failure means no sub-questions, which simply
falls back to ordinary single-pass retrieval.

## 3. Turning the question into the same kind of fingerprint *(retrieve step)*

The question is embedded with the exact same model used for every stored chunk — that
is what makes the two comparable at all.

## 4. Hybrid search — two different ways of matching, combined *(retrieve step)*

Two fundamentally different techniques run every time, side by side, over candidates
the asker is allowed to see:

- **Meaning-based (dense) search** — comparing the question's fingerprint against
  stored fingerprints and finding the numerically closest, regardless of wording.
- **Exact-word (lexical) search** — an independent pass matching literal words and
  phrases, so a ticket number, part code or other distinctive exact term is never
  missed just because a meaning-based comparison under-weights it.

Both run, and their two independent result lists are combined by **reciprocal rank
fusion** (constant `k=60`) — each candidate earns credit from *where it ranked* in each
list rather than from its raw score, sidestepping the fact that the two techniques
produce scores on completely incomparable scales.

The automatically extracted tags from ingestion — topics, named entities, likely
author — are folded into the text the lexical side matches against, at write time
(dates are deliberately excluded, since a bare year would false-positive-match any
chunk that happens to mention it in passing).

**The two storage backends fetch candidates differently, but fuse them identically:**

- The **local** implementation loads the tenant's whole corpus, scores dense similarity
  across all of it, and computes lexical scores in-process — with corpus-wide,
  undistorted term-rarity statistics, since the candidate pool genuinely *is* the whole
  corpus.
- The **production** implementation fetches two genuinely independent candidate lists —
  one from the approximate-nearest-neighbour vector index, one from Postgres's own
  full-text search (`ts_rank_cd` / `websearch_to_tsquery`). If the lexical side fails for
  any reason, the query degrades to dense-only rather than erroring.

Both then fuse with plain, unweighted 50/50 reciprocal rank fusion — deliberately
identical on both backends, so switching `VECTOR_BACKEND` cannot change a query's
ranking. This is a **measured decision**, not a missing feature: an A/B showed a
query-shape heuristic that leans fusion toward lexical matching for identifier-heavy
queries (natural-language prose weighted (1.0, 1.0); a mostly-identifier query weighted
(1.0, 2.0); a short query with no stopwords but at least one identifier weighted
(1.0, 1.5)) never actually fired on real prose queries, so it was removed from both
retrieval paths in favour of letting the reranker be the precision arbiter. The heuristic
itself still exists as a tested, standalone utility
(`app/shared/adapters/bm25.py::classify_query_weights`) — reachable for a future caller
that measures a real need for it (e.g. genuinely identifier-heavy traffic) — it is just
not wired into either adapter's default search path today. See `docs/BENCHMARKS.md`.

## 5. Who may see this — enforced before ranking *(retrieve step)*

Every candidate from both techniques is checked against the permission rule from Part 1
**before** it is allowed to influence the final ranking — never filtered afterwards,
which would let a search rank things a person isn't allowed to know exist. Tenant
scoping happens in the query itself (only this tenant's records, plus anything
published firm-wide); the per-user visibility and sharing rule is applied on top,
because it is an arbitrary predicate that cannot be pushed into SQL.

## 6. Reranking, and a relevance floor *(retrieve step)*

Hybrid search is deliberately asked for a **wider pool** than the question needs — four
times the requested number of results, with a floor of 50 candidates. That pool is then
handed to a model that looks at the question and one candidate passage **together, in a
single pass**, rather than comparing two independently computed fingerprints. That
joint attention is what makes it far more precise; it is also why it only ever runs
over an already-narrowed pool and never the whole corpus. It re-orders and trims; it
never finds anything new.

**Why the "wide net, then careful re-sort" shape matters:** a reranker can only reorder
what it was handed, so retrieval is tuned for recall (make sure the right answer is in
the pool at all) and reranking turns that wide, roughly ordered pool into a small,
precisely ordered one. The floor of 50 is not a guess: it was raised from 20 after
measuring that the correct document sits in a deduped top-20 pool about 91% of the time
but in a top-50 pool about 95%.

After scoring and truncating to the requested `top_k`, a **relevance floor**
(`RERANK_MIN_SCORE`, default `-3.0`) drops any surviving hit that still scores below it
— it is not padded back out to `top_k`, so a question with genuinely few relevant chunks
can come back with fewer contexts than asked for, rather than padding the answer with
irrelevant filler. The default was calibrated against a real measured case where a
genuine match scored 3.4 and irrelevant hits in the same pool clustered at -11.3..-11.45.
Set `RERANK_MIN_SCORE` to `None` to disable the floor and keep the old always-pad-to-`top_k`
behaviour.

The reranker also runs locally, at no per-query external cost. When it is configured
off, retrieval simply returns its fused order (no floor is applied); when it is on, the
scores reported are the reranker's, because those are what actually determined the
final order. A listwise "ask a chat model to score the whole pool at once" reranker is
also implemented — it produced a large quality gain at roughly 80× the latency, so it
stays an evaluation lever rather than the default.

Note that reranking always scores against the **original** question, even when the
question was decomposed — because the final answer is generated against the original
question too.

## 7. Generating a grounded, cited answer *(generate step)*

The final, reranked passages are handed to a chat model with a strict, narrow
instruction: answer **only** from these passages, and if the answer genuinely isn't
there, say exactly "I don't know." rather than inventing something plausible.

If the model does say that, the system does not stop at a dead end. It makes a second,
clearly separated call that answers from the model's own general knowledge — so an
ordinary greeting or an unrelated general question gets a sensible reply instead of a
blunt refusal. That answer is flagged as **not grounded**, and **no citations are
attached to it**, because it did not come from any document. Callers are expected to
surface that distinction to the user. (An empty context set — because retrieval found
nothing above the relevance floor — drives this same fallback.)

A grounded answer comes back with one citation per contributing passage: the chunk's
identity, its document, the filename, the human-readable location (page, page range,
sheet, or section), the score that determined its rank, and a short snippet.

The response also carries a small **trace** narrating the control flow actually taken:
whether the question was decomposed and into how many parts, how many candidates
hybrid retrieval produced, whether reranking ran, which model answered, and whether
the answer fell back to general knowledge.

## 8. Storing the answer for next time *(generate step)*

If the cache is enabled, the finished result is stored under the asking user, keyed on
the question's embedding and the answering model, under the tenant's current
generation, with a TTL. Both grounded and ungrounded answers are cached — per-user
scoping makes both safe to replay to the same person.

---

# PART 4 — THE SURFACES BUILT ON TOP

## The HTTP API

Unprefixed:

- **`POST /ingest`** — accepts a file and an optional scope; returns a document and job
  identity, or a deduplication notice. Requires ingest rights.
- **`POST /query`** — runs the **retrieve** step only (Part 3, steps 1–6): decompose,
  hybrid search, ACL filter, rerank. No generation, no cache. Returns contexts, chunk
  identities, scores, and any sub-questions.
- **`POST /answer`** — runs the **generate** step only (Part 3, steps 7–8): takes
  caller-supplied contexts/citations (typically from a prior `/query` call, but any
  source works), checks the answer cache, generates a grounded (or honestly ungrounded)
  answer, and caches it. Lets a caller inspect/adjust retrieved chunks before paying
  for generation, or plug in a different retrieval source entirely.
- **`GET /jobs/{id}`** and **`GET /jobs/{id}/trace`** — job status, and the full
  per-stage trace: where the job is, how long each stage took and what it produced, how
  the document was routed, and what ended up indexed.
- **`GET /documents`** — documents the caller may read, newest first, each with its
  most recent job. (The store pages by tenant and the permission rule then filters the
  page, so a page can come back shorter than requested without meaning the end was
  reached — proper permission-aware pagination is a known roadmap item.)
- **`GET /documents/{id}`** and **`GET /documents/{id}/chunks`** — document detail and
  the actual indexed chunks, with extractor, route reason, page, token count, and a
  text preview.
- **`POST /documents/{id}/reprocess`** — re-runs the pipeline; idempotent by
  construction.
- **`DELETE /documents/{id}`** — removes vectors, stored bytes, chunks, trace and job
  rows, invalidates cached answers, writes an audit entry. Admin only, own tenant only.
- **`GET /healthz`** (reports which backends are active) and **`GET /metrics`**
  (admin only).

Under `/onboarding` (Part 1 has the identity semantics):

- **`POST /onboarding/register`** / **`POST /onboarding/login`** — self-service
  signup/login, provisioning a tenant + admin transparently.
- **`GET /onboarding/status`** — unauthenticated; reports whether an LLM gateway is
  configured.
- **`POST /onboarding/gateway-config`** — admin only. Writes the LiteLLM `base_url` /
  `api_key` into a small `system_config` table living inside the metadata store (not a
  new adapter — just a new use of the existing one). `app/shared/gateway/client.py` checks
  that table on **every** outbound call and prefers it over the env-sourced default
  whenever it's non-blank — live, no restart. This is the one exception to "env is the
  source of truth" in this codebase.

## The onboarding UI

A single static page (`app/api/static/onboarding/`), mounted at `/`, no build step. A
signup-or-login form posts to `/onboarding/register` or `/login`, gets back an API
token, then checks `/onboarding/status`; if no gateway is configured yet, an admin-only
form appears to set the LiteLLM base URL/key via `/onboarding/gateway-config`
(skippable, settable later). Only the token is kept client-side (in `localStorage`) —
the password is never persisted. From there it hands off to the trace UI below.

## The Document Trace UI

A single self-contained page, served by the API itself at `/ui/trace/`, with no build
step, no framework and no package manager. Everything it needs is already exposed by
the API. Being served same-origin means it needs no cross-origin configuration and no
separate deployment.

It gates on an API token held only in the browser, lists ingested documents, and lets
you drop files onto it to ingest them. Selecting a document shows its journey: four
summary tiles (time in pipeline, searchable chunks produced, **how many pieces were
read for free**, and **how many needed the vision model**), a "why it cost what it
cost" breakdown of routing reasons tagged FREE or AI, the eight stages as cards with
status, duration bars and what each stage actually did for *this* document, the
extracted tags, and finally the real chunk table — each row traceable to the extractor
that produced it and the page it came from. While a job is running it polls and
updates live; failures show their error text. It also has an "Ask" tab that drives
`/query` then `/answer` and renders the answer with its clickable Sources panel.

Deletion from this UI is a two-step confirmation that spells out exactly what
disappears, rather than a bare "are you sure?", with destructive styling appearing only
at the second step.

## Running it in a container

`docker compose up` brings up Postgres+pgvector, a RediSearch-capable Redis
(`redis-stack-server` — plain Redis lacks the module the answer cache needs), the API,
and the worker — fully self-contained, no `.env` editing required. This path forces
the production backends on (Postgres metadata/queue, pgvector, Redis cache), unlike the
local defaults described in Part 6. `LITELLM_BASE_URL`/`LITELLM_API_KEY` ship blank on
purpose in `docker-compose.yml`; the onboarding UI's gateway-config step is the
intended way to set them, live, immediately after first signup. `scripts/seed.py
--platform` remains the way to provision the platform/global-knowledge-base tenant in
this path.

---

# PART 5 — OBSERVABILITY

**Structured logs with automatic correlation.** Every log line is one JSON object. A
per-request, per-job context (request id, tenant, user, job, document, stage, attempt)
is bound once at the edges and then attached automatically to every line inside that
scope. The practical effect: grep one identifier and you see the whole lifecycle —
request, ingest accepted, job claimed, each stage with its timings and counts, each
model call with its duration and retry count, job done. The API also returns its
request id in a response header, and deliberately keeps static-asset and health-check
requests out of the log.

**Counters and timings**, stored in the same database both processes share, so a
single metrics read covers API and worker together: ingest requests, accepted and
deduplicated; query requests; cache hits and misses; per-stage counts and accumulated
durations; jobs done, failed and dead; reprocess requests; documents deleted. Averages
are derived on read.

**The per-stage trace table** described in Part 2.9, which is the durable, queryable
counterpart to the logs and the sole data source for the trace UI.

What is deliberately *not* here yet: Prometheus-format export, distributed tracing, and
alerting. The correlation plumbing is already trace-shaped, so that is mostly exporter
wiring rather than redesign.

---

# PART 6 — PORTS, ADAPTERS, AND THE FLIP POINTS

Every infrastructural dependency sits behind an interface, and exactly one file — the
composition root (`app/shared/container.py`) — reads configuration and chooses concrete
implementations. Nothing in the pipeline, the retrieval logic or the API imports a
concrete adapter. That is the core architectural discipline of this repo.

| What | Interface | Implementations built |
|---|---|---|
| Metadata, jobs, chunks, audit, metrics, live gateway config | metadata store | embedded SQL database (local) · PostgreSQL (production) |
| Raw uploaded bytes | blob store | local filesystem, content-addressed |
| Searchable vectors | vector store | local file/embedded database · PostgreSQL + vector extension with ANN and full-text indexes |
| Work handoff | task queue | embedded database (single-writer claim) · PostgreSQL (skip-locked, multi-worker); both share one retry-backoff schedule |
| Text → vector | embedder | local compact model (default) · configurable local model · gateway-hosted |
| Candidate re-ordering | reranker | local cross-encoder (default) · listwise LLM (evaluation) · none |
| Answer reuse | answer cache | Redis with its query engine (optional; absent = disabled) |

The `system_config` table (Part 4) is not a new port — it's a new capability inside the
existing metadata-store port, used only for the live LiteLLM gateway override.

**Switchable by configuration, with no code change:** which metadata store, blob store,
vector store and queue are used; which embedding provider and which reranker; whether
chunk sizes derive automatically from the active embedder or are pinned; the relevance
floor after reranking; the vector index tuning parameters; the candidate-pool widening
factors; upload size limit; retry ceiling; worker poll interval; the platform tenant
that may publish firm-wide; the vision and chat models; and everything about the answer
cache including whether it exists at all.

**What is actually configured in this repository's `.env` right now:** the local
path — embedded database for metadata and queue, local filesystem for blobs, local-file
vector store — with the local embedding model, the local cross-encoder reranker
enabled, automatic chunk sizing on, and **the answer cache enabled** against a local
Redis. The PostgreSQL + vector-extension path is fully built and has been exercised end
to end, including the full benchmark suite in `docs/BENCHMARKS.md`, and is a
configuration change away — it's also what `docker-compose.yml` forces on
unconditionally (Part 4). (`CLAUDE.md` describes the Postgres/pgvector path as this
deployment's active backend; the checked-in `.env` says otherwise. Trust the `.env`
file and the `/healthz` endpoint, which reports the live backends, over either
document.)

**What has not been built:** cloud blob storage (so the blob store is single-machine
today — the API and worker must share a filesystem), a Celery/Redis-backed queue (some
adapter code comments still describe this as a future direction — that's stale
phrasing left over from before the Postgres queue adapter was built, not a real gap),
and binary-compressed embeddings. Each of those is deferred behind an interface that
already exists, not a redesign waiting to happen.

---

# PART 7 — HOW IT IS VERIFIED

**Tests.** 286 test functions across 28 modules (CLAUDE.md's "345 tests" is pytest's
collected-case count, which runs higher once parametrized variants are counted), all
hermetic: each runs against a throwaway container rooted in a temporary directory, with
the real local embedding model (process-cached, so it loads once per session) and
stubbed model calls, so the suite never touches the network. File fixtures — CSV, text,
Word, Excel, PDF — are built in memory, so the repo carries no binary test blobs.
Coverage spans identity and permissions, every loader, the block splitter, table
conversion, the chunker, tokenization, both stores, the queue's retry and dead-letter
behaviour, the vector store, BM25 scoring and hybrid fusion, reranking, query
decomposition, citations, job-event tracing, observability, onboarding (signup/login,
gateway-config override, per-email/per-IP rate limiting), the rate limiter itself, the
gateway client's retry/backoff, the API surface end to end, and the full ingestion
pipeline end to end.

Two things opt in deliberately rather than running by default: reranking (so the suite
isn't coupled to a second real model download) and the answer cache (which skips
entirely unless a suitable Redis is reachable — and which uses its own index name, so a
developer with a real cache configured doesn't get throwaway test tenants writing
permanent keys into the real cache's namespace).

**Evaluation and benchmarking.** A dozen scripts run the *real* shipped retrieval path
— not a hand-rolled approximation — against a public benchmark corpus with expert gold
relevance judgments, measuring recall, nDCG and MRR: dense-only and lexical-only
honesty references, the shipped hybrid path, and the shipped hybrid-plus-rerank path.
Others run an LLM-judged end-to-end quality evaluation, A/B a stronger embedder, A/B
the reranker with the embedder held fixed, A/B the listwise LLM reranker, A/B the
fusion simplification, diagnose where recall is actually lost (and what the ceiling
is, given that a reranker can only reorder what retrieval fetched), probe how sensitive
retrieval is to mere rephrasing, and run a head-to-head against an independently built
system using pure information-retrieval mathematics with no judge model anywhere.

Every one of those runs, in order, with its finding and whether it shipped, is written
up in `docs/BENCHMARKS.md` — including the honest negatives: the one clearly validated
improvement not yet adopted (a stronger embedder, held back by the real cost of
re-embedding an entire corpus at a different dimensionality), and the fact that no
naive baseline was ever run, so the comparisons prove two systems are comparable to
each other rather than that either is objectively good.

---

# PART 8 — WHAT IS DELIBERATELY NOT BUILT YET

This is a correct, tested reference implementation with a production storage path
already built and verified. It is explicitly not production-hardened. The full,
prioritised list lives in `CLAUDE.md` (read that version if this one ever drifts —
this section is a summary, not the source of truth); the headline items, as they
actually stand today:

- **Stuck-job recovery, plaintext API tokens, and missing file-safety limits are all
  fixed**, not open gaps — a lease + reaper reclaims jobs a crashed worker abandoned,
  API tokens are sha256-hashed at rest and compared by hash (never recoverable, same
  posture as the onboarding password hashes), and page/pixel/row/time limits are
  enforced on every loader. See CLAUDE.md roadmap items #1-#3 for exactly what's built.
- **Rate limiting is partial, not absent.** `/onboarding/register` and
  `/onboarding/login` are guarded per-email and per-IP (`app/shared/rate_limit.py`,
  in-process). Nothing else is rate-limited yet — no ingress-level throttle (API
  gateway/reverse proxy) and no per-process-coordinated limit for a horizontally
  scaled API (CLAUDE.md item #5).
- **Secrets live in an environment file** for everything except the LiteLLM gateway
  config, which now has a live, DB-backed override (Part 4) — but there is still no
  secrets manager integration for the rest (`DATABASE_URL` credentials, etc.).
- **No CI/CD, no infrastructure as code, and no load or chaos testing.**
  (Containerization now exists — `docker-compose.yml` + `Dockerfile` — but that's
  deployment packaging, not CI/CD.)
- **No migrations framework** — schema changes are raw `schema.sql` plus ad-hoc,
  idempotent `ALTER`/`CREATE INDEX IF NOT EXISTS` statements run on every `init_schema()`
  call, not a real migration tool (Alembic or similar).
- **No tenant/user/token admin API.** Self-service signup exists (Part 1), but there is
  still no way to invite a teammate, list/rotate tokens, or manage tenants short of
  each admin re-registering or re-running `scripts/seed.py`.
- **Blob storage is local-filesystem only**, which caps the system at one machine until
  a cloud blob adapter exists.
- **No document versioning, and tombstoned vectors are never compacted.**
- **No image preprocessing** (deskew, denoise), and per-page vision calls are serial.
- **No endpoint to change a document's visibility or sharing** after upload, even
  though the underlying rules are fully built and enforced.

---

# CLOSING — THE ONE THING THAT DOESN'T CHANGE

Every storage choice, every model choice, and every optional component described above
can be swapped by configuration. What never changes across any of those swaps is the
ingestion pipeline and the retrieval logic: they depend only on the abstract ideas of
"a place to put chunks and fingerprints", "a model that embeds", "a model that
reranks", and "somewhere to hand work to a worker" — never on which concrete thing is
currently plugged in. That is what makes it possible to develop the whole system on one
laptop with nothing installed, and run it on real infrastructure (or in one
`docker compose up`) without touching a line of the logic that makes it work.
