# Architecture — Ingestion & Retrieval, End to End

This document explains how the system works, in plain language, with no code. It is
split into the two flows the system supports — **getting a document in** (ingestion)
and **getting an answer out** (retrieval) — because they run at different times, on
different processes, and solve different problems, even though they share the same
storage.

```
INGESTION:  file upload ──▶ parse/route/extract ──▶ chunk ──▶ tag ──▶ embed ──▶ store
RETRIEVAL:  question ──▶ [decompose?] ──▶ embed ──▶ hybrid search ──▶ ACL filter
            ──▶ rerank ──▶ grounded answer
```

Both flows are built on the same **ports-and-adapters** discipline: every piece of
infrastructure (where metadata lives, where files live, where vectors live, how work
is queued) sits behind an interface, with a local, zero-setup implementation and a
production implementation both available today, selected purely by configuration.
Nothing in the ingestion or retrieval logic below knows or cares which one is active.

---

# PART 1 — INGESTION

Ingestion has exactly one entry point and two distinct speeds: a fast, synchronous
"accept this file" step, and a slower, asynchronous "actually process it" step that
runs completely separately.

## 1. The front door — accepting a file

There is exactly one way a document enters the system: an authenticated upload. No
batch importer, no special-case path, no second door — a one-page text file and a
fifty-page scanned archive box both go through the same handler.

What happens the moment that request arrives is deliberately small and fast:

1. **Who is this, and are they allowed to add documents?** The caller's identity and
   role are resolved from their access token. Two of the system's three roles may
   ingest; the third (read-only) may not.
2. **Is this document meant to be private to this client, or firm-wide?** Publishing
   into the firm-wide, cross-client knowledge pool is a separate, tightly-gated
   decision — only an administrator belonging to one specifically designated
   "platform" tenant may do it. Every other document defaults to staying inside the
   uploader's own tenant.
3. **Is this a file type the system understands?** Recognized by file extension
   only — PDF, Word, Markdown, plain text/RTF, CSV, HTML, Excel, or a handful of
   image formats. Anything else is rejected immediately.
4. **Is it empty, or too large?** Both are rejected before any real work happens.
5. **Have we already seen these exact bytes, for this exact client?** The file's
   content is fingerprinted (a cryptographic hash of its raw bytes). If that fingerprint
   already exists for this tenant, the upload is a no-op — the caller gets back the
   existing document's identity rather than a duplicate. This makes re-uploading the
   same file by accident (or on purpose, as a "did this already work?" check) free and
   safe.
6. **Store the raw bytes, untouched**, addressed by that same content fingerprint.
7. **Write down two facts**: a document now exists, and a job exists to process it,
   currently just sitting in a "waiting" state. Every ingest is also written to an
   audit trail — who uploaded what, when.
8. **Respond immediately** with the new document's identity and the job's identity.
   No parsing, chunking, or model call has happened yet.

**Why split it this way?** The actual processing work is wildly unpredictable in
duration — a one-page invoice might take under a second; a fifty-page scanned archive
with no typed text at all might need dozens of individual AI transcription calls and
take minutes. If the upload request had to wait for all of that, every single upload
would tie up a web request for as long as the slowest possible document takes, and one
large backlog of scans would make the entire system feel broken for everyone, not just
whoever uploaded them. By making the front door do only the fast, cheap checks above
and handing the real work to a separate process, the API stays fast and responsive no
matter how much or how hard work is queued up behind it.

## 2. The worker — claiming and running jobs, safely

A separate, continuously-running process is responsible for all the actual work. It
repeatedly asks "is there a waiting job?", and whenever one exists, it claims it
(marking it as being worked on so no other worker instance can pick up the same job
twice) and runs it through every processing stage in order.

If the worker crashes partway through a job, or a stage raises an unexpected error, the
job is not lost — it is retried with a short backoff delay, and only after a fixed
number of failed attempts is it marked permanently dead ("dead-lettered") rather than
retried forever. A dead-lettered job doesn't quietly disappear either; it's a
first-class, inspectable end state, not a crash. This means a bad file, a transient
network blip talking to an AI model, or a worker restart never silently loses work or
requires someone to notice and manually recover it.

## 3. Reading the document — structure first, AI only when structure runs out

This is the governing idea behind everything that happens next, across every single
file type the system understands: **if a piece of content is already structured or
already typed, ordinary, free software reads it perfectly, and no AI model is ever
involved.** A real table stored as an actual table is extracted as a table. Real typed
paragraphs are read directly. Word documents keep their heading structure and their
tables intact instead of being flattened into one wall of text. Spreadsheets keep every
sheet as its own clean table. None of this costs anything or calls out to any AI model.

AI enters the picture only for the one case ordinary software genuinely cannot handle:
content that is fundamentally a **picture of information** rather than encoded
information — a scanned page, a photographed form, an embedded screenshot or chart.

The routing decision is made at a fine grain — not "is this whole file a scan?" but,
page by page (for a PDF) or element by element (for any file type):

- **Is there a real, extractable table on this page?** Pulled out deterministically
  and converted straight to a clean markdown table — no AI, no cost. Its location on
  the page is then excluded from the page's plain-text extraction, so the same content
  never appears twice.
- **Is there a real, typed text layer?** Read directly — free.
- **Is this page overwhelmingly a picture, with almost no real extractable text at
  all?** That combination — very little text *and* an embedded image present — is the
  specific, narrow signal that a page was scanned rather than typed. Only in this case
  is the page rendered to an image and sent to a vision-capable AI model, under a
  deliberately strict instruction: transcribe exactly what's visible, in reading order,
  never summarize, never guess, never invent a word or number that isn't legible, and
  mark anything genuinely unreadable rather than fabricate a plausible replacement.
  If that transcription itself contains something that looks like a table, it's
  re-detected and split back out as a proper table element — a scanned bank statement's
  transaction table ends up just as usable as one typed directly into a spreadsheet.
- **Is there a real embedded photo or figure sitting on an otherwise ordinary typed
  page?** Only images above a minimum size (so logos, icons, and decorative rules are
  skipped) are individually cropped out and sent to the vision model — not the whole
  page, just the picture. The rest of that page is still read for free.

Word documents and Markdown files apply the same "read structure for free" idea their
own way: both track a running awareness of the current heading and all of its parent
headings as they walk through the document (not just the single nearest heading), so
that every piece of content can later be traced back to exactly which section, and
which section's section, it came from. A table can be written two different ways in
plain text — classic markdown table syntax, or simple tab-separated rows — and both are
recognized and normalized to the same internal representation, so nothing downstream
needs to know or care which format the original file used.

Every single one of these decisions — what a piece of content actually is, which method
read it, and *why* that specific method was chosen — is recorded and carried forward
with the content itself, all the way into the final stored record. Nothing is read
anonymously; every stored piece of text can be traced back to exactly how it got there.

**Why this matters in practice:** a real client's document archive — years of support
tickets, SOPs, contracts, scanned forms — is overwhelmingly clean, typed,
already-structured content. If every page of every document required a paid AI call
just to be read, the cost of bringing a new client's entire history on board would
scale linearly with how much history they have, making full access to years of
accumulated knowledge financially unrealistic on day one. By reading the easy, common
case for free and reserving AI spend strictly for the genuinely hard minority of pages,
the cost of ingesting a client's full archive stays small and predictable regardless of
its size — the bill tracks the amount of genuinely unreadable content, not the total
page count.

## 4. Chunking — breaking a document into pieces that are individually meaningful

A whole document is never searched as one unit. It's broken into right-sized pieces
("chunks"), because a search system needs to compare a question against something small
and specific enough that a good match actually means something — and because the final
answer needs to point back to an exact passage, not "somewhere in this document."

Several ideas govern exactly how that breaking-apart happens, each one addressing a way
this quietly goes wrong if done carelessly:

- **Pieces are sized against what the search model can actually understand — its real,
  measured limit, not a generic estimate.** The model that turns text into a searchable
  "fingerprint of meaning" has a hard ceiling on how much text it can look at in one go;
  anything past that ceiling is silently dropped before the fingerprint is even
  computed. Chunk sizes are measured directly against that specific model's real
  counting method, not a generic, unrelated approximation that happens to sound
  reasonable. (This system was once bitten by exactly that mistake: chunks were sized
  against a stand-in counting method that undercounted relative to the real one, so
  chunks were routinely twice the model's actual limit and silently truncated before
  they were ever searchable — with nothing anywhere reporting the loss. That's now
  fixed, and the sizing automatically adapts if the underlying search model is ever
  swapped for a different one with a different limit, rather than needing to be
  re-tuned by hand.)
- **A piece is never cut off mid-thought.** Splitting respects paragraph and sentence
  boundaries first — a sentence is never torn in half, a numbered list is never
  shredded at each list marker — and only hard-splits mid-sentence on the rare
  oversized paragraph that can't otherwise fit. Where one piece needs to carry a little
  context from its neighbor across a boundary, that overlap is always a whole trailing
  sentence, never a raw slice of characters, so a boundary never reads as an arbitrary
  cut, to a person or to the AI generating an answer from it.
- **Tables get their own treatment.** A table that doesn't fit in one piece is split by
  rows rather than by an arbitrary character count, and the header row is repeated at
  the top of every resulting piece, so a table fragment is never missing the column
  meaning that makes it readable. A row that would otherwise sit right at a split
  boundary is carried into both neighboring pieces, so it's findable either way.
- **Every piece remembers exactly where it came from and what it's part of.** Each
  chunk carries its own section identity — which heading, and which heading's heading,
  all the way up to the document title. This matters most on documents that repeat the
  same boilerplate structure many times under different topic headings (a real,
  observed pattern in operational documentation): without this, two chunks describing
  completely different topics could be textually almost identical, and a search system
  would have no way to tell them apart. Carrying this ancestry forward means every
  chunk stays uniquely identifiable no matter where a piece happens to be cut, without
  needing to repeat information that's already visible at the top of the chunk itself.

None of this was designed in the abstract — the section-identity fix in particular was
made in direct response to a real, measured retrieval-quality problem found during a
head-to-head comparison against an independently-built system (see
`docs/BENCHMARKS.md`), not discovered by inspection or guessed at ahead of time.

## 5. Automatic tagging — metadata for free

A single call to a small, fast AI model reads through the document's text once and
returns a handful of structured facts about it: who likely wrote it, when, what topics
it covers, and what named things (systems, products, people, standards) it mentions.
The model is explicitly told not to guess — anything it can't determine comes back
blank rather than a confident-but-wrong value.

This step is treated as auxiliary, not essential: if the call fails, or the model's
response can't be understood, the document simply ends up with blank tags rather than
the whole ingestion job failing. A tagging hiccup never blocks a document from being
searchable.

This isn't a dead-end nicety — those tags are carried forward and become part of what a
later search can match against (see Part 2, retrieval), so a topic or a named entity
mentioned in a document can be found even if a searcher's question doesn't use the
document's exact wording.

## 6. Turning text into searchable meaning

Every chunk's text is converted into a numeric "fingerprint of meaning" — a vector —
using a model that runs entirely on the system's own infrastructure, at no incremental
cost per document, rather than a paid external call. Two vectors that are close to each
other numerically represent text that means something similar, even if the wording is
completely different, which is what lets someone ask a question in their own words and
still find the right passage.

A placeholder step exists to eventually compress these fingerprints into a smaller,
faster-to-search representation; today the system stores the full-precision
fingerprint as-is.

## 7. What gets stored, and how it stays secure

Once a document has been read, routed, chunked, tagged, and turned into searchable
fingerprints, two things are written down together, permanently, and kept in lockstep:

- **The record of ownership and permission** — which client this belongs to, who
  uploaded it, who else is allowed to see it and under what sharing rule, and the exact
  text of every chunk it was broken into, each with its own stable, predictable
  identity.
- **The searchable index** — every chunk's fingerprint-of-meaning, carrying just enough
  of that same ownership/visibility information alongside it so that a search can be
  narrowed down to only what a given person is actually allowed to see *before*
  anything gets ranked, never after the fact.

Two independent, real implementations of this storage exist side by side today,
selected purely by configuration:

- A **local, zero-setup implementation**, everything in a single embedded database
  file, requiring no external infrastructure at all — meant for running the whole
  system on a single machine with nothing installed beyond the application itself.
- A **production implementation**, a real relational database plus that same
  database's dedicated vector-search extension, purpose-built for large-scale
  similarity search. The relational half holds the same ownership/permission
  bookkeeping; the vector-search half stores one row per chunk with its fingerprint as
  a natively indexed column, so finding the best matches becomes a fast, index-accelerated
  lookup rather than pulling every stored fingerprint into memory and comparing them
  all by hand on every single question. This production path is not aspirational — it
  is the one actually active in this deployment today.

Re-processing an existing document (say, after a chunking or tagging improvement) is
safe to run repeatedly: any prior chunks and fingerprints for that document are dropped
first, so re-running never leaves stale, duplicate, or orphaned data behind.

---

# PART 2 — RETRIEVAL

Retrieval answers a question against everything already ingested. It runs synchronously
— a person asks a question and waits for an answer in the same request — because,
unlike ingestion, the work here is small and bounded (a handful of database lookups and
one or two AI calls), not the open-ended, unpredictable-duration work ingestion does.

## 1. Is this really one question, or several stitched together?

Before anything is searched, the system makes a cheap, free check: does this question
show a real surface signal of actually being two or more distinct questions glued
together — an explicit comparison ("how does X relate to Y"), two question marks, or
similar? This check costs nothing and runs on every question.

Only if that free check flags something does the system spend a small AI call to
actually decide — and that call can itself say "no, this is genuinely one focused
question" and override the free check's guess. This is a deliberate safety net against
the free check being overzealous, not a rubber stamp.

**Why bother at all?** A single search pass over a combined, multi-topic question
dilutes the ranking signal for every topic inside it — neither half scores as strongly
as it would on its own. If a question is genuinely multi-part, it's split into a small
number of focused, independent sub-questions, each one is searched for separately, and
the results are merged and de-duplicated (a chunk relevant to more than one sub-question
keeps its best score and only counts once). Critically, **the final written answer is
still generated against the original question**, using the combined evidence from every
sub-question — the person asking never sees their question mechanically broken apart,
only a well-supported answer to what they actually asked.

## 2. Turning the question into the same kind of fingerprint

The question is converted into a fingerprint-of-meaning using the exact same model used
for every stored chunk during ingestion — this is what makes the two comparable at all.

## 3. Hybrid search — two different ways of matching, combined

Two fundamentally different techniques run every single time, side by side, over the
candidates a given person is allowed to see:

- **Meaning-based (dense) search** — comparing the question's fingerprint against every
  stored chunk's fingerprint, finding the ones that are numerically closest, i.e. the
  ones that mean something similar to the question, regardless of exact wording.
- **Exact-word (lexical) search** — an independent pass that matches literal words and
  phrases, so a ticket number, a part code, or another distinctive exact term is never
  missed just because a meaning-based comparison doesn't weight it strongly.

Neither is reliable alone: meaning-based search can under-weight a rare, specific
identifier a person typed verbatim; exact-word search misses a paraphrased question
entirely. So both run every time, and their two independent result lists are combined
into one final ranking using **reciprocal rank fusion** — every candidate earns a score
based on *where it ranked* in each list (not its raw score), and a candidate found by
both lists earns credit from both. Combining by rank position, rather than by raw score,
sidesteps the problem of the two techniques producing scores on completely different,
incomparable scales. Both signals count equally by default.

The automatically-extracted tags from ingestion (topics, named entities, likely author)
are folded into what the exact-word search matches against — so a document can be found
by a topic or a named entity it mentions, even if the chunk's own wording doesn't
happen to contain that exact term. This is a real, already-paid-for signal from
ingestion that would otherwise sit unused.

## 4. Who's allowed to see this — enforced before ranking, not after

Every single candidate, from both search techniques, is checked against a permission
rule before it's allowed to influence the final ranking at all — never filtered out
afterward, which would let a search rank things a person isn't even allowed to know
exist. The rule, checked in a fixed order, first match wins:

1. Is this firm-wide institutional knowledge, published deliberately to be visible to
   every authenticated person at every client? If so, visible — full stop, regardless
   of any of the checks below.
2. Is the person asking an administrator of their own client? Administrators see
   everything that belongs to their own client.
3. Did the person asking personally upload this? A person can always see their own
   work.
4. Is this document shared with the person's entire client/team? If so, visible to
   anyone on that team.
5. Is this document shared with a specific, named list of people, and is the person
   asking on that list? If so, visible.
6. Otherwise: not visible.

A client's data never crosses into another client's search results, with one
deliberate, narrow exception: content explicitly published as firm-wide knowledge
(gated separately, and only by a designated administrator role, so no client can
self-publish into that shared pool). Being able to *read* firm-wide content never
implies being able to *modify or delete* it — only the team that owns it can do that,
even though everyone else can see it.

## 5. Reranking — a second, more careful look at the top candidates

Hybrid search is deliberately asked to fetch a **wider pool** of candidates than the
question actually needs — say, several times more than will ultimately be shown. That
wider pool is then handed to a second, more computationally expensive step: a model
that looks at the question and one candidate passage *together, at the same time*
(rather than comparing two independently-computed fingerprints), which lets it judge
relevance far more precisely. Because this joint comparison is expensive, it only ever
runs over the already-narrowed pool from hybrid search, never the entire stored
collection — its job is strictly to re-order and trim what retrieval already found, not
to find anything new. The final, small set of results shown to the answer-generation
step is picked from this re-ordered pool, not from the original hybrid-search order.

**Why does this two-step "cast a wide net, then carefully re-sort" shape matter?**
A reranker can only ever reorder the candidates handed to it — it can never surface
something retrieval never fetched in the first place. So retrieval is deliberately
tuned to favor *recall* (make sure the right answer is somewhere in the pool at all),
and reranking is what turns that wide, roughly-ranked pool into a small, precisely
-ordered one worth actually reading. Skipping the wide-pool step and reranking only
what a narrow top-k already returned would starve the reranker of exactly the
borderline-but-correct candidates it exists to rescue.

## 6. Generating a grounded, cited answer

The final, reranked set of passages is handed to a language model with a strict,
narrow instruction: answer **only** using what's actually in those passages, and if the
answer genuinely isn't there, say so explicitly rather than invent something
plausible-sounding. The answer returned is always paired with the exact identifiers of
the chunks it was generated from, so any answer can be traced back to a specific,
real, stored passage — never "somewhere in this document," and never a free-floating
claim with no source.

---

# Two flip points, one set of logic

Everything described above runs identically regardless of which concrete
infrastructure is behind it. Two separate axes can each be flipped independently, purely
by configuration, with no change to any of the logic above:

- **Where things are stored** — a local, zero-setup mode (an embedded database, local
  files) for running the whole system on a single machine with nothing installed, or a
  production mode (a real relational database plus its native vector-search extension)
  for real deployments. This deployment currently runs the production mode.
- **Which model does the meaning-based embedding, and which model reranks** — the
  system has been benchmarked against several alternatives for both (see
  `docs/BENCHMARKS.md`); today it runs a compact, fully local embedding model and a
  compact, fully local reranking model, both chosen so that neither ingestion nor
  retrieval depends on paying for every single embedding or every single rerank call
  against an external AI vendor.

The one thing that never changes across any of these flips is the ingestion pipeline
and retrieval logic described above — they depend only on the abstract idea of "a place
to store chunks and fingerprints" and "a model that embeds/reranks," never on which
concrete implementation is currently plugged in.
