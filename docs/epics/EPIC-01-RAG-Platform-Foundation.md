# RKB-EPIC-01: RAG Platform Foundation

**Status: ✅ Complete (build-and-test level). Not yet staged/production-validated.**

**DRI:** John Paul Gummapu (AI Data Engineer) · **Sign-off:** Amith K A
**Depends on:** None (this is the foundation everything else builds on)

---

## What this epic is, in plain terms

This is the core of the system: a service that can take a document a user
uploads (a PDF, a Word file, a spreadsheet, a scanned image, whatever), read
it, break it into searchable pieces, and later let a user ask a question and
get back an answer with citations pointing to exactly where in their
documents that answer came from — while making sure one tenant (one
customer/organization) can never see another tenant's documents.

Three stories make up this epic:

1. **Ingestion, parsing & metadata** — reading files in and understanding them.
2. **Retrieval, storage, observability & testing** — finding the right
   information later and proving the whole thing actually works.
3. **Answering UI & chat integration** — the part a human actually looks at.

---

## Story RKB-STORY-01: Ingestion, parsing & metadata

**Status: ✅ Done and tested.**

### What it does

When a file is uploaded, the system:

- Figures out what kind of file it is (PDF, Word, Excel, CSV, HTML, Markdown,
  image, or plain text) and hands it to the right reader for that format.
- Reads real text and tables directly wherever it can (fast, free, no AI
  needed) — a Word document's paragraphs and tables, a spreadsheet's rows, a
  CSV's columns.
- Only calls in an AI model (a "vision" model that can read images) when the
  content genuinely can't be read as plain text — a scanned page, a photo, a
  chart. This keeps cost tied to how much of a document is actually
  unreadable, not to how many pages it has.
- Breaks the extracted content into small, meaningfully-sized chunks (not too
  big, not too small) that keep track of which section of the document they
  came from, so a search result can say "this came from page 4, under the
  'Financials' heading."
- Makes a best-effort attempt to pull out useful metadata — author, date,
  topics, named entities (people/companies/products mentioned) — using an AI
  model, but never lets that fail the whole upload if the AI call has a
  problem.

### How the flow works, step by step

```
File uploaded → what type of file is it? → read it (structure first, AI only
if needed) → split into elements → group into right-sized chunks (with page/
section info attached) → best-effort extract author/date/topics/entities →
ready for the next stage (turning it into something searchable)
```

### What's verified

- Every file type (PDF, DOCX, XLSX, CSV, HTML, Markdown, images, plain text)
  has dedicated automated tests confirming it's read correctly.
- Tests confirm the "structure first, AI only when needed" rule actually
  holds — e.g. a text-based PDF page is never sent to the AI model, only a
  genuinely scanned one is.
- Tests confirm a failure in the metadata-extraction step (author/topics/etc.)
  never blocks or breaks the upload — the document still gets ingested with
  that metadata simply left blank.

---

## Story RKB-STORY-02: Retrieval, storage, observability & testing

**Status: ✅ Done and tested.**

### What it does

- **Finding the right chunks:** when a user asks a question, the system
  searches using two methods at once — a meaning-based search ("vector"
  search, which understands *what the question means*, not just the exact
  words) and a keyword search ("BM25", the same style of ranking used by
  classic search engines). Combining both catches things a single method
  would miss.
- **Handling complex questions:** if a question is really two questions
  glued together ("how does X compare to Y"), the system detects that and
  searches for each part separately, then merges the results — this
  measurably improves answer quality on multi-part questions.
- **Where the data lives:** there are two interchangeable storage options —
  a fully local, no-setup option for development, and a
  production-grade option (PostgreSQL with the pgvector extension) that this
  deployment actually runs on today. Switching between them is a
  configuration change, not a code change.
- **Seeing what's happening:** every request and every pipeline step writes
  structured logs with a shared "request ID" so a single upload or a single
  question can be traced end-to-end across every step it went through.
- **Optional answer cache:** if a question (or a very similarly worded one)
  was already answered for a user, the system can serve that answer back
  instantly instead of re-searching and re-generating — this is optional and
  only turns on if a specific caching backend (Redis) is configured; it
  changes nothing for a deployment that doesn't set it up.

### What's verified

- Automated test suite: **300+ tests**, covering ingestion, retrieval, access
  control, hybrid search ranking, decomposition, the observability/logging
  behavior, and the optional cache.
- Retrieval quality has been independently benchmarked, not just
  unit-tested: against a standard public benchmark (BEIR), against an
  LLM-graded answer-quality benchmark (RAGAS), and in a direct head-to-head
  comparison against a separately-built internal system (referred to as
  "KHUB" in the benchmark reports) — all documented in
  `docs/BENCHMARKS.md` with the actual numbers, not just a claim that it was
  done.
- 3 known, pre-existing test failures remain open (a UI static-file
  detail, a validation edge case, and a bug in one test's own sample data) —
  flagged, not hidden, and none of them affect ingestion, retrieval, or
  access-control correctness.

---

## Story RKB-STORY-03: Answering UI & chat integration

**Status: ⚠️ Partially done — flagged below.**

### What's done

- **Ingestion & Document Trace UI**: a working web page where someone can
  drag in a document, watch it move through each processing stage in real
  time, and see exactly how it was parsed, chunked, and indexed.
- **"Ask" view**: the same UI has a tab to ask a question and see the answer
  come back with a "Sources" panel — clickable citations that jump back to
  the exact passage the answer was grounded in.
- **Grounded answers with safe fallback**: the answer generator is
  instructed to answer only from the retrieved document content. If nothing
  relevant is found, it says so plainly and falls back to a general answer
  *without* attaching document citations to it — so a citation is never
  shown next to an answer that didn't actually come from the user's
  documents.

### What's flagged as pending

- **Chat front-end integration (Open WebUI)** — not built yet. There is no
  existing integration with Open WebUI (or any other third-party chat
  front-end) in the codebase today. The `/query` and `/answer` API endpoints
  are ready to be called by an external chat UI, but the actual integration
  work hasn't started. **Flagging this now, not claiming it's done.**

---

## Bottom line for this epic

The foundation — read documents in, search them intelligently, answer with
proof — is built, tested, and benchmarked. What's not yet done is wiring a
third-party chat interface (Open WebUI) on top of it; the custom-built Trace
UI already works as a first-party way to use the system today.
