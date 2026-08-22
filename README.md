# RAG Ingestion Layer

Multi-tenant, RBAC-scoped ingestion for Docs / PDFs / Images / Tables / Excel.
Extracts (with intelligent routing to OCR/vision), chunks, embeds via a LiteLLM
gateway, **binarizes** the embeddings, and stores them in the `knowledgebase`
vector collection.

Runs **fully local** (SQLite + local-file vectors + local FS, no Docker). Every
backend sits behind a port so it flips to Postgres / Qdrant / Azure Blob / Celery
by changing `*_BACKEND` in `.env`.

A full, live, un-mocked run of everything below — real document, real model
calls, every chunk and embedding printed in full — lives in
[`notebooks/pipeline_walkthrough.ipynb`](notebooks/pipeline_walkthrough.ipynb).
This README describes the same flow it demonstrates; the numbers quoted below
(chunk counts, scores, example JSON) are taken directly from that notebook's
last real run.

## Run it end-to-end (step by step, no prior knowledge assumed)

There are two ways to try this system. **Option A (chat UI)** is the easiest way
to actually *see* the pipeline work — you drop a file into a chat window and
watch it move through every stage, then ask a question and see exactly where
the answer came from. **Option B (raw API)** is the bare FastAPI service —
useful if you're writing code against it, not exploring what it does.

Pick **whichever terminal you actually use** below — the two are not
interchangeable, and copy-pasting a bash command into PowerShell (or vice
versa) is the single most common thing that goes wrong here. If you're
double-clicking a "PowerShell" or "Windows Terminal" icon, or your prompt
looks like `PS C:\...>`, use the **PowerShell** blocks. If you opened "Git
Bash", or your prompt looks like `user@machine MINGW64 ~`, use the **Git
Bash** blocks. Every step below is in the repo folder (`cd rag-ingestion`
first).

### Option A — Chat UI (recommended: the visual, guided way)

This wires the pipeline up behind [Open WebUI](https://docs.openwebui.com/), a
free chat interface, skinned in Rimini Street's colors. You'll end up with a
normal-looking chat window where you can drag in a document and ask questions
about it.

**Step 1 — Start this repo's backend** (the part that does the actual
parsing/embedding/searching).

PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
# if PowerShell blocks this with an execution-policy error, run once:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
pip install -r requirements-dev.txt
Copy-Item .env.example .env            # fill in LITELLM_API_KEY (and model IDs) if you have one

python -m scripts.seed                 # prints a tenant + a bearer token -- copy it, you'll need it in Step 3
```

Git Bash:
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements-dev.txt
cp .env.example .env                   # fill in LITELLM_API_KEY (and model IDs) if you have one

python -m scripts.seed                 # prints a tenant + a bearer token -- copy it, you'll need it in Step 3
```

Then, in **two separate terminal windows** (same shell type, doesn't matter
which), leave these running the whole time — if either stops, ingestion/
queries stop working:

```
# terminal 1 -- the API
uvicorn app.api.app:app --port 8000

# terminal 2 -- the worker (this is what actually processes each document)
python -m app.worker
```

**Step 2 — Install and start Open WebUI** (the chat window itself). This is a
separate Python application with its own dependencies, so it needs its **own**
virtual environment — don't install it into the `.venv` from Step 1. It also
currently only supports **Python 3.11 or 3.12** (not 3.13) — check what you
have with `python --version`; if you're on 3.13+, install a 3.11/3.12
interpreter first (e.g. from [python.org](https://www.python.org/downloads/))
and swap it in below (`py -3.12` / `python3.12`).

PowerShell (new window, terminal 3):
```powershell
py -3.12 -m venv .venv-openwebui
.venv-openwebui\Scripts\Activate.ps1
pip install open-webui
open-webui serve                        # starts a chat UI at http://localhost:8080
```

Git Bash (new window, terminal 3):
```bash
py -3.12 -m venv .venv-openwebui        # or: python3.12 -m venv .venv-openwebui
source .venv-openwebui/Scripts/activate
pip install open-webui
open-webui serve                        # starts a chat UI at http://localhost:8080
```

Open `http://localhost:8080` in your browser and create the first account —
it automatically becomes the admin account.

**Step 3 — Connect Open WebUI to this repo's backend.** In the Open WebUI
browser tab:

1. Go to **Settings → Account → API Keys → Create new key** and copy it (you'll
   paste it into a Valve in the next step).
2. Go to **Admin Panel → Functions → +** (create a new function), and paste in
   the entire contents of
   [`integrations/openwebui/rimini_rag_pipe.py`](integrations/openwebui/rimini_rag_pipe.py).
   Save it, then make sure it's toggled **on**.
3. Click the little gear/**Valves** icon on that function and fill in:
   - `RAG_BASE_URL` → `http://localhost:8000` (this repo's API from Step 1)
   - `RAG_API_TOKEN` → the token printed by `scripts.seed` in Step 1
   - `OPENWEBUI_BASE_URL` → `http://localhost:8080`
   - `OPENWEBUI_API_KEY` → the API key you just created above
4. Start a new chat and pick **"Rimini Street Knowledge Assistant"** from the
   model dropdown at the top.

**Step 4 — Apply the Rimini Street look** (optional, purely cosmetic — skip
this if you just want it working). Full instructions, including what's
officially supported vs. best-effort, are in
[`integrations/openwebui/branding/README.md`](integrations/openwebui/branding/README.md).
The short version, still in the `.venv-openwebui` environment from Step 2:

PowerShell:
```powershell
$env:WEBUI_NAME = "Rimini Street Knowledge Assistant"
$env:WEBUI_FAVICON_URL = "integrations/openwebui/branding/favicon.png"
python integrations/openwebui/branding/apply_theme.py
```

Git Bash:
```bash
set -a && source integrations/openwebui/branding/.env.openwebui && set +a
python integrations/openwebui/branding/apply_theme.py
```
Then restart `open-webui serve` and refresh the browser. Also paste
[`integrations/openwebui/branding/banner.html`](integrations/openwebui/branding/banner.html)
into **Admin Panel → Settings → General → Banners** for the yellow/black brand
strip.

**Step 5 — Try it.**

1. In the chat, click the attachment (paperclip) icon and pick a file from
   `sample_pdfs/` (or any PDF/DOCX/CSV/XLSX/image of your own).
2. Send it. Watch the little status line above the chat message tick through
   each real pipeline stage — *Parsing document… → Routing… → Extracting…
   → Chunking… → Extracting metadata… → Embedding chunks… → Preparing
   vectors… → Indexing… → indexed and ready.* That's the actual backend
   working, not a fake progress bar.
3. Once it says "indexed and ready," ask a question about the document (e.g.
   *"What does this document say about revenue?"*). You'll see a second status
   trail — this time narrating the *retrieval* pipeline (whether your question
   got split into sub-questions, how many chunks it found, whether the
   re-ranker re-sorted them, which model generated the answer) — followed by
   numbered **citations** (which file and which page/section each fact came
   from) and finally the answer itself.

If anything doesn't behave as described, the detailed troubleshooting notes
are in [`integrations/openwebui/README.md`](integrations/openwebui/README.md).

### Option B — Raw API (curl / Swagger, for developers)

PowerShell:
```powershell
cd rag-ingestion
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env           # fill LITELLM_API_KEY + model IDs when available

# 1. seed a tenant + admin token
python -m scripts.seed

# 2. run the API (terminal 1)
uvicorn app.api.app:app --reload

# 3. run the worker (terminal 2)
python -m app.worker

# 4. ingest a file (use the token from step 1)
curl.exe -H "Authorization: Bearer <TOKEN>" -F "file=@some.pdf" http://127.0.0.1:8000/ingest
curl.exe -H "Authorization: Bearer <TOKEN>" http://127.0.0.1:8000/jobs/<JOB_ID>
```

Git Bash:
```bash
cd rag-ingestion
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env                  # fill LITELLM_API_KEY + model IDs when available

# 1. seed a tenant + admin token
python -m scripts.seed

# 2. run the API (terminal 1)
uvicorn app.api.app:app --reload

# 3. run the worker (terminal 2)
python -m app.worker

# 4. ingest a file (use the token from step 1)
curl -H "Authorization: Bearer <TOKEN>" -F "file=@some.pdf" http://127.0.0.1:8000/ingest
curl -H "Authorization: Bearer <TOKEN>" http://127.0.0.1:8000/jobs/<JOB_ID>
```

Interactive API docs (Swagger UI) are auto-served at `http://127.0.0.1:8000/docs`
once the API is running.

Note: in PowerShell, plain `curl`/`curl.exe` on some systems is aliased to
`Invoke-WebRequest`, which doesn't understand `-F`/`-H` the same way — use
`curl.exe` explicitly (as above) to force the real curl binary, or use
`Invoke-RestMethod` instead.

## End-to-end pipeline walkthrough

This is the full journey of one document through the system, step by step,
matching the stages in `notebooks/pipeline_walkthrough.ipynb`. Every example
below is real output from that notebook's last run against
`sample_pdfs/test-2.pdf` (a 5-page "SmartHome Hub" product launch report).

### Step 1 — Intelligent extraction (pay for AI only where it's needed)

For every page of a PDF, the router (`extract_pdf` in the notebook; the
equivalent production logic lives under `app/pipeline`) makes a sequence of
cheap-first decisions:

1. **Is there a structured table on the page?** `pdfplumber.find_tables()`
   pulls it out and converts it straight to markdown via `rows_to_markdown` —
   no AI, no cost. The table's bounding box is then excluded from the page's
   plain-text extraction so it isn't duplicated.
2. **Is there a real text layer?** If the page has typed text, it's read
   directly via `page.extract_text()` — free.
3. **Is the page mostly a picture with almost no extractable text?**
   (`len(text) < MIN_CHARS_SCANNED` — 20 characters — *and* the page has
   embedded images). That's the signal it's a scan. Only now is the page
   rendered to a PNG at ~200 DPI (`RENDER_SCALE = 200/72`) and sent to the
   vision model (`VISION_MODEL`, e.g. `claude-sonnet-5`) with a strict,
   no-hallucination transcription prompt that forbids summarizing or
   inventing content, and asks for a `[illegible]` marker instead of a guess.
4. **Is there a real embedded photo/figure on an otherwise-typed page?**
   Only images above `MIN_IMG_AREA_PT` (5000pt², to skip logos/icons) are
   cropped and sent individually to the vision model — the rest of the page
   is still read for free.

If a vision response itself contains a markdown table (e.g. a scanned form),
`split_blocks` re-splits it back into separate table/text elements so
downstream chunking still treats it as a table, not a wall of prose.

**Why it matters:** a client's multi-year document archive is typically
80-90% clean, typed content. Only genuinely hard pages (scans, handwriting,
photos) ever hit a paid AI call, so onboarding a client's full history stays
cheap and predictable rather than scaling linearly with document count.

On the sample document, this produced **5 elements, all free** (0 pages
needed the vision model) — one per page, each tagged with its page number,
modality (`text`/`table`/`image`), the extractor that produced it
(`pdf_text`, `pdf_table`, `vision`, `vision_table`), and *why* that path was
chosen (`text_layer`, `structured_table`, `scanned_page`, `figure_detected`).

### Step 2 — Chunking into traceable pieces

Extracted elements are packed into right-sized chunks (`ChunkSpec`), sized against
the **active embedder's real token limit** (`ChunkSpec.auto`, the default) rather than
a fixed guess — so a chunk is never silently truncated at embed time, and sizing
automatically scales if the embedding model is ever swapped:

- `target_tokens = 180` — aim for pieces around this size (MiniLM's 256-token limit)
- `max_tokens = 220` — hard ceiling, a chunk is never allowed past this
- `overlap_tokens = 20` — the tail sentences of one chunk are carried into
  the start of the next, so context isn't lost at a boundary
- `min_tokens = 16` — a leftover chunk this small is merged into the
  previous one rather than left as a fragment

Splitting respects paragraph (`\n\s*\n`) and sentence boundaries first, and
only hard-splits mid-sentence on the rare oversized paragraph that exceeds
`max_tokens` on its own. Tables get their own path (`_chunk_table`): if a
table doesn't fit in one chunk, it's split by rows, but the **header row is
repeated** at the top of every resulting chunk so a table piece is never
missing its column meaning. Every chunk keeps the full list of source pages
it was drawn from and gets a stable, predictable ID (`doc001001`,
`doc001002`, ...).

On the sample document (a short, 5-page report with light real text per page), the
five page-elements packed into a small number of chunks spanning pages
`[1, 2, 3, 4, 5]`; the notebook's saved run predates the current 180-token auto-sized
default and shows the numbers under the older fixed 512-token target — re-run the
notebook to see current chunk boundaries under the active default.

### Step 3 — Automatic metadata tagging

A single call to a small, fast chat model (`CHAT_MODEL`, e.g. `gpt-5-nano`)
reads the document text (capped at `TEXT_BUDGET_TOKENS = 3000` so cost per
document stays bounded) and returns strict JSON with exactly four keys:

```json
{"author": string|null, "date": string|null, "topics": string[], "entities": string[]}
```

The prompt explicitly forbids guessing — any field the model can't determine
comes back `null`/`[]` rather than a confident-but-wrong value, and if the
call fails or returns unparseable output, extraction never blocks
ingestion — it just leaves the fields blank.

Real output on the sample document:

```json
{
  "author": "Alex Johnson",
  "date": null,
  "topics": [
    "SmartHome Hub", "Market analysis", "Competitor analysis",
    "Marketing strategy", "Product launch", "AI-powered assistant",
    "Universal compatibility", "Security encryption"
  ],
  "entities": [
    "Innovative Tech Solutions, Inc.", "SmartHome Hub", "Alex Johnson",
    "sample-files.com", "CES", "IFA", "SmartTech Co.", "HomeGenius", "ConnectAll"
  ]
}
```

**Why it matters:** this is the same mechanism that will eventually
auto-match a document to the right SAP/Oracle access level (e.g. "Finance
module, Business Unit 1000") instead of a human manually tagging every
document.

### Step 4 — Embedding (turning text into searchable meaning)

Every chunk's text is converted into a 384-dimensional vector using
**all-MiniLM-L6-v2**, run **locally** via ONNX Runtime (`onnxruntime`,
CPU provider) — the model and tokenizer are pulled once from
`Xenova/all-MiniLM-L6-v2` on Hugging Face and cached. Token sequences are
truncated/padded to 256 tokens, mean-pooled over the attention mask, and
L2-normalized, so cosine similarity between two vectors reduces to a plain
dot product.

This step runs entirely on our own infrastructure: the gateway's AI vendor
doesn't offer embeddings as a paid endpoint, so it costs nothing per
document and never leaves the building. On the sample document, embedding
the 1 chunk took **40ms**.

### Step 5 — What actually gets stored

One record is persisted per document, with every chunk (and its full
embedding) nested inside it — this is the exact shape the production
`knowledgebase` collection stores:

```json
{
  "_id": "doc001",
  "tenant_id": "acme",
  "user_id": "u_admin",
  "visibility": "tenant",
  "acl_user_ids": [],
  "scope": "tenant",
  "source_type": "pdf",
  "filename": "test-2.pdf",
  "extracted_metadata": { "author": "Alex Johnson", "date": null, "topics": [...], "entities": [...] },
  "chunks": [
    { "chunk_id": "doc001001", "content": "...", "modality": "text", "embeddings": [384 floats] }
  ]
}
```

(In production the embeddings are additionally **binarized** before storage
in the vector index — see Status below; the notebook stores/prints the raw
float vectors for clarity.)

### Step 6 — Role-based access control (RBAC)

Every client already runs SAP/Oracle-style access rules (a warehouse clerk
doesn't see payroll data; one business unit doesn't see another's; one
client never sees another client's data). The system enforces the same
three-tier model on every single read, via one function, `can_view`,
evaluated top-to-bottom (first match wins):

1. `scope == "global"` → visible to **everyone, at every client** — this is
   our own firm-wide institutional knowledge, published by our team.
2. `role == "admin"` → an admin sees everything belonging to their own
   client/tenant.
3. `payload.user_id == user_id` → you can always see something you
   personally uploaded.
4. `visibility == "tenant"` → shared with the whole client, visible to
   anyone at that tenant.
5. `visibility == "shared"` and `user_id in acl_user_ids` → explicitly
   shared with specific people only.
6. Otherwise → blocked.

Verified against four real scenarios in the notebook, using this exact
document:

| Who's asking | Document | Result | Why |
|---|---|---|---|
| Acme's own admin | Acme's private doc | **VISIBLE** | Admins see all of their own tenant's data |
| A different employee at Acme | Acme's private doc | **BLOCKED** | Never shared with them |
| Someone at a different client entirely | Acme's private doc | **BLOCKED** | Client data never crosses tenants |
| That same different-client person | Our firm-wide (`scope=global`) doc | **VISIBLE** | Institutional knowledge is global by design |

### Step 7 — Hybrid retrieval: exact-word + meaning, fused

Two different ways a person might search call for two different techniques:

- **Exact wording** (a ticket number, part code, distinctive term) — handled
  by **BM25** (`rank_bm25.BM25Okapi`) over simple lowercased word tokens.
- **Described in their own words** — handled by **dense/meaning search**:
  cosine similarity between the query's MiniLM embedding and every chunk's
  embedding.

Neither is reliable alone (BM25 misses paraphrases; dense search can
under-weight a rare specific code), so the system runs **both every time**
and fuses the two rankings positionally with **Reciprocal Rank Fusion**:

```
score(chunk) = 1/(k + dense_rank) + 1/(k + bm25_rank)     # k = RRF_K = 60
```

Fusing by rank position (not raw score) avoids ever having to compare two
incomparable scoring scales directly. In the notebook, the rarest term in
the sample document (`"2025"`, appearing in only 1 of its chunks) is searched
this way and every chunk is shown ranked by dense score, BM25 score, and the
fused score.

### Step 8 — Reranking: a second, more careful pass

Hybrid search deliberately fetches a **wider pool** than requested (several times
`top_k`, see `rerank_min_candidates`/`rerank_candidate_multiplier`), and a
**cross-encoder** (`Xenova/ms-marco-MiniLM-L-6-v2`, run locally via ONNX — no gateway
call) re-scores every `(question, passage)` pair **jointly** in one forward pass, which
judges relevance more precisely than comparing two independently-computed embeddings.
The pool is then truncated back down to `top_k` in the reranker's order, not hybrid
search's order. This is on by default (`RERANKER_PROVIDER=cross_encoder`) — it was
benchmarked against alternatives (a stronger cross-encoder, and an LLM-as-reranker
approach) before being adopted; see `docs/BENCHMARKS.md`.

### Step 9 — Measuring retrieval quality (not just eyeballing it)

A single short document only produces one chunk, so it can't test whether
retrieval can tell the *right* passage apart from *wrong but plausible* ones.
To measure that properly, the notebook pools the sample document's chunks
together with a second, unrelated real document (a bank statement,
`Sbizhub_C2219080509040.pdf`) — 5 chunks total across 2 documents — then:

1. Asks the chat model to generate **one specific, factual test question per
   chunk**, written so it can only be answered from that exact passage
   (nobody hand-writes these, to avoid bias toward whichever search
   technique is being tested).
2. Runs every generated question back through both dense-only and hybrid
   search across the whole pool, and records the rank at which the
   question's true source chunk comes back.
3. Reports three standard IR metrics, computed twice (dense-only vs. hybrid):
   - **Recall@1** — correct chunk was the #1 result
   - **Recall@3** — correct chunk was somewhere in the top 3
   - **Mean Reciprocal Rank (MRR)** — average of `1/rank`; 1.0 = always #1

Real result from the notebook's last run (5 test questions, 5 pooled chunks):

| Metric | Meaning-only search | Combined (hybrid) search |
|---|---|---|
| Recall@1 | 80.0% | 80.0% |
| Recall@3 | 80.0% | 80.0% |
| MRR | 0.850 | 0.850 |

(On this small pool the two techniques tie — expected on content with little
repeated/distinctive vocabulary for BM25 to add on top of dense search; the
one miss, rank 4, was an account-statement-number question where the number
itself wasn't distinctive enough to separate it from similar-looking
statement text.)

For an external, published reference point beyond this small live test: the
same retrieval approach was also measured on **BEIR/SciFact** (5,183
documents, 300 expert-verified queries) — **nDCG@10 of 0.648**, essentially
matching a classic keyword-search baseline (0.652), while separately
achieving **92.5% recall** of all truly relevant results within the top 100.

### Step 10 — Generating a grounded, cited answer

The top-ranked chunks from Step 8 are handed to the chat model with a system
prompt that restricts it to **only** that content and requires it to say
`"I don't know."` verbatim if the answer isn't present — never inventing
anything. The answer is returned alongside the exact chunk ID(s) it came
from, so it's always traceable back to a specific passage, never "somewhere
in this document."

Real example from the notebook:

> **Question:** What does the document say about 2025?
> **Answer:** The global smart home market is projected to reach $135.3
> billion by 2025, growing at a CAGR of 11.6% from 2020 to 2025.
> **Traceable to:** `doc001001`

### Cost/step summary

| Step | AI model used | Cost | Why |
|---|---|---|---|
| Reading typed text & tables | None | Free | Ordinary software reads this perfectly |
| Reading scanned pages / images | Vision model | Paid, only when needed | Only a genuine scan needs AI to "read" it like a photo |
| Breaking into chunks | None | Free | Deterministic rules, no AI needed |
| Auto-tagging (author/date/topics) | Chat model | Cheap, one small call per document | Saves manual labeling work |
| Turning text into searchable meaning | all-MiniLM-L6-v2 (local) | Free | We run it ourselves; not offered as a paid gateway endpoint |
| Exact-word search | None | Free | Simple, local computation |
| Reranking | Local cross-encoder (ONNX) | Free | Runs on our own infrastructure, no gateway call |
| Retrieval quality testing | Chat model | Cheap, one small call per chunk | Generates independent test questions to measure accuracy |
| Final answer generation | Chat model | Cheap, small model | Short, focused generation only |

## Status
- **Day 1 (done):** ports, SQLite metadata store (7 tables + vector index), local
  blob store, SQLite queue, gateway client, `/ingest` + `/jobs` + `/documents`,
  worker loop with retry/dead-letter.
- **Day 2 (done):** loaders for all 5 shapes (pdf/docx/text/csv-html/xlsx),
  per-asset intelligent router, structure-aware sentence-safe chunker with
  header-repeating table splits, deterministic table→markdown (LLM only for
  image tables), chunk persistence + `GET /documents/{id}/chunks`.
- **Day 3 (done):** pluggable `Embedder` (gateway | local dev), embed + upsert
  stages, local float `knowledgebase` vector store (numpy float32 append-file +
  `vector_points` index), tenant-scoped cosine search, chunk↔vector linking.
  **Float embeddings for now — binarization deferred behind the same port.**
- **Day 4 (done):** full `tests/` suite (61 tests), tenant-isolation +
  delete-cascade tests, and **observability** — structured JSON logs with context
  (job/doc/tenant/stage/duration) + SQLite-backed cross-process metrics at
  `GET /metrics` (per-stage timings, job/ingest counters).
- **Day 5 (done):** production backends (Postgres metadata/queue + pgvector, built
  and exercised end-to-end — active in this deployment's `.env` today), a
  cross-encoder reranker (default-on), query decomposition for multi-part
  questions, and chunk sizing that auto-derives from whichever embedding model is
  active. Validated by five rounds of retrieval-quality benchmarking — see
  `docs/BENCHMARKS.md`.

See `docs/ARCHITECTURE.md` for the full, in-depth explanation of both the ingestion
and retrieval flows described above, and `docs/BENCHMARKS.md` for every retrieval
benchmark run against this system, in order, with a "what actually shipped" summary.

## Tests
```bash
pip install -r requirements-dev.txt
pytest                       # 132 tests, real MiniLM embedder (model cached after first load)
```
Each test runs against a fresh container in a temp directory using the real
ONNX MiniLM embedder (process-cached, loads once). Coverage:
`ids, chunker, tables, embedder, vector_store, metadata_store, queue, blob_store,
loaders, pipeline (E2E, all 5 shapes + dead-letter), api (auth/RBAC/dedup/
isolation/delete-cascade), query decomposition, reranking, BM25 (core + hybrid)`.

### Embeddings & vision
- **Embeddings**: real local **all-MiniLM-L6-v2 (384-dim) via ONNX** (`EMBEDDING_PROVIDER=minilm`,
  default) — no torch, no gateway. `gateway` provider is available if an embedding
  model is ever provisioned on the gateway. The vector store dim follows the embedder.
- **OCR/vision**: the gateway has **no OCR or embedding model** — only chat/vision.
  So there is no OCR concept; image/scanned-page transcription is done by the
  **vision LLM** (`VISION_MODEL=claude-sonnet-5`, `temperature=0`) with a **strict
  no-hallucination transcription prompt** (`app/pipeline/prompts.py`).
- **Windows note**: onnxruntime needs the MSVC C++ runtime, which this box lacks;
  the `msvc-runtime` pip package supplies the DLLs and `app/runtime.py` makes them
  discoverable at import.

### PDF stack note
Uses pure-Python **pdfplumber** (text/tables) + **pypdfium2** (rasterization) so
no MSVC runtime is required on Windows. Scanned-page OCR and figure captioning
call the gateway vision model, so they need `LITELLM_API_KEY` + model IDs set.

## Layout
```
app/ports        interfaces (MetadataStore, BlobStore, VectorStore, TaskQueue, Reranker)
app/adapters     sqlite / postgres / localfs / pgvector / queue / embedders / rerankers
app/gateway      LiteLLM client (embeddings/ocr/vision)
app/pipeline     stage runner
app/rag          query answering, decomposition, access control
app/api          FastAPI app + auth + routes
app/worker.py    queue-polling worker
scripts/seed.py  create a tenant + token
docs/            ARCHITECTURE.md (ingestion + retrieval deep dive), BENCHMARKS.md (retrieval-quality history)
notebooks/pipeline_walkthrough.ipynb   live, end-to-end demo of the full flow above
integrations/openwebui/   chat UI front end (Open WebUI Pipe + Rimini Street theme) -- see "Run it end-to-end" above
```
