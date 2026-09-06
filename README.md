# RAG Ingestion Layer

A multi-tenant, permission-aware system that takes in your documents — PDFs,
Word docs, spreadsheets, scanned images, whatever — and lets you ask plain
questions and get back answers grounded in the exact passages that back them
up, cited every time.

Here's the whole story, start to finish.

## 1. Start the stack

All you need is Docker.

```bash
git clone <repo-url>
cd Rimini-Vector-RAG
docker compose up
```

That's it — no `.env` to hand-edit first, no Python environment to set up.
Compose brings up four things behind the scenes: Postgres (with the pgvector
extension, for storing chunks and their embeddings), Redis (for an optional
answer cache), the API, and a worker that quietly processes documents in the
background. Give it a minute the first time — the images have to build and
Postgres has to initialize.

## 2. Open it in a browser

Go to `http://localhost:8000/`. You'll land on the onboarding page.

Pick "Create account", give it an email and password, and submit. Behind the
scenes this quietly creates your own tenant — a private space, walled off
from anyone else using this same deployment — and makes you its first admin.

Next you'll see a gateway screen asking for a LiteLLM base URL and key. This
is where the system sends its LLM calls — generating answers, and reading
scanned pages that can't be parsed as plain text (embeddings run locally, so
no key is needed for those). Paste yours in once; it's saved live, no restart
needed, and you can change it later. If you don't have one handy yet, you can
skip this screen and come back to it.

Once you're through, you'll land on a screen with your API token already
generated — you won't need to touch it directly, the UI carries it forward
for you automatically.

## 3. Give it a document

Open `http://localhost:8000/ui/trace` — the Document Trace UI. Drag a PDF,
Word doc, spreadsheet, or image onto the left panel.

The file appears in the list right away, and a live trace ticks through each
pipeline stage as the worker picks it up in the background:

- **parse** — reading the file into raw pieces: text, tables, images
- **route** — deciding what actually needs a vision model (a scan, a photo)
  versus what's plain typed text, read for free
- **chunk** — breaking the document into small, individually meaningful pieces
- **metadata** — a quick pass to guess the author, date, topics, key entities
- **embed** — turning each chunk into a vector, so it can be found by meaning
- **store** — writing it all into the knowledge base, ready to be searched

A short document usually finishes in a few seconds. Click any finished
document afterward to replay its trace at any time.

## 4. Ask it something

Switch to the **Ask** tab in the same UI, and type a question in plain
English about anything you've ingested.

The answer comes back with a **Sources** panel underneath — the exact
passages it was actually grounded in, word for word, each one clickable to
jump straight to that spot in the original document. If nothing relevant was
found, it says so honestly ("I don't know") instead of making something up.

Under the hood, that one question runs its own small pipeline: it's embedded,
checked against a semantic cache (in case this exact question was asked
before), searched two different ways at once — exact keyword matching and
meaning-based search — and merged, re-ranked by a second, more careful pass,
and only then handed to the model, with strict instructions to answer only
from what it's been given.

## 5. That's the loop

Upload more documents, ask more questions. Everything stays scoped to your
own tenant unless you, as an admin, explicitly mark something as shared —
nobody else's documents leak into your answers, and yours don't leak into
theirs.

---

## Want to go deeper?

- **`docs/ARCHITECTURE.md`** — the full technical narrative behind every step
  above: routing logic, chunking rules, the exact access-control model, hybrid
  retrieval math, reranking, all of it, in plain language, no code required.
- **`docs/BENCHMARKS.md`** — every retrieval-quality experiment run against
  this system, in order, including a head-to-head comparison against an
  independently built system.
- **`notebooks/pipeline_walkthrough.ipynb`** — a live, real, un-mocked run of
  the whole ingestion pipeline against a real document, every intermediate
  output printed in full.

## Running it without Docker

```bash
python -m venv .venv && source .venv/Scripts/activate     # Windows/Git Bash
pip install -r requirements-dev.txt
cp .env.example .env              # set LITELLM_API_KEY
python -m scripts.seed            # prints an admin API token
uvicorn app.api.app:app --port 8000     # terminal 1
python -m app.ingest.worker              # terminal 2
pytest                                   # 345 tests
```

## Layout
```
app/shared       config, composition root, domain models, gateway client, and the
                 ports/adapters genuinely used by both flows (metadata_store,
                 vector_store, embedder, sqlite/postgres/pgvector/localfs, bm25)
app/ingest       ingestion pipeline (parse -> route -> chunk -> metadata -> embed -> store),
                 worker, task queue + blob store ports/adapters
app/retrieval    query answering, decomposition, access control, reranker +
                 answer-cache ports/adapters
app/api          FastAPI app + auth + ingest/retrieval routes + the onboarding & trace UIs
app/ingest/worker.py   the background queue-polling worker
scripts/seed.py  create a tenant + token from the command line, no UI needed
docs/            ARCHITECTURE.md (deep dive), BENCHMARKS.md (retrieval-quality history),
                 adr/0001-ingest-retrieval-split.md (why app/ is split this way)
```
