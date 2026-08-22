# RAG Ingestion Layer — CLAUDE.md

Read this first, before exploring the repo. It's a map, not a copy of the code —
verify specifics by reading the actual file before relying on line numbers or
claims below (this file can drift as the code changes).

**Full narrative documentation:** `docs/ARCHITECTURE.md` is the authoritative deep
dive — the full ingestion and retrieval flows, explained in depth, in plain language
(no code). Read it when you need depth; this file is the fast-orientation index.
Also see `docs/BENCHMARKS.md` — every retrieval-quality benchmark run against this
system, in order, including the head-to-head comparison against an independently-built
system that drove the chunking hardening, and a "what actually shipped" summary tying
every finding to the current default configuration.

## What this is

A multi-tenant, RBAC-scoped RAG ingestion pipeline (Python/FastAPI). Two flows:

```
INGEST:  file ─▶ parse ─▶ route ─▶ extract ─▶ chunk ─▶ metadata ─▶ embed ─▶ store
QUERY:   question ─▶ embed ─▶ [semantic answer-cache hit? ─▶ return instantly]
         ─▶ [decompose if multi-part] ─▶ hybrid retrieve
         (dense+BM25, ACL-filtered) ─▶ assemble context ─▶ LLM answer ─▶ cache
```

The answer cache is OPTIONAL (per-user scoped, Redis Stack vector KNN); it is
inert unless `REDIS_URL` is set, so the local/default path is unchanged.

~5,164 lines, 71 modules, 132 tests. Runs **fully local** (SQLite + local-file
vectors + local filesystem, no Docker) — everything sits behind a port/adapter
so it "flips" to Postgres/pgvector/Azure Blob/Celery via `.env`, no code change.
Postgres + pgvector adapters are **already built, exercised end-to-end, and are
the backends actually active in this deployment's `.env` today**; Azure Blob and
Celery are not built yet (still local-only).

Key design principle: **structure first, AI only when structure runs out.**
Real tables/text (DOCX, CSV, typed PDF text, Markdown) are parsed with plain
libraries, free. AI (vision LLM) is only invoked for genuinely unreadable
content — scanned pages, photos, embedded figures — so ingestion cost tracks
the amount of unreadable content, not total page count.

## Directory map

```
app/
  config.py            Settings (pydantic-settings) — *_BACKEND flip switches live here
  container.py         composition root: build_container() wires adapters from config
  ids.py                BSON-compatible ObjectId generator (no pymongo)
  runtime.py            Windows MSVC DLL bootstrap for onnxruntime
  observability.py      structured JSON logging + correlation contextvars
  worker.py             `python -m app.worker` — polls queue, runs pipeline per job

  domain/models.py      enums + dataclasses: Role, Modality, Job, ChunkRecord, Principal…

  ports/                interfaces: metadata_store, blob_store, vector_store, task_queue, embedder, answer_cache
  adapters/
    sqlite/              metadata_store + queue + metrics (LOCAL default)
    postgres/            metadata_store + queue + metrics (PRODUCTION, built & tested)
    localfs/              blob_store + vector_store "knowledgebase" (LOCAL default)
    pgvector/             vector_store, native vector column + HNSW index (PRODUCTION, built & tested)
    shared/bm25.py         RRF fusion (dense + BM25), used by both vector store adapters
    queue/sqlite_queue.py  claim/complete/retry/dead-letter logic
    embedders/             minilm (local ONNX, default) | onnx_embedder (configurable
                           HF ONNX model, e.g. bge) | gateway (LiteLLM)
    rerankers/              cross_encoder (local ONNX, default) | llm_reranker (gateway, eval-only)
    cache/redis_stack.py   semantic answer cache — RediSearch vector KNN (OPTIONAL, off unless REDIS_URL set)
  gateway/client.py      LiteLLM client: embed / vision / chat

  pipeline/
    runner.py            stage machine: parse→route→extract→chunk→metadata→embed→binarize→upsert
    loaders/              pdf, docx_loader, markdown, text, table (csv/html), excel, image
    blocks.py             shared prose/table/code block splitter + heading-stack tracker (section_path)
    chunker.py            structure-aware, token-budget-aware chunking (target 180/max 220/overlap 20)
    tables.py             deterministic table → markdown
    tokens.py             count_tokens() against the REAL MiniLM WordPiece tokenizer; EMBED_MAX_TOKENS=256
    elements.py            Element dataclass
    metadata_extract.py    best-effort LLM author/date/topics/entities extraction
    provenance.py          per-element extractor/route_reason bookkeeping
    prompts.py             STRICT_TRANSCRIBE_PROMPT, METADATA_EXTRACTION_PROMPT, QUERY_DECOMPOSITION_PROMPT

  api/
    app.py               FastAPI factory + request-correlation middleware
    auth.py              HTTPBearer → Principal; RBAC guards
    routes.py            /ingest /query /jobs /documents /healthz /metrics

  rag/
    query.py             answer_query(): semantic-cache check → retrieve + generate, routes through decompose.py first
    decompose.py         query decomposition (free heuristic gate + cheap LLM call)
    access.py            can_view() — the ACL/scope predicate, enforced pre-ranking

tests/     138 tests + 8 answer-cache tests (skip unless a Redis Stack is reachable):
           unit + e2e + api + acl + observability + decomposition + reranking + BM25 + cache
eval/      run_retrieval.py (BEIR SciFact qrels benchmark), run_ragas.py, golden*.json,
           khub_compare_*.csv (head-to-head comparison raw data), run_tier1.py/run_tier2.py/
           run_rerank_ab.py/run_rerank_llm.py/run_retrieval_sota.py (benchmark scripts, see docs/BENCHMARKS.md)
docs/      ARCHITECTURE.md (ingestion + retrieval deep dive, no code), BENCHMARKS.md (retrieval-quality history)
notebooks/ pipeline_walkthrough.ipynb (narrated live run), khub_vs_pipeline_comparison.ipynb
scripts/seed.py   create a tenant + admin API token
```

## Identity / auth / RBAC (quick reference)

- `tenant_id`/`user_id`/`document_id` are 24-char hex ObjectIds (`app/ids.py`).
- Auth: `Authorization: Bearer <api_token>` → `HTTPBearer` → `Principal(tenant_id, user_id, role)`.
- Roles: `admin` (manage+ingest+delete), `member` (ingest own), `viewer` (read-only).
- Tenant isolation is default; `scope=global` documents are readable cross-tenant
  (only an admin of the one `PLATFORM_TENANT_ID` env-configured tenant may publish
  to global scope). Within a tenant, `visibility`/`acl_user_ids` gates sharing.
  Full rule: `app/rag/access.py::can_view`.

## Config flip points (`.env`, see `app/config.py`)

`METADATA_BACKEND` / `BLOB_BACKEND` / `VECTOR_BACKEND` / `QUEUE_BACKEND`,
`EMBEDDING_PROVIDER` (minilm|gateway), `RERANKER_PROVIDER` (cross_encoder|none),
`CHUNK_AUTO_SIZE` (default on — sizes chunks from the active embedder's real token
limit), `DATABASE_URL`, `VISION_MODEL`, `CHAT_MODEL`, `MAX_UPLOAD_MB`, `MAX_ATTEMPTS`,
`PLATFORM_TENANT_ID`, `REDIS_URL` (empty = answer cache off; set to enable the
per-user semantic cache — needs a Redis with the RediSearch module) plus
`CACHE_SIMILARITY_THRESHOLD` / `CACHE_TTL_SECONDS` / `CACHE_INDEX_NAME`.
`app/container.py` is the one file that reads these and picks
concrete adapters — nothing else should import a concrete adapter directly. This
deployment's `.env` currently runs `METADATA_BACKEND=postgres`,
`VECTOR_BACKEND=pgvector`, `QUEUE_BACKEND=postgres` — the production path, not the
local defaults shown above.

## Running it

```bash
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements-dev.txt
cp .env.example .env                    # set LITELLM_API_KEY
python -m scripts.seed                  # prints an admin API token
uvicorn app.api.app:app --port 8000     # terminal 1
python -m app.worker                    # terminal 2
pytest                                  # 132 tests
```

Gateway (LiteLLM) has chat + vision models only — **no embedding model, no OCR
model**. Embeddings run locally via MiniLM/ONNX; OCR is done by the vision LLM
under a strict no-hallucination prompt.

## Known gaps — read before assuming something is production-ready

This is a **correct, tested reference implementation**, with a Postgres+pgvector path
already built, verified end-to-end, and active in this deployment's `.env` today. It is
explicitly **not yet production-hardened** in the ways listed below.

## Production-readiness assessment & roadmap

Organized by priority. 🔴 = correctness/security (blocks any real deployment),
🟠 = scaling/ops (blocks running with real load or a real team), 🟡 = quality
(works, but leaves value on the table), 🟢 = later/roadmap.

### 🔴 Must-fix before any production traffic

1. **Stuck-job recovery.** A crashed worker leaves a job in `running` forever —
   no lease/heartbeat/reaper. Add a lease timestamp + a reaper that requeues jobs
   whose lease expired (standard for SQLite/Postgres-backed queues; trivial once
   on Postgres since `SELECT ... FOR UPDATE SKIP LOCKED` + a `lease_expires_at`
   column covers it).
2. **API tokens stored in plaintext** (`users.api_token`). Store `sha256(token)`,
   compare by hash; only ever show the raw token once, at creation time.
3. **No file-safety limits.** No zip/PDF-bomb guard, no page cap, no memory cap
   on rasterization. A malicious or malformed file can exhaust worker memory/CPU.
   Add: max page count, max rasterized-image dimensions, a wall-clock timeout per
   pipeline stage, and reject decompression-bomb-shaped inputs before parsing.
4. **Secrets in `.env` only.** `LITELLM_API_KEY`, `DATABASE_URL` credentials — no
   secrets manager integration. Fine for local dev; needs Key Vault/Secrets
   Manager/Vault before any shared environment.
5. **No rate limiting / no request size enforcement beyond `MAX_UPLOAD_MB` at the
   app layer** — add it at the ingress (API gateway / reverse proxy) too, not just
   in-process, so a flood of requests can't reach the app process at all.

### 🟠 Scaling & operability

6. **Task queue**: SQLite-polling queue works for one worker; the Postgres queue
   adapter exists and supports safe multi-worker claim, but there's still no
   Celery/Redis-backed queue for true horizontal worker scaling, backpressure,
   or priority queues. If ingestion volume grows, move to the Postgres queue in
   production (already built) and consider Celery/Redis only if queue depth or
   worker fan-out becomes the bottleneck — don't build it speculatively.
7. **Blob storage**: local filesystem only. Needs Azure Blob/S3 adapter before
   running on more than one machine (worker and API must share the blob store;
   local FS doesn't survive horizontal scaling or container redeploys).
8. **No migrations framework** — schema changes are raw `schema.sql` + ad-hoc
   `ALTER`. Adopt Alembic (or similar) before the schema needs to evolve under
   live data; retrofitting migrations after prod data exists is much harder.
9. **No admin API** — tenant/user/token management is script-only
   (`scripts/seed.py`). Needs authenticated admin endpoints (or an internal
   admin tool) for tenant onboarding, token rotation, and job-queue inspection
   (list/retry/dead-letter-requeue) without shelling into the database.
10. **Metrics/observability**: JSON logs + a SQLite/Postgres-backed `/metrics`
    counter table today. For real ops: emit Prometheus-format metrics (or push
    to an existing metrics pipeline), add distributed tracing (OpenTelemetry —
    the correlation-context plumbing in `observability.py` is already
    trace-shaped, this is mostly exporter wiring), and configure alerting on
    dead-letter rate, stage latency, and gateway error rate.
11. **No containerization/CI/CD/IaC** yet — needed for reproducible deploys.
    Dockerfile + a CI pipeline (lint, `pytest`, build) is the near-term bar;
    IaC (Terraform/Bicep) for the Postgres+pgvector infra follows once a target
    cloud is chosen.
12. **No load/chaos testing** — worker throughput under concurrent multi-tenant
    load, gateway rate-limit/backoff behavior under sustained traffic, and
    behavior when Postgres/pgvector is under contention are all unverified.
13. **No horizontal API scaling verification** — FastAPI/uvicorn should scale
    horizontally behind a load balancer in principle (no in-process state beyond
    the composition root), but this hasn't been load-tested with multiple API
    replicas hitting the same Postgres instance.

### 🟡 Retrieval quality — works, but leaves value on the table

14. ~~No re-ranker.~~ **Done** — a cross-encoder second pass
    (`RERANKER_PROVIDER=cross_encoder`, default) runs over the fused dense+lexical
    top-k on every query. Validated against a stronger cross-encoder and an
    LLM-as-reranker alternative before being kept as the default — see
    `docs/BENCHMARKS.md`.
15. **Fusion weighting is static** (RRF, 50/50 dense/lexical) — this is now
    **deliberate, not an oversight**: an A/B showed query-adaptive weighting never
    actually fired on real prose queries (see the fusion-simplification section of
    `docs/BENCHMARKS.md`), so it was removed in favor of letting the reranker (#14)
    be the precision arbiter. Revisit only if real identifier-heavy traffic
    (part numbers, ticket codes) is measured to need it — that case was explicitly
    not covered by the A/B that removed the adaptive path.
16. ~~Extracted metadata (`topics`/`entities`) has zero effect on ranking.~~
    **Fixed** — `topics`/`entities`/`author` are folded into the lexical
    (BM25/Postgres full-text) searchable text at upsert, so they now do affect
    ranking. Still not used for a metadata *filter*, if that's ever wanted.
17. **BM25 index rebuilt from scratch per query** (no persistence) — matches the
    dense path's own brute-force-scan profile today, so not a new bottleneck,
    but both need addressing together once corpus size grows past what fits in
    a per-query in-memory scan (this is where pgvector's HNSW index and a
    persisted/incremental BM25 index — or a real search engine — become
    necessary, not optional).
18. **MiniLM is the default embedder, and a stronger one is a validated, ready
    next step, not just a hope.** A benchmark A/B found `bge-base-en-v1.5` a clean,
    uniform win over MiniLM on every retrieval metric (`docs/BENCHMARKS.md`) — it
    has not been adopted in default config because it requires a full re-embed of
    the corpus (768-dim vs 384, ~5x slower to embed on CPU), a real migration cost,
    not because the improvement is unproven. Binary embeddings are still deferred
    behind the port, not implemented.
19. **No image preprocessing** (deskew/denoise) — OCR quality degrades on noisy
    scans; vector-drawn charts are undetected; no encrypted/corrupt-file
    handling; per-page vision calls are serial (could be parallelized with a
    bounded concurrency limit for large scanned documents).
20. **No document versioning**; tombstoned vectors are never compacted —
    storage grows monotonically with reprocessing/deletion churn.
21. **No naive-baseline validation** — the khub comparison proves the two
    systems are comparable to *each other*, not that either is *objectively*
    good (no keyword-only or no-retrieval baseline was run). Worth doing once,
    cheaply, for an absolute quality floor.

### 🟢 Roadmap / explicitly out of scope for Phase 1

Phase-2 connectors (email/Teams/Slack), PII redaction, webhooks, response
pagination on list endpoints.

## Working conventions for this repo

- Ports/adapters is the core architectural discipline — never import a concrete
  adapter (e.g. `app.adapters.postgres.*`) from `app/pipeline`, `app/rag`, or
  `app/api`; go through `app/container.py` and the `app/ports/` interfaces.
- Chunk sizing must always be measured against the real embedder's tokenizer
  (`app/pipeline/tokens.py`), never a generic proxy like `tiktoken` — this was a
  real, measured bug (see `docs/ARCHITECTURE.md`, Part 1 §4) and easy to reintroduce.
- Every extractor/loader records `extractor` + `route_reason` provenance on each
  Element — preserve this when touching loaders; it's load-bearing for
  debugging routing decisions and for the job's `route_summary`.
- Run `pytest` (132 tests, hermetic, real MiniLM model cached after first load)
  before considering any pipeline/chunking/retrieval change done.
