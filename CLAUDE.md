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

~7,500 lines, 88 modules, 345 tests. Runs **fully local** (SQLite + local-file
vectors + local filesystem) with no external infra required — everything sits
behind a port/adapter so it "flips" to Postgres/pgvector/Azure Blob/Celery via
`.env`, no code change. A `docker compose up` path also exists
(`docker-compose.yml` + `Dockerfile`) that self-contains Postgres+pgvector and
Redis alongside the API/worker, with a browser-based onboarding flow (signup +
live LiteLLM gateway config, no `.env` editing) — see "Running it" below.
Postgres + pgvector adapters are **already built, exercised end-to-end, and are
the backends actually active in this deployment's `.env` today**; Azure Blob and
Celery are not built yet (still local-only).

Key design principle: **structure first, AI only when structure runs out.**
Real tables/text (DOCX, CSV, typed PDF text, Markdown) are parsed with plain
libraries, free. AI (vision LLM) is only invoked for genuinely unreadable
content — scanned pages, photos, embedded figures — so ingestion cost tracks
the amount of unreadable content, not total page count.

## Directory map

`app/` is physically split into three top-level packages — `app/shared/`
(infrastructure both flows depend on), `app/ingest/` (ingest-only), and
`app/retrieval/` (retrieval-only) — plus `app/api/` (the transport layer that
composes both). See `docs/adr/0001-ingest-retrieval-split.md` for the reasoning
and the specific cases (container, config, embedders, gateway client,
metadata_store, vector_store) that are shared by necessity rather than
duplicated.

```
app/
  shared/                        infrastructure BOTH flows depend on
    config.py                    Settings (pydantic-settings) — *_BACKEND flip switches live here
    container.py                 composition root: build_container() wires adapters from config
    ids.py                       BSON-compatible ObjectId generator (no pymongo)
    runtime.py                   Windows MSVC DLL bootstrap for onnxruntime
    observability.py             structured JSON logging + correlation contextvars
    security.py                  hash_token() — sha256 API-token hashing, at rest and at lookup
    rate_limit.py                RateLimiter — in-process fixed-window abuse guard
                                 (onboarding register/login; see roadmap item #5)
    domain/models.py             enums + dataclasses: Role, Modality, Job, ChunkRecord, Principal…
    gateway/client.py            LiteLLM client: embed / vision / chat
    ports/                       metadata_store, vector_store, embedder — the 3 ports genuinely
                                 used by both ingest writes/queries and retrieval reads
    adapters/
      sqlite/                    metadata_store + metrics + db (LOCAL default)
      postgres/                  metadata_store + metrics + db (PRODUCTION, built & tested)
      localfs/vector_store.py    local file-based "knowledgebase" vector store (LOCAL default)
      pgvector/                  vector_store, native vector column + HNSW index (PRODUCTION, built & tested)
      bm25.py                    RRF fusion (dense + BM25), used by both vector store adapters
      embedders/                 minilm (local ONNX, default) | onnx_embedder (configurable
                                 HF ONNX model, e.g. bge) | gateway (LiteLLM) — same instance
                                 embeds chunks at ingest time and queries at retrieval time

  ingest/                        code that only ever runs during ingestion
    worker.py                    `python -m app.ingest.worker` — polls queue, runs pipeline per job
    pipeline/
      runner.py                  stage machine: parse→route→extract→chunk→metadata→embed→binarize→upsert
      loaders/                   pdf, docx_loader, markdown, text, table (csv/html), excel, image
      blocks.py                  shared prose/table/code block splitter + heading-stack tracker (section_path)
      chunker.py                 structure-aware, token-budget-aware chunking (target 180/max 220/overlap 20)
      tables.py                  deterministic table → markdown
      tokens.py                  count_tokens() against the REAL MiniLM WordPiece tokenizer; EMBED_MAX_TOKENS=256
      elements.py                Element dataclass
      metadata_extract.py        best-effort LLM author/date/topics/entities extraction
      provenance.py              per-element extractor/route_reason bookkeeping
      prompts.py                 STRICT_TRANSCRIBE_PROMPT, METADATA_EXTRACTION_PROMPT
    ports/                       task_queue, blob_store (ingest-only in every current caller)
    adapters/
      queue/                     sqlite.py + postgres.py — claim/complete/retry/dead-letter logic
      localfs/blob_store.py      content-addressed raw file bytes

  retrieval/                     code that only ever runs during a query
    rag/
      query.py                   answer_query(): semantic-cache check → retrieve + generate, routes through decompose.py first
      decompose.py                query decomposition (free heuristic gate + cheap LLM call)
      access.py                   can_view() — the ACL/scope predicate, enforced pre-ranking
      prompts.py                  QUERY_DECOMPOSITION_PROMPT
    ports/                       reranker, answer_cache (retrieval-only)
    adapters/
      rerankers/                  cross_encoder (local ONNX, default) | llm_reranker (gateway, eval-only)
      cache/redis_stack.py        semantic answer cache — RediSearch vector KNN (OPTIONAL, off unless REDIS_URL set)

  api/
    app.py                       FastAPI factory + request-correlation middleware
    auth.py                      HTTPBearer → Principal; RBAC guards
    ingest_routes.py             /ingest /jobs /jobs/{id}/trace /documents
                                 /documents/{id}/chunks /healthz /metrics
    retrieval_routes.py          /query /answer
    _common.py                   shared route helpers: _read_capped, _visible_or_404, _owned_or_404
    onboarding_routes.py         self-serve /onboarding/register|login (provisions a
                                 tenant+admin transparently) + /onboarding/gateway-config
                                 (admin-only, live LiteLLM base_url/api_key override,
                                 persisted to the `system_config` table — wins over
                                 env, no restart needed) + /onboarding/status
    static/onboarding/           onboarding UI (signup/login → optional gateway
                                 config gate → API token), mounted at `/`
    static/trace/                Document Trace UI — an ingestion-trace view AND an
                                 "Ask" view that calls /query and renders the answer's
                                 Sources panel (clickable citations back to source
                                 passages); single self-contained page, no build step,
                                 mounted at `/ui` (trace at `/ui/trace`)
```

```
tests/     mirrors app/: tests/{ingest,retrieval,shared,api}/, conftest.py at tests/ root.
           ~331 tests + 14 answer-cache tests (skip unless a Redis Stack is reachable):
           unit + e2e + api + acl + observability + decomposition + reranking +
           BM25 + cache + onboarding (incl. rate limiting) + gateway client
           (retry/backoff + DB-override config) + rate limiter unit tests
eval/      run_retrieval.py (BEIR SciFact qrels benchmark), run_ragas.py, golden*.json,
           khub_compare_*.csv (head-to-head comparison raw data), run_tier1.py/run_tier2.py/
           run_rerank_ab.py/run_rerank_llm.py/run_retrieval_sota.py (benchmark scripts, see docs/BENCHMARKS.md)
docs/      ARCHITECTURE.md (ingestion + retrieval deep dive, no code), BENCHMARKS.md (retrieval-quality history),
           adr/0001-ingest-retrieval-split.md (the app/shared+ingest+retrieval split decision)
notebooks/ pipeline_walkthrough.ipynb (narrated live run), khub_vs_pipeline_comparison.ipynb
scripts/seed.py   create a tenant + admin API token from the command line —
                  mainly for `--platform` (the global-knowledge-base tenant)
                  now that self-serve /onboarding/register is the primary path
docker-compose.yml / Dockerfile   self-contained stack (Postgres+pgvector,
                  Redis, api, worker) — no `.env` required, see "Running it"
```

## Identity / auth / RBAC (quick reference)

- `tenant_id`/`user_id`/`document_id` are 24-char hex ObjectIds (`app/shared/ids.py`).
- Auth: `Authorization: Bearer <api_token>` → `HTTPBearer` → `Principal(tenant_id, user_id, role)`.
- Roles: `admin` (manage+ingest+delete), `member` (ingest own), `viewer` (read-only).
- Tenant isolation is default; `scope=global` documents are readable cross-tenant
  (only an admin of the one `PLATFORM_TENANT_ID` env-configured tenant may publish
  to global scope). Within a tenant, `visibility`/`acl_user_ids` gates sharing.
  Full rule: `app/retrieval/rag/access.py::can_view`.

## Config flip points (`.env`, see `app/shared/config.py`)

`METADATA_BACKEND` / `BLOB_BACKEND` / `VECTOR_BACKEND` / `QUEUE_BACKEND`,
`EMBEDDING_PROVIDER` (minilm|gateway), `RERANKER_PROVIDER` (cross_encoder|none),
`RERANK_MIN_SCORE` (default `-3.0` — cross-encoder hits scoring below this are
dropped rather than padded into `top_k`; `None` disables the floor — see
`app/retrieval/rag/query.py::_rerank`), `CHUNK_AUTO_SIZE` (default on — sizes chunks from
the active embedder's real token limit), `DATABASE_URL`, `VISION_MODEL`,
`CHAT_MODEL`, `MAX_UPLOAD_MB`, `MAX_ATTEMPTS`, `PLATFORM_TENANT_ID`, `REDIS_URL`
(empty = answer cache off; set to enable the per-user semantic cache — needs a
Redis with the RediSearch module) plus `CACHE_SIMILARITY_THRESHOLD` /
`CACHE_TTL_SECONDS` / `CACHE_INDEX_NAME`, and the onboarding abuse guards
`LOGIN_RATE_LIMIT_PER_EMAIL` (default 10) / `LOGIN_RATE_LIMIT_PER_IP` (default
30) / `LOGIN_RATE_LIMIT_WINDOW_SECONDS` (default 900) /
`REGISTER_RATE_LIMIT_PER_IP` (default 10) / `REGISTER_RATE_LIMIT_WINDOW_SECONDS`
(default 3600) — see roadmap item #5.
`app/shared/container.py` is the one file that reads these and picks
concrete adapters — nothing else should import a concrete adapter directly. This
deployment's `.env` currently runs `METADATA_BACKEND=postgres`,
`VECTOR_BACKEND=pgvector`, `QUEUE_BACKEND=postgres` — the production path, not the
local defaults shown above.

`LITELLM_BASE_URL`/`LITELLM_API_KEY` are the one exception to "env is the
source of truth": `POST /onboarding/gateway-config` (admin-only) writes an
override into the `system_config` table, and `LiteLLMClient._resolve_config`
prefers that DB value over the env-sourced constructor default on **every**
call, live, no restart — this is what lets the onboarding UI's gateway screen
take effect immediately. `docker-compose.yml` ships both as blank on purpose,
by design (see the comment there); the onboarding UI's gateway gate is the
intended way to set them in that path.

## Running it

**Docker Compose (self-contained, no `.env` needed):**

```bash
docker compose up
```

Brings up Postgres+pgvector, Redis, the API, and the worker. Open
`http://localhost:8000/` — the onboarding page walks you through creating an
account (provisions a tenant+admin transparently) and, live, configuring the
LiteLLM gateway (skippable, settable later). Then `http://localhost:8000/ui/trace`
to drag in a document and watch it ingest, and its "Ask" tab to query it.
`docker compose exec api python -m scripts.seed --platform` remains available
for the platform/global-knowledge-base tenant. Note: the runtime image
deliberately does **not** ship `pytest`/`tests/` (dev-only, see the
`Dockerfile` comment) — run the test suite from a local venv (below), not
inside the container.

**Local venv (no Docker):**

```bash
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements-dev.txt
cp .env.example .env                    # set LITELLM_API_KEY
python -m scripts.seed                  # prints an admin API token
uvicorn app.api.app:app --port 8000     # terminal 1
python -m app.ingest.worker              # terminal 2
pytest                                  # 345 tests
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

1. ~~Stuck-job recovery.~~ **Done** — `ingestion_jobs.lease_expires_at` is set by
   `claim_next()` (both SQLite and Postgres queue adapters,
   `JOB_LEASE_SECONDS` default 300s), and `TaskQueue.reap_expired()` reclaims
   (or dead-letters, past `MAX_ATTEMPTS`) any job stuck `running` past its
   lease. `app/ingest/worker.py`'s poll loop calls it once per iteration,
   isolated in its own try/except so a reaper failure can't take the worker
   down. See `app/ingest/adapters/queue/{sqlite,postgres}.py`.
2. ~~API tokens stored in plaintext~~ (`users.api_token`). **Done** — tokens are
   sha256-hashed before storage and compared by hash
   (`app/shared/security.py::hash_token`, both metadata_store adapters).
   `POST /onboarding/login` can no longer recover the original token (it's a
   one-way hash), so it mints and returns a fresh one instead, invalidating
   the previous — see `app/api/onboarding_routes.py::login` and
   `MetadataStore.rotate_api_token`.
3. ~~No file-safety limits.~~ **Done** — `MAX_PDF_PAGES` (default 500),
   `MAX_IMAGE_PIXELS` (default 40M, a decompression-bomb guard via PIL header
   inspection before decode), and `MAX_TABLE_ROWS` (default 200k) are enforced
   across every loader (`app/ingest/pipeline/safety.py`); `_stage_parse` in
   `app/ingest/pipeline/runner.py` wraps the whole parse+extract call in a
   soft, thread-based `PARSE_TIMEOUT_SECONDS` (default 120s) timeout. The
   timeout is deliberately scoped to the parse stage only, not all 8 pipeline
   stages — later stages (embed/upsert) touch the DB/network, and abandoning
   an in-flight write on timeout risks corrupting shared state, whereas
   parsing untrusted bytes is pure CPU with nothing to corrupt if abandoned.
4. **Secrets in `.env` only.** `LITELLM_API_KEY`, `DATABASE_URL` credentials — no
   secrets manager integration. Fine for local dev; needs Key Vault/Secrets
   Manager/Vault before any shared environment. **Not yet started** — needs a
   target platform decision (Azure Key Vault / AWS Secrets Manager / Vault)
   before any code changes; the choice isn't this codebase's to make alone.
5. **No rate limiting at the ingress; app-layer rate limiting exists only on
   `/onboarding/register` and `/onboarding/login`.** `app/shared/rate_limit.py`
   (`RateLimiter`, in-process, thread-safe, fixed-window) guards those two
   endpoints per-email and per-IP (`app/api/onboarding_routes.py`) — the two
   unauthenticated endpoints most exposed to credential-stuffing/mass-signup
   abuse. Every other route still has no throttle beyond `MAX_UPLOAD_MB`.
   Two gaps remain, both **not yet started**: (a) nothing at the ingress
   (API gateway/reverse proxy) to stop a flood from reaching the app process
   at all, and (b) the in-process limiter is per-API-process, so it doesn't
   coordinate across horizontally-scaled replicas (needs a shared store, e.g.
   Redis, once API scaling — item #13 — is real). Needs a decision on ingress
   placement before that half is implemented.

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
11. **CI/CD partially done; no IaC.** ~~No containerization/CI~~ — `Dockerfile` +
    `.github/workflows/ci.yml` now cover the near-term bar. CI v1 runs four jobs
    on every PR and push to `main`: `test` (blocking — the full suite on Python
    3.12, with a `redis/redis-stack-server` service so the 15 answer-cache tests
    actually run rather than silently skipping, and an `actions/cache` of the two
    HuggingFace ONNX models), `secrets` (blocking — gitleaks over working tree
    *and* full history), `build` (blocking — image builds, nothing pushed), and
    `lint` (**advisory only** — `ruff check` + `ruff format --check`, config in
    `ruff.toml`). Deliberately absent, and why: no Postgres/pgvector service (no
    test connects to either — see item #23), and no LiteLLM credentials (the
    suite stubs that boundary, so a real key would buy zero coverage). Still
    open: flip `lint` to blocking once its backlog clears, no image push to a
    registry, and IaC (Terraform/Bicep) for the Postgres+pgvector infra once a
    target cloud is chosen.
12. **No load/chaos testing** — worker throughput under concurrent multi-tenant
    load, gateway rate-limit/backoff behavior under sustained traffic, and
    behavior when Postgres/pgvector is under contention are all unverified.
13. **No horizontal API scaling verification** — FastAPI/uvicorn should scale
    horizontally behind a load balancer in principle (no in-process state beyond
    the composition root), but this hasn't been load-tested with multiple API
    replicas hitting the same Postgres instance.
23. **No Postgres/pgvector integration tests** — surfaced while building CI v1.
    Both adapters are the backends this deployment actually runs, yet *no test
    opens a connection to either*: `tests/retrieval/test_access.py` exercises
    `acl_pushdown` as a pure function, asserting on generated SQL strings only.
    So a `VECTOR_BACKEND`/`METADATA_BACKEND` flip is verified by hand, never by
    the suite. This is why CI deliberately ships no Postgres service container —
    the gap is the tests, not the infrastructure. Write connection-level tests
    first, then add a `pgvector/pgvector:pg16` service (the image
    `docker-compose.yml` already uses) to the `test` job. Note item #15's
    standing requirement that a backend flip must not change ranking for the
    same query — that is exactly the invariant these tests should pin.
    (Numbered 23, out of section order, on purpose: this list is append-only so
    that existing cross-references like "item #5"/"#14"/"#22" stay valid.)

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
    not covered by the A/B that removed the adaptive path. Applies identically on
    both vector-store backends (`pgvector` and `localfs`) — a `VECTOR_BACKEND` flip
    must not change ranking for the same query; the adaptive-weighting helper
    (`app/shared/adapters/bm25.py::classify_query_weights`) still exists, tested,
    but is not called by either adapter's default search path.
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
22. ~~No relevance floor after reranking — `/query` always padded results out to
    `top_k`, however irrelevant, whenever a tenant's truly-relevant chunk count
    was smaller than `top_k`.~~ **Fixed** — `RERANK_MIN_SCORE` (default `-3.0`)
    drops hits below the floor in `app/retrieval/rag/query.py::_rerank`; calibrated
    against a real measured case where a genuine match scored 3.4 and
    irrelevant hits in the same pool clustered at -11.3..-11.45.

### 🟢 Roadmap / explicitly out of scope for Phase 1

Phase-2 connectors (email/Teams/Slack), PII redaction, webhooks, response
pagination on list endpoints.

## Working conventions for this repo

- Ports/adapters is the core architectural discipline — never import a concrete
  adapter (e.g. `app.shared.adapters.postgres.*`) from `app/ingest/pipeline`,
  `app/retrieval/rag`, or `app/api`; go through `app/shared/container.py` and
  the `ports/` interfaces (split across `app/shared/ports/`, `app/ingest/ports/`,
  `app/retrieval/ports/` by which flow(s) actually consume them — see
  `docs/adr/0001-ingest-retrieval-split.md`).
- Chunk sizing must always be measured against the real embedder's tokenizer
  (`app/ingest/pipeline/tokens.py`), never a generic proxy like `tiktoken` — this
  was a real, measured bug (see `docs/ARCHITECTURE.md`, Part 1 §4) and easy to reintroduce.
- Every extractor/loader records `extractor` + `route_reason` provenance on each
  Element — preserve this when touching loaders; it's load-bearing for
  debugging routing decisions and for the job's `route_summary`.
- Run `pytest` (345 tests, hermetic, real MiniLM model cached after first load)
  before considering any pipeline/chunking/retrieval change done.
