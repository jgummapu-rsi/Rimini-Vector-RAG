# Rimini Vector RAG

Upload documents, search their contents, and ask questions with answers linked back
to the original sources. Each workspace has its own users, document permissions,
and version history. A browser interface and REST API are included.

**Stack:** FastAPI · PostgreSQL/pgvector · Redis Stack · LiteLLM · local ONNX reranking

## Contents

- [Quick start with Docker](#quick-start-with-docker)
- [Your first document and answer](#your-first-document-and-answer)
- [Configuration](#configuration)
- [Run Python directly](#run-python-directly)
- [How it works](#how-it-works)
- [Development and tests](#development-and-tests)
- [Evaluation tools](#evaluation-tools)
- [Production and upgrades](#production-and-upgrades)
- [Troubleshooting](#troubleshooting)
- [Repository layout](#repository-layout)

## Quick start with Docker

### 1. Prerequisites

- Git and Docker Engine/Desktop with **Docker Compose v2** and Linux containers.
- An accessible **LiteLLM gateway** with an API key and embedding, chat, and vision
  model aliases. The gateway is external; this repository does not start one.
- Network access to Python/PyTorch package registries and Hugging Face for the
  initial build and model downloads. Subsequent runs reuse cached models.
- Sufficient Docker resources: API and worker each have a 4 GiB memory limit;
  PostgreSQL, Redis, and image builds need additional capacity.

On Windows, use Docker Desktop's WSL2 backend and run the shell commands below in
WSL. On macOS, use Docker Desktop. Direct Python execution requires Linux.

### 2. Get the code and configure the gateway

```sh
git clone https://github.com/jgummapu-rsi/Rimini-Vector-RAG.git
cd Rimini-Vector-RAG
cp .env.example .env
```

Edit `.env` before starting. Supply values from your gateway administrator:

```dotenv
COMPOSE_PROFILES=development
LITELLM_BASE_URL=https://your-gateway.example.com
LITELLM_API_KEY=your-gateway-api-key
CHAT_MODEL=your-chat-model-alias
VISION_MODEL=your-vision-model-alias
EMBEDDING_PROVIDER=gateway
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_REVISION=text-embedding-3-large
EMBEDDING_DIM=1536
```

The chat/vision names shipped in `.env.example` are deployment-specific defaults;
replace them with aliases your gateway actually serves. The embedding deployment
must support 1,536-dimensional output. Keep `.env` local; it is ignored by Git.

### 3. Build the image and download the Docling models

```sh
docker compose build api worker
docker compose run --rm --no-deps worker python -m scripts.download_docling
```

The one-off container provisions CPU layout, table and OCR models in the shared
model volume and writes a model manifest. No host Python installation is needed.
Allow at least 16 GiB host RAM for the full stack. See [Docling setup](docs/docling.md)
for CPU sizing, configuration and the native parser option.

### 4. Start and check readiness

```sh
docker compose up -d
docker compose ps
docker compose logs -f api worker
```

Wait for the API to become healthy. First startup downloads the reranker and
tokenizers and prepares the configured embedding model. Stop following logs with
`Ctrl+C`; the services keep running.

```sh
curl --fail http://localhost:8000/readyz
```

| Address | Purpose |
|---|---|
| <http://localhost:8000/> | Create an account or sign in |
| <http://localhost:8000/ui/trace> | Upload, inspect processing, and ask questions |
| <http://localhost:8000/docs> | Interactive API documentation |
| <http://localhost:8000/healthz> | Process liveness |
| <http://localhost:8000/readyz> | Storage/cache readiness after startup preparation |

The development profile starts `api`, `worker`, `postgres`, and `redis`. PostgreSQL
is available on `127.0.0.1:55433`, Redis on `127.0.0.1:56380`, and the API on
`127.0.0.1:8000`. Containers use internal service addresses automatically.

### Everyday commands and persistence

```sh
docker compose logs -f worker            # Follow ingestion jobs
docker compose up -d --build api worker  # Apply code/configuration changes
docker compose down                     # Stop; retain named data volumes
```

Uploads, database records, Redis data, and Hugging Face models live in named volumes.
The Compose project name is `enterprise-rag`; keep it stable to reuse those volumes.
The layout checkpoint lives in the host directory downloaded above.
**`docker compose down -v` deletes the named volumes and their application data.**

## Your first document and answer

### In the browser

1. Open the home page and register. This creates a workspace and its admin account.
2. Open the workspace, upload a document, and wait for its ingestion job to finish.
3. Ask a question whose answer is in that document.
4. Expand the citations to inspect source passages and available page highlights.
5. Add colleagues through workspace member management. Each new registration creates
   a separate workspace; members should use the accounts their workspace admin creates.

Supported inputs: **PDF, DOCX, TXT, Markdown, RTF, CSV, HTML, XLSX, XLS, PNG, JPEG,
TIFF, and WebP**. The default upload limit is 50 MiB. Native text and tables are
extracted directly; image/scanned content uses the vision gateway. Uploads are
private by default, and retrieval honors the current user's permissions.

### Through the API

Use an account's API token from onboarding, or register a development account:

```sh
curl --fail-with-body http://localhost:8000/onboarding/register \
  -H 'Content-Type: application/json' \
  -d '{"email":"demo@example.com","password":"change-this-demo-password","display_name":"Demo workspace"}'
```

Copy the returned `api_token` into your shell. Logging in again through
`POST /onboarding/login` issues a new token and invalidates the previous one.

```sh
export RAG_TOKEN='paste-api_token-here'

curl --fail-with-body http://localhost:8000/ingest \
  -H "Authorization: Bearer $RAG_TOKEN" \
  -F 'file=@notebooks/test_data/1.md'
```

The upload returns HTTP **202** with a `document_id` and `job_id`. Use that job ID:

```sh
curl --fail-with-body "http://localhost:8000/jobs/JOB_ID" \
  -H "Authorization: Bearer $RAG_TOKEN"
```

Wait for `status` to become `done`, then ask a question about the uploaded file:

```sh
curl --fail-with-body http://localhost:8000/ask \
  -H "Authorization: Bearer $RAG_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question":"What does this document describe?","top_k":5,"allow_general_answer":false}'
```

`allow_general_answer=false` restricts the response to document evidence. With the
default `true`, an unscoped question without sufficient retrieved evidence may
receive a clearly labeled general-knowledge answer. Inspect `answer_status`,
`evidence_origin`, and `citations` to distinguish the two.

| Endpoint | Purpose |
|---|---|
| `POST /ingest` | Upload a document and queue processing |
| `GET /jobs/{job_id}` | Inspect processing status |
| `GET /jobs/{job_id}/trace` | Inspect pipeline stages |
| `GET /documents` | List accessible documents |
| `GET /documents/{document_id}/versions` | Inspect version history |
| `POST /query` | Retrieve and rerank passages without generating an answer |
| `POST /answer` | Generate from previously retrieved, authorized chunk IDs |
| `POST /ask` | Retrieve and generate in one call, with answer caching |

The interactive `/docs` page documents all request fields. Use **Authorize** with
the raw API token. `document_ids` can restrict a query to selected accessible files.

## Configuration

[`.env.example`](.env.example) is the starting template;
[`app/shared/config.py`](app/shared/config.py) defines all application settings.
Host-run Python reads environment variables and `.env`. Compose forwards the
settings explicitly listed in [`docker-compose.yml`](docker-compose.yml); adding
an arbitrary variable to `.env` does not automatically pass it to containers.

| Setting | Default / purpose |
|---|---|
| `LITELLM_BASE_URL`, `LITELLM_API_KEY` | Gateway origin and credential |
| `CHAT_MODEL`, `VISION_MODEL` | Gateway aliases for answers and visual extraction |
| `EMBEDDING_PROVIDER` | `gateway`; `minilm` enables local 384-dimensional embeddings |
| `EMBEDDING_MODEL`, `EMBEDDING_DIM` | `text-embedding-3-large`, `1536` |
| `EMBEDDING_REVISION` | Recorded deployment identity; use an immutable revision if available |
| `RERANKER_PROVIDER` | `cross_encoder`; `none` disables reranking |
| `CACHE_INDEX_NAME` | Redis answer-cache namespace, specific to the embedding profile |
| `METADATA_EXTRACTION_ENABLED` | `true`; best-effort author/topic enrichment |
| `PARSING_BACKEND` | `docling`; CPU PDF/image extraction with layout, OCR and tables |
| `LAYOUT_ENABLED` | `false`; optional YOLO layout for the native parser |
| `PARSE_TIMEOUT_SECONDS` | `1800`; parser wall-time budget |
| `PARSE_MEMORY_MB` | `16384`; parser virtual address-space ceiling |
| `DATABASE_URL`, `REDIS_URL` | Host-run storage addresses; Compose supplies internal addresses |
| `DATA_DIR` | `data`; original uploads and extraction artifacts |

The 1,536-dimensional embedding default fits the current pgvector HNSW vector index
limit of 2,000 dimensions. **Changing an existing collection's embedding model or
dimension requires migration or re-ingestion**, plus an appropriate cache namespace.

Deployment-wide gateway updates require a provisioned operator. Set
`OPERATOR_USER_IDS` and `GATEWAY_ALLOWED_ORIGINS` to JSON arrays of user IDs and
permitted gateway origins. Workspace-admin status alone does not grant operator
access. `/metrics` is also operator-only.

## Run Python directly

Use **Linux or WSL2, Python 3.12**, and the same PostgreSQL/Redis services. Python
3.13 is also used by the local test environment. Run commands from the repository
root. Bounded native parsing uses Linux process/resource controls.

On Debian/Ubuntu, install native libraries and venv support if needed:

```sh
sudo apt-get update
sudo apt-get install -y python3-venv libgl1 libglib2.0-0
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Copy `.env.example` to `.env` if you have not already configured it, then set the
gateway values as described above. `requirements.txt` is the single dependency list
for the app, tests, notebooks, and evaluations; installation includes those tools.

```sh
docker compose --profile development up -d postgres redis
python -m scripts.download_docling
python -m scripts.init_database
python -m uvicorn app.api.app:app --host 127.0.0.1 --port 8000
```

In a second terminal, from the same repository and virtual environment:

```sh
. .venv/bin/activate
python -m app.ingest.worker
```

API and worker must share `.env`, storage addresses, and `DATA_DIR`. If switching
from a fully containerized run, stop its application services first with
`docker compose stop api worker` to free port 8000 and avoid competing workers.
The host `data/` directory is separate from Docker's upload volume.

## How it works

```text
Browser / API client
        |
        v
FastAPI ----> original uploads + PostgreSQL ingestion queue
        |                              |
        |                              v
        |                    Background ingestion worker
        |                    parse -> chunk -> embed -> publish
        |                              |
        v                              v
Permission-aware search <---- PostgreSQL + pgvector
        |
        v
Local cross-encoder reranking -> source context -> LiteLLM answer
        |                                               |
        +----------- Redis answer cache <---------------+
```

- **Ingestion:** parses text, tables, and visual content; creates token-bounded
  chunks; records source locations; publishes the new searchable generation.
- **Retrieval:** combines vector similarity and lexical search, checks access,
  reranks candidates locally, and expands supporting context where appropriate.
- **Answers:** use gateway chat models and validate citation references. Redis
  caches answers with permission/freshness checks.
- **Versions:** retain upload history and track changed chunks when a document is
  replaced. Existing documents can be reprocessed through
  `POST /documents/{document_id}/reprocess`, including after enabling layout detection.

## Development and tests

Activate the Python environment and run the same style checks as CI:

```sh
python -m ruff check .
python -m ruff format --check .
```

Apply formatting with `python -m ruff format .`. Imports are grouped at module tops
and enforced by Ruff. The few deferred native-runtime imports carry explicit
reasons and targeted exceptions. CI blocks on lint/format failures.

### Provision disposable test services

Tests require **dedicated PostgreSQL with pgvector 0.8.0+ and Redis Stack**. A plain
Redis image lacks the search commands used by the cache. The PostgreSQL test user
must be able to create/drop databases and install pgvector. Node.js 22+ runs the
browser checks. Recovery tests use PostgreSQL tools from the container below.

```sh
docker run -d --name rag-test-postgres \
  -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=rag-test-only -e POSTGRES_DB=rag_test \
  -p 127.0.0.1:55432:5432 pgvector/pgvector:0.8.2-pg16
docker run -d --name rag-test-redis \
  -p 127.0.0.1:56379:6379 redis/redis-stack-server:7.4.0-v8
docker exec rag-test-postgres pg_isready -U postgres -d rag_test
docker exec rag-test-redis redis-cli ping
```

Wait for `accepting connections` and `PONG`, then:

```sh
export TEST_DATABASE_URL=postgresql://postgres:rag-test-only@127.0.0.1:55432/rag_test
export TEST_REDIS_URL=redis://127.0.0.1:56379
export RESTORE_DRILL_TOOLS_CONTAINER=rag-test-postgres
python -m pytest -n 2 --dist loadfile -rs
node tests/api/inline_source.test.cjs
```

Fixtures create isolated databases, schemas, temporary upload directories, and
Redis namespaces. Tests use local embeddings and stub gateway responses; no live
gateway key is needed. Initial local model downloads still require network access.
The test services use different ports from the development application.

When finished, remove only these disposable containers:

```sh
docker rm -f rag-test-postgres rag-test-redis
```

## Evaluation tools

Stateful evaluations also require disposable storage. They read `EVAL_*` variables
explicitly rather than reusing the application's database:

```sh
export EVAL_DATABASE_URL="$TEST_DATABASE_URL"
export EVAL_REDIS_URL="$TEST_REDIS_URL"
export EVAL_LITELLM_BASE_URL='https://your-gateway.example.com'
export EVAL_LITELLM_API_KEY='your-gateway-api-key'
python -m eval.run_retrieval_postgres --pipeline --rerank --workers 4 \
  --output eval/retrieval_gateway_large_pipeline.xlsx
```

This downloads SciFact, runs ingestion and retrieval, and writes JSON/Excel reports
with per-query scores and latency. Metadata generation is disabled for this run.
Embedding overrides are `EVAL_EMBEDDING_PROVIDER`, `EVAL_EMBEDDING_MODEL`,
`EVAL_EMBEDDING_DIM`, and `EVAL_EMBEDDING_REVISION`.

| Module | Purpose |
|---|---|
| `eval.run_retrieval_postgres` | End-to-end SciFact retrieval benchmark |
| `eval.compare_subset_reranking` | Compare reranking variants |
| `eval.run_ragas` | Model-judged answer quality |
| `eval.answer_acceptance` | Reviewed answer acceptance cases |
| `eval.chunking_audit` | Chunk/source fidelity diagnostics |
| `eval.filtered_recall`, `eval.lexical_compare` | Access-filtered recall and lexical behavior |
| `eval.network_load`, `eval.multi_replica_load` | Load and multi-replica behavior |
| `eval.restore_drill`, `eval.deployment_acceptance` | Recovery and deployment checks |

Consult each module's CLI help or docstring for inputs. Golden question sets and
sample documents remain versioned; generated reports, logs, and model caches are
ignored. Local `RAG_System.md` / `rag_system.md` notes are also ignored.

## Production and upgrades

The `production` profile uses a prebuilt image and externally provisioned
PostgreSQL/pgvector and Redis Stack. It runs API/worker as UID/GID **10001**, with
read-only root filesystems, mounted credentials, TLS, and startup schema changes
disabled. Use a separate Compose project name.

| Setting | Required value |
|---|---|
| `COMPOSE_PROFILES` | `production` only |
| `RAG_RELEASE_IMAGE` | Release image pinned by digest |
| `RUNTIME_DATABASE_SECRET`, `RUNTIME_REDIS_SECRET` | Files containing restricted runtime connection URLs |
| `GATEWAY_KEY_SECRET`, `PRODUCTION_GATEWAY_ORIGIN` | Gateway credential file and origin |
| `TLS_CERTIFICATE_FILE`, `TLS_KEY_FILE`, `PRODUCTION_TLS_HOSTNAME` | TLS certificate, key, and certificate hostname |
| `PRODUCTION_CACHE_INDEX` | Release-specific Redis namespace |
| `PRODUCTION_BIND_ADDRESS`, `PRODUCTION_HTTPS_PORT` | Binding; defaults to loopback port 8443 |

1. Provision external storage and a migration identity. Initialize schema with
   `python -m scripts.init_database --schema` using that identity's environment.
2. Use `python -m scripts.grant_runtime --help` to provision restricted runtime grants.
3. Make mounted secrets and model files readable by UID 10001. Provision writable
   upload/model volumes, and download the layout checkpoint if enabled.
4. Set the production variables above, including operator IDs/allowed gateway
   origins where required, then start:

```sh
docker compose -p rag-production --profile production up -d --no-build
```

Select only one profile; the `.env.example` default is `development`. Record the
production project name because it determines volume identities.

**Upgrades:** drain workers; back up PostgreSQL and referenced upload files; verify
an isolated restore; migrate schema and embedding profiles as required; apply
runtime grants/volume ownership; start matching API and worker images; verify
readiness, upload, retrieval, and permissions. Keep a recovery copy and previous
image. Restoring an old snapshot after new uploads requires preserving those writes.

Utilities: `scripts.migrate_embeddings`, `scripts.encrypted_backup`,
`scripts.grant_runtime`, and `scripts.production_preflight` expose CLI help.

## Troubleshooting

| Symptom | Check / resolution |
|---|---|
| API startup fails or readiness stays unavailable | Run `docker compose logs api worker`. Confirm gateway aliases/key, PostgreSQL/Redis health, and model-download access. |
| `Layout checkpoint missing` | Run the checkpoint download step; confirm the read-only mount. Or explicitly disable `LAYOUT_ENABLED` and recreate API/worker. |
| Upload stays queued | Confirm `worker` is running and using the same database and upload storage as the API. Inspect `/jobs/{job_id}/trace`. |
| `unknown command FT...` from Redis | Use Redis Stack, not plain Redis. |
| Embedding profile/dimension mismatch | Restore the matching configuration or run an embedding migration/re-ingestion. Changing only `.env` cannot convert stored vectors. |
| Gateway works on the host but not in Docker | `localhost` inside a container is the container. Use a reachable hostname or `host.docker.internal`; a loopback-only tunnel may need the relay below. |
| HTTP 401 after signing in again | Replace the old API token; login rotates it. |
| No passages or an insufficient-evidence answer | Wait for ingestion, inspect extraction status, check document permissions/selection, and ask a question supported by the uploaded content. |
| `KeyError: TEST_DATABASE_URL` / storage connection failure during tests | Export the test variables and start the dedicated test services above. |
| Port already allocated | Stop the conflicting service or adjust Compose host bindings; the default API port is 8000. |

For a host-only gateway tunnel, install `socat` and run
`bash scripts/gateway-relay.sh TUNNEL_PORT RELAY_PORT` with a free relay port. Point
`LITELLM_BASE_URL` at the relay using the gateway's expected hostname and set
`GATEWAY_HOSTNAME` to that hostname so development containers route it to the host.
The development Compose definition contains a default tunnel-host mapping; override
it for your environment. Recreate API and worker after changing `.env`.

## Repository layout

```text
app/
  api/          FastAPI routes, authentication, browser pages
  ingest/       Loaders, chunking, publication, queue adapters, worker
  retrieval/    Search, reranking, grounding, result models, answer cache
  shared/       Settings, domain models, interfaces, storage/model adapters
tests/          API, ingestion, retrieval, storage, and recovery checks
scripts/        Setup, checkpoint download, migration, backup, deployment utilities
eval/           Evaluation programs and golden question sets
notebooks/      Walkthroughs and sample text documents
sample_pdfs/    Sample source documents
.env.example    Configuration template
requirements.txt
docker-compose.yml
Dockerfile
ruff.toml
```

`data/`, `models/`, generated evaluation reports, local agent files, and caches are
local artifacts excluded by [`.gitignore`](.gitignore). This README is the setup
and operations guide; the live `/docs` page is the API reference.

## Docling parsing

PDF/image ingestion uses a pinned, vendored Docling CPU pipeline. Install with
`pip install -r requirements.txt`, then provision models with
`python -m scripts.download_docling` before starting the worker. See
[Docling setup, CPU sizing and architecture](docs/docling.md).
