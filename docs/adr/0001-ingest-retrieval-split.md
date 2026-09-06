# ADR-0001: Split `app/` into `app/shared/`, `app/ingest/`, and `app/retrieval/`

Status: Accepted
Date: 2026-09-01
Deciders: Platform Engineering (via AI-assisted session, see CLAUDE.md)

## Context

The ingestion and retrieval flows documented in `docs/ARCHITECTURE.md` share a
single flat `app/` package (`app/pipeline`, `app/rag`, `app/adapters/*`,
`app/ports/*`). As the codebase grows, the boundary between "code that runs
during ingestion" and "code that runs during a query" is implicit — visible
only by reading imports, not by directory layout. The request driving this
change: make the ingest/retrieval boundary physically explicit, while keeping
the existing ports/adapters discipline intact (never import a concrete adapter
from outside the composition root).

A pre-restructuring inventory of every file in `app/` (see the three research
passes summarized in the approved plan) found that most code is cleanly
one-sided (e.g. `app/pipeline/*` has zero adapter imports and is ingest-only;
`app/rag/*` is retrieval-only), but several pieces are genuinely used by both
flows against the same underlying resource:

- `app/container.py` — the single composition root; wires every adapter for
  both flows from one `Settings` object.
- `app/config.py` — one `Settings` class with both ingest and retrieval
  flip-switches (`*_BACKEND`, `RERANKER_PROVIDER`, `CHUNK_*`, etc.).
- The embedder adapters (`minilm.py`, `gateway.py`, `onnx_embedder.py`) — the
  same instance embeds chunks at ingest time and queries at retrieval time.
- `app/gateway/client.py` — `.vision()`/`.chat()` used by ingest loaders for
  OCR/transcription, `.chat()` used by retrieval for answer generation,
  `.embed()` used by both.
- The `metadata_store` port + its sqlite/postgres adapters — auth token
  lookup is used by every request, job/document writes are ingest-only,
  document/chunk reads are retrieval-only, all against the same tables.
- The `vector_store` port + its localfs/pgvector adapters — `upsert`/
  `ensure_collection` are ingest-only, `search` is retrieval-only, against one
  schema/connection per backend.
- `app/pipeline/prompts.py` was assumed ingest-only but `app/rag/decompose.py`
  imports `QUERY_DECOMPOSITION_PROMPT` from it — it is a mixed file, not an
  ingest-only one.

## Decision

Create three top-level packages under `app/`:

- `app/shared/` — genuinely cross-cutting infrastructure: domain models,
  config, the composition root, the gateway client, the embedder adapters,
  the metadata-store/vector-store ports and their sqlite/postgres/localfs/
  pgvector adapters, plus `ids.py`/`runtime.py`/`observability.py`.
- `app/ingest/` — everything that only ever runs during ingestion: the
  pipeline (parse/route/extract/chunk/metadata/embed/upsert stages), the
  worker, the task-queue port + adapters, the blob-store port + adapter, and
  the ingest-only prompt constants.
- `app/retrieval/` — everything that only ever runs during a query: the RAG
  orchestration (`query.py`/`decompose.py`/`access.py`), the reranker port +
  adapters, the answer-cache port + adapter, and the retrieval-only prompt
  constants.

`app/api/` keeps `app.py`/`auth.py`/`onboarding_routes.py` where they are
(already shared glue), but `routes.py` is split into `ingest_routes.py` and
`retrieval_routes.py` plus a small `_common.py` for helpers used by both
(`_visible_or_404`, `_owned_or_404`, `_read_capped`).

`tests/` mirrors the same three-way split (`tests/{shared,ingest,retrieval,api}/`),
with `conftest.py` staying at `tests/` root as shared fixture infrastructure.

## Alternatives Considered

- **Duplicate the shared pieces into both `app/ingest/` and `app/retrieval/`.**
  Rejected: this would mean two copies of `Settings`, two composition roots,
  and two copies of the ONNX-loading embedder code, which will drift over
  time and directly contradicts this repo's own stated invariant that
  `app/container.py` is the *one* file that picks concrete adapters. It also
  can't actually be done cleanly for `metadata_store`/`vector_store`, since
  both flows read/write the same physical tables — a duplicated adapter pair
  would need two separate connection pools against the same database, which
  is not a real architectural option, just a cosmetic split.
- **Logical grouping only, no files physically moved.** Rejected: doesn't
  satisfy the goal of making the boundary visible in the directory layout,
  which was the explicit ask. Lower risk, but achieves less.
- **Split `app/ports/` as one block into `app/shared/ports/`.** Rejected in
  favor of placing each port next to its real consumer(s): `task_queue`/
  `blob_store` are ingest-only in every current caller, `reranker`/
  `answer_cache` are retrieval-only, so those four move into
  `app/ingest/ports/` and `app/retrieval/ports/` respectively; only the three
  ports that are genuinely mixed (`metadata_store`, `vector_store`,
  `embedder`) live in `app/shared/ports/`. This gives a more honest picture
  of the boundary than lumping all seven ports together by convention alone.

## Consequences

**Easier:** the ingest/retrieval boundary is now visible by directory alone;
onboarding a new engineer to "just the query path" or "just the ingestion
pipeline" means reading one subtree, not filtering imports across a flat
`app/`; the 4 pre-existing ports/adapters violations found during the
inventory (queue adapters reaching into metadata_store's private
`_row_to_job` helper, pgvector's `db.py` importing postgres's `db.py`) get
cleaned up as part of the same file moves.

**Harder:** `app/shared/` is now a larger, higher-fan-in package than any
single package was before — changes to `Settings`, `container.py`, or the
metadata/vector-store adapters still require understanding both flows' needs,
same as today, just concentrated in one place instead of scattered. New
contributors need to learn the three-way split (shared vs ingest vs
retrieval) as a first orientation step, which this ADR and the updated
`CLAUDE.md` directory map exist to shorten.

**Must watch:** any new port or adapter added later needs a deliberate
decision about which of the three packages it belongs in, using the same
test applied here — is it used by one flow, or does it read/write a resource
both flows touch? Defaulting new code into `app/shared/` "to be safe" would
erode the whole point of this split over time.
