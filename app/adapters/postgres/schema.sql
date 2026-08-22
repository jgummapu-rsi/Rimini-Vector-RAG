-- Production metadata store (Postgres). Postgres translation of
-- app/adapters/sqlite/schema.sql: ObjectIds still stored as 24-char hex TEXT
-- (app-generated, no gen_random_uuid() needed); JSON columns are native JSONB
-- here (vs TEXT in the SQLite mirror, which has no JSON type).
--
-- Does NOT include vector_documents -- that table is SQLite/local-only; its
-- Postgres replacement is `vector_chunks` in app/adapters/pgvector/schema.sql.

CREATE TABLE IF NOT EXISTS tenants (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL REFERENCES tenants(id),
    email       TEXT NOT NULL,
    role        TEXT NOT NULL DEFAULT 'member',   -- admin | member | viewer
    api_token   TEXT NOT NULL UNIQUE,             -- bearer token (local auth)
    status      TEXT NOT NULL DEFAULT 'active',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, email)
);

CREATE TABLE IF NOT EXISTS documents (
    id                 TEXT PRIMARY KEY,
    tenant_id          TEXT NOT NULL REFERENCES tenants(id),
    owner_user_id      TEXT NOT NULL REFERENCES users(id),
    source_type        TEXT NOT NULL,                -- pdf | docx | image | table | xlsx
    blob_path          TEXT NOT NULL,
    content_sha256     TEXT NOT NULL,
    mime               TEXT,
    filename           TEXT,
    visibility         TEXT NOT NULL DEFAULT 'private',  -- tenant | private | shared
    acl_user_ids       JSONB NOT NULL DEFAULT '[]',
    scope              TEXT NOT NULL DEFAULT 'tenant',  -- tenant | global (cross-tenant reach)
    extracted_metadata JSONB NOT NULL DEFAULT '{}',  -- {author, date, topics[], entities[]}
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, content_sha256)            -- dedup key
);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id              TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES documents(id),
    tenant_id       TEXT NOT NULL REFERENCES tenants(id),
    stage           TEXT NOT NULL DEFAULT 'parse',
    status          TEXT NOT NULL DEFAULT 'queued', -- queued|running|done|failed|dead
    attempts        INTEGER NOT NULL DEFAULT 0,
    error           TEXT,
    route_summary   JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_jobs_claim ON ingestion_jobs(status, created_at);

CREATE TABLE IF NOT EXISTS chunks (
    id              TEXT PRIMARY KEY,              -- = document_id + zero-padded ordinal
    document_id     TEXT NOT NULL REFERENCES documents(id),
    tenant_id       TEXT NOT NULL REFERENCES tenants(id),
    ordinal         INTEGER NOT NULL,
    modality        TEXT NOT NULL,                 -- text | image | table
    extractor       TEXT,
    route_reason    TEXT,
    token_count     INTEGER,
    content_sha256  TEXT,
    text            TEXT,                          -- chunk content (also used by embed stage)
    meta            JSONB,                         -- page range, sheet, etc.
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(tenant_id, document_id);

CREATE TABLE IF NOT EXISTS connectors (
    id          TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL REFERENCES tenants(id),
    kind        TEXT NOT NULL,                     -- email | teams | slack (Phase 2)
    config      JSONB NOT NULL DEFAULT '{}',       -- secret refs
    cursor      TEXT,
    enabled     BOOLEAN NOT NULL DEFAULT false,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit_log (
    id          TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    user_id     TEXT,
    action      TEXT NOT NULL,
    target      TEXT,
    meta        JSONB,
    at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Observability: counters + accumulated timings (shared by API + worker).
CREATE TABLE IF NOT EXISTS metrics (
    name      TEXT PRIMARY KEY,
    count     BIGINT NOT NULL DEFAULT 0,
    total_ms  DOUBLE PRECISION NOT NULL DEFAULT 0
);
