-- Local metadata store (SQLite). Mirrors the production Postgres schema.
-- ObjectIds stored as 24-char hex TEXT; JSON columns as TEXT.

CREATE TABLE IF NOT EXISTS tenants (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active',
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS users (
    id          TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL REFERENCES tenants(id),
    email       TEXT NOT NULL,
    role        TEXT NOT NULL DEFAULT 'member',   -- admin | member | viewer
    api_token   TEXT NOT NULL UNIQUE,             -- bearer token (local auth)
    status      TEXT NOT NULL DEFAULT 'active',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (tenant_id, email)
);

CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    tenant_id       TEXT NOT NULL REFERENCES tenants(id),
    owner_user_id   TEXT NOT NULL REFERENCES users(id),
    source_type     TEXT NOT NULL,                -- pdf | docx | image | table | xlsx
    blob_path       TEXT NOT NULL,
    content_sha256  TEXT NOT NULL,
    mime            TEXT,
    filename        TEXT,
    visibility      TEXT NOT NULL DEFAULT 'private',  -- tenant | private | shared
    acl_user_ids    TEXT NOT NULL DEFAULT '[]',   -- JSON array
    scope           TEXT NOT NULL DEFAULT 'tenant',  -- tenant | global (cross-tenant reach)
    extracted_metadata TEXT NOT NULL DEFAULT '{}',  -- JSON: {author, date, topics[], entities[]}
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
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
    route_summary   TEXT,                          -- JSON
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
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
    meta            TEXT,                          -- JSON (page range, sheet, etc.)
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(tenant_id, document_id);

CREATE TABLE IF NOT EXISTS connectors (
    id          TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL REFERENCES tenants(id),
    kind        TEXT NOT NULL,                     -- email | teams | slack (Phase 2)
    config      TEXT NOT NULL DEFAULT '{}',        -- JSON (secret refs)
    cursor      TEXT,
    enabled     INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS audit_log (
    id          TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL,
    user_id     TEXT,
    action      TEXT NOT NULL,
    target      TEXT,
    meta        TEXT,                              -- JSON
    at          TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Local VectorStore = the `knowledgebase` collection (replaced by a Qdrant/Mongo
-- collection on flip). ONE record per document; each document's chunks (with
-- their content + embeddings) are embedded in the record's chunks[] array.
CREATE TABLE IF NOT EXISTS vector_documents (
    _id           TEXT PRIMARY KEY,                -- document id
    tenant_id     TEXT NOT NULL,
    deleted       INTEGER NOT NULL DEFAULT 0,
    scope         TEXT NOT NULL DEFAULT 'tenant',   -- tenant | global (denormalized from record for SQL filtering)
    record        TEXT NOT NULL,                   -- JSON document with embedded chunks[]
    created_at    INTEGER NOT NULL,                -- epoch seconds
    updated_at    INTEGER NOT NULL                 -- epoch seconds
);
CREATE INDEX IF NOT EXISTS idx_vd_tenant ON vector_documents(tenant_id, deleted);

-- Observability: counters + accumulated timings (shared by API + worker).
CREATE TABLE IF NOT EXISTS metrics (
    name      TEXT PRIMARY KEY,
    count     INTEGER NOT NULL DEFAULT 0,
    total_ms  REAL NOT NULL DEFAULT 0
);
