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
    api_token   TEXT NOT NULL UNIQUE,             -- sha256 hash of the bearer token
    password_hash TEXT,                            -- PBKDF2 hash (set by /onboarding/register);
                                                    -- NULL for seed-script users that never log
                                                    -- in with a password
    status      TEXT NOT NULL DEFAULT 'active',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, email)
);

-- app.api.onboarding_routes.get_user_by_email looks up by email ACROSS every
-- tenant (onboarding login has no tenant_id up front) and treats "the row with
-- a password_hash" as the one true password-based account for that email. The
-- UNIQUE(tenant_id, email) above does not stop that invariant from being
-- violated -- two concurrent /onboarding/register calls with the same email
-- land in two DIFFERENT freshly-created tenants, so that constraint never
-- fires. This partial index is the actual guard: at most one password-holding
-- row per email, globally.
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_password_unique
    ON users(email) WHERE password_hash IS NOT NULL;

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
    -- Earliest time this job may be claimed. Retry backoff lives here rather
    -- than in the worker's own sleep: a sleeping worker doesn't stop a SECOND
    -- worker from claiming the same failing job immediately, so the delay has
    -- to be a property of the row. Defaults to now() = claimable at once.
    available_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Set to now()+JOB_LEASE_SECONDS when claimed, NULL otherwise. If a worker
    -- crashes mid-job the row stays `running` forever with no lease check --
    -- this is what lets the reaper reclaim it once the lease expires.
    lease_expires_at TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_jobs_claim
    ON ingestion_jobs(status, available_at, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_lease
    ON ingestion_jobs(status, lease_expires_at);

-- Per-stage trace: one row per pipeline stage per attempt. The runner logs the
-- same information, but logs aren't queryable -- this is what lets the trace UI
-- replay a document's journey after the job is long finished.
CREATE TABLE IF NOT EXISTS job_events (
    id              TEXT PRIMARY KEY,
    job_id          TEXT NOT NULL,
    document_id     TEXT NOT NULL,
    tenant_id       TEXT NOT NULL,
    stage           TEXT NOT NULL,                 -- parse | route | ... | upsert
    seq             INTEGER NOT NULL DEFAULT 0,    -- stage position; ids/timestamps are
                                                   -- too coarse to order ms-apart stages
    status          TEXT NOT NULL,                 -- ok | error
    duration_ms     DOUBLE PRECISION NOT NULL DEFAULT 0,
    attempt         INTEGER NOT NULL DEFAULT 0,
    detail          JSONB,                         -- counts, routing breakdown
    at              TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(job_id, attempt, seq);
CREATE INDEX IF NOT EXISTS idx_job_events_doc ON job_events(tenant_id, document_id);

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

-- Live-editable deployment-wide config, currently just the LiteLLM gateway
-- override (base_url/api_key). Read by app.gateway.client.LiteLLMClient's
-- config_provider closure (wired in app.container.build_container()) on every
-- gateway call, so POST /onboarding/gateway-config takes effect immediately,
-- no restart. Key/value rather than fixed columns so future overrides don't
-- need another migration.
CREATE TABLE IF NOT EXISTS system_config (
    key         TEXT PRIMARY KEY,
    value       TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
