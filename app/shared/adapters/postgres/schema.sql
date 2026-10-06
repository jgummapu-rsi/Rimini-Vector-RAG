
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
    role        TEXT NOT NULL DEFAULT 'member',
    api_token   TEXT NOT NULL UNIQUE,
    password_hash TEXT,

    status      TEXT NOT NULL DEFAULT 'active',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, email)
);

CREATE TABLE IF NOT EXISTS workspace_models (
    tenant_id TEXT PRIMARY KEY REFERENCES tenants(id),
    chat_model TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_password_unique
    ON users(email) WHERE password_hash IS NOT NULL;

CREATE TABLE IF NOT EXISTS documents (
    id                 TEXT PRIMARY KEY,
    tenant_id          TEXT NOT NULL REFERENCES tenants(id),
    owner_user_id      TEXT NOT NULL REFERENCES users(id),
    source_type        TEXT NOT NULL,
    blob_path          TEXT NOT NULL,
    content_sha256     TEXT NOT NULL,
    mime               TEXT,
    filename           TEXT,
    visibility         TEXT NOT NULL DEFAULT 'private',
    acl_user_ids       JSONB NOT NULL DEFAULT '[]',
    scope              TEXT NOT NULL DEFAULT 'tenant',
    extracted_metadata JSONB NOT NULL DEFAULT '{}',

    version            INTEGER NOT NULL DEFAULT 1,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (tenant_id, owner_user_id, content_sha256)
);

CREATE INDEX IF NOT EXISTS idx_documents_owner_filename
    ON documents(tenant_id, owner_user_id, filename);

CREATE TABLE IF NOT EXISTS document_versions (
    id               TEXT PRIMARY KEY,
    document_id      TEXT NOT NULL REFERENCES documents(id),
    tenant_id        TEXT NOT NULL REFERENCES tenants(id),
    version          INTEGER NOT NULL,
    content_sha256   TEXT NOT NULL,
    blob_path        TEXT NOT NULL,
    filename         TEXT,
    mime             TEXT,
    byte_size        BIGINT NOT NULL DEFAULT 0,
    uploaded_by      TEXT NOT NULL,
    job_id           TEXT,

    chunks_added     INTEGER,
    chunks_removed   INTEGER,
    chunks_unchanged INTEGER,
    delta            JSONB NOT NULL DEFAULT '{}',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, version)
);
CREATE INDEX IF NOT EXISTS idx_docver_doc ON document_versions(tenant_id, document_id);
CREATE INDEX IF NOT EXISTS idx_docver_job ON document_versions(job_id);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id              TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES documents(id),
    tenant_id       TEXT NOT NULL REFERENCES tenants(id),
    stage           TEXT NOT NULL DEFAULT 'parse',
    status          TEXT NOT NULL DEFAULT 'queued',
    attempts        INTEGER NOT NULL DEFAULT 0,
    error           TEXT,
    route_summary   JSONB,

    blob_path       TEXT,
    content_sha256  TEXT,
    version         INTEGER,

    available_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    lease_expires_at TIMESTAMPTZ,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_jobs_claim
    ON ingestion_jobs(status, available_at, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_lease
    ON ingestion_jobs(status, lease_expires_at);

CREATE TABLE IF NOT EXISTS job_events (
    id              TEXT PRIMARY KEY,
    job_id          TEXT NOT NULL,
    document_id     TEXT NOT NULL,
    tenant_id       TEXT NOT NULL,
    stage           TEXT NOT NULL,
    seq             INTEGER NOT NULL DEFAULT 0,

    status          TEXT NOT NULL,
    duration_ms     DOUBLE PRECISION NOT NULL DEFAULT 0,
    attempt         INTEGER NOT NULL DEFAULT 0,
    detail          JSONB,
    at              TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(job_id, attempt, seq);
CREATE INDEX IF NOT EXISTS idx_job_events_doc ON job_events(tenant_id, document_id);

CREATE TABLE IF NOT EXISTS chunks (
    id              TEXT PRIMARY KEY,
    document_id     TEXT NOT NULL REFERENCES documents(id),
    tenant_id       TEXT NOT NULL REFERENCES tenants(id),
    ordinal         INTEGER NOT NULL,
    modality        TEXT NOT NULL,
    extractor       TEXT,
    route_reason    TEXT,
    token_count     INTEGER,
    content_sha256  TEXT,
    text            TEXT,
    meta            JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(tenant_id, document_id);

CREATE TABLE IF NOT EXISTS connectors (
    id          TEXT PRIMARY KEY,
    tenant_id   TEXT NOT NULL REFERENCES tenants(id),
    kind        TEXT NOT NULL,
    config      JSONB NOT NULL DEFAULT '{}',
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

CREATE TABLE IF NOT EXISTS metrics (
    name      TEXT PRIMARY KEY,
    count     BIGINT NOT NULL DEFAULT 0,
    total_ms  DOUBLE PRECISION NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS system_config (
    key         TEXT PRIMARY KEY,
    value       TEXT,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS extraction_artifacts (
    job_id TEXT NOT NULL REFERENCES ingestion_jobs(id) ON DELETE CASCADE,
    content_sha256 TEXT NOT NULL,
    artifact_key TEXT NOT NULL,
    artifact JSONB NOT NULL,
    PRIMARY KEY (job_id, content_sha256, artifact_key)
);

ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS filename TEXT;
ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS source_type TEXT;
ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS lease_token TEXT;
ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS generation_id TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS active_generation_id TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS indexed_version INTEGER;

CREATE TABLE IF NOT EXISTS ingestion_generations (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    content_sha256 TEXT NOT NULL,
    blob_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    source_type TEXT NOT NULL,
    parent_generation_id TEXT,
    chunks JSONB NOT NULL,
    extracted_metadata JSONB NOT NULL,
    published_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_generations_document ON ingestion_generations(document_id);

CREATE OR REPLACE FUNCTION reject_generation_update() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'Published generations are immutable';
END;
$$;
DROP TRIGGER IF EXISTS generation_immutable ON ingestion_generations;
CREATE TRIGGER generation_immutable BEFORE UPDATE ON ingestion_generations
FOR EACH ROW EXECUTE FUNCTION reject_generation_update();

CREATE TABLE IF NOT EXISTS corpus_epochs (
    scope_key TEXT PRIMARY KEY,
    revision BIGINT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS deleted_documents (
    document_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    deleted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE ingestion_generations ADD COLUMN IF NOT EXISTS embedding_profile_id TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_document_job
    ON ingestion_jobs(document_id) WHERE status IN ('queued','running');

ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_tenant_id_content_sha256_key;
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_owner_hash
    ON documents(tenant_id, owner_user_id, content_sha256);

CREATE OR REPLACE FUNCTION document_authorization_epoch() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF (OLD.visibility,OLD.scope,OLD.acl_user_ids,OLD.owner_user_id)
       IS DISTINCT FROM (NEW.visibility,NEW.scope,NEW.acl_user_ids,NEW.owner_user_id) THEN
        INSERT INTO corpus_epochs(scope_key,revision)
        VALUES (CASE WHEN OLD.scope='global' OR NEW.scope='global' THEN 'global' ELSE NEW.tenant_id END,1)
        ON CONFLICT(scope_key) DO UPDATE SET revision=corpus_epochs.revision+1;
    END IF;
    RETURN NEW;
END;
$$;
DROP TRIGGER IF EXISTS document_authorization_epoch ON documents;
CREATE TRIGGER document_authorization_epoch AFTER UPDATE OF visibility,scope,acl_user_ids,owner_user_id
    ON documents FOR EACH ROW EXECUTE FUNCTION document_authorization_epoch();
