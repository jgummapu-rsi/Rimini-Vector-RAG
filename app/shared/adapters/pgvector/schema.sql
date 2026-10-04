

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS vector_chunks (
    chunk_id     TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    document_id  TEXT NOT NULL,
    scope        TEXT NOT NULL DEFAULT 'tenant',
    deleted      BOOLEAN NOT NULL DEFAULT false,
    embedding    vector({dim}) NOT NULL,
    payload      JSONB NOT NULL DEFAULT '{{}}',

    tsv          tsvector,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_vc_tenant ON vector_chunks(tenant_id, deleted);
CREATE INDEX IF NOT EXISTS idx_vc_doc    ON vector_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_vc_tsv     ON vector_chunks USING gin (tsv);

CREATE INDEX IF NOT EXISTS idx_vc_hnsw ON vector_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = {hnsw_m}, ef_construction = {hnsw_ef_construction});
