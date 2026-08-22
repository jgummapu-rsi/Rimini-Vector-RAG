-- pgvector VectorStore = ONE ROW PER CHUNK in `vector_chunks` (unlike the
-- local adapter's one-JSON-blob-per-document layout, which was a SQLite-only
-- trick). One row per chunk matches VectorPoint 1:1 and lets pgvector's HNSW
-- index do real index-accelerated ANN search (ORDER BY embedding <=> query
-- LIMIT n) instead of loading every candidate into Python/numpy.
--
-- This file is a TEMPLATE, not run verbatim: {dim}/{hnsw_m}/{hnsw_ef_construction}
-- are .format()-substituted by PgVectorStore.ensure_collection() before
-- executing, since a type modifier (vector(N)) and index storage options can't
-- be %s-bound parameters.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS vector_chunks (
    chunk_id     TEXT PRIMARY KEY,
    tenant_id    TEXT NOT NULL,
    document_id  TEXT NOT NULL,
    scope        TEXT NOT NULL DEFAULT 'tenant',
    deleted      BOOLEAN NOT NULL DEFAULT false,
    embedding    vector({dim}) NOT NULL,
    payload      JSONB NOT NULL DEFAULT '{{}}',
    -- Lexical index for hybrid retrieval: a full-text vector over the chunk's
    -- searchable text (content + LLM-extracted topics/entities/author, built by
    -- app.adapters.shared.bm25.searchable_text and populated on upsert). This is
    -- an INDEPENDENT candidate source at query time -- it makes a pure lexical
    -- match reachable even when dense ANN ranks it outside the vector pool, which
    -- a BM25 pass over the dense pool alone can never do.
    tsv          tsvector,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_vc_tenant ON vector_chunks(tenant_id, deleted);
CREATE INDEX IF NOT EXISTS idx_vc_doc    ON vector_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_vc_tsv     ON vector_chunks USING gin (tsv);

-- HNSW over ivfflat: no training/ANALYZE step needed, better fit for a table
-- that starts empty and grows via streaming ingestion. Cosine distance to
-- match the local adapter's cosine-similarity ranking.
CREATE INDEX IF NOT EXISTS idx_vc_hnsw ON vector_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = {hnsw_m}, ef_construction = {hnsw_ef_construction});
