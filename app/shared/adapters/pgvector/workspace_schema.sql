-- Additional embedding spaces; the existing default index is left intact.
CREATE TABLE IF NOT EXISTS workspace_embedding_profiles (
    profile_id TEXT PRIMARY KEY,
    manifest JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS workspace_vector_chunks (
    chunk_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT 'tenant',
    deleted BOOLEAN NOT NULL DEFAULT false,
    embedding vector NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}',
    tsv tsvector,
    generation_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (vector_dims(embedding) IN (384, 1536))
);
CREATE INDEX IF NOT EXISTS idx_wvc_tenant ON workspace_vector_chunks(tenant_id,deleted);
CREATE INDEX IF NOT EXISTS idx_wvc_doc ON workspace_vector_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_wvc_generation ON workspace_vector_chunks(generation_id);
CREATE INDEX IF NOT EXISTS idx_wvc_profile ON workspace_vector_chunks((payload->>'embedding_profile_id'));
CREATE INDEX IF NOT EXISTS idx_wvc_tsv ON workspace_vector_chunks USING gin(tsv);
CREATE INDEX IF NOT EXISTS idx_wvc_384 ON workspace_vector_chunks
    USING hnsw ((embedding::vector(384)) vector_cosine_ops) WHERE vector_dims(embedding)=384;
CREATE INDEX IF NOT EXISTS idx_wvc_1536 ON workspace_vector_chunks
    USING hnsw ((embedding::vector(1536)) vector_cosine_ops) WHERE vector_dims(embedding)=1536;
CREATE OR REPLACE VIEW workspace_retrieval_vectors AS
SELECT v.chunk_id,v.tenant_id,v.document_id,v.deleted,v.embedding,v.tsv,v.generation_id,
    CASE WHEN d.id IS NULL THEN v.scope ELSE d.scope END AS scope,
    CASE WHEN d.id IS NULL THEN v.payload ELSE v.payload || jsonb_build_object(
        'user_id',d.owner_user_id,'visibility',d.visibility,'acl_user_ids',d.acl_user_ids,'scope',d.scope) END AS payload
FROM workspace_vector_chunks v LEFT JOIN documents d ON d.id=v.document_id AND d.tenant_id=v.tenant_id
WHERE (v.generation_id IS NULL OR d.active_generation_id=v.generation_id)
AND NOT EXISTS (SELECT 1 FROM deleted_documents dead WHERE dead.document_id=v.document_id AND dead.tenant_id=v.tenant_id);
