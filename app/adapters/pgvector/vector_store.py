"""Postgres/pgvector VectorStore: ONE ROW PER CHUNK in `vector_chunks`, unlike
the local adapter's one-JSON-blob-per-document layout (a SQLite-only trick).
One row per chunk matches VectorPoint 1:1 and lets pgvector's HNSW index do
real index-accelerated ANN search (ORDER BY embedding <=> query LIMIT n)
instead of loading every candidate into Python.

Hybrid retrieval uses TWO independent candidate sources, then fuses them:
  - dense: HNSW ANN over the embedding column (semantic neighbourhood)
  - lexical: a GIN-indexed `tsvector` (Postgres full-text search, ts_rank_cd)

Fusing two *independent* candidate lists (not BM25 re-scoring the dense pool) is
what lets a pure lexical match surface even when the modest embedder ranks it
outside the dense pool -- the exact identifier-heavy case (part numbers, ticket
codes) where lexical beats dense. Postgres's own full-text ranking is
corpus-consistent, so unlike a Python BM25 pass over a tiny dense pool, its IDF
is not distorted by a narrow candidate set. The two rank positions are combined
with plain 50/50 reciprocal rank fusion (see app.adapters.shared.bm25): both
channels count equally and every candidate is eligible -- the cross-encoder
reranker in app.rag.query is the downstream precision arbiter, so retrieval
favors recall (surface everything plausible) and lets the reranker decide order.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import psycopg2.errors
from psycopg2.extras import Json

from app.adapters.pgvector.db import transaction
from app.adapters.shared import bm25
from app.ports.vector_store import SearchHit, VectorPoint, VectorStore

_SCHEMA = Path(__file__).with_name("schema.sql")
log = logging.getLogger("pipeline")

# Postgres text-search configuration used for both indexing and querying. Must
# match on both sides or lexemes won't line up.
_TS_CONFIG = "english"


class PgVectorStore(VectorStore):
    def __init__(
        self, dsn: str, dim: int,
        candidate_multiplier: int = 4, min_candidates: int = 50,
        hnsw_m: int = 16, hnsw_ef_construction: int = 64, hnsw_ef_search: int = 100,
        lexical_only_cap: int = 10,
    ):
        self.dsn = dsn
        self.dim = dim
        self.candidate_multiplier = candidate_multiplier
        self.min_candidates = min_candidates
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        # Max number of LEXICAL-ONLY candidates (FTS matches dense didn't surface)
        # allowed into the fused pool, and only for lexical-leaning queries -- see
        # the gate in search(). Bounds how much the lexical source can perturb a
        # ranking that dense already covers well.
        self.lexical_only_cap = lexical_only_cap

    def ensure_collection(self, dim: int) -> None:
        self.dim = dim
        try:
            with transaction(self.dsn) as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        except psycopg2.errors.InsufficientPrivilege as e:
            raise RuntimeError(
                "CREATE EXTENSION vector requires superuser privilege; run "
                "'CREATE EXTENSION vector;' once as a superuser against this "
                "database, then retry."
            ) from e

        with transaction(self.dsn) as cur:
            cur.execute("SELECT to_regclass('vector_chunks') AS reg")
            exists = cur.fetchone()["reg"] is not None
            if exists:
                cur.execute(
                    "SELECT atttypmod AS dim FROM pg_attribute "
                    "WHERE attrelid = 'vector_chunks'::regclass "
                    "AND attname = 'embedding' AND NOT attisdropped"
                )
                existing = cur.fetchone()["dim"]
                if existing != dim:
                    raise ValueError(
                        f"vector_chunks dim mismatch: store={existing} embedder={dim}. "
                        f"Drop/migrate vector_chunks to re-init."
                    )
            else:
                ddl = _SCHEMA.read_text(encoding="utf-8").format(
                    dim=dim, hnsw_m=self.hnsw_m,
                    hnsw_ef_construction=self.hnsw_ef_construction,
                )
                cur.execute(ddl)

            # Idempotent migration so a table created before the lexical index
            # existed picks up the tsvector column + GIN index in place.
            cur.execute("ALTER TABLE vector_chunks ADD COLUMN IF NOT EXISTS tsv tsvector")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_vc_tsv "
                        "ON vector_chunks USING gin (tsv)")

    def upsert(self, points: list[VectorPoint]) -> None:
        if not points:
            return
        with transaction(self.dsn) as cur:
            for p in points:
                if len(p.vector) != self.dim:
                    raise ValueError(f"vector dim {len(p.vector)} != collection dim {self.dim}")
                # Same searchable text the localfs adapter feeds BM25: chunk
                # content plus LLM-extracted topics/entities/author.
                searchable = bm25.searchable_text(
                    p.payload.get("content", ""), p.payload.get("topics"),
                    p.payload.get("entities"), p.payload.get("author"),
                )
                cur.execute(
                    "INSERT INTO vector_chunks "
                    "(chunk_id, tenant_id, document_id, scope, deleted, embedding, "
                    "payload, tsv, updated_at) "
                    "VALUES (%s, %s, %s, %s, false, %s, %s, "
                    "to_tsvector(%s, %s), now()) "
                    "ON CONFLICT (chunk_id) DO UPDATE SET "
                    "tenant_id=excluded.tenant_id, document_id=excluded.document_id, "
                    "scope=excluded.scope, deleted=false, embedding=excluded.embedding, "
                    "payload=excluded.payload, tsv=excluded.tsv, updated_at=now()",
                    (p.chunk_id, p.tenant_id, p.payload.get("_id", ""),
                     p.payload.get("scope", "tenant"), p.vector, Json(p.payload),
                     _TS_CONFIG, searchable),
                )

    def delete_by_document(self, tenant_id: str, document_id: str) -> int:
        with transaction(self.dsn) as cur:
            cur.execute(
                "UPDATE vector_chunks SET deleted=true, updated_at=now() "
                "WHERE tenant_id=%s AND document_id=%s AND deleted=false",
                (tenant_id, document_id),
            )
            removed = cur.rowcount
        return removed

    def count(self, tenant_id: str) -> int:
        with transaction(self.dsn) as cur:
            cur.execute(
                "SELECT COUNT(*) AS n FROM vector_chunks WHERE tenant_id=%s AND deleted=false",
                (tenant_id,),
            )
            n = cur.fetchone()["n"]
        return n

    def _dense_candidates(self, cur, tenant_id, query, pool_size):
        cur.execute(
            "SELECT chunk_id, payload, 1 - (embedding <=> %s::vector) AS dense_score "
            "FROM vector_chunks "
            "WHERE deleted=false AND (tenant_id=%s OR scope='global') "
            "ORDER BY embedding <=> %s::vector LIMIT %s",
            (query, tenant_id, query, pool_size),
        )
        return cur.fetchall()

    def _lexical_candidates(self, cur, tenant_id, query_text, pool_size):
        """Independent lexical candidate list via full-text search, ranked by
        ts_rank_cd. Returns [] (not an error) when the query has no lexemes."""
        cur.execute(
            "SELECT chunk_id, payload, "
            "ts_rank_cd(tsv, q) AS lex_score "
            "FROM vector_chunks, websearch_to_tsquery(%s, %s) AS q "
            "WHERE deleted=false AND (tenant_id=%s OR scope='global') "
            "AND tsv @@ q "
            "ORDER BY lex_score DESC LIMIT %s",
            (_TS_CONFIG, query_text, tenant_id, pool_size),
        )
        return cur.fetchall()

    def search(
        self,
        tenant_id: str,
        query: list[float],
        top_k: int = 5,
        access: Optional[Callable[[dict], bool]] = None,
        query_text: str = "",
    ) -> list[SearchHit]:
        # Standard hybrid retrieval: fetch top_k candidates from each independent
        # channel (dense HNSW + lexical FTS), fuse with plain 50/50 reciprocal
        # rank fusion, return top_k. No query-adaptive weighting and no gated
        # lexical-only injection: the cross-encoder reranker downstream is the
        # precision arbiter, so every fused candidate is eligible and both
        # channels count equally. `candidate_multiplier`/`min_candidates`/
        # `lexical_only_cap` are retained on the constructor for config
        # compatibility (and as a ready lever to deepen per-channel fetch) but
        # are intentionally not used here.
        with transaction(self.dsn) as cur:
            # SET can't take a bind parameter; hnsw_ef_search is internal config,
            # not user input, so an f-string is safe here.
            cur.execute(f"SET LOCAL hnsw.ef_search = {int(self.hnsw_ef_search)}")
            dense_rows = self._dense_candidates(cur, tenant_id, query, top_k)

            lexical_rows = []
            if query_text:
                try:
                    lexical_rows = self._lexical_candidates(cur, tenant_id, query_text, top_k)
                except psycopg2.Error as e:
                    # A malformed tsquery or a pre-migration table must never break
                    # the query -- degrade to dense-only ranking.
                    log.warning("pgvector lexical search failed, dense-only", extra={
                        "event": "pgvector_fts_failed", "error": str(e)[:200],
                    })

        # --- access filter (arbitrary predicate, can't be pushed into SQL) ---
        def _visible(rows):
            out = []
            for r in rows:
                payload = r["payload"]
                if access is None or access(payload):
                    out.append(r)
            return out

        dense_rows = _visible(dense_rows)
        lexical_rows = _visible(lexical_rows)
        if not dense_rows and not lexical_rows:
            return []

        # Pure-dense path (no query_text): preserve raw cosine as the score,
        # exactly like the localfs adapter, so callers see the same semantics.
        if not query_text:
            hits = []
            for r in dense_rows[:top_k]:
                hits.append(self._hit(r["chunk_id"], float(r["dense_score"]),
                                      r["payload"], float(r["dense_score"]), None, False))
            return hits

        # Every candidate from either channel is eligible; a chunk found by both
        # accumulates rank contributions from both (which is exactly why fusion
        # beats either channel alone). Fusing on RANK POSITION (not raw score)
        # sidesteps calibrating bounded cosine against unbounded ts_rank_cd.
        k = bm25.RRF_K
        payloads: dict[str, dict] = {}
        dense_score: dict[str, float] = {}
        lex_score: dict[str, float] = {}
        fused: dict[str, float] = {}

        for rank, r in enumerate(dense_rows, start=1):
            cid = r["chunk_id"]
            payloads[cid] = r["payload"]
            dense_score[cid] = float(r["dense_score"])
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)

        for rank, r in enumerate(lexical_rows, start=1):
            cid = r["chunk_id"]
            payloads.setdefault(cid, r["payload"])
            lex_score[cid] = float(r["lex_score"])
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)

        ordered = sorted(fused, key=lambda c: fused[c], reverse=True)[:top_k]
        return [
            self._hit(cid, float(fused[cid]), payloads[cid],
                      dense_score.get(cid), lex_score.get(cid), True)
            for cid in ordered
        ]

    @staticmethod
    def _hit(chunk_id, score, payload, dense, lex, hybrid) -> SearchHit:
        return SearchHit(
            chunk_id=chunk_id,
            score=score,
            payload={
                "_id": payload.get("_id"),
                "modality": payload.get("modality"),
                "content": payload.get("content"),
                "location": payload.get("location", ""),
                "filename": payload.get("filename"),
                "user_id": payload.get("user_id"),
                "visibility": payload.get("visibility"),
                "acl_user_ids": payload.get("acl_user_ids", []),
                "scope": payload.get("scope", "tenant"),
                "source_type": payload.get("source_type"),
                "topics": payload.get("topics", []),
                "entities": payload.get("entities", []),
                "author": payload.get("author"),
                "dense_score": dense,
                "bm25_score": lex if hybrid else None,
            },
        )
