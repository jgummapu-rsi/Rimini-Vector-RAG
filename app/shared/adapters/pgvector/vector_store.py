"""Postgres/pgvector VectorStore: ONE ROW PER CHUNK in `vector_chunks`.
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
with plain 50/50 reciprocal rank fusion (see app.shared.adapters.bm25): both
channels count equally and every candidate is eligible -- the cross-encoder
reranker in app.retrieval.rag.query is the downstream precision arbiter, so retrieval
favors recall (surface everything plausible) and lets the reranker decide order.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path

import psycopg2.errors
from psycopg2.extras import Json

from app.retrieval.rag.context import matching_roster
from app.retrieval.rag.provenance import citation_provenance
from app.shared.adapters import bm25
from app.shared.adapters.pgvector.db import transaction
from app.shared.adapters.pgvector.profile import (
    check_profile,
    ensure_profile,
    refresh_retrieval_view,
)
from app.shared.domain.embedding import EmbeddingProfile, validate_vectors
from app.shared.ports.vector_store import SearchHit, VectorPoint, VectorStore

_SCHEMA = Path(__file__).with_name("schema.sql")
log = logging.getLogger(__name__)

_TS_CONFIG = "english"


def lexical_query(text: str) -> str:
    if '"' in text or re.search(r"(?:^|\s)-\w|\bOR\b", text):
        return text
    return " OR ".join(re.findall(r"[\w]+(?:[-./][\w]+)*", text, flags=re.UNICODE))


_ACTIVE = " AND (generation_id IS NULL OR EXISTS (SELECT 1 FROM documents d WHERE d.id=vector_chunks.document_id AND d.tenant_id=vector_chunks.tenant_id AND d.active_generation_id=vector_chunks.generation_id))"

_ACL_SQL = (
    " AND (scope='global'"
    " OR payload->>'user_id'=%s"
    " OR payload->>'visibility'='tenant'"
    " OR (payload->>'visibility'='shared'"
    "     AND jsonb_exists(payload->'acl_user_ids', %s)))"
)


def acl_pushdown(access) -> tuple[str, tuple]:
    """Translate an `access` predicate into a SQL fragment + bind params, so the
    visibility filter runs BEFORE the LIMIT instead of thinning an already
    truncated page (the bug this exists to fix).

    Returns ("", ()) when there is nothing to push down -- no predicate at all,
    an admin (who sees the whole tenant anyway), or an opaque callable that
    carries no principal. In that last case the Python-side filter in `search`
    is still applied, so an untranslatable predicate degrades to the old
    behaviour rather than to no filtering at all.
    """
    if access is None:
        return "", ()
    clauses = ""
    params: tuple = ()
    document_ids = getattr(access, "document_ids", None)
    if document_ids is not None:
        clauses = " AND document_id = ANY(%s)"
        params = (sorted(document_ids),)
    if getattr(access, "role", None) == "admin":
        return clauses, params
    user_id = getattr(access, "user_id", None)
    if not user_id:
        return clauses, params
    return clauses + _ACL_SQL, (*params, user_id, user_id)


class PgVectorStore(VectorStore):
    """Postgres/pgvector `VectorStore`: one row per chunk, HNSW dense search
    fused with GIN-indexed full-text lexical search."""

    def __init__(
        self,
        dsn: str,
        dim: int,
        candidate_multiplier: int = 4,
        min_candidates: int = 50,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
        hnsw_ef_search: int = 100,
        lexical_only_cap: int = 10,
        profile: EmbeddingProfile | None = None,
        workspace: bool = False,
    ):
        """Bind to the Postgres database at `dsn`; `dim` is the embedding size."""
        self.dsn = dsn
        self.dim = dim
        self.profile = profile
        self.workspace = workspace
        self.table = "workspace_vector_chunks" if workspace else "vector_chunks"
        self.view = "workspace_retrieval_vectors" if workspace else "retrieval_vectors"
        self.candidate_multiplier = candidate_multiplier
        self.min_candidates = min_candidates
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search

        self.lexical_only_cap = lexical_only_cap

    def ensure_collection(self, dim: int) -> None:
        """Create the `vector` extension and `vector_chunks` table/indexes if
        absent, or verify an existing table already matches `dim`."""
        self.dim = dim
        with transaction(self.dsn) as cur:
            cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
            extension = cur.fetchone()
            if extension is None or tuple(map(int, extension["extversion"].split("."))) < (0, 8, 0):
                raise RuntimeError("pgvector >= 0.8.0 must be installed before startup")

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
                    dim=dim,
                    hnsw_m=self.hnsw_m,
                    hnsw_ef_construction=self.hnsw_ef_construction,
                )
                cur.execute(ddl)

            cur.execute("ALTER TABLE vector_chunks ADD COLUMN IF NOT EXISTS tsv tsvector")
            cur.execute("ALTER TABLE vector_chunks ADD COLUMN IF NOT EXISTS generation_id TEXT")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_vc_generation ON vector_chunks(generation_id)"
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_vc_tsv ON vector_chunks USING gin (tsv)")
            if self.profile is None:
                raise ValueError("An explicit embedding profile is required")
            ensure_profile(cur, self.profile)
            refresh_retrieval_view(cur)
            cur.execute(_SCHEMA.with_name("workspace_schema.sql").read_text(encoding="utf-8"))

    def upsert(self, points: list[VectorPoint]) -> None:
        """Insert or replace the given chunk points."""
        if not points:
            return
        validate_vectors([p.vector for p in points], len(points), self.dim)
        with transaction(self.dsn) as cur:
            check_profile(cur, self.profile.id, self.workspace)
            for p in points:
                cur.execute(
                    "SELECT 1 FROM deleted_documents WHERE tenant_id=%s AND document_id=%s",
                    (p.tenant_id, p.payload.get("_id", "")),
                )
                if cur.fetchone() is not None:
                    raise ValueError("Deleted document cannot receive vectors")
                if len(p.vector) != self.dim:
                    raise ValueError(f"vector dim {len(p.vector)} != collection dim {self.dim}")
                searchable = bm25.searchable_text(
                    p.payload.get("content", ""),
                    p.payload.get("topics"),
                    p.payload.get("entities"),
                    p.payload.get("author"),
                )
                cur.execute(
                    f"INSERT INTO {self.table} "
                    "(chunk_id, tenant_id, document_id, scope, deleted, embedding, "
                    "payload, tsv, updated_at) "
                    "VALUES (%s, %s, %s, %s, false, %s, %s, "
                    "to_tsvector(%s, %s), now()) "
                    "ON CONFLICT (chunk_id) DO UPDATE SET "
                    "tenant_id=excluded.tenant_id, document_id=excluded.document_id, "
                    "scope=excluded.scope, deleted=false, embedding=excluded.embedding, "
                    "payload=excluded.payload, tsv=excluded.tsv, updated_at=now() "
                    f"WHERE {self.table}.generation_id IS NULL",
                    (
                        p.chunk_id,
                        p.tenant_id,
                        p.payload.get("_id", ""),
                        p.payload.get("scope", "tenant"),
                        p.vector,
                        Json(dict(p.payload, embedding_profile_id=self.profile.id)),
                        _TS_CONFIG,
                        searchable,
                    ),
                )
                if cur.rowcount != 1:
                    raise ValueError("Published generation vectors are immutable")

    def delete_by_document(self, tenant_id: str, document_id: str) -> int:
        """Tombstone all chunks of a document; return the number removed."""
        with transaction(self.dsn) as cur:
            cur.execute(
                f"UPDATE {self.table} SET deleted=true, updated_at=now() "
                "WHERE tenant_id=%s AND document_id=%s AND deleted=false",
                (tenant_id, document_id),
            )
            removed = cur.rowcount
            if not self.workspace:
                cur.execute(
                    "UPDATE workspace_vector_chunks SET deleted=true,updated_at=now() "
                    "WHERE tenant_id=%s AND document_id=%s AND deleted=false",
                    (tenant_id, document_id),
                )
                removed += cur.rowcount
        return removed

    def count(self, tenant_id: str) -> int:
        """Number of non-deleted chunk rows for a tenant."""
        with transaction(self.dsn) as cur:
            cur.execute(
                f"SELECT COUNT(*) AS n FROM {self.view} vector_chunks WHERE tenant_id=%s AND deleted=false"
                + _ACTIVE,
                (tenant_id,),
            )
            n = cur.fetchone()["n"]
            if not self.workspace:
                cur.execute(
                    "SELECT COUNT(*) AS n FROM workspace_retrieval_vectors "
                    "WHERE tenant_id=%s AND deleted=false",
                    (tenant_id,),
                )
                n += cur.fetchone()["n"]
        return n

    def validate_sources(self, tenant_id: str, chunk_ids: list[str], access=None) -> bool:
        if not chunk_ids or len(set(chunk_ids)) != len(chunk_ids):
            return False
        acl, params = acl_pushdown(access)
        with transaction(self.dsn) as cur:
            check_profile(cur, self.profile.id, self.workspace)
            cur.execute(
                f"SELECT chunk_id,payload FROM {self.view} vector_chunks "
                "WHERE deleted=false AND (tenant_id=%s OR scope='global') AND chunk_id=ANY(%s) "
                + self._profile_filter()
                + acl,
                (tenant_id, chunk_ids, *params),
            )
            rows = cur.fetchall()
        return len(rows) == len(chunk_ids) and all(
            access is None or access(row["payload"]) for row in rows
        )

    def surrounding_chunks(self, tenant_id, chunk_ids, access=None, radius=2, limit=50):
        """Read adjacent published evidence, never cross a source section boundary.

        Join through the authoritative chunk ordinal, not the textual chunk ID.
        Both anchors and neighbors use the live permission/publication view.
        """
        if not chunk_ids:
            return []
        acl, params = acl_pushdown(access)
        with transaction(self.dsn) as cur:
            check_profile(cur, self.profile.id, self.workspace)
            cur.execute(
                "WITH visible AS MATERIALIZED ("
                f"SELECT chunk_id,payload,document_id,tenant_id,generation_id FROM {self.view} vector_chunks "
                "WHERE deleted=false AND (tenant_id=%s OR scope='global') "
                + self._profile_filter()
                + acl
                + "), anchors AS (SELECT v.*,c.ordinal FROM visible v JOIN chunks c ON c.id=v.chunk_id "
                "WHERE v.chunk_id=ANY(%s)) "
                "SELECT v.chunk_id,v.payload,c.ordinal,min(abs(c.ordinal-a.ordinal)) AS distance "
                "FROM anchors a JOIN chunks c ON c.document_id=a.document_id AND c.tenant_id=a.tenant_id "
                "AND c.ordinal BETWEEN a.ordinal-%s AND a.ordinal+%s "
                "JOIN visible v ON v.chunk_id=c.id AND v.generation_id IS NOT DISTINCT FROM a.generation_id "
                "WHERE (v.payload->'meta'->>'section_path') IS NOT DISTINCT FROM "
                "(a.payload->'meta'->>'section_path') "
                "GROUP BY v.chunk_id,v.payload,c.ordinal ORDER BY distance,c.ordinal,v.chunk_id LIMIT %s",
                (
                    tenant_id,
                    *params,
                    chunk_ids,
                    min(5, max(0, radius)),
                    min(5, max(0, radius)),
                    min(50, max(1, limit)),
                ),
            )
            rows = cur.fetchall()
        return [
            self._hit(
                row["chunk_id"],
                0.0,
                dict(row["payload"], source_ordinal=row["ordinal"]),
                None,
                None,
                False,
            )
            for row in rows
            if access is None or access(row["payload"])
        ]

    def collection_candidates(self, tenant_id, question, access=None):

        acl, params = acl_pushdown(access)
        with transaction(self.dsn) as cur:
            check_profile(cur, self.profile.id, self.workspace)
            rows = self._lexical_candidates(
                cur, tenant_id, question, 200, acl + " AND payload->>'modality'='table'", params
            )
        return [
            self._hit(
                row["chunk_id"],
                float(row["lex_score"]),
                row["payload"],
                None,
                float(row["lex_score"]),
                True,
            )
            for row in rows
            if (access is None or access(row["payload"]))
            and matching_roster(question, row["payload"].get("content", ""))
        ][:20]

    def _dense_candidates(self, cur, tenant_id, query, pool_size, acl_sql="", acl_params=()):
        """HNSW nearest-neighbour candidates, ACL-filtered before the LIMIT."""
        cur.execute(
            f"SELECT count(*) AS n FROM (SELECT 1 FROM {self.view} vector_chunks "
            "WHERE deleted=false AND (tenant_id=%s OR scope='global') "
            + _ACTIVE
            + self._profile_filter()
            + acl_sql
            + " LIMIT 1001) eligible",
            (tenant_id, *acl_params),
        )
        eligible = cur.fetchone()["n"]
        cur.execute("SET LOCAL hnsw.iterative_scan='strict_order'")
        cur.execute("SET LOCAL hnsw.max_scan_tuples=20000")
        cur.execute("SET LOCAL hnsw.scan_mem_multiplier=2")
        cur.execute(
            "SELECT set_config('hnsw.ef_search',%s,true)",
            (str(max(self.hnsw_ef_search, min(1000, pool_size * 40))),),
        )
        if eligible <= 1000:
            cur.execute(
                f"WITH eligible AS MATERIALIZED (SELECT chunk_id,payload,embedding FROM {self.view} vector_chunks "
                "WHERE deleted=false AND (tenant_id=%s OR scope='global') "
                + _ACTIVE
                + self._profile_filter()
                + acl_sql
                + ") SELECT chunk_id,payload,1-(embedding <=> %s::vector) AS dense_score FROM eligible "
                "ORDER BY embedding <=> %s::vector,chunk_id LIMIT %s",
                (tenant_id, *acl_params, query, query, pool_size),
            )
            return cur.fetchall()
        distance = f"embedding::vector({self.dim})" if self.workspace else "embedding"
        dimension_filter = f" AND vector_dims(embedding)={self.dim}" if self.workspace else ""
        cur.execute(
            f"SELECT chunk_id, payload, 1 - ({distance} <=> %s::vector) AS dense_score "
            f"FROM {self.view} vector_chunks "
            "WHERE deleted=false AND (tenant_id=%s OR scope='global') "
            + _ACTIVE
            + self._profile_filter()
            + acl_sql
            + dimension_filter
            + f" ORDER BY {distance} <=> %s::vector LIMIT %s",
            (query, tenant_id, *acl_params, query, pool_size),
        )
        return cur.fetchall()

    def _lexical_candidates(self, cur, tenant_id, query_text, pool_size, acl_sql="", acl_params=()):
        """Independent lexical candidate list via full-text search, ranked by
        ts_rank_cd. Returns [] (not an error) when the query has no lexemes."""
        cur.execute(
            "SELECT chunk_id, payload, "
            "ts_rank_cd(tsv, q) AS lex_score "
            f"FROM {self.view} vector_chunks, websearch_to_tsquery(%s, %s) AS q "
            "WHERE deleted=false AND (tenant_id=%s OR scope='global') "
            "AND tsv @@ q "
            + _ACTIVE
            + self._profile_filter()
            + acl_sql
            + " ORDER BY lex_score DESC, chunk_id LIMIT %s",
            (_TS_CONFIG, lexical_query(query_text), tenant_id, *acl_params, pool_size),
        )
        return cur.fetchall()

    def search(
        self,
        tenant_id: str,
        query: list[float],
        top_k: int = 5,
        access: Callable[[dict], bool] | None = None,
        query_text: str = "",
    ) -> list[SearchHit]:
        """Tenant-scoped nearest-neighbour search, hybrid-fused with lexical
        full-text search when `query_text` is given.

        Fetches a wider pool from each independent channel (dense HNSW +
        lexical FTS), fuses with plain 50/50 reciprocal rank fusion, returns
        top_k. No query-adaptive weighting and no gated lexical-only
        injection: the cross-encoder reranker downstream is the precision
        arbiter, so every fused candidate is eligible and both channels count
        equally. Channel depth is controlled by `candidate_multiplier` and
        `min_candidates`, before fusion narrows to top_k. The visibility rule is pushed into BOTH candidate queries
        so each channel's LIMIT counts visible rows only -- see `acl_pushdown`.
        """
        acl_sql, acl_params = acl_pushdown(access)
        validate_vectors([query], 1, self.dim)
        pool_size = (
            max(top_k, top_k * self.candidate_multiplier, self.min_candidates)
            if query_text
            else top_k
        )

        with transaction(self.dsn) as cur:
            check_profile(cur, self.profile.id, self.workspace)

            cur.execute(f"SET LOCAL hnsw.ef_search = {int(self.hnsw_ef_search)}")
            dense_rows = self._dense_candidates(
                cur, tenant_id, query, pool_size, acl_sql, acl_params
            )

            lexical_rows = []
            if query_text:
                cur.execute("SAVEPOINT lexical_search")
                try:
                    lexical_rows = self._lexical_candidates(
                        cur, tenant_id, query_text, pool_size, acl_sql, acl_params
                    )
                except psycopg2.Error as e:
                    cur.execute("ROLLBACK TO SAVEPOINT lexical_search")
                    log.warning(
                        "pgvector lexical search failed, dense-only",
                        extra={
                            "event": "pgvector_fts_failed",
                            "error": str(e)[:200],
                        },
                    )
                finally:
                    cur.execute("RELEASE SAVEPOINT lexical_search")

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

        if not query_text:
            hits = []
            for r in dense_rows[:top_k]:
                hits.append(
                    self._hit(
                        r["chunk_id"],
                        float(r["dense_score"]),
                        r["payload"],
                        float(r["dense_score"]),
                        None,
                        False,
                    )
                )
            return hits

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
            self._hit(
                cid,
                float(fused[cid]),
                payloads[cid],
                dense_score.get(cid),
                lex_score.get(cid),
                True,
            )
            for cid in ordered
        ]

    def _profile_filter(self) -> str:
        # The profile ID is a SHA-256 produced by EmbeddingProfile, never user SQL.
        return (
            f" AND payload->>'embedding_profile_id'='{self.profile.id}'" if self.workspace else ""
        )

    @staticmethod
    def _hit(chunk_id, score, payload, dense, lex, hybrid) -> SearchHit:
        """Build a SearchHit from a row, exposing raw per-channel scores."""

        source_type = payload.get("source_type")
        return SearchHit(
            chunk_id=chunk_id,
            score=score,
            payload={
                "_id": payload.get("_id"),
                "modality": payload.get("modality"),
                "content": payload.get("content"),
                "location": payload.get("location", ""),
                "section_path": (payload.get("meta") or {}).get("section_path"),
                "source_ordinal": payload.get("source_ordinal"),
                "filename": payload.get("filename"),
                "user_id": payload.get("user_id"),
                "visibility": payload.get("visibility"),
                "acl_user_ids": payload.get("acl_user_ids", []),
                "scope": payload.get("scope", "tenant"),
                "source_type": source_type,
                "generation_id": payload.get("generation_id"),
                "version": payload.get("version"),
                "provenance": citation_provenance(payload.get("meta"), source_type),
                "topics": payload.get("topics", []),
                "entities": payload.get("entities", []),
                "author": payload.get("author"),
                "dense_score": dense,
                "bm25_score": lex if hybrid else None,
            },
        )
