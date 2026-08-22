"""Local VectorStore = the `knowledgebase` collection (document-oriented).

ONE record per document; each document's chunks are embedded in a `chunks[]`
array. The record schema (sqlite `vector_documents.record`, JSON):

    {
      "_id":          <document id>,
      "tenant_id":    <ObjectId>,          # hard isolation key
      "user_id":      <ObjectId>,          # owner
      "visibility":   "private|tenant|shared",
      "acl_user_ids": [...],
      "source_type":  "pdf|docx|image|table|xlsx",
      "created_at":   <epoch>, "updated_at": <epoch>,
      "chunks": [
        { "chunk_id":  <_id + zero-padded NNN>,   # e.g. <_id>001
          "content":   "<chunk text>",
          "modality":  "text|image|table",
          "embeddings":[ ...float32... ],
          "created_at":<epoch>, "updated_at": <epoch> },
        ...
      ]
    }

Vector search indexes one vector per point, so `search()` UNWINDS chunks[] into
candidate vectors, tenant-filtered, and ranks by cosine. On a flip to Qdrant the
same unwind produces one point per chunk.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from app.adapters.shared import bm25
from app.adapters.sqlite.db import transaction
from app.ports.vector_store import SearchHit, VectorPoint, VectorStore

COLLECTION = "knowledgebase"


class LocalFileVectorStore(VectorStore):
    def __init__(self, vector_dir: Path, db_path: Path, dim: int):
        self.dir = vector_dir / COLLECTION
        self.db_path = db_path
        self.dim = dim

    @property
    def _meta_path(self) -> Path:
        return self.dir / "meta.json"

    def ensure_collection(self, dim: int) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        if self._meta_path.exists():
            existing = json.loads(self._meta_path.read_text()).get("dim")
            if existing != dim:
                raise ValueError(
                    f"knowledgebase dim mismatch: store={existing} embedder={dim}. "
                    f"Delete {self.dir} + reset vector_documents to re-init."
                )
        else:
            self._meta_path.write_text(json.dumps({"dim": dim}))

    def upsert(self, points: list[VectorPoint]) -> None:
        """Points are per-chunk; they are grouped by document `_id` into one
        record each (all chunks of a document are upserted together)."""
        if not points:
            return
        now = int(time.time())
        groups: dict[str, list[VectorPoint]] = {}
        for p in points:
            if len(p.vector) != self.dim:
                raise ValueError(f"vector dim {len(p.vector)} != collection dim {self.dim}")
            groups.setdefault(p.payload.get("_id", ""), []).append(p)

        with transaction(self.db_path) as c:
            for doc_id, pts in groups.items():
                record = self._build_document(doc_id, pts, now)
                c.execute(
                    "INSERT OR REPLACE INTO vector_documents "
                    "(_id, tenant_id, deleted, scope, record, created_at, updated_at) "
                    "VALUES (?, ?, 0, ?, ?, ?, ?)",
                    (doc_id, pts[0].tenant_id, record["scope"], json.dumps(record), now, now),
                )

    @staticmethod
    def _build_document(doc_id: str, pts: list[VectorPoint], now: int) -> dict:
        head = pts[0].payload
        chunks = [{
            "chunk_id": p.chunk_id,
            "content": p.payload.get("content", ""),
            "modality": p.payload.get("modality"),
            "location": p.payload.get("location", ""),   # provenance for citations
            "meta": p.payload.get("meta", {}),
            "embeddings": [float(x) for x in p.vector],
            "created_at": now,
            "updated_at": now,
        } for p in pts]
        return {
            "_id": doc_id,
            "tenant_id": pts[0].tenant_id,
            "user_id": head.get("user_id"),
            "visibility": head.get("visibility"),
            "acl_user_ids": head.get("acl_user_ids", []),
            "scope": head.get("scope", "tenant"),
            "source_type": head.get("source_type"),
            "filename": head.get("filename"),            # doc-level provenance
            "topics": head.get("topics", []),
            "entities": head.get("entities", []),
            "author": head.get("author"),
            "created_at": now,
            "updated_at": now,
            "chunks": chunks,
        }

    def delete_by_document(self, tenant_id: str, document_id: str) -> int:
        """Tombstone the document record; return the number of chunk-vectors removed."""
        with transaction(self.db_path) as c:
            row = c.execute(
                "SELECT record FROM vector_documents "
                "WHERE _id=? AND tenant_id=? AND deleted=0",
                (document_id, tenant_id),
            ).fetchone()
            if not row:
                return 0
            n = len(json.loads(row["record"]).get("chunks", []))
            c.execute(
                "UPDATE vector_documents SET deleted=1, updated_at=? "
                "WHERE _id=? AND tenant_id=?",
                (int(time.time()), document_id, tenant_id),
            )
            return n

    def count(self, tenant_id: str) -> int:
        """Total chunk-vectors for the tenant (summed across document records)."""
        with transaction(self.db_path) as c:
            rows = c.execute(
                "SELECT record FROM vector_documents WHERE tenant_id=? AND deleted=0",
                (tenant_id,),
            ).fetchall()
        return sum(len(json.loads(r["record"]).get("chunks", [])) for r in rows)

    def search(self, tenant_id, query, top_k=5, access=None, query_text=""):
        """Candidates are this tenant's own documents PLUS any scope=global
        document (readable cross-tenant); the `access` predicate still applies
        per-document before either ranking signal sees it.

        Dense cosine similarity is always computed. If `query_text` is given,
        BM25 lexical scoring is computed over the same candidates and the two
        rankings are fused via reciprocal rank fusion (see `shared/bm25.py`) —
        `SearchHit.score` is then the fused RRF value, not raw cosine; the raw
        `dense_score`/`bm25_score` are exposed on the payload for transparency.
        """
        with transaction(self.db_path) as c:
            rows = c.execute(
                "SELECT record FROM vector_documents "
                "WHERE (tenant_id=? OR scope='global') AND deleted=0",
                (tenant_id,),
            ).fetchall()

        vecs: list[list[float]] = []
        meta: list[tuple[dict, dict]] = []
        for r in rows:
            rec = json.loads(r["record"])
            # per-user visibility/ACL filter (doc-level) before ranking
            if access is not None and not access(rec):
                continue
            for ch in rec.get("chunks", []):
                vecs.append(ch["embeddings"])
                meta.append((ch, rec))
        if not vecs:
            return []

        mat = np.asarray(vecs, dtype=np.float32)
        q = np.asarray(query, dtype=np.float32)
        qn = np.linalg.norm(q) or 1.0
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1.0
        dense_scores = (mat @ q) / (norms * qn)

        lexical_scores = None
        final_scores = dense_scores
        if query_text:
            # The candidate pool here IS the whole tenant corpus (every non-deleted
            # chunk was loaded above), so corpus statistics are genuinely
            # corpus-wide and BM25 IDF is not distorted by a narrow pool.
            corpus_tokens = [
                bm25.tokenize(bm25.searchable_text(
                    ch.get("content", ""), rec.get("topics"), rec.get("entities"),
                    rec.get("author")))
                for ch, rec in meta
            ]
            stats = bm25.corpus_stats(corpus_tokens)
            lexical_scores = bm25.bm25_scores(bm25.tokenize(query_text), corpus_tokens, stats)
            w_dense, w_lex = bm25.classify_query_weights(query_text)
            final_scores = bm25.weighted_rrf(dense_scores, lexical_scores, w_dense, w_lex)

        order = np.argsort(-final_scores)[:top_k]
        hits: list[SearchHit] = []
        for i in order:
            ch, rec = meta[i]
            hits.append(SearchHit(
                chunk_id=ch["chunk_id"],
                score=float(final_scores[i]),
                payload={
                    "_id": rec["_id"],
                    "modality": ch.get("modality"),
                    "content": ch.get("content"),
                    "location": ch.get("location", ""),      # citation-ready
                    "filename": rec.get("filename"),
                    "user_id": rec.get("user_id"),
                    "visibility": rec.get("visibility"),
                    "acl_user_ids": rec.get("acl_user_ids", []),
                    "scope": rec.get("scope", "tenant"),
                    "source_type": rec.get("source_type"),
                    "topics": rec.get("topics", []),
                    "entities": rec.get("entities", []),
                    "author": rec.get("author"),
                    "dense_score": float(dense_scores[i]),
                    "bm25_score": float(lexical_scores[i]) if lexical_scores is not None else None,
                },
            ))
        return hits
