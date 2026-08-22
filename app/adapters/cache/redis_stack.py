"""Redis Stack answer-cache adapter (native RediSearch vector KNN).

Each cached answer is a Redis HASH keyed `ans:{tenant}:{user}:{qhash}`:
  tenant (TAG), user (TAG), model (TAG)  -- filter a KNN result to this principal
                                            and the model that produced it
  gen    (NUMERIC)                        -- tenant generation the entry was built
                                            under; a stale entry stops matching
                                            once invalidate_tenant() bumps the gen
  vec    (VECTOR, FLOAT32, HNSW, COSINE)  -- the question embedding
  data                                    -- JSON of the full QueryResult body

Lookup is a single filtered KNN (`FT.SEARCH ... =>[KNN 1 @vec $blob]`); a hit
requires cosine similarity >= threshold (RediSearch COSINE distance = 1 - sim).
Each entry carries a TTL as a staleness backstop.

`redis` is imported at module load, so this module is imported LAZILY by
app.container only when REDIS_URL is set -- keeping redis an optional dependency.

NOTE: the index is created with the ACTIVE embedder's dim. If the embedder is
ever flipped to a different dimensionality (e.g. MiniLM 384 -> bge 768), the
index must be dropped and recreated (FT.DROPINDEX) or KNN will reject the query
vector.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

import numpy as np
import redis
from redis.commands.search.field import NumericField, TagField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from redis.exceptions import ResponseError

from app.ports.answer_cache import AnswerCache, CacheHit


def _normalize(question: str) -> str:
    """Collapse whitespace + lowercase so trivially-different spellings of the
    same question share a storage key (exact re-asks overwrite rather than pile
    up). Semantic matching is the KNN's job; this only dedupes the key."""
    return " ".join(question.lower().split())


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class RedisStackAnswerCache(AnswerCache):
    def __init__(self, url: str, index_name: str, dim: int,
                 threshold: float, ttl_seconds: int):
        # decode_responses=False: vectors are written/read as raw float32 bytes.
        # (Search RESULT fields still come back as str, so `data` decodes cleanly.)
        self._r = redis.Redis.from_url(url, decode_responses=False)
        self._index = index_name
        # Hash keys are namespaced by the index name (e.g. index "ans_idx" ->
        # keys "ans_idx:..."). This ties an index to exactly its own keys, so two
        # indexes (or a test index alongside prod) never collide. NOTE: RediSearch
        # only permits FT.CREATE on logical DB 0.
        self._prefix = f"{index_name}:"
        self._dim = dim
        self._threshold = threshold
        self._ttl = ttl_seconds

    # --- index lifecycle ---------------------------------------------------
    def init_index(self) -> None:
        try:
            self._r.ft(self._index).info()
            return                                    # already exists
        except ResponseError:
            pass
        schema = (
            TagField("tenant"),
            TagField("user"),
            TagField("model"),
            NumericField("gen"),
            VectorField("vec", "HNSW", {
                "TYPE": "FLOAT32", "DIM": self._dim, "DISTANCE_METRIC": "COSINE",
            }),
        )
        self._r.ft(self._index).create_index(
            schema,
            definition=IndexDefinition(prefix=[self._prefix], index_type=IndexType.HASH),
        )

    # --- read/write --------------------------------------------------------
    def _gen(self, tenant_id: str) -> int:
        raw = self._r.get(f"{self._prefix}gen:{tenant_id}")
        return int(raw) if raw is not None else 0

    def get(self, tenant_id: str, user_id: str, qvec: list[float],
            model: str) -> Optional[CacheHit]:
        gen = self._gen(tenant_id)
        mhash = _short_hash(model)
        q = (
            Query(
                f"(@tenant:{{{tenant_id}}} @user:{{{user_id}}} "
                f"@model:{{{mhash}}} @gen:[{gen} {gen}])"
                "=>[KNN 1 @vec $blob AS dist]"
            )
            .sort_by("dist")
            .return_fields("data", "dist")
            .dialect(2)
        )
        blob = np.asarray(qvec, dtype=np.float32).tobytes()
        try:
            res = self._r.ft(self._index).search(q, query_params={"blob": blob})
        except ResponseError:
            return None                               # index missing / transient
        if not res.docs:
            return None
        doc = res.docs[0]
        similarity = 1.0 - float(doc.dist)            # COSINE distance -> similarity
        if similarity < self._threshold:
            return None
        return CacheHit(payload=json.loads(doc.data), similarity=similarity)

    def put(self, tenant_id: str, user_id: str, question: str,
            qvec: list[float], model: str, payload: dict) -> None:
        gen = self._gen(tenant_id)
        mhash = _short_hash(model)
        key = f"{self._prefix}{tenant_id}:{user_id}:{_short_hash(_normalize(question) + '|' + model)}"
        mapping = {
            "tenant": tenant_id,
            "user": user_id,
            "model": mhash,
            "gen": gen,
            "vec": np.asarray(qvec, dtype=np.float32).tobytes(),
            "data": json.dumps(payload),
        }
        pipe = self._r.pipeline()
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, self._ttl)
        pipe.execute()

    def invalidate_tenant(self, tenant_id: str) -> None:
        # O(1): bump the generation. Entries built under the old gen no longer
        # match the KNN filter and are eventually reaped by their TTL.
        self._r.incr(f"{self._prefix}gen:{tenant_id}")
