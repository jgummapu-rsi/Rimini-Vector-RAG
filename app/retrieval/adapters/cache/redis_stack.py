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
app.shared.container only when REDIS_URL is set -- keeping redis an optional dependency.

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
from redis.commands.search.query import Query

# redis-py renamed this module to snake_case in 6.x and dropped the camelCase
# alias. Support both, since requirements.txt deliberately pins a range rather
# than one exact version (see the redis dependency's comment there).
try:
    from redis.commands.search.index_definition import IndexDefinition, IndexType
except ImportError:  # redis-py <= 5.x
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.exceptions import ResponseError

from app.retrieval.ports.answer_cache import AnswerCache, CacheHit


# Counter name for the cross-tenant generation. Not a valid ObjectId, so it can
# never collide with a real tenant's own counter key.
_GLOBAL_GEN = "__all__"


def _normalize(question: str) -> str:
    """Collapse whitespace + lowercase so trivially-different spellings of the
    same question share a storage key (exact re-asks overwrite rather than pile
    up). Semantic matching is the KNN's job; this only dedupes the key."""
    return " ".join(question.lower().split())


def _short_hash(text: str) -> str:
    """First 16 hex chars of the SHA-1 of `text`, used as a compact key component."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


class RedisStackAnswerCache(AnswerCache):
    """Per-user semantic answer cache backed by Redis Stack's RediSearch vector KNN."""

    def __init__(self, url: str, index_name: str, dim: int,
                 threshold: float, ttl_seconds: int):
        """Connect to Redis at `url` and configure the index name, embedding
        dimension, similarity threshold, and entry TTL."""
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
        """Create the RediSearch index if it doesn't already exist."""
        try:
            self._r.ft(self._index).info()
            return
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
    def _counter(self, key: str) -> int:
        """Read an integer counter key, defaulting to 0 if unset."""
        raw = self._r.get(key)
        return int(raw) if raw is not None else 0

    def _gen(self, tenant_id: str) -> int:
        """Effective generation for a tenant = its own counter PLUS the global
        one, so bumping either invalidates the tenant's entries.

        Summing is safe because both counters are monotonically increasing and
        never reset: the effective generation therefore only ever goes up, so a
        stored entry's generation is always strictly below the current one after
        any bump. Two different (tenant, global) pairs summing to the same value
        can only happen going forward in time, never back to a superseded value.
        """
        return (self._counter(f"{self._prefix}gen:{tenant_id}")
                + self._counter(f"{self._prefix}gen:{_GLOBAL_GEN}"))

    def get(self, tenant_id: str, user_id: str, qvec: list[float],
            model: str) -> Optional[CacheHit]:
        """Return the best cached answer within the similarity threshold, or None."""
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
        """Store (or overwrite) the answer for this question under the current generation."""
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
        """Bump the tenant's generation counter (O(1)); entries built under the
        old generation stop matching the KNN filter and are reaped by their TTL."""
        self._r.incr(f"{self._prefix}gen:{tenant_id}")

    def invalidate_all(self) -> None:
        """Bump the global generation counter (O(1)), invalidating every
        tenant's cache at once since each tenant's effective generation includes
        this counter (see `_gen`)."""
        self._r.incr(f"{self._prefix}gen:{_GLOBAL_GEN}")
