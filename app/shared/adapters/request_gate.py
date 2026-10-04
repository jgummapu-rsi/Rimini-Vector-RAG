from __future__ import annotations

import logging
from uuid import uuid4

import redis

from app.shared.ports.request_gate import RequestCapacityError, RequestGate

log = logging.getLogger(__name__)

_ACQUIRE = """
local clock = redis.call('TIME')
local now = tonumber(clock[1])*1000 + math.floor(tonumber(clock[2])/1000)
for i=1,3 do
    redis.call('ZREMRANGEBYSCORE',KEYS[i],'-inf',now)
    if redis.call('ZCARD',KEYS[i]) >= tonumber(ARGV[i]) then return 0 end
end
for i=4,5 do
    redis.call('ZREMRANGEBYSCORE',KEYS[i],'-inf',now-60000)
    if redis.call('ZCARD',KEYS[i]) >= tonumber(ARGV[i]) then return 0 end
end
for i=1,3 do
    redis.call('ZADD',KEYS[i],now+120000,ARGV[6])
    redis.call('PEXPIRE',KEYS[i],120000)
end
for i=4,5 do
    redis.call('ZADD',KEYS[i],now,ARGV[6])
    redis.call('PEXPIRE',KEYS[i],60000)
end
return 1
"""
_RENEW = """
local clock = redis.call('TIME')
local now = tonumber(clock[1])*1000 + math.floor(tonumber(clock[2])/1000)
for i=1,3 do
    local expiry = redis.call('ZSCORE',KEYS[i],ARGV[1])
    if not expiry or tonumber(expiry) <= now then return 0 end
end
for i=1,3 do
    redis.call('ZADD',KEYS[i],now+120000,ARGV[1])
    redis.call('PEXPIRE',KEYS[i],120000)
end
return 1
"""
_RELEASE = """
for i=1,3 do redis.call('ZREM',KEYS[i],ARGV[1]) end
return 1
"""


class RedisRequestGate(RequestGate):
    def __init__(
        self,
        url: str,
        namespace: str,
        *,
        global_concurrency: int,
        tenant_concurrency: int,
        user_concurrency: int,
        tenant_per_minute: int,
        user_per_minute: int,
    ):
        self._redis = redis.Redis.from_url(url, socket_connect_timeout=3, socket_timeout=3)
        self._prefix = namespace + ":admission:"
        self._limits = (
            global_concurrency,
            tenant_concurrency,
            user_concurrency,
            tenant_per_minute,
            user_per_minute,
        )
        self._acquire = self._redis.register_script(_ACQUIRE)
        self._renew = self._redis.register_script(_RENEW)
        self._release = self._redis.register_script(_RELEASE)

    def _keys(self, tenant_id: str, user_id: str) -> list[str]:
        tenant = self._prefix + "tenant:" + tenant_id
        user = tenant + ":user:" + user_id
        return [
            self._prefix + "global",
            tenant + ":active",
            user + ":active",
            tenant + ":rate",
            user + ":rate",
        ]

    def close(self) -> None:
        self._redis.close()

    def acquire(self, tenant_id: str, user_id: str) -> str:
        token = uuid4().hex
        if not self._acquire(keys=self._keys(tenant_id, user_id), args=[*self._limits, token]):
            raise RequestCapacityError("Request rate or concurrency limit reached")
        return token

    def renew(self, tenant_id: str, user_id: str, token: str) -> None:
        if not self._renew(keys=self._keys(tenant_id, user_id)[:3], args=[token]):
            raise RequestCapacityError("Request admission lease expired")

    def release(self, tenant_id: str, user_id: str, token: str) -> None:
        self._release(keys=self._keys(tenant_id, user_id)[:3], args=[token])
