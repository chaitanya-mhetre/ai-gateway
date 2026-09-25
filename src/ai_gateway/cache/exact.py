"""Exact cache.

Key = sha256(canonical JSON of tenant + alias + messages + tools + sampling params).
- Canonical JSON (sorted keys, fixed separators) means semantically identical requests hash the same.
- The tenant is part of the key, so one customer can never receive another customer's cached answer.
- The *alias* (not the resolved provider) is keyed: any target of the alias is an acceptable answer.

When is caching valid? Only when the same input should give the same output:
- temperature == 0 (deterministic-ish), or the client explicitly opts in (`X-Gateway-Cache: exact`);
- not for tool-calling requests by default (tool results depend on the outside world);
- never when the client sends `X-Gateway-Cache: bypass`.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Protocol

from redis.asyncio import Redis

from ai_gateway.models import ChatRequest, ChatResponse


def cache_key(tenant: str, req: ChatRequest) -> str:
    payload = {
        "tenant": tenant,
        "alias": req.model,
        "messages": [m.model_dump(exclude_none=True) for m in req.messages],
        "tools": [t.model_dump(exclude_none=True) for t in req.tools],
        "tool_choice": req.tool_choice,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "max_tokens": req.max_tokens,
        "stop": req.stop,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def exact_cache_allowed(req: ChatRequest, mode: str | None) -> bool:
    if mode == "bypass":
        return False
    if mode == "exact":
        return True
    return req.temperature == 0 and not req.tools


class CacheStore(Protocol):
    async def get(self, key: str) -> ChatResponse | None: ...
    async def set(self, key: str, value: ChatResponse, ttl_s: int) -> None: ...


class InMemoryCacheStore:
    """Bounded LRU with per-entry expiry (single replica)."""

    def __init__(
        self, max_entries: int = 10_000, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.max_entries = max_entries
        self.clock = clock
        self._data: OrderedDict[str, tuple[float, ChatResponse]] = OrderedDict()

    async def get(self, key: str) -> ChatResponse | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires, value = item
        if expires <= self.clock():
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    async def set(self, key: str, value: ChatResponse, ttl_s: int) -> None:
        self._data[key] = (self.clock() + ttl_s, value)
        self._data.move_to_end(key)
        while len(self._data) > self.max_entries:
            self._data.popitem(last=False)


class RedisCacheStore:
    def __init__(self, redis: Redis, prefix: str = "gw:cache") -> None:
        self.redis = redis
        self.prefix = prefix

    async def get(self, key: str) -> ChatResponse | None:
        raw = await self.redis.get(f"{self.prefix}:{key}")
        return None if raw is None else ChatResponse.model_validate_json(raw)

    async def set(self, key: str, value: ChatResponse, ttl_s: int) -> None:
        await self.redis.set(f"{self.prefix}:{key}", value.model_dump_json(), ex=ttl_s)
