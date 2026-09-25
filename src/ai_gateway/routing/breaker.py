"""Circuit breaker per provider.

State machine:
    CLOSED ──(failure ratio ≥ threshold over the rolling window, with ≥ min_requests)──▶ OPEN
    OPEN ──(cool-down elapsed; the next caller becomes the single probe)──▶ HALF_OPEN
    HALF_OPEN ──probe succeeds──▶ CLOSED (window reset)
    HALF_OPEN ──probe fails──▶ OPEN (cool-down restarts)

Why: when a provider is down, every request would otherwise burn its timeout budget (and add
retry load) before falling back. An open breaker skips the provider instantly.

Two implementations with the same interface:
- `InMemoryBreaker`: per process. Fine for one replica and for tests.
- `RedisBreaker`: shared by all replicas, using Lua scripts so check-and-update is atomic.
"""

from __future__ import annotations

import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from redis.asyncio import Redis

from ai_gateway.config import BreakerConfig


class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


STATE_GAUGE_VALUE = {BreakerState.CLOSED: 0, BreakerState.HALF_OPEN: 1, BreakerState.OPEN: 2}


class CircuitBreaker(Protocol):
    async def allow(self, provider: str) -> bool: ...
    async def record_success(self, provider: str) -> None: ...
    async def record_failure(self, provider: str) -> None: ...
    async def state(self, provider: str) -> BreakerState: ...


@dataclass
class _Circuit:
    state: BreakerState = BreakerState.CLOSED
    opened_at: float = 0.0
    probe_in_flight: bool = False
    outcomes: deque[tuple[float, bool]] = field(default_factory=deque)


class InMemoryBreaker:
    def __init__(self, cfg: BreakerConfig, clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg
        self.clock = clock
        self._circuits: dict[str, _Circuit] = {}

    def _get(self, provider: str) -> _Circuit:
        return self._circuits.setdefault(provider, _Circuit())

    def _trim(self, c: _Circuit, now: float) -> None:
        while c.outcomes and c.outcomes[0][0] < now - self.cfg.window_s:
            c.outcomes.popleft()

    async def allow(self, provider: str) -> bool:
        c, now = self._get(provider), self.clock()
        if c.state is BreakerState.CLOSED:
            return True
        if c.state is BreakerState.OPEN:
            if now - c.opened_at < self.cfg.cooldown_s:
                return False
            c.state, c.probe_in_flight = BreakerState.HALF_OPEN, True
            return True  # this caller is the probe
        # HALF_OPEN: exactly one probe at a time
        if c.probe_in_flight:
            return False
        c.probe_in_flight = True
        return True

    async def record_success(self, provider: str) -> None:
        c, now = self._get(provider), self.clock()
        if c.state is BreakerState.HALF_OPEN:
            self._circuits[provider] = _Circuit()  # recovered: close and forget old failures
            return
        c.outcomes.append((now, True))
        self._trim(c, now)

    async def record_failure(self, provider: str) -> None:
        c, now = self._get(provider), self.clock()
        if c.state is BreakerState.HALF_OPEN:
            c.state, c.opened_at, c.probe_in_flight = BreakerState.OPEN, now, False
            return
        c.outcomes.append((now, False))
        self._trim(c, now)
        total = len(c.outcomes)
        failures = sum(1 for _, ok in c.outcomes if not ok)
        if total >= self.cfg.min_requests and failures / total >= self.cfg.failure_ratio:
            c.state, c.opened_at = BreakerState.OPEN, now
            c.outcomes.clear()

    async def state(self, provider: str) -> BreakerState:
        c = self._get(provider)
        if c.state is BreakerState.OPEN and self.clock() - c.opened_at >= self.cfg.cooldown_s:
            return BreakerState.HALF_OPEN  # would admit a probe
        return c.state


# --- Redis implementation ------------------------------------------------------------------------
# Keys per provider: {prefix}:{p}:state (hash: state, opened_at, probe) and {prefix}:{p}:events
# (a ZSET of "<uuid>:<0|1>" members scored by timestamp).

_ALLOW = """
local st = redis.call('HGET', KEYS[1], 'state') or 'closed'
local now = tonumber(ARGV[1]); local cooldown = tonumber(ARGV[2])
if st == 'closed' then return 1 end
if st == 'open' then
  local opened = tonumber(redis.call('HGET', KEYS[1], 'opened_at') or '0')
  if now - opened < cooldown then return 0 end
  redis.call('HSET', KEYS[1], 'state', 'half_open', 'probe', '1')
  return 1
end
if redis.call('HGET', KEYS[1], 'probe') == '1' then return 0 end
redis.call('HSET', KEYS[1], 'probe', '1')
return 1
"""

_RECORD = """
local st = redis.call('HGET', KEYS[1], 'state') or 'closed'
local now = tonumber(ARGV[1]); local ok = ARGV[2]; local window = tonumber(ARGV[3])
local min_req = tonumber(ARGV[4]); local ratio = tonumber(ARGV[5]); local member = ARGV[6]
if st == 'half_open' then
  if ok == '1' then
    redis.call('DEL', KEYS[1], KEYS[2])
  else
    redis.call('HSET', KEYS[1], 'state', 'open', 'opened_at', tostring(now), 'probe', '0')
  end
  return 0
end
if st == 'open' then return 0 end
redis.call('ZADD', KEYS[2], now, member .. ':' .. ok)
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now - window)
redis.call('EXPIRE', KEYS[2], math.ceil(window) + 60)
if ok == '1' then return 0 end
local members = redis.call('ZRANGE', KEYS[2], 0, -1)
local total = #members
local failures = 0
for _, m in ipairs(members) do
  if string.sub(m, -1) == '0' then failures = failures + 1 end
end
if total >= min_req and failures / total >= ratio then
  redis.call('HSET', KEYS[1], 'state', 'open', 'opened_at', tostring(now), 'probe', '0')
  redis.call('DEL', KEYS[2])
  return 1
end
return 0
"""


class RedisBreaker:
    def __init__(
        self,
        redis: Redis,
        cfg: BreakerConfig,
        *,
        prefix: str = "gw:breaker",
        clock: Callable[[], float] = time.time,  # wall clock: shared across machines
    ) -> None:
        self.redis = redis
        self.cfg = cfg
        self.prefix = prefix
        self.clock = clock
        self._allow = redis.register_script(_ALLOW)
        self._record = redis.register_script(_RECORD)

    def _keys(self, provider: str) -> list[str]:
        return [f"{self.prefix}:{provider}:state", f"{self.prefix}:{provider}:events"]

    async def allow(self, provider: str) -> bool:
        result = await self._allow(
            keys=self._keys(provider)[:1], args=[self.clock(), self.cfg.cooldown_s]
        )
        return int(result) == 1

    async def _record_outcome(self, provider: str, ok: bool) -> None:
        await self._record(
            keys=self._keys(provider),
            args=[
                self.clock(),
                "1" if ok else "0",
                self.cfg.window_s,
                self.cfg.min_requests,
                self.cfg.failure_ratio,
                uuid.uuid4().hex,
            ],
        )

    async def record_success(self, provider: str) -> None:
        await self._record_outcome(provider, True)

    async def record_failure(self, provider: str) -> None:
        await self._record_outcome(provider, False)

    async def state(self, provider: str) -> BreakerState:
        raw = await self.redis.hgetall(self._keys(provider)[0])
        fields = {_text(k): _text(v) for k, v in raw.items()}
        st = BreakerState(fields.get("state", "closed"))
        if st is BreakerState.OPEN:
            opened = float(fields.get("opened_at", "0"))
            if self.clock() - opened >= self.cfg.cooldown_s:
                return BreakerState.HALF_OPEN
        return st


def _text(value: bytes | str) -> str:
    return value.decode() if isinstance(value, bytes) else value
