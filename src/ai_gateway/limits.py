"""Rate limits (RPM, TPM) and monthly budgets.

Token bucket: each key has a bucket of `limit` tokens that refills continuously at `limit/60` per
second. A request takes `cost` tokens; if the bucket has fewer, it's rejected with the time until
enough tokens exist (`Retry-After`). Bursts up to `limit` are allowed, and the average is bounded.

TPM "reserve → reconcile": we can't know a response's token count in advance, so we
1. reserve an upper-bound estimate (prompt estimate + max_tokens) before calling the provider;
2. after the response, refund `reserved - actual` (or charge extra if we under-estimated; the
   bucket may go negative, which delays the key's next requests).

Budgets: monthly counters per project (tokens, micro-USD). Checked before a call and incremented
after it. Concurrency means in-flight requests can overshoot a hard budget by at most
(in-flight requests x their cost). That is documented, and it's acceptable for metering.

In-memory implementations are exact within one process (asyncio is single-threaded and these
methods don't await between check and update). Redis implementations use Lua scripts so that
check-and-update is atomic across replicas.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from redis.asyncio import Redis

from ai_gateway.auth import Principal
from ai_gateway.db import month_bucket
from ai_gateway.errors import BudgetExceededError, InvalidRequestError, RateLimitedError

MICRO = Decimal(1_000_000)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    retry_after: float = 0.0
    remaining: float = 0.0


class Limiter(Protocol):
    async def take(self, key: str, limit: int, cost: int) -> Decision: ...
    async def give_back(self, key: str, limit: int, amount: int) -> None: ...


class InMemoryLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last_refill_ts)

    def _refill(self, key: str, limit: int) -> float:
        now = self.clock()
        tokens, last = self._buckets.get(key, (float(limit), now))
        tokens = min(float(limit), tokens + (now - last) * limit / 60.0)
        self._buckets[key] = (tokens, now)
        return tokens

    async def take(self, key: str, limit: int, cost: int) -> Decision:
        tokens = self._refill(key, limit)
        if tokens >= cost:
            self._buckets[key] = (tokens - cost, self.clock())
            return Decision(True, remaining=tokens - cost)
        return Decision(False, retry_after=(cost - tokens) * 60.0 / limit, remaining=tokens)

    async def give_back(self, key: str, limit: int, amount: int) -> None:
        tokens = self._refill(key, limit)
        # Positive amount = refund (capped at capacity); negative = extra charge (may go below 0).
        self._buckets[key] = (min(float(limit), tokens + amount), self.clock())


_TAKE = """
local limit = tonumber(ARGV[1]); local cost = tonumber(ARGV[2]); local now = tonumber(ARGV[3])
local state = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(state[1]) or limit
local ts = tonumber(state[2]) or now
tokens = math.min(limit, tokens + (now - ts) * limit / 60.0)
local allowed = 0
if tokens >= cost then tokens = tokens - cost; allowed = 1 end
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('EXPIRE', KEYS[1], 120)
return {allowed, tostring(tokens)}
"""

_GIVE = """
local limit = tonumber(ARGV[1]); local amount = tonumber(ARGV[2]); local now = tonumber(ARGV[3])
local state = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(state[1]) or limit
local ts = tonumber(state[2]) or now
tokens = math.min(limit, tokens + (now - ts) * limit / 60.0 + amount)
redis.call('HSET', KEYS[1], 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('EXPIRE', KEYS[1], 120)
return tostring(tokens)
"""


class RedisLimiter:
    def __init__(
        self, redis: Redis, *, prefix: str = "gw:rl", clock: Callable[[], float] = time.time
    ) -> None:
        self.redis = redis
        self.prefix = prefix
        self.clock = clock
        self._take = redis.register_script(_TAKE)
        self._give = redis.register_script(_GIVE)

    async def take(self, key: str, limit: int, cost: int) -> Decision:
        allowed, tokens_raw = await self._take(
            keys=[f"{self.prefix}:{key}"], args=[limit, cost, self.clock()]
        )
        tokens = float(tokens_raw)
        if int(allowed) == 1:
            return Decision(True, remaining=tokens)
        return Decision(False, retry_after=(cost - tokens) * 60.0 / limit, remaining=tokens)

    async def give_back(self, key: str, limit: int, amount: int) -> None:
        await self._give(keys=[f"{self.prefix}:{key}"], args=[limit, amount, self.clock()])


class BudgetStore(Protocol):
    async def get(self, project_id: str, month: str) -> tuple[int, int]: ...  # (tokens, micro_usd)
    async def add(self, project_id: str, month: str, tokens: int, micro_usd: int) -> None: ...


class InMemoryBudgetStore:
    def __init__(self) -> None:
        self._used: dict[tuple[str, str], tuple[int, int]] = {}

    async def get(self, project_id: str, month: str) -> tuple[int, int]:
        return self._used.get((project_id, month), (0, 0))

    async def add(self, project_id: str, month: str, tokens: int, micro_usd: int) -> None:
        t, c = self._used.get((project_id, month), (0, 0))
        self._used[(project_id, month)] = (t + tokens, c + micro_usd)


class RedisBudgetStore:
    def __init__(self, redis: Redis, prefix: str = "gw:budget") -> None:
        self.redis = redis
        self.prefix = prefix

    def _key(self, project_id: str, month: str) -> str:
        return f"{self.prefix}:{project_id}:{month}"

    async def get(self, project_id: str, month: str) -> tuple[int, int]:
        tokens, micro = await self.redis.hmget(
            self._key(project_id, month), ["tokens", "micro_usd"]
        )
        return int(tokens or 0), int(micro or 0)

    async def add(self, project_id: str, month: str, tokens: int, micro_usd: int) -> None:
        key = self._key(project_id, month)
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.hincrby(key, "tokens", tokens)
            pipe.hincrby(key, "micro_usd", micro_usd)
            pipe.expire(key, 60 * 60 * 24 * 40)
            await pipe.execute()


@dataclass
class Reservation:
    principal: Principal
    reserved_tokens: int


class Guard:
    """Admission control for one request: RPM, TPM reservation, budget. Then settlement."""

    def __init__(self, limiter: Limiter, budgets: BudgetStore) -> None:
        self.limiter = limiter
        self.budgets = budgets

    async def admit(self, principal: Principal, estimated_tokens: int) -> Reservation:
        if principal.key_id is None:
            return Reservation(principal, 0)  # anonymous (auth disabled): no limits
        await self._check_budget(principal)
        if principal.rpm_limit:
            d = await self.limiter.take(f"rpm:{principal.key_id}", principal.rpm_limit, 1)
            if not d.allowed:
                raise RateLimitedError(
                    f"requests-per-minute limit ({principal.rpm_limit}) exceeded",
                    retry_after=d.retry_after,
                )
        reserved = 0
        if principal.tpm_limit:
            if estimated_tokens > principal.tpm_limit:
                raise InvalidRequestError(
                    f"request needs ~{estimated_tokens} tokens, above this key's per-minute limit of {principal.tpm_limit}; lower max_tokens"
                )
            d = await self.limiter.take(
                f"tpm:{principal.key_id}", principal.tpm_limit, estimated_tokens
            )
            if not d.allowed:
                raise RateLimitedError(
                    f"tokens-per-minute limit ({principal.tpm_limit}) exceeded",
                    retry_after=d.retry_after,
                )
            reserved = estimated_tokens
        return Reservation(principal, reserved)

    async def _check_budget(self, p: Principal) -> None:
        if p.project_id is None or p.budget_mode != "hard":
            return
        if p.monthly_token_budget is None and p.monthly_cost_budget_usd is None:
            return
        tokens, micro = await self.budgets.get(p.project_id, month_bucket())
        if p.monthly_token_budget is not None and tokens >= p.monthly_token_budget:
            raise BudgetExceededError("monthly token budget exhausted for this project")
        if (
            p.monthly_cost_budget_usd is not None
            and Decimal(micro) / MICRO >= p.monthly_cost_budget_usd
        ):
            raise BudgetExceededError("monthly cost budget exhausted for this project")

    async def settle(self, reservation: Reservation, actual_tokens: int, cost_usd: Decimal) -> None:
        p = reservation.principal
        if p.key_id is None:
            return
        if p.tpm_limit and reservation.reserved_tokens:
            delta = reservation.reserved_tokens - actual_tokens  # >0 refund, <0 extra charge
            if delta:
                await self.limiter.give_back(f"tpm:{p.key_id}", p.tpm_limit, delta)
        if p.project_id:
            await self.budgets.add(
                p.project_id, month_bucket(), actual_tokens, math.ceil(cost_usd * MICRO)
            )
