from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from decimal import Decimal

import pytest
from redis.asyncio import Redis

from ai_gateway.auth import Principal
from ai_gateway.db import month_bucket
from ai_gateway.errors import BudgetExceededError, InvalidRequestError, RateLimitedError
from ai_gateway.limits import (
    Guard,
    InMemoryBudgetStore,
    InMemoryLimiter,
    Limiter,
    RedisBudgetStore,
    RedisLimiter,
)
from ai_gateway.redis_client import make_redis


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def principal(**kw: object) -> Principal:
    base: dict[str, object] = {
        "key_id": "k1",
        "key_prefix": "abc",
        "project_id": "p1",
        "tenant_id": "t1",
        "rpm_limit": None,
        "tpm_limit": None,
    }
    base.update(kw)
    return Principal(**base)  # type: ignore[arg-type]


async def test_token_bucket_burst_then_refill() -> None:
    clock = Clock()
    lim = InMemoryLimiter(clock)
    results = [(await lim.take("k", 60, 1)).allowed for _ in range(61)]
    assert results.count(True) == 60  # full burst allowed
    denied = await lim.take("k", 60, 1)
    assert not denied.allowed and denied.retry_after == pytest.approx(1.0)  # 1 token/s refill
    clock.now += 1
    assert (await lim.take("k", 60, 1)).allowed


async def test_give_back_refund_and_extra_charge() -> None:
    clock = Clock()
    lim = InMemoryLimiter(clock)
    await lim.take("t", 1000, 800)
    await lim.give_back("t", 1000, 700)  # refund: actual was only 100
    assert (await lim.take("t", 1000, 900)).allowed
    await lim.give_back("t", 1000, -500)  # under-estimated by 500 -> bucket goes negative
    d = await lim.take("t", 1000, 1)
    assert not d.allowed and d.retry_after > 25


async def test_guard_rpm_limit_and_retry_after() -> None:
    guard = Guard(InMemoryLimiter(Clock()), InMemoryBudgetStore())
    p = principal(rpm_limit=2)
    await guard.admit(p, 10)
    await guard.admit(p, 10)
    with pytest.raises(RateLimitedError) as exc:
        await guard.admit(p, 10)
    assert exc.value.headers()["Retry-After"] == "30"


async def test_guard_tpm_reserve_and_reconcile() -> None:
    guard = Guard(InMemoryLimiter(Clock()), InMemoryBudgetStore())
    p = principal(tpm_limit=1000)
    r1 = await guard.admit(p, 900)
    with pytest.raises(RateLimitedError):
        await guard.admit(p, 200)  # only 100 left while r1 is reserved
    await guard.settle(r1, actual_tokens=100, cost_usd=Decimal(0))  # refund 800
    await guard.admit(p, 800)


async def test_request_bigger_than_tpm_is_a_400_not_a_429() -> None:
    guard = Guard(InMemoryLimiter(Clock()), InMemoryBudgetStore())
    with pytest.raises(InvalidRequestError):
        await guard.admit(principal(tpm_limit=100), 500)


async def test_hard_budget_blocks_soft_budget_does_not() -> None:
    budgets = InMemoryBudgetStore()
    guard = Guard(InMemoryLimiter(Clock()), budgets)
    hard = principal(monthly_token_budget=100)
    r = await guard.admit(hard, 10)
    await guard.settle(r, 100, Decimal("0.002"))
    with pytest.raises(BudgetExceededError):
        await guard.admit(hard, 10)
    await guard.admit(principal(monthly_token_budget=100, budget_mode="soft"), 10)
    assert await budgets.get("p1", month_bucket()) == (100, 2000)


async def test_cost_budget() -> None:
    guard = Guard(InMemoryLimiter(Clock()), InMemoryBudgetStore())
    p = principal(monthly_cost_budget_usd=Decimal("0.01"))
    r = await guard.admit(p, 10)
    await guard.settle(r, 10, Decimal("0.01"))
    with pytest.raises(BudgetExceededError):
        await guard.admit(p, 10)


async def burst(limiter: Limiter, n: int, limit: int) -> int:
    guard = Guard(limiter, InMemoryBudgetStore())
    p = principal(rpm_limit=limit)

    async def one() -> bool:
        try:
            await guard.admit(p, 1)
            return True
        except RateLimitedError:
            return False

    return sum(await asyncio.gather(*(one() for _ in range(n))))


async def test_concurrent_requests_never_exceed_limit_in_memory() -> None:
    assert await burst(InMemoryLimiter(Clock()), 100, 50) == 50


REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:56382/0")


@pytest.fixture
async def redis() -> AsyncIterator[Redis]:
    client: Redis = make_redis(REDIS_URL)
    try:
        await client.ping()
    except Exception:
        await client.aclose()
        pytest.skip(f"Redis not reachable at {REDIS_URL} (run `make up`)")
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()


@pytest.mark.redis
async def test_concurrent_requests_never_exceed_limit_redis(redis: Redis) -> None:
    # Lua makes check-and-decrement atomic: exactly `limit` of 200 concurrent requests pass.
    assert await burst(RedisLimiter(redis, clock=Clock()), 200, 50) == 50


@pytest.mark.redis
async def test_redis_refund(redis: Redis) -> None:
    clock = Clock()
    lim = RedisLimiter(redis, clock=clock)
    assert (await lim.take("t", 1000, 900)).allowed
    assert not (await lim.take("t", 1000, 200)).allowed
    await lim.give_back("t", 1000, 800)
    assert (await lim.take("t", 1000, 800)).allowed


@pytest.mark.redis
async def test_redis_budget_store(redis: Redis) -> None:
    store = RedisBudgetStore(redis)
    await store.add("p", "202609", 10, 500)
    await store.add("p", "202609", 5, 250)
    assert await store.get("p", "202609") == (15, 750)
