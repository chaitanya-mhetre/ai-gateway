from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
from redis.asyncio import Redis

from ai_gateway.config import BreakerConfig
from ai_gateway.redis_client import make_redis
from ai_gateway.routing.breaker import BreakerState, CircuitBreaker, InMemoryBreaker, RedisBreaker

CFG = BreakerConfig(window_s=10, min_requests=4, failure_ratio=0.5, cooldown_s=5)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


async def run_state_machine(b: CircuitBreaker, clock: FakeClock) -> None:
    # Below min_requests nothing trips, even at 100% failures.
    for _ in range(3):
        await b.record_failure("p")
    assert await b.state("p") is BreakerState.CLOSED
    # 4th request: 4/4 failures >= 50% -> OPEN
    await b.record_failure("p")
    assert await b.state("p") is BreakerState.OPEN
    assert not await b.allow("p")
    # After the cool-down exactly one probe is admitted
    clock.now += 5
    assert await b.allow("p")
    assert not await b.allow("p")
    # Probe fails -> OPEN again, cool-down restarts
    await b.record_failure("p")
    assert not await b.allow("p")
    clock.now += 5
    assert await b.allow("p")
    await b.record_success("p")  # probe succeeds -> CLOSED
    assert await b.state("p") is BreakerState.CLOSED
    assert await b.allow("p") and await b.allow("p")


async def test_in_memory_breaker_state_machine() -> None:
    clock = FakeClock()
    await run_state_machine(InMemoryBreaker(CFG, clock=clock), clock)


async def test_successes_keep_ratio_below_threshold() -> None:
    b = InMemoryBreaker(CFG, clock=FakeClock())
    for _ in range(3):
        await b.record_success("p")
    await b.record_failure("p")
    await b.record_success("p")
    await b.record_failure("p")
    assert await b.state("p") is BreakerState.CLOSED  # 2/6 failures


async def test_old_outcomes_leave_the_window() -> None:
    clock = FakeClock()
    b = InMemoryBreaker(CFG, clock=clock)
    for _ in range(3):
        await b.record_failure("p")
    clock.now += 11  # those failures are now outside the 10 s window
    await b.record_failure("p")
    assert await b.state("p") is BreakerState.CLOSED


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
async def test_redis_breaker_state_machine(redis: Redis) -> None:
    clock = FakeClock()
    await run_state_machine(RedisBreaker(redis, CFG, clock=clock), clock)


@pytest.mark.redis
async def test_redis_breaker_is_shared_between_replicas(redis: Redis) -> None:
    clock = FakeClock()
    replica_a, replica_b = (
        RedisBreaker(redis, CFG, clock=clock),
        RedisBreaker(redis, CFG, clock=clock),
    )
    for _ in range(4):
        await replica_a.record_failure("p")
    assert not await replica_b.allow("p")  # B sees the circuit A opened
