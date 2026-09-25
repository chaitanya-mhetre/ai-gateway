from __future__ import annotations

import os
from collections.abc import AsyncIterator
from decimal import Decimal

import pytest
from redis.asyncio import Redis
from sqlalchemy import func, select

from ai_gateway.db import Database, UsageDaily, UsageEvent
from ai_gateway.metering.usage import (
    InProcessSink,
    RedisStreamSink,
    UsageRecord,
    UsageWorker,
    write_batch,
)
from ai_gateway.redis_client import make_redis


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    database = Database("sqlite+aiosqlite:///:memory:")
    await database.create_all()
    yield database
    await database.dispose()


def rec(request_id: str, **kw: object) -> UsageRecord:
    base: dict[str, object] = {
        "request_id": request_id,
        "project_id": "proj",
        "alias": "chat-default",
        "provider": "p",
        "model": "m",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "est_cost_usd": Decimal("0.001"),
    }
    base.update(kw)
    return UsageRecord.model_validate(base)


async def count(db: Database) -> int:
    async with db.session() as s:
        return int((await s.execute(select(func.count()).select_from(UsageEvent))).scalar_one())


async def test_write_batch_is_idempotent_and_rolls_up(db: Database) -> None:
    assert await write_batch(db, [rec("a"), rec("b"), rec("b")]) == 2
    assert await write_batch(db, [rec("a"), rec("c", status_code=502)]) == 1  # 'a' is a duplicate
    assert await count(db) == 3
    async with db.session() as s:
        daily = (await s.execute(select(UsageDaily))).scalars().one()
    assert daily.requests == 3 and daily.prompt_tokens == 30 and daily.errors == 1
    assert daily.est_cost_usd == Decimal("0.003")


async def test_unknown_cost_is_stored_as_zero_but_record_keeps_none(db: Database) -> None:
    r = rec("x", est_cost_usd=None)
    assert r.est_cost_usd is None
    await write_batch(db, [r])
    assert await count(db) == 1


async def test_in_process_sink_flushes_on_stop(db: Database) -> None:
    sink = InProcessSink(db, flush_interval_s=3600)
    await sink.start()
    for i in range(5):
        await sink.publish(rec(f"r{i}"))
    await sink.stop()
    assert await count(db) == 5


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
async def test_stream_worker_writes_and_acks(redis: Redis, db: Database) -> None:
    sink = RedisStreamSink(redis, stream="t:usage")
    worker = UsageWorker(redis, db, stream="t:usage", group="g", block_ms=10)
    await worker.ensure_group()
    for i in range(3):
        await sink.publish(rec(f"s{i}"))
    assert await worker.process_once() == 3
    pending = await redis.xpending("t:usage", "g")
    assert pending["pending"] == 0


@pytest.mark.redis
async def test_crashed_consumer_entries_are_reclaimed(redis: Redis, db: Database) -> None:
    sink = RedisStreamSink(redis, stream="t:usage")
    crashed = UsageWorker(redis, db, stream="t:usage", group="g", consumer="crashed")
    await crashed.ensure_group()
    await sink.publish(rec("lost-1"))
    # Simulate a crash: the consumer reads the entry but dies before writing/acking it.
    await redis.xreadgroup("g", "crashed", {"t:usage": ">"}, count=10)
    assert await count(db) == 0
    rescuer = UsageWorker(
        redis, db, stream="t:usage", group="g", consumer="rescuer", claim_idle_ms=0, block_ms=10
    )
    assert await rescuer.process_once() == 1  # XAUTOCLAIM took over the pending entry
    assert await count(db) == 1
