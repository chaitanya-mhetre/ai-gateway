"""Usage events: produced on the request path, persisted asynchronously.

Why asynchronous? A synchronous INSERT per request would add DB latency (and a DB dependency) to
every LLM call. Instead the request publishes an event and returns:

- With Redis: `XADD` to a Redis Stream. A separate worker (`ai-gateway usage-worker`) reads the stream
  as a consumer group, inserts in batches, updates daily rollups, and only then `XACK`s. If the
  worker crashes mid-batch, the un-acked entries stay pending and are re-claimed (`XAUTOCLAIM`),
  so delivery is at-least-once. Inserts are idempotent on `request_id`, which gives effectively-once rows.
- Without Redis (single replica / dev): an in-process asyncio queue flushed by a background task.
  Events still in the queue can be lost on a crash. That's the documented trade-off of this mode.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Protocol

from pydantic import BaseModel, Field
from redis.asyncio import Redis
from redis.exceptions import ResponseError
from sqlalchemy import select
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.ext.asyncio import AsyncSession

from ai_gateway.db import Database, UsageDaily, UsageEvent

log = logging.getLogger(__name__)

STREAM = "gw:usage"
GROUP = "usage-writers"


class UsageRecord(BaseModel):
    """The usage-event schema. Shared in spirit with oss/aiwatch (same field names)."""

    request_id: str
    project_id: str | None = None
    api_key_id: str | None = None
    alias: str
    provider: str = ""
    model: str = ""
    attempt_count: int = 0
    fallback_used: bool = False
    cache: str = "miss"
    stream: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    tokens_estimated: bool = False
    est_cost_usd: Decimal | None = None  # None = no price entry for this model
    latency_ms: int = 0
    ttft_ms: int | None = None
    status_code: int = 200
    error_type: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class UsageSink(Protocol):
    async def publish(self, record: UsageRecord) -> None: ...
    async def start(self) -> None: ...
    async def stop(self) -> None: ...


# --- persistence (shared by both modes) ---------------------------------------------------------


def _insert_ignore(dialect: str) -> Any:
    return (postgresql if dialect == "postgresql" else sqlite).insert


async def write_batch(db: Database, records: Sequence[UsageRecord]) -> int:
    """Insert events (skipping duplicates) and fold NEW ones into daily rollups. Returns #new."""
    if not records:
        return 0
    dialect = db.engine.dialect.name
    insert = _insert_ignore(dialect)
    async with db.session() as session:
        existing = set(
            (
                await session.execute(
                    select(UsageEvent.request_id).where(
                        UsageEvent.request_id.in_([r.request_id for r in records])
                    )
                )
            ).scalars()
        )
        fresh = [
            r for r in {r.request_id: r for r in records}.values() if r.request_id not in existing
        ]
        if not fresh:
            return 0
        rows = [{**r.model_dump(), "est_cost_usd": r.est_cost_usd or Decimal(0)} for r in fresh]
        await session.execute(
            insert(UsageEvent).values(rows).on_conflict_do_nothing(index_elements=["request_id"])
        )
        await _rollup(session, fresh, insert)
        await session.commit()
        return len(fresh)


async def _rollup(session: AsyncSession, records: list[UsageRecord], insert: Any) -> None:
    agg: dict[tuple[str, Any, str, str], dict[str, Any]] = {}
    for r in records:
        if r.project_id is None:
            continue
        key = (r.project_id, r.created_at.date(), r.provider, r.model)
        a = agg.setdefault(
            key,
            {
                "requests": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "est_cost_usd": Decimal(0),
                "errors": 0,
            },
        )
        a["requests"] += 1
        a["prompt_tokens"] += r.prompt_tokens
        a["completion_tokens"] += r.completion_tokens
        a["est_cost_usd"] += r.est_cost_usd or Decimal(0)
        a["errors"] += 1 if r.status_code >= 400 else 0
    for (project_id, day, provider, model), a in agg.items():
        stmt = insert(UsageDaily).values(
            project_id=project_id, day=day, provider=provider, model=model, **a
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["project_id", "day", "provider", "model"],
            set_={col: getattr(UsageDaily, col) + getattr(stmt.excluded, col) for col in a},
        )
        await session.execute(stmt)


# --- in-process mode ------------------------------------------------------------------------------


class InProcessSink:
    def __init__(
        self, db: Database, *, batch_size: int = 200, flush_interval_s: float = 1.0
    ) -> None:
        self.db = db
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self.queue: asyncio.Queue[UsageRecord] = asyncio.Queue(maxsize=100_000)
        self._task: asyncio.Task[None] | None = None

    async def publish(self, record: UsageRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except asyncio.QueueFull:
            log.warning("usage queue full; dropping event %s", record.request_id)

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self.flush_interval_s)
            await self.flush()

    async def flush(self) -> int:
        batch: list[UsageRecord] = []
        while not self.queue.empty() and len(batch) < self.batch_size:
            batch.append(self.queue.get_nowait())
        try:
            return await write_batch(self.db, batch)
        except Exception:
            log.exception("failed to write %d usage events", len(batch))
            return 0

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        while not self.queue.empty():
            await self.flush()


# --- Redis Streams mode ---------------------------------------------------------------------------


class RedisStreamSink:
    def __init__(self, redis: Redis, *, stream: str = STREAM, maxlen: int = 1_000_000) -> None:
        self.redis = redis
        self.stream = stream
        self.maxlen = maxlen

    async def publish(self, record: UsageRecord) -> None:
        # MAXLEN ~ caps memory if the worker is down for a long time (oldest entries trimmed).
        await self.redis.xadd(
            self.stream, {"data": record.model_dump_json()}, maxlen=self.maxlen, approximate=True
        )

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


class UsageWorker:
    """Consumer-group reader: batch → DB → XACK. Safe to run several replicas of it."""

    def __init__(
        self,
        redis: Redis,
        db: Database,
        *,
        consumer: str = "worker-1",
        stream: str = STREAM,
        group: str = GROUP,
        batch_size: int = 500,
        block_ms: int = 1000,
        claim_idle_ms: int = 60_000,
    ) -> None:
        self.redis, self.db = redis, db
        self.consumer, self.stream, self.group = consumer, stream, group
        self.batch_size, self.block_ms, self.claim_idle_ms = batch_size, block_ms, claim_idle_ms

    async def ensure_group(self) -> None:
        try:
            await self.redis.xgroup_create(self.stream, self.group, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    @staticmethod
    def _decode(entries: list[Any]) -> tuple[list[bytes], list[UsageRecord]]:
        ids: list[bytes] = []
        records: list[UsageRecord] = []
        for entry_id, fields in entries:
            ids.append(entry_id)
            raw = fields.get(b"data") or fields.get("data")
            try:
                records.append(UsageRecord.model_validate(json.loads(raw)))
            except Exception:
                log.exception(
                    "dropping malformed usage entry %s", entry_id
                )  # ack it: poison message
        return ids, records

    async def process_once(self) -> int:
        """Handle one batch: re-claimed stale entries first, then new ones. Returns #rows written."""
        claimed = await self.redis.xautoclaim(
            self.stream, self.group, self.consumer, self.claim_idle_ms, "0-0", count=self.batch_size
        )
        entries: list[Any] = list(claimed[1]) if claimed else []
        if not entries:
            resp = await self.redis.xreadgroup(
                self.group,
                self.consumer,
                {self.stream: ">"},
                count=self.batch_size,
                block=self.block_ms,
            )
            entries = _read_entries(resp)
        if not entries:
            return 0
        ids, records = self._decode(entries)
        written = await write_batch(self.db, records)  # raises → no XACK → retried later
        await self.redis.xack(self.stream, self.group, *ids)
        return written

    async def run_forever(self) -> None:
        await self.ensure_group()
        while True:
            try:
                await self.process_once()
            except Exception:
                log.exception("usage batch failed; will retry")
                await asyncio.sleep(1)


def _read_entries(resp: Any) -> list[Any]:
    """XREADGROUP returns [[stream, entries]] under RESP2 and {stream: entries} under RESP3."""
    if not resp:
        return []
    if isinstance(resp, dict):
        return [e for entries in resp.values() for e in entries]
    return list(resp[0][1])
