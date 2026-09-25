"""Database schema (SQLAlchemy 2.0, async) and a small repository.

Postgres in production (asyncpg), SQLite in tests (aiosqlite). Types are kept portable on purpose,
e.g. JSON instead of Postgres ARRAY. Tables are created with `metadata.create_all` on startup; a
real deployment would use Alembic migrations (see docs/limitations in the README).
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    Numeric,
    String,
    UniqueConstraint,
    func,
    select,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool


def _uuid() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[str]: JSON}


class Tenant(Base):
    __tablename__ = "tenants"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Project(Base):
    __tablename__ = "projects"
    __table_args__ = (UniqueConstraint("tenant_id", "name"),)
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(ForeignKey("tenants.id"), index=True)
    name: Mapped[str] = mapped_column(String(200))
    monthly_token_budget: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    monthly_cost_budget_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 4), nullable=True)
    budget_mode: Mapped[str] = mapped_column(String(8), default="hard")  # hard | soft
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ApiKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    prefix: Mapped[str] = mapped_column(String(16), unique=True)  # lookup handle, safe to log
    key_hash: Mapped[bytes] = mapped_column(LargeBinary(32))  # HMAC-SHA256(pepper, key)
    name: Mapped[str] = mapped_column(String(200))
    allowed_aliases: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    rpm_limit: Mapped[int] = mapped_column(Integer, default=60)
    tpm_limit: Mapped[int] = mapped_column(Integer, default=100_000)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class UsageEvent(Base):
    __tablename__ = "usage_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(32), unique=True)  # idempotent inserts
    project_id: Mapped[str | None] = mapped_column(String(32), index=True, nullable=True)
    api_key_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    alias: Mapped[str] = mapped_column(String(200))
    provider: Mapped[str] = mapped_column(String(64))
    model: Mapped[str] = mapped_column(String(200))
    attempt_count: Mapped[int] = mapped_column(Integer, default=1)
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False)
    cache: Mapped[str] = mapped_column(String(16), default="miss")
    stream: Mapped[bool] = mapped_column(Boolean, default=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cached_tokens: Mapped[int] = mapped_column(Integer, default=0)
    tokens_estimated: Mapped[bool] = mapped_column(Boolean, default=False)
    est_cost_usd: Mapped[Decimal] = mapped_column(Numeric(14, 8), default=Decimal(0))
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    ttft_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status_code: Mapped[int] = mapped_column(Integer, default=200)
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, index=True
    )


class UsageDaily(Base):
    __tablename__ = "usage_daily"
    project_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    provider: Mapped[str] = mapped_column(String(64), primary_key=True)
    model: Mapped[str] = mapped_column(String(200), primary_key=True)
    requests: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    completion_tokens: Mapped[int] = mapped_column(BigInteger, default=0)
    est_cost_usd: Mapped[Decimal] = mapped_column(Numeric(14, 6), default=Decimal(0))
    errors: Mapped[int] = mapped_column(Integer, default=0)


class Database:
    def __init__(self, url: str) -> None:
        kwargs: dict[str, Any] = {}
        if url.startswith("sqlite") and ":memory:" in url:
            # One shared connection, otherwise every session would see a different empty DB.
            kwargs = {"poolclass": StaticPool, "connect_args": {"check_same_thread": False}}
        self.engine: AsyncEngine = create_async_engine(url, **kwargs)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)

    async def create_all(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self.engine.dispose()

    def session(self) -> AsyncSession:
        return self.sessions()


async def key_with_project(session: AsyncSession, prefix: str) -> tuple[ApiKey, Project] | None:
    row = (
        await session.execute(
            select(ApiKey, Project)
            .join(Project, Project.id == ApiKey.project_id)
            .where(ApiKey.prefix == prefix)
        )
    ).first()
    return None if row is None else (row[0], row[1])


def month_bucket(now: datetime | None = None) -> str:
    return (now or utcnow()).strftime("%Y%m")


__all__ = [
    "ApiKey",
    "Base",
    "Database",
    "Project",
    "Tenant",
    "UsageDaily",
    "UsageEvent",
    "func",
    "key_with_project",
]
