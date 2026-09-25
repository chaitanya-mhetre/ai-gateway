"""Alembic environment.

Two ways in:
- Programmatic (`Database.migrate()`, used by `ai-gateway migrate` and the app's dev auto-migrate):
  the caller passes an open sync connection in `config.attributes["connection"]`.
- The `alembic` CLI (`alembic upgrade head`, `alembic revision --autogenerate`): no connection is
  passed, so we build an async engine from `GATEWAY_DATABASE_URL` like the app does.

`render_as_batch=True` lets the same scripts run on SQLite, which can't ALTER most things in place.
"""

from __future__ import annotations

import asyncio

from alembic import context
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from ai_gateway.config import Settings
from ai_gateway.db import Base

target_metadata = Base.metadata


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def _run_async(url: str) -> None:
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        await conn.run_sync(_run)
        await conn.commit()
    await engine.dispose()


def run_offline() -> None:
    context.configure(
        url=Settings().database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    connection = context.config.attributes.get("connection")
    if connection is not None:
        _run(connection)
    else:
        asyncio.run(_run_async(Settings().database_url))


if context.is_offline_mode():
    run_offline()
else:
    run_online()
