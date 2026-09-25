"""Alembic migrations: upgrade from empty, downgrade to base, and no drift from the ORM models.

SQLite runs everywhere. The Postgres variant runs when POSTGRES_URL is set (CI sets it), because
the production schema lives on Postgres and some DDL differs between the two.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection

from ai_gateway.db import Base, Database, alembic_config

APP_TABLES = {"tenants", "projects", "api_keys", "usage_events", "usage_daily", "admin_users"}


def head_revision() -> str:
    head = ScriptDirectory.from_config(alembic_config()).get_current_head()
    assert head is not None
    return head


def _tables(conn: Connection) -> set[str]:
    return set(inspect(conn).get_table_names()) - {"alembic_version"}


def _diff(conn: Connection) -> list[Any]:
    ctx = MigrationContext.configure(conn, opts={"compare_type": True})
    return list(compare_metadata(ctx, Base.metadata))


def _urls() -> list[str]:
    urls = ["sqlite+aiosqlite:///:memory:"]
    if pg := os.environ.get("POSTGRES_URL"):
        urls.append(pg)
    return urls


@pytest.fixture(params=_urls(), ids=lambda u: u.split("+", 1)[0])
async def empty_db(request: pytest.FixtureRequest) -> AsyncIterator[Database]:
    db = Database(request.param)
    # Postgres is shared between runs: start from a genuinely empty schema.
    if not request.param.startswith("sqlite"):
        async with db.engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
    yield db
    await db.dispose()


async def test_upgrade_from_empty_creates_every_table(empty_db: Database) -> None:
    assert await empty_db.current_revision() is None
    await empty_db.migrate()
    assert await empty_db.current_revision() == head_revision()
    async with empty_db.engine.connect() as conn:
        assert await conn.run_sync(_tables) == APP_TABLES


async def test_models_match_migrations(empty_db: Database) -> None:
    """If this fails, someone changed a model without writing a migration (or vice versa).

    Fix: `GATEWAY_DATABASE_URL=... uv run alembic revision --autogenerate -m "<what changed>"`.
    """
    await empty_db.migrate()
    async with empty_db.engine.connect() as conn:
        assert await conn.run_sync(_diff) == []


async def test_migrate_is_idempotent(empty_db: Database) -> None:
    await empty_db.migrate()
    await empty_db.migrate()
    assert await empty_db.current_revision() == head_revision()


async def test_downgrade_to_base_then_upgrade_again(empty_db: Database) -> None:
    await empty_db.migrate()
    await empty_db.downgrade("base")
    async with empty_db.engine.connect() as conn:
        assert await conn.run_sync(_tables) == set()
    await empty_db.migrate()
    async with empty_db.engine.connect() as conn:
        assert await conn.run_sync(_tables) == APP_TABLES


async def test_step_by_step_upgrade_keeps_data(empty_db: Database) -> None:
    """Rows written at revision 0001 survive the later migrations."""
    await empty_db.migrate("0001")
    async with empty_db.engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO tenants (id, name, created_at) VALUES ('t1', 'acme', :now)"),
            {"now": datetime(2026, 1, 1, tzinfo=UTC)},
        )
    await empty_db.migrate()
    async with empty_db.engine.connect() as conn:
        names: list[str] = list((await conn.execute(text("SELECT name FROM tenants"))).scalars())
    assert names == ["acme"]
