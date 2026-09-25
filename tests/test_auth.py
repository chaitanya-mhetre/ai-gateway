from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta

import pytest

from ai_gateway.auth import KeyAuthenticator, generate_key, hash_key, parse_prefix
from ai_gateway.db import ApiKey, Database, Project, Tenant, utcnow
from ai_gateway.errors import AuthenticationError, PermissionDeniedError

PEPPER = "test-pepper"


def test_key_format_and_prefix() -> None:
    key, prefix = generate_key()
    assert key.startswith(f"gk_{prefix}_")
    assert parse_prefix(key) == prefix
    assert parse_prefix("sk-something") is None
    assert parse_prefix("gk_short_x") is None


def test_hash_depends_on_pepper() -> None:
    key, _ = generate_key()
    assert hash_key(key, "a") != hash_key(key, "b")
    assert len(hash_key(key, "a")) == 32


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    database = Database("sqlite+aiosqlite:///:memory:")
    await database.migrate()
    yield database
    await database.dispose()


async def seed_key(db: Database, **key_fields: object) -> tuple[str, str]:
    key, prefix = generate_key()
    async with db.session() as s:
        tenant = Tenant(name=f"t-{prefix}")
        s.add(tenant)
        await s.flush()
        project = Project(tenant_id=tenant.id, name="p")
        s.add(project)
        await s.flush()
        record = ApiKey(
            project_id=project.id,
            prefix=prefix,
            key_hash=hash_key(key, PEPPER),
            name="k",
            **key_fields,
        )
        s.add(record)
        await s.commit()
        return key, record.id


async def test_valid_key_yields_principal(db: Database) -> None:
    key, key_id = await seed_key(db, allowed_aliases=["chat-default"], rpm_limit=5)
    p = await KeyAuthenticator(db, PEPPER).authenticate(f"Bearer {key}")
    assert p.key_id == key_id and p.rpm_limit == 5 and p.tenant_id is not None
    p.check_alias("chat-default")
    with pytest.raises(PermissionDeniedError):
        p.check_alias("chat-premium")


@pytest.mark.parametrize("header", [None, "Basic abc", "Bearer not-a-gateway-key"])
async def test_missing_or_malformed(db: Database, header: str | None) -> None:
    with pytest.raises(AuthenticationError):
        await KeyAuthenticator(db, PEPPER).authenticate(header)


async def test_wrong_secret_with_real_prefix_is_rejected(db: Database) -> None:
    key, _ = await seed_key(db)
    tampered = key[:-4] + ("aaaa" if not key.endswith("aaaa") else "bbbb")
    with pytest.raises(AuthenticationError, match="invalid API key"):
        await KeyAuthenticator(db, PEPPER).authenticate(f"Bearer {tampered}")


async def test_wrong_pepper_is_rejected(db: Database) -> None:
    key, _ = await seed_key(db)
    with pytest.raises(AuthenticationError):
        await KeyAuthenticator(db, "other-pepper").authenticate(f"Bearer {key}")


async def test_expired_key(db: Database) -> None:
    key, _ = await seed_key(db, expires_at=utcnow() - timedelta(seconds=1))
    with pytest.raises(AuthenticationError, match="expired"):
        await KeyAuthenticator(db, PEPPER).authenticate(f"Bearer {key}")


async def test_revocation_respects_cache_ttl_and_invalidate(db: Database) -> None:
    key, key_id = await seed_key(db)
    clock = Clock()
    auth = KeyAuthenticator(db, PEPPER, cache_ttl_s=30, clock=clock)
    await auth.authenticate(f"Bearer {key}")  # now cached
    async with db.session() as s:
        record = await s.get(ApiKey, key_id)
        assert record is not None
        record.revoked_at = utcnow()
        await s.commit()
    await auth.authenticate(f"Bearer {key}")  # still served from cache (documented staleness)
    clock.now = 31
    with pytest.raises(AuthenticationError, match="revoked"):
        await auth.authenticate(f"Bearer {key}")


async def test_invalidate_makes_revocation_immediate(db: Database) -> None:
    key, key_id = await seed_key(db)
    auth = KeyAuthenticator(db, PEPPER, cache_ttl_s=300)
    await auth.authenticate(f"Bearer {key}")
    async with db.session() as s:
        record = await s.get(ApiKey, key_id)
        assert record is not None
        record.revoked_at = utcnow()
        await s.commit()
    auth.invalidate(key.split("_")[1])
    with pytest.raises(AuthenticationError):
        await auth.authenticate(f"Bearer {key}")
