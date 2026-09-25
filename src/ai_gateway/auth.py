"""API keys: generation, hashing, verification, and a short-TTL lookup cache.

Key format: `gk_<prefix>_<secret>`
- `prefix` (12 chars) is stored in plaintext. It's the lookup handle and is safe to show in logs/UIs.
- The full key is stored only as HMAC-SHA256(pepper, key). A database leak alone doesn't reveal
  usable keys, and the pepper lives outside the DB (env / secret manager).
- Comparison uses `hmac.compare_digest` (constant time) to avoid timing side channels.
- A plain fast hash is fine here (unlike passwords): keys have 256 bits of entropy, so brute force
  is hopeless and we need per-request speed.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from ai_gateway.db import ApiKey, Database, Project, key_with_project, utcnow
from ai_gateway.errors import AuthenticationError, PermissionDeniedError

PREFIX_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def generate_key() -> tuple[str, str]:
    """Return (plaintext_key, prefix)."""
    prefix = "".join(secrets.choice(PREFIX_ALPHABET) for _ in range(12))
    return f"gk_{prefix}_{secrets.token_urlsafe(32)}", prefix


def hash_key(key: str, pepper: str) -> bytes:
    return hmac.new(pepper.encode(), key.encode(), hashlib.sha256).digest()


def parse_prefix(key: str) -> str | None:
    parts = key.split("_", 2)
    if len(parts) != 3 or parts[0] != "gk" or len(parts[1]) != 12:
        return None
    return parts[1]


@dataclass(frozen=True)
class Principal:
    """The authenticated caller. `anonymous` is used when auth is disabled (local dev)."""

    key_id: str | None
    key_prefix: str
    project_id: str | None
    tenant_id: str | None
    allowed_aliases: tuple[str, ...] | None = None
    rpm_limit: int | None = None
    tpm_limit: int | None = None
    monthly_token_budget: int | None = None
    monthly_cost_budget_usd: Decimal | None = None
    budget_mode: str = "hard"

    @property
    def tenant_scope(self) -> str:
        """Namespace for tenant-isolated state (cache keys etc.)."""
        return self.tenant_id or "anonymous"

    def check_alias(self, alias: str) -> None:
        if self.allowed_aliases is not None and alias not in self.allowed_aliases:
            raise PermissionDeniedError(f"this key may not use model '{alias}'")


ANONYMOUS = Principal(key_id=None, key_prefix="anonymous", project_id=None, tenant_id=None)


def principal_from(key: ApiKey, project: Project) -> Principal:
    return Principal(
        key_id=key.id,
        key_prefix=key.prefix,
        project_id=project.id,
        tenant_id=project.tenant_id,
        allowed_aliases=tuple(key.allowed_aliases) if key.allowed_aliases is not None else None,
        rpm_limit=key.rpm_limit,
        tpm_limit=key.tpm_limit,
        monthly_token_budget=project.monthly_token_budget,
        monthly_cost_budget_usd=project.monthly_cost_budget_usd,
        budget_mode=project.budget_mode,
    )


def _aware(dt: datetime | None) -> datetime | None:
    # SQLite returns naive datetimes; treat them as UTC.
    return None if dt is None or dt.tzinfo is not None else dt.replace(tzinfo=utcnow().tzinfo)


class KeyAuthenticator:
    """Verify bearer keys against the DB, with an in-process TTL cache on the hot path.

    Revocation latency is bounded by `cache_ttl_s` (default 30 s), a documented trade-off between
    DB load and how quickly a revoked key stops working. `invalidate()` makes it immediate on
    the replica that performed the revocation.
    """

    def __init__(
        self,
        db: Database,
        pepper: str,
        *,
        cache_ttl_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.db = db
        self.pepper = pepper
        self.cache_ttl_s = cache_ttl_s
        self.clock = clock
        self._cache: dict[str, tuple[float, ApiKey, Project]] = {}

    def invalidate(self, prefix: str) -> None:
        self._cache.pop(prefix, None)

    async def _lookup(self, prefix: str) -> tuple[ApiKey, Project] | None:
        hit = self._cache.get(prefix)
        if hit and self.clock() - hit[0] < self.cache_ttl_s:
            return hit[1], hit[2]
        async with self.db.session() as session:
            found = await key_with_project(session, prefix)
        if found is not None:
            self._cache[prefix] = (self.clock(), found[0], found[1])
        return found

    async def authenticate(self, authorization: str | None) -> Principal:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise AuthenticationError("missing bearer API key")
        key = authorization[7:].strip()
        prefix = parse_prefix(key)
        if prefix is None:
            raise AuthenticationError("malformed API key")
        found = await self._lookup(prefix)
        # Same error for "unknown prefix" and "wrong secret": don't leak which prefixes exist.
        if found is None or not hmac.compare_digest(found[0].key_hash, hash_key(key, self.pepper)):
            raise AuthenticationError("invalid API key")
        api_key, project = found
        if api_key.revoked_at is not None:
            raise AuthenticationError("API key revoked")
        expires = _aware(api_key.expires_at)
        if expires is not None and expires <= utcnow():
            raise AuthenticationError("API key expired")
        return principal_from(api_key, project)
