"""Control-plane identities: admin users, roles and permissions.

Replaces the old single shared admin token. Each operator gets a personal token
(`ga_<prefix>_<secret>`, stored as HMAC-SHA256 with the pepper, exactly like API keys), so access
can be granted per person, limited by role, and revoked one person at a time.

Authorization is "rules as data": `ROLE_PERMISSIONS` maps each role to the permissions it holds and
every admin route declares the one permission it needs (see `admin.require`).
"""

from __future__ import annotations

import hmac
import secrets
from dataclasses import dataclass
from enum import StrEnum

from email_validator import EmailNotValidError, validate_email
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ai_gateway.auth import PREFIX_ALPHABET, hash_key
from ai_gateway.db import AdminUser, Database, utcnow

TOKEN_KIND = "ga"


class Role(StrEnum):
    OWNER = "owner"  # everything, including managing other admins
    OPERATOR = "operator"  # day-to-day: tenants, projects, keys
    VIEWER = "viewer"  # read-only: usage, keys (metadata only), health, prices


class Permission(StrEnum):
    TENANTS_WRITE = "tenants:write"
    PROJECTS_WRITE = "projects:write"
    KEYS_READ = "keys:read"
    KEYS_WRITE = "keys:write"
    USAGE_READ = "usage:read"
    PROVIDERS_READ = "providers:read"
    PRICES_READ = "prices:read"
    ADMINS_MANAGE = "admins:manage"


_READ = frozenset(
    {Permission.KEYS_READ, Permission.USAGE_READ, Permission.PROVIDERS_READ, Permission.PRICES_READ}
)
_OPERATE = _READ | {Permission.TENANTS_WRITE, Permission.PROJECTS_WRITE, Permission.KEYS_WRITE}

ROLE_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.VIEWER: _READ,
    Role.OPERATOR: frozenset(_OPERATE),
    Role.OWNER: frozenset(_OPERATE | {Permission.ADMINS_MANAGE}),
}


@dataclass(frozen=True)
class AdminPrincipal:
    id: str
    email: str
    role: Role

    def can(self, permission: Permission) -> bool:
        return permission in ROLE_PERMISSIONS[self.role]


class AdminAuthError(Exception):
    """Missing, malformed, unknown, wrong or disabled admin token. One message for all: no oracle."""


class LastOwnerError(Exception):
    """Refusing to disable the only active owner (that would lock everyone out)."""


def generate_admin_token() -> tuple[str, str]:
    """Return (plaintext_token, prefix)."""
    prefix = "".join(secrets.choice(PREFIX_ALPHABET) for _ in range(12))
    return f"{TOKEN_KIND}_{prefix}_{secrets.token_urlsafe(32)}", prefix


def parse_admin_prefix(token: str) -> str | None:
    parts = token.split("_", 2)
    if len(parts) != 3 or parts[0] != TOKEN_KIND or len(parts[1]) != 12 or not parts[2]:
        return None
    if any(c not in PREFIX_ALPHABET for c in parts[1]):
        return None
    return parts[1]


async def create_admin_user(
    db: Database,
    pepper: str,
    *,
    email: str,
    role: Role,
    created_by: str | None = None,
    token: str | None = None,
) -> tuple[AdminUser, str]:
    """Create an admin and return (row, plaintext_token). The token is never stored or shown again.

    `token` lets tests (and secret-manager provisioning) supply a pre-generated token; it must
    still have the `ga_<prefix>_<secret>` shape.
    """
    try:
        email = validate_email(email, check_deliverability=False).normalized.lower()
    except EmailNotValidError as exc:
        raise ValueError(f"invalid email: {exc}") from exc
    if token is None:
        token, prefix = generate_admin_token()
    else:
        parsed = parse_admin_prefix(token)
        if parsed is None:
            raise ValueError("admin token must look like ga_<12 chars>_<secret>")
        prefix = parsed
    async with db.session() as s:
        user = AdminUser(
            email=email,
            role=str(role),
            token_prefix=prefix,
            token_hash=hash_key(token, pepper),
            created_by=created_by,
        )
        s.add(user)
        await s.commit()
    return user, token


async def active_owner_count(session: AsyncSession) -> int:
    stmt = (
        select(func.count())
        .select_from(AdminUser)
        .where(AdminUser.role == str(Role.OWNER), AdminUser.disabled_at.is_(None))
    )
    return int((await session.execute(stmt)).scalar_one())


async def disable_admin_user(db: Database, admin_id: str) -> AdminUser | None:
    """Disable (soft-delete) an admin. Takes effect on their next request: there's no auth cache."""
    async with db.session() as s:
        user = await s.get(AdminUser, admin_id)
        if user is None:
            return None
        is_active_owner = user.disabled_at is None and user.role == str(Role.OWNER)
        if is_active_owner and await active_owner_count(s) <= 1:
            raise LastOwnerError("cannot disable the last active owner")
        user.disabled_at = user.disabled_at or utcnow()
        await s.commit()
        return user


class AdminAuthenticator:
    """Verify `Authorization: Bearer ga_...` against the admin_users table.

    Deliberately uncached (unlike data-plane keys): the admin plane is low-traffic, and disabling an
    admin must take effect immediately on every replica.
    """

    def __init__(self, db: Database, pepper: str) -> None:
        self.db = db
        self.pepper = pepper

    async def authenticate(self, authorization: str | None) -> AdminPrincipal:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise AdminAuthError("admin token required")
        token = authorization[7:].strip()
        prefix = parse_admin_prefix(token)
        if prefix is None:
            raise AdminAuthError("admin token required")
        async with self.db.session() as s:
            user = (
                await s.execute(select(AdminUser).where(AdminUser.token_prefix == prefix))
            ).scalar_one_or_none()
        if (
            user is None
            or not hmac.compare_digest(user.token_hash, hash_key(token, self.pepper))
            or user.disabled_at is not None
        ):
            raise AdminAuthError("admin token required")
        return AdminPrincipal(id=user.id, email=user.email, role=Role(user.role))
