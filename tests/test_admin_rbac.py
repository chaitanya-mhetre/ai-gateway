"""Admin plane: per-person tokens, roles, permissions, and admin lifecycle."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from fastapi.routing import APIRoute

from ai_gateway import admin as admin_module
from ai_gateway.admin_users import (
    ROLE_PERMISSIONS,
    AdminAuthenticator,
    AdminAuthError,
    Permission,
    Role,
    create_admin_user,
    disable_admin_user,
    generate_admin_token,
    parse_admin_prefix,
)
from ai_gateway.app import create_app
from ai_gateway.config import Settings
from ai_gateway.db import Database
from ai_gateway.providers.base import Provider
from tests.conftest import ADMIN, OPERATOR, VIEWER, app_client, make_config


@pytest.fixture
async def gw(providers: dict[str, Provider]) -> AsyncIterator[httpx.AsyncClient]:
    settings = Settings(
        auth_enabled=True,
        key_pepper="pep",
        redis_url=None,
        database_url="sqlite+aiosqlite:///:memory:",
    )
    async with app_client(create_app(settings, config=make_config(), providers=providers)) as c:
        yield c


# --- tokens -------------------------------------------------------------------------------------


def test_token_format_round_trip() -> None:
    token, prefix = generate_admin_token()
    assert token.startswith(f"ga_{prefix}_")
    assert parse_admin_prefix(token) == prefix


@pytest.mark.parametrize(
    "token",
    ["", "ga_short_x", "gk_abcdefghijkl_secret", "ga_ABCDEFGHIJKL_secret", "ga_abcdefghijkl_"],
)
def test_malformed_tokens_are_rejected(token: str) -> None:
    assert parse_admin_prefix(token) is None


async def test_authenticator_rejects_wrong_secret_and_disabled_admin() -> None:
    db = Database("sqlite+aiosqlite:///:memory:")
    await db.migrate()
    user, token = await create_admin_user(db, "pep", email="A@Example.com", role=Role.VIEWER)
    auth = AdminAuthenticator(db, "pep")

    principal = await auth.authenticate(f"Bearer {token}")
    assert principal.email == "a@example.com"  # normalised
    assert principal.role is Role.VIEWER

    prefix = parse_admin_prefix(token)
    with pytest.raises(AdminAuthError):
        await auth.authenticate(f"Bearer ga_{prefix}_not-the-secret")
    with pytest.raises(AdminAuthError):
        await AdminAuthenticator(db, "other-pepper").authenticate(f"Bearer {token}")

    await disable_admin_user(db, user.id)
    with pytest.raises(AdminAuthError):
        await auth.authenticate(f"Bearer {token}")
    await db.dispose()


# --- role → permission table ---------------------------------------------------------------------


def test_roles_are_strictly_nested() -> None:
    assert ROLE_PERMISSIONS[Role.VIEWER] < ROLE_PERMISSIONS[Role.OPERATOR]
    assert ROLE_PERMISSIONS[Role.OPERATOR] < ROLE_PERMISSIONS[Role.OWNER]
    assert ROLE_PERMISSIONS[Role.OWNER] == frozenset(Permission)


def test_every_admin_route_declares_a_permission() -> None:
    """Guard rail: a new admin route without `require(...)` would be open to every admin role."""
    missing = []
    for route in admin_module.router.routes:
        assert isinstance(route, APIRoute)
        if getattr(route.endpoint, "public_to_any_admin", False):
            continue
        deps = [d.call for d in route.dependant.dependencies]
        if not any(hasattr(d, "required_permission") for d in deps):
            missing.append(f"{sorted(route.methods or ())} {route.path}")
    assert missing == []


# --- HTTP matrix ---------------------------------------------------------------------------------

# (method, path, json) for one representative call per permission.
CALLS: dict[Permission, tuple[str, str, dict[str, Any] | None]] = {
    Permission.TENANTS_WRITE: ("POST", "/admin/v1/tenants", {"name": "matrix"}),
    Permission.PROJECTS_WRITE: ("POST", "/admin/v1/projects", {"tenant_id": "nope", "name": "x"}),
    Permission.KEYS_READ: ("GET", "/admin/v1/keys?project_id=nope", None),
    Permission.KEYS_WRITE: ("DELETE", "/admin/v1/keys/nope", None),
    Permission.USAGE_READ: ("GET", "/admin/v1/usage?project_id=nope", None),
    Permission.PROVIDERS_READ: ("GET", "/admin/v1/providers/health", None),
    Permission.PRICES_READ: ("GET", "/admin/v1/prices", None),
    Permission.ADMINS_MANAGE: ("GET", "/admin/v1/admins", None),
}
HEADERS = {Role.OWNER: ADMIN, Role.OPERATOR: OPERATOR, Role.VIEWER: VIEWER}


@pytest.mark.parametrize("role", list(Role))
@pytest.mark.parametrize("permission", list(Permission))
async def test_permission_matrix(gw: httpx.AsyncClient, role: Role, permission: Permission) -> None:
    method, path, body = CALLS[permission]
    r = await gw.request(method, path, json=body, headers=HEADERS[role])
    if permission in ROLE_PERMISSIONS[role]:
        # Allowed: whatever the handler says (201/200, or 404 for the made-up ids), never 401/403.
        assert r.status_code not in (401, 403), r.text
    else:
        assert r.status_code == 403
        assert str(permission) in r.json()["detail"]


@pytest.mark.parametrize("permission", list(Permission))
async def test_every_admin_call_needs_a_token(
    gw: httpx.AsyncClient, permission: Permission
) -> None:
    method, path, body = CALLS[permission]
    r = await gw.request(method, path, json=body)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


async def test_data_plane_key_is_not_an_admin_token(gw: httpx.AsyncClient) -> None:
    t = (await gw.post("/admin/v1/tenants", json={"name": "acme"}, headers=ADMIN)).json()
    p = (
        await gw.post("/admin/v1/projects", json={"tenant_id": t["id"], "name": "x"}, headers=ADMIN)
    ).json()
    k = (
        await gw.post("/admin/v1/keys", json={"project_id": p["id"], "name": "k"}, headers=ADMIN)
    ).json()
    r = await gw.get("/admin/v1/prices", headers={"Authorization": f"Bearer {k['key']}"})
    assert r.status_code == 401


async def test_me_lists_my_permissions(gw: httpx.AsyncClient) -> None:
    me = (await gw.get("/admin/v1/me", headers=VIEWER)).json()
    assert me["role"] == "viewer"
    assert me["permissions"] == sorted(str(p) for p in ROLE_PERMISSIONS[Role.VIEWER])


# --- admin lifecycle -------------------------------------------------------------------------------


async def test_owner_creates_admin_who_can_then_log_in(gw: httpx.AsyncClient) -> None:
    r = await gw.post(
        "/admin/v1/admins", json={"email": "new@example.com", "role": "operator"}, headers=ADMIN
    )
    assert r.status_code == 201
    created = r.json()
    new_headers = {"Authorization": f"Bearer {created['token']}"}
    assert (await gw.get("/admin/v1/me", headers=new_headers)).json()["role"] == "operator"
    assert (
        await gw.post("/admin/v1/tenants", json={"name": "made-by-new"}, headers=new_headers)
    ).status_code == 201

    listed = (await gw.get("/admin/v1/admins", headers=ADMIN)).json()
    assert "token" not in listed[0] and "token_hash" not in listed[0]
    me = (await gw.get("/admin/v1/me", headers=ADMIN)).json()
    assert next(a for a in listed if a["id"] == created["id"])["created_by"] == me["id"]


async def test_duplicate_email_conflicts(gw: httpx.AsyncClient) -> None:
    r = await gw.post(
        "/admin/v1/admins", json={"email": "owner@example.com", "role": "viewer"}, headers=ADMIN
    )
    assert r.status_code == 409


async def test_invalid_email_or_role_rejected(gw: httpx.AsyncClient) -> None:
    for body in ({"email": "not-an-email", "role": "viewer"}, {"email": "a@b.co", "role": "god"}):
        assert (await gw.post("/admin/v1/admins", json=body, headers=ADMIN)).status_code in (
            400,
            422,
        )


async def test_disabled_admin_is_locked_out_immediately(gw: httpx.AsyncClient) -> None:
    created = (
        await gw.post(
            "/admin/v1/admins", json={"email": "temp@example.com", "role": "viewer"}, headers=ADMIN
        )
    ).json()
    headers = {"Authorization": f"Bearer {created['token']}"}
    assert (await gw.get("/admin/v1/prices", headers=headers)).status_code == 200
    assert (await gw.delete(f"/admin/v1/admins/{created['id']}", headers=ADMIN)).status_code == 200
    assert (await gw.get("/admin/v1/prices", headers=headers)).status_code == 401


async def test_cannot_disable_the_last_owner(gw: httpx.AsyncClient) -> None:
    me = (await gw.get("/admin/v1/me", headers=ADMIN)).json()
    r = await gw.delete(f"/admin/v1/admins/{me['id']}", headers=ADMIN)
    assert r.status_code == 409
    # With a second owner, the first one can be disabled.
    await gw.post(
        "/admin/v1/admins", json={"email": "owner2@example.com", "role": "owner"}, headers=ADMIN
    )
    assert (await gw.delete(f"/admin/v1/admins/{me['id']}", headers=ADMIN)).status_code == 200


async def test_operator_cannot_escalate(gw: httpx.AsyncClient) -> None:
    r = await gw.post(
        "/admin/v1/admins", json={"email": "evil@example.com", "role": "owner"}, headers=OPERATOR
    )
    assert r.status_code == 403
