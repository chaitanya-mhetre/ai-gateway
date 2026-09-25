"""Control-plane API: tenants, projects, keys, admins, provider health, usage.

Authentication: a personal admin token per operator (`admin_users.py`), never a data-plane key.
Authorization: every route declares exactly one `Permission` via `require(...)`; the role → permission
table lives in `admin_users.ROLE_PERMISSIONS`. `tests/test_admin_rbac.py` fails if a route is added
without a permission.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime, time
from decimal import Decimal
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import case, func, select
from sqlalchemy.exc import IntegrityError

from ai_gateway.admin_users import (
    AdminAuthenticator,
    AdminAuthError,
    AdminPrincipal,
    LastOwnerError,
    Permission,
    Role,
    create_admin_user,
    disable_admin_user,
)
from ai_gateway.auth import KeyAuthenticator, generate_key, hash_key
from ai_gateway.config import Settings
from ai_gateway.db import AdminUser, ApiKey, Database, Project, Tenant, UsageEvent, utcnow
from ai_gateway.gateway import Gateway
from ai_gateway.metering.pricing import PriceTable

audit_log = logging.getLogger("ai_gateway.admin.audit")


async def authenticate_admin(
    request: Request, authorization: str | None = Header(default=None)
) -> AdminPrincipal:
    """Router-wide dependency: 401 unless the caller presents a valid, enabled admin token."""
    authenticator: AdminAuthenticator = request.app.state.admin_authenticator
    try:
        principal = await authenticator.authenticate(authorization)
    except AdminAuthError as exc:
        raise HTTPException(401, str(exc), headers={"WWW-Authenticate": "Bearer"}) from exc
    request.state.admin = principal
    return principal


def require(permission: Permission) -> Callable[..., Awaitable[AdminPrincipal]]:
    """Per-route dependency: 403 unless the authenticated admin's role grants `permission`."""

    async def check(
        request: Request, admin: Annotated[AdminPrincipal, Depends(authenticate_admin)]
    ) -> AdminPrincipal:
        if not admin.can(permission):
            raise HTTPException(403, f"role '{admin.role}' lacks permission '{permission}'")
        if request.method != "GET":
            # Who changed what: the minimum audit trail for a control plane.
            audit_log.info(
                "admin_action admin_id=%s email=%s role=%s permission=%s %s %s",
                admin.id,
                admin.email,
                admin.role,
                permission,
                request.method,
                request.url.path,
            )
        return admin

    check.required_permission = permission  # type: ignore[attr-defined]  # read by the RBAC test
    return check


router = APIRouter(prefix="/admin/v1", dependencies=[Depends(authenticate_admin)], tags=["admin"])


class TenantIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class ProjectIn(BaseModel):
    tenant_id: str
    name: str = Field(min_length=1, max_length=200)
    monthly_token_budget: int | None = Field(default=None, ge=0)
    monthly_cost_budget_usd: Decimal | None = Field(default=None, ge=0)
    budget_mode: Literal["hard", "soft"] = "hard"


class KeyIn(BaseModel):
    project_id: str
    name: str = Field(min_length=1, max_length=200)
    allowed_aliases: list[str] | None = None
    rpm_limit: int = Field(default=60, ge=1)
    tpm_limit: int = Field(default=100_000, ge=1)
    expires_at: datetime | None = None


class AdminIn(BaseModel):
    email: EmailStr
    role: Role


def _db(request: Request) -> Database:
    db: Database = request.app.state.db
    return db


@router.post("/tenants", status_code=201, dependencies=[Depends(require(Permission.TENANTS_WRITE))])
async def create_tenant(body: TenantIn, request: Request) -> dict[str, Any]:
    async with _db(request).session() as s:
        tenant = Tenant(name=body.name)
        s.add(tenant)
        try:
            await s.commit()
        except IntegrityError as exc:
            raise HTTPException(409, "tenant name already exists") from exc
        return {"id": tenant.id, "name": tenant.name}


@router.post(
    "/projects", status_code=201, dependencies=[Depends(require(Permission.PROJECTS_WRITE))]
)
async def create_project(body: ProjectIn, request: Request) -> dict[str, Any]:
    async with _db(request).session() as s:
        if await s.get(Tenant, body.tenant_id) is None:
            raise HTTPException(404, "tenant not found")
        project = Project(**body.model_dump())
        s.add(project)
        try:
            await s.commit()
        except IntegrityError as exc:
            raise HTTPException(409, "project name already exists for this tenant") from exc
        return {"id": project.id, "tenant_id": project.tenant_id, "name": project.name}


@router.post("/keys", status_code=201, dependencies=[Depends(require(Permission.KEYS_WRITE))])
async def create_key(body: KeyIn, request: Request) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    plaintext, prefix = generate_key()
    async with _db(request).session() as s:
        if await s.get(Project, body.project_id) is None:
            raise HTTPException(404, "project not found")
        key = ApiKey(
            **body.model_dump(), prefix=prefix, key_hash=hash_key(plaintext, settings.key_pepper)
        )
        s.add(key)
        await s.commit()
        # The plaintext key is returned exactly once and never stored.
        return {"id": key.id, "prefix": prefix, "key": plaintext, "project_id": key.project_id}


@router.get("/keys", dependencies=[Depends(require(Permission.KEYS_READ))])
async def list_keys(project_id: str, request: Request) -> list[dict[str, Any]]:
    async with _db(request).session() as s:
        rows = (
            (await s.execute(select(ApiKey).where(ApiKey.project_id == project_id))).scalars().all()
        )
        return [
            {
                "id": k.id,
                "prefix": k.prefix,
                "name": k.name,
                "rpm_limit": k.rpm_limit,
                "tpm_limit": k.tpm_limit,
                "allowed_aliases": k.allowed_aliases,
                "revoked": k.revoked_at is not None,
            }
            for k in rows
        ]


@router.delete("/keys/{key_id}", dependencies=[Depends(require(Permission.KEYS_WRITE))])
async def revoke_key(key_id: str, request: Request) -> dict[str, Any]:
    async with _db(request).session() as s:
        key = await s.get(ApiKey, key_id)
        if key is None:
            raise HTTPException(404, "key not found")
        key.revoked_at = utcnow()
        await s.commit()
        authenticator: KeyAuthenticator = request.app.state.authenticator
        authenticator.invalidate(
            key.prefix
        )  # immediate on this replica; others within the cache TTL
        return {"id": key.id, "revoked": True}


@router.get("/providers/health", dependencies=[Depends(require(Permission.PROVIDERS_READ))])
async def providers_health(request: Request) -> dict[str, Any]:
    gateway: Gateway = request.app.state.gateway
    latency = gateway.latency.snapshot()
    out: dict[str, Any] = {}
    for name in gateway.providers:
        out[name] = {
            "circuit": str(await gateway.breaker.state(name)),
            "ewma_latency_s": {
                k: round(v, 4) for k, v in latency.items() if k.startswith(f"{name}:")
            },
        }
    return out


@router.get("/usage", dependencies=[Depends(require(Permission.USAGE_READ))])
async def usage(
    request: Request,
    project_id: str,
    from_date: Annotated[date | None, Query(alias="from")] = None,
    to_date: Annotated[date | None, Query(alias="to")] = None,
    group_by: Literal["model", "day", "key"] = "model",
) -> list[dict[str, Any]]:
    """Aggregated usage from the raw events (the daily rollup table serves dashboards)."""
    group_cols: dict[str, Any] = {
        "model": (UsageEvent.provider, UsageEvent.model),
        "day": (func.date(UsageEvent.created_at),),
        "key": (UsageEvent.api_key_id,),
    }
    cols = group_cols[group_by]
    stmt = (
        select(
            *cols,
            func.count().label("requests"),
            func.sum(UsageEvent.prompt_tokens).label("prompt_tokens"),
            func.sum(UsageEvent.completion_tokens).label("completion_tokens"),
            func.sum(UsageEvent.est_cost_usd).label("est_cost_usd"),
            func.sum(case((UsageEvent.status_code >= 400, 1), else_=0)).label("errors"),
            func.sum(case((UsageEvent.fallback_used, 1), else_=0)).label("fallbacks"),
        )
        .where(UsageEvent.project_id == project_id)
        .group_by(*cols)
    )
    if from_date:
        stmt = stmt.where(UsageEvent.created_at >= datetime.combine(from_date, time.min, UTC))
    if to_date:
        stmt = stmt.where(UsageEvent.created_at < datetime.combine(to_date, time.max, UTC))
    async with _db(request).session() as s:
        rows = (await s.execute(stmt)).all()
    out: list[dict[str, Any]] = []
    for row in rows:
        m = dict(row._mapping)
        group = {
            k: (str(v) if isinstance(v, date) else v) for k, v in m.items() if k not in _AGG_FIELDS
        }
        if group_by == "day":
            group = {"day": str(next(iter(group.values())))}
        out.append(
            {
                **group,
                "requests": int(m["requests"]),
                "prompt_tokens": int(m["prompt_tokens"] or 0),
                "completion_tokens": int(m["completion_tokens"] or 0),
                "est_cost_usd": str(Decimal(m["est_cost_usd"] or 0).quantize(Decimal("0.000001"))),
                "errors": int(m["errors"] or 0),
                "fallbacks": int(m["fallbacks"] or 0),
            }
        )
    return out


_AGG_FIELDS = {
    "requests",
    "prompt_tokens",
    "completion_tokens",
    "est_cost_usd",
    "errors",
    "fallbacks",
}


@router.get("/prices", dependencies=[Depends(require(Permission.PRICES_READ))])
async def prices(request: Request) -> list[dict[str, Any]]:
    table: PriceTable = request.app.state.prices
    return [e.model_dump(mode="json") for e in table.entries]


# --- admin users (owner only) --------------------------------------------------------------------


def _admin_out(user: AdminUser) -> dict[str, Any]:
    return {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "token_prefix": user.token_prefix,
        "disabled": user.disabled_at is not None,
        "created_by": user.created_by,
    }


@router.get("/me")
async def me(admin: Annotated[AdminPrincipal, Depends(authenticate_admin)]) -> dict[str, Any]:
    """Who am I and what may I do? Any authenticated admin may call this."""
    return {
        "id": admin.id,
        "email": admin.email,
        "role": str(admin.role),
        "permissions": sorted(str(p) for p in Permission if admin.can(p)),
    }


me.public_to_any_admin = True  # type: ignore[attr-defined]  # read by the RBAC test


@router.post("/admins", status_code=201)
async def create_admin(
    body: AdminIn,
    request: Request,
    admin: Annotated[AdminPrincipal, Depends(require(Permission.ADMINS_MANAGE))],
) -> dict[str, Any]:
    settings: Settings = request.app.state.settings
    try:
        user, token = await create_admin_user(
            _db(request), settings.key_pepper, email=body.email, role=body.role, created_by=admin.id
        )
    except IntegrityError as exc:
        raise HTTPException(409, "an admin with this email already exists") from exc
    # The plaintext token is returned exactly once and never stored.
    return {**_admin_out(user), "token": token}


@router.get("/admins", dependencies=[Depends(require(Permission.ADMINS_MANAGE))])
async def list_admins(request: Request) -> list[dict[str, Any]]:
    async with _db(request).session() as s:
        rows = (await s.execute(select(AdminUser).order_by(AdminUser.created_at))).scalars().all()
    return [_admin_out(u) for u in rows]


@router.delete("/admins/{admin_id}", dependencies=[Depends(require(Permission.ADMINS_MANAGE))])
async def disable_admin(admin_id: str, request: Request) -> dict[str, Any]:
    try:
        user = await disable_admin_user(_db(request), admin_id)
    except LastOwnerError as exc:
        raise HTTPException(409, str(exc)) from exc
    if user is None:
        raise HTTPException(404, "admin not found")
    return _admin_out(user)
