"""Control-plane API: tenants, projects, keys, provider health.

Protected by a separate admin bearer token (`GATEWAY_ADMIN_TOKEN`), so data-plane keys can never
reach it. A production system would use SSO/JWT with roles; see README "Limitations".
"""

from __future__ import annotations

import hmac
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ai_gateway.auth import KeyAuthenticator, generate_key, hash_key
from ai_gateway.config import Settings
from ai_gateway.db import ApiKey, Database, Project, Tenant, utcnow
from ai_gateway.gateway import Gateway


def require_admin(request: Request, authorization: str | None = Header(default=None)) -> None:
    settings: Settings = request.app.state.settings
    token = (
        authorization[7:] if authorization and authorization.lower().startswith("bearer ") else ""
    )
    if not token or not hmac.compare_digest(token, settings.admin_token):
        raise HTTPException(status_code=401, detail="admin token required")


router = APIRouter(prefix="/admin/v1", dependencies=[Depends(require_admin)], tags=["admin"])


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


def _db(request: Request) -> Database:
    db: Database = request.app.state.db
    return db


@router.post("/tenants", status_code=201)
async def create_tenant(body: TenantIn, request: Request) -> dict[str, Any]:
    async with _db(request).session() as s:
        tenant = Tenant(name=body.name)
        s.add(tenant)
        try:
            await s.commit()
        except IntegrityError as exc:
            raise HTTPException(409, "tenant name already exists") from exc
        return {"id": tenant.id, "name": tenant.name}


@router.post("/projects", status_code=201)
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


@router.post("/keys", status_code=201)
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


@router.get("/keys")
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


@router.delete("/keys/{key_id}")
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


@router.get("/providers/health")
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
