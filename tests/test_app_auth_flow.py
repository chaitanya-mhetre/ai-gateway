"""End-to-end control plane + data plane: admin creates a key, clients use it."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from ai_gateway.app import create_app
from ai_gateway.config import Settings
from ai_gateway.providers.base import Provider
from tests.conftest import app_client, make_config

ADMIN = {"Authorization": "Bearer admin-secret"}
CHAT = {"model": "chat-default", "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
async def gw(providers: dict[str, Provider]) -> AsyncIterator[httpx.AsyncClient]:
    settings = Settings(
        auth_enabled=True,
        admin_token="admin-secret",
        key_pepper="pep",
        redis_url=None,
        database_url="sqlite+aiosqlite:///:memory:",
    )
    async with app_client(create_app(settings, config=make_config(), providers=providers)) as c:
        yield c


async def make_key(c: httpx.AsyncClient, **key_fields: Any) -> dict[str, Any]:
    t = (await c.post("/admin/v1/tenants", json={"name": "acme"}, headers=ADMIN)).json()
    p = (
        await c.post(
            "/admin/v1/projects", json={"tenant_id": t["id"], "name": "search"}, headers=ADMIN
        )
    ).json()
    r = await c.post(
        "/admin/v1/keys",
        json={"project_id": p["id"], "name": "backend", **key_fields},
        headers=ADMIN,
    )
    assert r.status_code == 201
    key: dict[str, Any] = r.json()
    return key


async def test_admin_requires_token(gw: httpx.AsyncClient) -> None:
    assert (await gw.post("/admin/v1/tenants", json={"name": "x"})).status_code == 401
    assert (
        await gw.post(
            "/admin/v1/tenants", json={"name": "x"}, headers={"Authorization": "Bearer nope"}
        )
    ).status_code == 401


async def test_data_plane_requires_key(gw: httpx.AsyncClient) -> None:
    r = await gw.post("/v1/chat/completions", json=CHAT)
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "invalid_api_key"


async def test_key_works_and_plaintext_is_not_listed(gw: httpx.AsyncClient) -> None:
    key = await make_key(gw)
    r = await gw.post(
        "/v1/chat/completions", json=CHAT, headers={"Authorization": f"Bearer {key['key']}"}
    )
    assert r.status_code == 200
    listed = (await gw.get(f"/admin/v1/keys?project_id={key['project_id']}", headers=ADMIN)).json()
    assert listed[0]["prefix"] == key["prefix"]
    assert "key" not in listed[0] and "key_hash" not in listed[0]


async def test_alias_allow_list(gw: httpx.AsyncClient) -> None:
    key = await make_key(gw, allowed_aliases=["embed-default"])
    auth = {"Authorization": f"Bearer {key['key']}"}
    assert (await gw.post("/v1/chat/completions", json=CHAT, headers=auth)).status_code == 403
    models = (await gw.get("/v1/models", headers=auth)).json()["data"]
    assert [m["id"] for m in models] == ["embed-default"]


async def test_rpm_limit_returns_429_with_retry_after(gw: httpx.AsyncClient) -> None:
    key = await make_key(gw, rpm_limit=2)
    auth = {"Authorization": f"Bearer {key['key']}"}
    codes = [
        (await gw.post("/v1/chat/completions", json=CHAT, headers=auth)).status_code
        for _ in range(3)
    ]
    assert codes == [200, 200, 429]
    r = await gw.post("/v1/chat/completions", json=CHAT, headers=auth)
    assert "retry-after" in r.headers


async def test_revocation_is_immediate_on_same_replica(gw: httpx.AsyncClient) -> None:
    key = await make_key(gw)
    auth = {"Authorization": f"Bearer {key['key']}"}
    assert (await gw.post("/v1/chat/completions", json=CHAT, headers=auth)).status_code == 200
    assert (await gw.delete(f"/admin/v1/keys/{key['id']}", headers=ADMIN)).status_code == 200
    assert (await gw.post("/v1/chat/completions", json=CHAT, headers=auth)).status_code == 401


async def test_provider_health_endpoint(gw: httpx.AsyncClient) -> None:
    key = await make_key(gw)
    await gw.post(
        "/v1/chat/completions", json=CHAT, headers={"Authorization": f"Bearer {key['key']}"}
    )
    health = (await gw.get("/admin/v1/providers/health", headers=ADMIN)).json()
    assert health["primary"]["circuit"] == "closed"
    assert "primary:p-model" in health["primary"]["ewma_latency_s"]


async def test_oversized_body_rejected(gw: httpx.AsyncClient) -> None:
    big = {"model": "chat-default", "messages": [{"role": "user", "content": "x" * 1_100_000}]}
    assert (await gw.post("/v1/chat/completions", json=big)).status_code == 413
