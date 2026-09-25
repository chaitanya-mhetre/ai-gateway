from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from redis.asyncio import Redis

from ai_gateway.app import create_app
from ai_gateway.cache.exact import (
    InMemoryCacheStore,
    RedisCacheStore,
    cache_key,
    exact_cache_allowed,
)
from ai_gateway.cache.semantic import SemanticCache, cosine
from ai_gateway.config import Settings
from ai_gateway.models import ChatRequest, ChatResponse, Message, ToolSpec
from ai_gateway.providers.base import Provider
from ai_gateway.providers.mock import MockProvider, fake_embedding
from ai_gateway.redis_client import make_redis
from tests.conftest import ADMIN, app_client, make_config


def req(content: str = "What is 2+2?", **kw: Any) -> ChatRequest:
    return ChatRequest(model="chat-default", messages=[Message(role="user", content=content)], **kw)


def resp(text: str = "4") -> ChatResponse:
    return ChatResponse(id="r", provider="p", model="m", content=text)


# --- unit ---------------------------------------------------------------------------------------


def test_cache_key_is_deterministic_and_tenant_scoped() -> None:
    assert cache_key("t1", req(temperature=0)) == cache_key("t1", req(temperature=0))
    assert cache_key("t1", req(temperature=0)) != cache_key("t2", req(temperature=0))
    assert cache_key("t1", req(temperature=0)) != cache_key("t1", req(temperature=0.5))
    assert cache_key("t1", req("a")) != cache_key("t1", req("b"))


def test_cache_eligibility_rules() -> None:
    assert exact_cache_allowed(req(temperature=0), None)
    assert not exact_cache_allowed(req(temperature=0.7), None)
    assert not exact_cache_allowed(req(temperature=0, tools=[ToolSpec(name="f")]), None)
    assert exact_cache_allowed(req(temperature=0.7), "exact")
    assert not exact_cache_allowed(req(temperature=0), "bypass")


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def test_memory_store_ttl_and_lru() -> None:
    clock = Clock()
    store = InMemoryCacheStore(max_entries=2, clock=clock)
    await store.set("a", resp("A"), ttl_s=10)
    await store.set("b", resp("B"), ttl_s=10)
    assert await store.get("a") is not None  # touch a → b becomes least recently used
    await store.set("c", resp("C"), ttl_s=10)
    assert await store.get("b") is None
    clock.now = 11
    assert await store.get("a") is None  # expired


def test_cosine() -> None:
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 1]) == 0.0


async def test_semantic_threshold_and_tenant_isolation() -> None:
    async def embed(text: str) -> list[float]:
        return fake_embedding(text, 64)

    cache = SemanticCache(embed, threshold=0.9)
    original = req("how do I reset my password")
    _, vec = await cache.lookup("t1", original)
    cache.store("t1", original, vec, resp("Go to settings."))
    hit, _ = await cache.lookup("t1", req("how do I reset my password?"))
    assert hit is not None and hit.similarity >= 0.9
    miss, _ = await cache.lookup("t1", req("what is the refund policy for india"))
    assert miss is None
    other_tenant, _ = await cache.lookup("t2", req("how do I reset my password"))
    assert other_tenant is None


REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:56382/0")


@pytest.fixture
async def redis() -> AsyncIterator[Redis]:
    client: Redis = make_redis(REDIS_URL)
    try:
        await client.ping()
    except Exception:
        await client.aclose()
        pytest.skip(f"Redis not reachable at {REDIS_URL} (run `make up`)")
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()


@pytest.mark.redis
async def test_redis_store_round_trip(redis: Redis) -> None:
    store = RedisCacheStore(redis)
    await store.set("k", resp("cached"), ttl_s=5)
    got = await store.get("k")
    assert got is not None and got.content == "cached"
    assert 0 < await redis.ttl("gw:cache:k") <= 5


# --- through the HTTP API -----------------------------------------------------------------------


@pytest.fixture
def mocks() -> dict[str, MockProvider]:
    return {
        "primary": MockProvider("primary"),
        "secondary": MockProvider("secondary"),
        "local": MockProvider("local", embedding_dim=64),
    }


@pytest.fixture
async def gw(mocks: dict[str, MockProvider]) -> AsyncIterator[httpx.AsyncClient]:
    config = make_config()
    config.cache.semantic_enabled = True
    config.cache.semantic_threshold = 0.9
    settings = Settings(
        auth_enabled=True,
        key_pepper="pep",
        redis_url=None,
        database_url="sqlite+aiosqlite:///:memory:",
    )
    providers: dict[str, Provider] = dict(mocks)
    async with app_client(create_app(settings, config=config, providers=providers)) as c:
        yield c


async def key(c: httpx.AsyncClient, tenant: str) -> dict[str, str]:
    t = (await c.post("/admin/v1/tenants", json={"name": tenant}, headers=ADMIN)).json()
    p = (
        await c.post("/admin/v1/projects", json={"tenant_id": t["id"], "name": "x"}, headers=ADMIN)
    ).json()
    k = (
        await c.post("/admin/v1/keys", json={"project_id": p["id"], "name": "k"}, headers=ADMIN)
    ).json()
    return {"Authorization": f"Bearer {k['key']}"}


def body(content: str = "What is 2+2?", **kw: Any) -> dict[str, Any]:
    return {"model": "chat-default", "messages": [{"role": "user", "content": content}], **kw}


async def test_exact_hit_skips_provider(
    gw: httpx.AsyncClient, mocks: dict[str, MockProvider]
) -> None:
    auth = await key(gw, "acme")
    first = await gw.post("/v1/chat/completions", json=body(temperature=0), headers=auth)
    second = await gw.post("/v1/chat/completions", json=body(temperature=0), headers=auth)
    assert first.headers["x-gateway-cache"] == "miss"
    assert second.headers["x-gateway-cache"] == "exact"
    assert (
        second.json()["choices"][0]["message"]["content"]
        == first.json()["choices"][0]["message"]["content"]
    )
    assert mocks["primary"].calls == 1


async def test_bypass_header(gw: httpx.AsyncClient, mocks: dict[str, MockProvider]) -> None:
    auth = await key(gw, "acme")
    await gw.post("/v1/chat/completions", json=body(temperature=0), headers=auth)
    r = await gw.post(
        "/v1/chat/completions",
        json=body(temperature=0),
        headers={**auth, "X-Gateway-Cache": "bypass"},
    )
    assert r.headers["x-gateway-cache"] == "miss" and mocks["primary"].calls == 2


async def test_no_cross_tenant_hits(gw: httpx.AsyncClient, mocks: dict[str, MockProvider]) -> None:
    await gw.post("/v1/chat/completions", json=body(temperature=0), headers=await key(gw, "acme"))
    r = await gw.post(
        "/v1/chat/completions", json=body(temperature=0), headers=await key(gw, "globex")
    )
    assert r.headers["x-gateway-cache"] == "miss" and mocks["primary"].calls == 2


async def test_stream_request_served_from_cache(
    gw: httpx.AsyncClient, mocks: dict[str, MockProvider]
) -> None:
    auth = await key(gw, "acme")
    await gw.post("/v1/chat/completions", json=body(temperature=0), headers=auth)
    r = await gw.post("/v1/chat/completions", json=body(temperature=0, stream=True), headers=auth)
    assert r.headers["x-gateway-cache"] == "exact"
    assert "Hello from the mock provider." in r.text and r.text.rstrip().endswith("data: [DONE]")
    assert mocks["primary"].calls == 1


async def test_semantic_cache_opt_in(gw: httpx.AsyncClient, mocks: dict[str, MockProvider]) -> None:
    auth = {**(await key(gw, "acme")), "X-Gateway-Cache": "semantic"}
    await gw.post(
        "/v1/chat/completions",
        json=body("how do I reset my password", temperature=0.7),
        headers=auth,
    )
    near = await gw.post(
        "/v1/chat/completions",
        json=body("How do I reset my password?", temperature=0.7),
        headers=auth,
    )
    assert near.headers["x-gateway-cache"] == "semantic"
    assert float(near.headers["x-gateway-cache-similarity"]) >= 0.9
    far = await gw.post(
        "/v1/chat/completions",
        json=body("cancel my subscription today", temperature=0.7),
        headers=auth,
    )
    assert far.headers["x-gateway-cache"] == "miss"
    assert mocks["primary"].calls == 2
    metrics = (await gw.get("/metrics")).text
    assert 'gateway_cache_requests_total{result="hit",type="semantic"} 1.0' in metrics


async def test_without_opt_in_semantic_is_not_used(
    gw: httpx.AsyncClient, mocks: dict[str, MockProvider]
) -> None:
    auth = await key(gw, "acme")
    await gw.post(
        "/v1/chat/completions",
        json=body("how do I reset my password", temperature=0.7),
        headers=auth,
    )
    r = await gw.post(
        "/v1/chat/completions",
        json=body("How do I reset my password?", temperature=0.7),
        headers=auth,
    )
    assert r.headers["x-gateway-cache"] == "miss"
