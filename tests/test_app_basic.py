from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import openai
import pytest

from ai_gateway.app import create_app
from ai_gateway.config import Settings
from ai_gateway.providers.base import Provider
from tests.conftest import app_client, make_config


async def test_chat_completion_via_alias(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "chat-default", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "Hello from the mock provider."
    assert r.headers["x-gateway-provider"] == "primary"
    assert r.headers["x-gateway-model"] == "p-model"


async def test_explicit_provider_model(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "secondary:any-model", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    assert r.headers["x-gateway-provider"] == "secondary"


async def test_unknown_model_is_404_in_openai_error_shape(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "model_not_found"


async def test_validation_error_is_400(client: httpx.AsyncClient) -> None:
    r = await client.post("/v1/chat/completions", json={"model": "chat-default", "messages": []})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


async def test_models_endpoint(client: httpx.AsyncClient) -> None:
    r = await client.get("/v1/models")
    assert {m["id"] for m in r.json()["data"]} == {"chat-default", "embed-default"}


async def test_embeddings_endpoint(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/v1/embeddings", json={"model": "embed-default", "input": ["hello world", "bye"]}
    )
    assert r.status_code == 200
    assert len(r.json()["data"]) == 2


@pytest.fixture
async def sdk(
    settings: Settings, providers: dict[str, Provider]
) -> AsyncIterator[openai.AsyncOpenAI]:
    app = create_app(settings, config=make_config(), providers=providers)
    await app.state.db.migrate()
    http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))
    # The SDK types its client against its vendored httpx fork; the stdlib httpx client is
    # runtime-compatible (duck-typed), which is all this in-process test needs.
    yield openai.AsyncOpenAI(base_url="http://gw/v1", api_key="unused", http_client=http_client)  # type: ignore[arg-type]
    await http_client.aclose()


async def test_official_openai_sdk_non_streaming(sdk: openai.AsyncOpenAI) -> None:
    resp = await sdk.chat.completions.create(
        model="chat-default", messages=[{"role": "user", "content": "hi"}]
    )
    assert resp.choices[0].message.content == "Hello from the mock provider."
    assert resp.usage is not None and resp.usage.total_tokens > 0


async def test_official_openai_sdk_streaming(sdk: openai.AsyncOpenAI) -> None:
    stream = await sdk.chat.completions.create(
        model="chat-default", messages=[{"role": "user", "content": "hi"}], stream=True
    )
    text = ""
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            text += chunk.choices[0].delta.content
    assert text == "Hello from the mock provider."


async def test_official_openai_sdk_embeddings(sdk: openai.AsyncOpenAI) -> None:
    resp = await sdk.embeddings.create(model="embed-default", input="hello")
    assert len(resp.data[0].embedding) == 16


async def test_invalid_policy_header_is_400(client: httpx.AsyncClient) -> None:
    r = await client.post(
        "/v1/chat/completions",
        json={"model": "chat-default", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Gateway-Route-Policy": "fastest"},
    )
    assert r.status_code == 400


async def test_fallback_visible_in_headers(settings: Settings) -> None:
    from ai_gateway.providers.mock import MockProvider, http_5xx

    providers: dict[str, Provider] = {
        "primary": MockProvider("primary", script=[http_5xx()]),
        "secondary": MockProvider("secondary"),
        "local": MockProvider("local"),
    }
    app = create_app(settings, config=make_config(), providers=providers)
    async with app_client(app) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={"model": "chat-default", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 200
    assert r.headers["x-gateway-provider"] == "secondary"
    assert r.headers["x-gateway-attempts"] == "2"


async def test_mid_stream_failure_becomes_in_band_error_event(settings: Settings) -> None:
    from ai_gateway.providers.mock import MockProvider

    providers: dict[str, Provider] = {
        "primary": MockProvider("primary", reply="abcdefghijkl", chunk_size=4, stream_fail_after=1),
        "secondary": MockProvider("secondary"),
        "local": MockProvider("local"),
    }
    app = create_app(settings, config=make_config(), providers=providers)
    async with app_client(app) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={
                "model": "chat-default",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert r.status_code == 200
    events = [line for line in r.text.split("\n\n") if line]
    assert '"content":"abcd"' in events[0]
    assert '"type": "provider_error"' in events[-2]
    assert events[-1] == "data: [DONE]"


async def test_all_targets_failing_returns_502(settings: Settings) -> None:
    from ai_gateway.providers.mock import MockProvider, http_5xx

    providers: dict[str, Provider] = {
        "primary": MockProvider("primary", script=[http_5xx()]),
        "secondary": MockProvider("secondary", script=[http_5xx()]),
        "local": MockProvider("local"),
    }
    app = create_app(settings, config=make_config(), providers=providers)
    async with app_client(app) as c:
        r = await c.post(
            "/v1/chat/completions",
            json={"model": "chat-default", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert r.status_code == 502
    assert r.json()["error"]["type"] == "all_targets_failed"
