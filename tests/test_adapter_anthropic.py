from __future__ import annotations

import json

import pytest
import respx

from ai_gateway.errors import HTTP_5XX, HTTP_429, ProviderError
from ai_gateway.models import ChatRequest, Message, ToolCall, ToolSpec
from ai_gateway.providers.anthropic import AnthropicProvider, build_payload
from tests.fixture_utils import load_bytes, load_json

BASE = "https://api.anthropic.test"
URL = f"{BASE}/v1/messages"


def conversation() -> ChatRequest:
    return ChatRequest(
        model="x",
        messages=[
            Message(role="system", content="be brief"),
            Message(role="user", content="weather in Pune and Mumbai?"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    ToolCall(id="t1", name="get_weather", arguments='{"city":"Pune"}'),
                    ToolCall(id="t2", name="get_weather", arguments='{"city":"Mumbai"}'),
                ],
            ),
            Message(role="tool", tool_call_id="t1", content="31C"),
            Message(role="tool", tool_call_id="t2", content="29C"),
        ],
        tools=[
            ToolSpec(
                name="get_weather",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            )
        ],
        tool_choice="required",
        temperature=1.5,
    )


def test_payload_translation() -> None:
    p = build_payload(conversation(), "claude-test", stream=False)
    assert p["system"] == "be brief"
    assert p["max_tokens"] == 1024  # required by Anthropic; default applied
    assert p["temperature"] == 1.0  # clamped to Anthropic's 0..1 range
    assert p["tool_choice"] == {"type": "any"}
    assert p["tools"][0]["input_schema"]["properties"]["city"]["type"] == "string"
    roles = [m["role"] for m in p["messages"]]
    assert roles == ["user", "assistant", "user"]  # two tool results merged into ONE user turn
    assert p["messages"][1]["content"][0] == {
        "type": "tool_use",
        "id": "t1",
        "name": "get_weather",
        "input": {"city": "Pune"},
    }
    assert [b["tool_use_id"] for b in p["messages"][2]["content"]] == ["t1", "t2"]


@respx.mock
async def test_text_response_and_usage_includes_cache_reads() -> None:
    route = respx.post(URL).respond(json=load_json("anthropic_text.json"))
    resp = await AnthropicProvider("anthropic", BASE, "k").chat(
        ChatRequest(model="x", messages=[Message(role="user", content="hi")]), "claude-test", 5
    )
    assert resp.content == "Hello!"
    assert resp.usage.prompt_tokens == 14 and resp.usage.cached_tokens == 4
    headers = route.calls.last.request.headers
    assert headers["x-api-key"] == "k" and headers["anthropic-version"] == "2023-06-01"


@respx.mock
async def test_tool_use_response() -> None:
    respx.post(URL).respond(json=load_json("anthropic_tool.json"))
    resp = await AnthropicProvider("anthropic", BASE, "k").chat(conversation(), "claude-test", 5)
    assert resp.finish_reason == "tool_calls"
    assert resp.content == "Checking."
    assert resp.tool_calls[0].id == "toolu_1"
    assert json.loads(resp.tool_calls[0].arguments) == {"city": "Pune"}


@respx.mock
async def test_stream_text_then_tool_call() -> None:
    respx.post(URL).respond(
        200,
        content=load_bytes("anthropic_stream.sse"),
        headers={"content-type": "text/event-stream"},
    )
    chunks = [
        c
        async for c in AnthropicProvider("anthropic", BASE, "k").stream(
            conversation(), "claude-test", 5
        )
    ]
    text = "".join(c.content or "" for c in chunks)
    assert text == "Let me check."
    tool_deltas = [d for c in chunks for d in c.tool_calls]
    assert (
        tool_deltas[0].id == "toolu_9"
        and tool_deltas[0].name == "get_weather"
        and tool_deltas[0].index == 0
    )
    assert json.loads("".join(d.arguments for d in tool_deltas)) == {"city": "Pune"}
    final = chunks[-1]
    assert final.finish_reason == "tool_calls"
    assert (
        final.usage is not None
        and final.usage.prompt_tokens == 12
        and final.usage.completion_tokens == 22
    )


@respx.mock
async def test_in_stream_overloaded_error_is_retryable() -> None:
    respx.post(URL).respond(
        200,
        content=load_bytes("anthropic_stream_error.sse"),
        headers={"content-type": "text/event-stream"},
    )
    with pytest.raises(ProviderError) as exc:
        _ = [
            c
            async for c in AnthropicProvider("anthropic", BASE, "k").stream(
                conversation(), "claude-test", 5
            )
        ]
    assert exc.value.reason == HTTP_5XX and exc.value.retryable


@respx.mock
async def test_http_429_with_retry_after() -> None:
    respx.post(URL).respond(
        429,
        json={"type": "error", "error": {"type": "rate_limit_error"}},
        headers={"retry-after": "3"},
    )
    with pytest.raises(ProviderError) as exc:
        await AnthropicProvider("anthropic", BASE, "k").chat(conversation(), "claude-test", 5)
    assert exc.value.reason == HTTP_429 and exc.value.retry_after == 3.0
