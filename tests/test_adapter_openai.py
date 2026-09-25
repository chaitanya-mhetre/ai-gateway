from __future__ import annotations

import json

import httpx
import pytest
import respx

from ai_gateway.errors import (
    CONTEXT_LENGTH,
    HTTP_5XX,
    HTTP_429,
    INVALID_REQUEST,
    TIMEOUT,
    ProviderError,
)
from ai_gateway.models import ChatRequest, EmbeddingRequest, Message, ToolSpec
from ai_gateway.providers.openai import OpenAIProvider, build_payload

BASE = "https://api.openai.test/v1"


def req(**kw: object) -> ChatRequest:
    return ChatRequest.model_validate(
        {"model": "x", "messages": [{"role": "user", "content": "hi"}], **kw}
    )


@pytest.fixture
def provider() -> OpenAIProvider:
    return OpenAIProvider("openai", BASE, "sk-test")


def test_payload_uses_max_completion_tokens_and_stream_usage() -> None:
    p = build_payload(req(max_tokens=10, temperature=0), "gpt-x", stream=True)
    assert p["max_completion_tokens"] == 10
    assert "max_tokens" not in p
    assert p["stream_options"] == {"include_usage": True}


@respx.mock
async def test_chat_parses_text_and_usage(provider: OpenAIProvider) -> None:
    route = respx.post(f"{BASE}/chat/completions").respond(
        json={
            "id": "c1",
            "model": "gpt-x",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 2},
            },
        }
    )
    resp = await provider.chat(req(), "gpt-x", 5)
    assert resp.content == "hello"
    assert resp.usage.prompt_tokens == 5 and resp.usage.cached_tokens == 2
    assert route.calls.last.request.headers["authorization"] == "Bearer sk-test"


@respx.mock
async def test_chat_parses_tool_calls(provider: OpenAIProvider) -> None:
    respx.post(f"{BASE}/chat/completions").respond(
        json={
            "id": "c1",
            "model": "gpt-x",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city":"Pune"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )
    resp = await provider.chat(req(tools=[ToolSpec(name="get_weather")]), "gpt-x", 5)
    assert resp.finish_reason == "tool_calls"
    assert resp.tool_calls[0].name == "get_weather"
    assert json.loads(resp.tool_calls[0].arguments) == {"city": "Pune"}


@pytest.mark.parametrize(
    ("status", "body", "headers", "reason", "retryable"),
    [
        (429, "slow down", {"retry-after": "7"}, HTTP_429, True),
        (503, "overloaded", {}, HTTP_5XX, True),
        (400, "This model's maximum context length is 8192 tokens", {}, CONTEXT_LENGTH, False),
        (400, "bad param", {}, INVALID_REQUEST, False),
    ],
)
@respx.mock
async def test_http_errors_are_classified(
    provider: OpenAIProvider,
    status: int,
    body: str,
    headers: dict[str, str],
    reason: str,
    retryable: bool,
) -> None:
    respx.post(f"{BASE}/chat/completions").respond(status, text=body, headers=headers)
    with pytest.raises(ProviderError) as exc:
        await provider.chat(req(), "gpt-x", 5)
    assert exc.value.reason == reason
    assert exc.value.retryable is retryable
    if status == 429:
        assert exc.value.retry_after == 7.0


@respx.mock
async def test_timeout_is_retryable(provider: OpenAIProvider) -> None:
    respx.post(f"{BASE}/chat/completions").mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(ProviderError) as exc:
        await provider.chat(req(), "gpt-x", 5)
    assert exc.value.reason == TIMEOUT and exc.value.retryable


SSE_TOOL_STREAM = (
    'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"f","arguments":""}}]},"finish_reason":null}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"a\\":1}"}}]},"finish_reason":null}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n'
    'data: {"choices":[],"usage":{"prompt_tokens":9,"completion_tokens":4}}\n\n'
    "data: [DONE]\n\n"
)


@respx.mock
async def test_stream_normalises_chunks(provider: OpenAIProvider) -> None:
    respx.post(f"{BASE}/chat/completions").respond(
        200, content=SSE_TOOL_STREAM.encode(), headers={"content-type": "text/event-stream"}
    )
    chunks = [c async for c in provider.stream(req(stream=True), "gpt-x", 5)]
    assert chunks[0].content == "Hi"
    assert chunks[1].tool_calls[0].id == "call_1" and chunks[1].tool_calls[0].name == "f"
    assert chunks[2].tool_calls[0].arguments == '{"a":1}'
    assert chunks[3].finish_reason == "tool_calls"
    assert chunks[4].usage is not None and chunks[4].usage.completion_tokens == 4


@respx.mock
async def test_stream_error_before_first_chunk_raises(provider: OpenAIProvider) -> None:
    respx.post(f"{BASE}/chat/completions").respond(500, text="boom")
    with pytest.raises(ProviderError) as exc:
        _ = [c async for c in provider.stream(req(stream=True), "gpt-x", 5)]
    assert exc.value.reason == HTTP_5XX


@respx.mock
async def test_embeddings(provider: OpenAIProvider) -> None:
    respx.post(f"{BASE}/embeddings").respond(
        json={
            "data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}],
            "usage": {"prompt_tokens": 4},
        }
    )
    resp = await provider.embed(EmbeddingRequest(model="e", input=["a", "b"]), "e", 5)
    assert resp.vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert resp.usage.prompt_tokens == 4


def test_message_builder_keeps_tool_fields() -> None:
    from ai_gateway.models import ToolCall
    from ai_gateway.providers.openai import to_openai_message

    m = Message(role="assistant", tool_calls=[ToolCall(id="c", name="f", arguments="{}")])
    assert to_openai_message(m)["tool_calls"][0]["function"]["name"] == "f"
