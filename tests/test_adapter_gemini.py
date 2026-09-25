from __future__ import annotations

import json

import pytest
import respx

from ai_gateway.errors import REFUSAL, ProviderError
from ai_gateway.models import ChatRequest, EmbeddingRequest, Message, ToolCall, ToolSpec
from ai_gateway.providers.gemini import GeminiProvider, build_payload
from tests.fixture_utils import load_bytes, load_json

BASE = "https://gemini.test/v1beta"


def simple() -> ChatRequest:
    return ChatRequest(
        model="x",
        messages=[Message(role="system", content="sys"), Message(role="user", content="hi")],
        max_tokens=64,
        temperature=0,
    )


def test_payload_translation_with_tool_round_trip() -> None:
    req = ChatRequest(
        model="x",
        messages=[
            Message(role="user", content="weather?"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="c1", name="get_weather", arguments='{"city":"Pune"}')],
            ),
            Message(role="tool", tool_call_id="c1", content="31C"),
        ],
        tools=[ToolSpec(name="get_weather")],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    p = build_payload(req)
    assert [c["role"] for c in p["contents"]] == ["user", "model", "user"]
    assert p["contents"][1]["parts"][0] == {
        "functionCall": {"name": "get_weather", "args": {"city": "Pune"}}
    }
    # functionResponse references the function NAME, recovered from the earlier call id
    assert p["contents"][2]["parts"][0]["functionResponse"]["name"] == "get_weather"
    assert p["toolConfig"] == {
        "functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": ["get_weather"]}
    }


def test_system_and_generation_config() -> None:
    p = build_payload(simple())
    assert p["systemInstruction"] == {"parts": [{"text": "sys"}]}
    assert p["generationConfig"] == {"temperature": 0, "maxOutputTokens": 64}


@respx.mock
async def test_text_response() -> None:
    route = respx.post(f"{BASE}/models/gemini-test:generateContent").respond(
        json=load_json("gemini_text.json")
    )
    resp = await GeminiProvider("gemini", BASE, "g-key").chat(simple(), "gemini-test", 5)
    assert resp.content == "Hello there"
    assert resp.usage.prompt_tokens == 8 and resp.usage.cached_tokens == 3
    assert route.calls.last.request.headers["x-goog-api-key"] == "g-key"


@respx.mock
async def test_function_call_gets_generated_id() -> None:
    respx.post(f"{BASE}/models/gemini-test:generateContent").respond(
        json=load_json("gemini_tool.json")
    )
    resp = await GeminiProvider("gemini", BASE, "k").chat(simple(), "gemini-test", 5)
    assert resp.finish_reason == "tool_calls"
    assert resp.tool_calls[0].id.startswith("call_")
    assert json.loads(resp.tool_calls[0].arguments) == {"city": "Pune"}


@respx.mock
async def test_blocked_prompt_is_non_retryable_refusal() -> None:
    respx.post(f"{BASE}/models/gemini-test:generateContent").respond(
        json=load_json("gemini_blocked.json")
    )
    with pytest.raises(ProviderError) as exc:
        await GeminiProvider("gemini", BASE, "k").chat(simple(), "gemini-test", 5)
    assert exc.value.reason == REFUSAL and not exc.value.retryable


@respx.mock
async def test_stream_uses_alt_sse_and_last_usage_wins() -> None:
    route = respx.post(f"{BASE}/models/gemini-test:streamGenerateContent").respond(
        200, content=load_bytes("gemini_stream.sse"), headers={"content-type": "text/event-stream"}
    )
    chunks = [
        c async for c in GeminiProvider("gemini", BASE, "k").stream(simple(), "gemini-test", 5)
    ]
    assert route.calls.last.request.url.params["alt"] == "sse"
    assert "".join(c.content or "" for c in chunks) == "Hello"
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].usage is not None and chunks[-1].usage.completion_tokens == 2


@respx.mock
async def test_batch_embeddings_flag_estimated_usage() -> None:
    respx.post(f"{BASE}/models/embed-test:batchEmbedContents").respond(
        json={"embeddings": [{"values": [0.1, 0.2]}, {"values": [0.3, 0.4]}]}
    )
    resp = await GeminiProvider("gemini", BASE, "k").embed(
        EmbeddingRequest(model="e", input=["a", "b"]), "embed-test", 5
    )
    assert resp.vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert resp.usage.estimated
