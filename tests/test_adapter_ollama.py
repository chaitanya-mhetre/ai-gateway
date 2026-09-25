from __future__ import annotations

import json

import respx

from ai_gateway.models import ChatRequest, EmbeddingRequest, Message, ToolSpec
from ai_gateway.providers.ollama import OllamaProvider, build_payload
from tests.fixture_utils import load_bytes, load_json

BASE = "http://ollama.test:11434"


def req(**kw: object) -> ChatRequest:
    return ChatRequest.model_validate(
        {"model": "x", "messages": [{"role": "user", "content": "hi"}], **kw}
    )


def test_options_mapping() -> None:
    p = build_payload(req(max_tokens=20, temperature=0.2, stop=["\n"]), "llama-test", stream=False)
    assert p["options"] == {"temperature": 0.2, "num_predict": 20, "stop": ["\n"]}
    assert p["stream"] is False


@respx.mock
async def test_text() -> None:
    respx.post(f"{BASE}/api/chat").respond(json=load_json("ollama_text.json"))
    resp = await OllamaProvider("ollama", BASE).chat(req(), "llama-test", 5)
    assert resp.content == "Hi!"
    assert resp.usage.prompt_tokens == 7 and resp.usage.completion_tokens == 2


@respx.mock
async def test_tool_call_arguments_object_becomes_json_string() -> None:
    respx.post(f"{BASE}/api/chat").respond(json=load_json("ollama_tool.json"))
    resp = await OllamaProvider("ollama", BASE).chat(
        req(tools=[ToolSpec(name="get_weather")]), "llama-test", 5
    )
    assert resp.finish_reason == "tool_calls"
    assert json.loads(resp.tool_calls[0].arguments) == {"city": "Pune"}


@respx.mock
async def test_ndjson_stream() -> None:
    respx.post(f"{BASE}/api/chat").respond(200, content=load_bytes("ollama_stream.ndjson"))
    chunks = [
        c async for c in OllamaProvider("ollama", BASE).stream(req(stream=True), "llama-test", 5)
    ]
    assert "".join(c.content or "" for c in chunks) == "Hello"
    assert chunks[-1].finish_reason == "length"
    assert chunks[-1].usage is not None and chunks[-1].usage.completion_tokens == 2


@respx.mock
async def test_embed() -> None:
    respx.post(f"{BASE}/api/embed").respond(
        json={"embeddings": [[1.0, 0.0]], "prompt_eval_count": 3}
    )
    resp = await OllamaProvider("ollama", BASE).embed(
        EmbeddingRequest(model="e", input=["a"]), "nomic-test", 5
    )
    assert resp.vectors == [[1.0, 0.0]] and resp.usage.prompt_tokens == 3


def test_tool_message_carries_name() -> None:
    p = build_payload(
        ChatRequest(model="x", messages=[Message(role="tool", name="get_weather", content="31C")]),
        "m",
        stream=False,
    )
    assert p["messages"][0]["tool_name"] == "get_weather"
