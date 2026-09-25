from __future__ import annotations

import json

import pytest

from ai_gateway.errors import InvalidRequestError
from ai_gateway.models import StreamChunk, ToolCallDelta, Usage
from ai_gateway.openai_compat import ChunkEncoder, OpenAIChatRequest, to_internal


def parse(body: dict[str, object]) -> OpenAIChatRequest:
    return OpenAIChatRequest.model_validate(body)


def test_developer_role_and_text_parts_are_normalised() -> None:
    req = to_internal(
        parse(
            {
                "model": "chat-default",
                "messages": [
                    {"role": "developer", "content": "be brief"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "hi "},
                            {"type": "text", "text": "there"},
                        ],
                    },
                ],
                "max_completion_tokens": 50,
                "stop": "END",
            }
        ),
        max_tokens_cap=1000,
    )
    assert req.messages[0].role == "system"
    assert req.messages[1].content == "hi there"
    assert req.max_tokens == 50
    assert req.stop == ["END"]
    assert req.system_prompt == "be brief"


def test_image_parts_rejected() -> None:
    body = parse(
        {
            "model": "m",
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}],
        }
    )
    with pytest.raises(InvalidRequestError):
        to_internal(body, max_tokens_cap=1000)


def test_max_tokens_cap_enforced() -> None:
    with pytest.raises(InvalidRequestError):
        to_internal(
            parse(
                {"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 5000}
            ),
            max_tokens_cap=1000,
        )


def test_tool_calls_round_trip_into_internal() -> None:
    req = to_internal(
        parse(
            {
                "model": "m",
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city":"Pune"}'},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "c1", "content": "31C"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "parameters": {"type": "object"}},
                    }
                ],
            }
        ),
        max_tokens_cap=1000,
    )
    assert req.messages[1].tool_calls[0].name == "get_weather"
    assert req.messages[2].tool_call_id == "c1"
    assert req.tools[0].name == "get_weather"


def _data(event: str) -> dict[str, object]:
    assert event.startswith("data: ")
    obj = json.loads(event[len("data: ") :])
    assert isinstance(obj, dict)
    return obj


def test_chunk_encoder_sends_role_once_then_content_then_usage() -> None:
    enc = ChunkEncoder("rid", "m")
    first = enc.encode(StreamChunk(content="Hel"))
    second = enc.encode(StreamChunk(content="lo"))
    last = enc.encode(
        StreamChunk(finish_reason="stop", usage=Usage(prompt_tokens=3, completion_tokens=2))
    )
    assert _data(first[0])["choices"] == [
        {"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}
    ]
    assert _data(second[0])["choices"] == [
        {"index": 0, "delta": {"content": "lo"}, "finish_reason": None}
    ]
    assert _data(last[0])["choices"] == [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    assert _data(last[1])["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }


def test_chunk_encoder_tool_call_delta() -> None:
    enc = ChunkEncoder("rid", "m")
    ev = enc.encode(
        StreamChunk(tool_calls=[ToolCallDelta(index=0, id="c1", name="f", arguments="")])
    )
    delta = _data(ev[0])["choices"][0]["delta"]  # type: ignore[index]
    assert delta["tool_calls"] == [
        {"index": 0, "id": "c1", "type": "function", "function": {"name": "f", "arguments": ""}}
    ]
