"""The public, OpenAI-compatible wire format ↔ internal model.

Clients keep using the official OpenAI SDK and only change `base_url`.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai_gateway.errors import InvalidRequestError
from ai_gateway.models import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    Message,
    StreamChunk,
    ToolCall,
    ToolSpec,
)

# --- incoming request schemas (validated strictly, unknown fields ignored) ----------------------


class _FunctionCall(BaseModel):
    name: str
    arguments: str = "{}"


class _ToolCallIn(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: _FunctionCall


class _MessageIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[_ToolCallIn] | None = None
    tool_call_id: str | None = None


class _FunctionDef(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class _ToolIn(BaseModel):
    type: Literal["function"] = "function"
    function: _FunctionDef


class OpenAIChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[_MessageIn] = Field(min_length=1, max_length=1000)
    tools: list[_ToolIn] | None = None
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = None
    stream: bool = False
    user: str | None = None

    @field_validator("stop")
    @classmethod
    def _stop_list(cls, v: str | list[str] | None) -> str | list[str] | None:
        if isinstance(v, list) and len(v) > 4:
            raise ValueError("at most 4 stop sequences")
        return v


class OpenAIEmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    input: str | list[str]


def _flatten_content(content: str | list[dict[str, Any]] | None) -> str | None:
    if content is None or isinstance(content, str):
        return content
    texts: list[str] = []
    for part in content:
        if part.get("type") == "text":
            texts.append(str(part.get("text", "")))
        else:
            raise InvalidRequestError(
                f"content part type '{part.get('type')}' is not supported (v1 is text-only)"
            )
    return "".join(texts)


def to_internal(body: OpenAIChatRequest, *, max_tokens_cap: int) -> ChatRequest:
    messages: list[Message] = []
    for m in body.messages:
        role = "system" if m.role == "developer" else m.role
        messages.append(
            Message(
                role=role,
                content=_flatten_content(m.content),
                name=m.name,
                tool_calls=[
                    ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
                    for tc in m.tool_calls or []
                ],
                tool_call_id=m.tool_call_id,
            )
        )
    max_tokens = body.max_completion_tokens or body.max_tokens
    if max_tokens is not None and max_tokens > max_tokens_cap:
        raise InvalidRequestError(
            f"max_tokens {max_tokens} exceeds the gateway cap of {max_tokens_cap}"
        )
    stop = [body.stop] if isinstance(body.stop, str) else list(body.stop or [])
    return ChatRequest(
        model=body.model,
        messages=messages,
        tools=[
            ToolSpec(
                name=t.function.name,
                description=t.function.description,
                parameters=t.function.parameters,
            )
            for t in body.tools or []
        ],
        tool_choice=body.tool_choice,
        temperature=body.temperature,
        top_p=body.top_p,
        max_tokens=max_tokens,
        stop=stop,
        stream=body.stream,
        user=body.user,
    )


def embedding_to_internal(body: OpenAIEmbeddingRequest) -> EmbeddingRequest:
    inputs = [body.input] if isinstance(body.input, str) else body.input
    if not inputs or len(inputs) > 2048:
        raise InvalidRequestError("input must contain 1..2048 strings")
    return EmbeddingRequest(model=body.model, input=inputs)


# --- outgoing ---------------------------------------------------------------------------------


def _tool_calls_out(calls: list[ToolCall]) -> list[dict[str, Any]]:
    return [
        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
        for c in calls
    ]


def response_to_openai(resp: ChatResponse, request_id: str) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": resp.content}
    if resp.tool_calls:
        message["tool_calls"] = _tool_calls_out(resp.tool_calls)
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": resp.model,
        "choices": [
            {"index": 0, "message": message, "finish_reason": resp.finish_reason, "logprobs": None}
        ],
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
            "prompt_tokens_details": {"cached_tokens": resp.usage.cached_tokens},
        },
    }


class ChunkEncoder:
    """Encodes internal StreamChunks as OpenAI `chat.completion.chunk` SSE events."""

    def __init__(self, request_id: str, model: str) -> None:
        self.id = f"chatcmpl-{request_id}"
        self.model = model
        self.created = int(time.time())
        self._sent_role = False

    def _event(self, choices: list[dict[str, Any]], usage: dict[str, Any] | None = None) -> str:
        obj: dict[str, Any] = {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": choices,
        }
        if usage is not None:
            obj["usage"] = usage
        return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"

    def encode(self, chunk: StreamChunk) -> list[str]:
        out: list[str] = []
        delta: dict[str, Any] = {}
        if not self._sent_role:
            delta["role"] = "assistant"
            self._sent_role = True
        if chunk.content:
            delta["content"] = chunk.content
        if chunk.tool_calls:
            delta["tool_calls"] = [
                {
                    "index": d.index,
                    **({"id": d.id, "type": "function"} if d.id else {}),
                    "function": {**({"name": d.name} if d.name else {}), "arguments": d.arguments},
                }
                for d in chunk.tool_calls
            ]
        if delta or chunk.finish_reason:
            out.append(
                self._event([{"index": 0, "delta": delta, "finish_reason": chunk.finish_reason}])
            )
        if chunk.usage is not None:
            u = chunk.usage
            out.append(
                self._event(
                    [],
                    {
                        "prompt_tokens": u.prompt_tokens,
                        "completion_tokens": u.completion_tokens,
                        "total_tokens": u.total_tokens,
                    },
                )
            )
        return out

    @staticmethod
    def error(message: str, error_type: str) -> str:
        return f"data: {json.dumps({'error': {'message': message, 'type': error_type}})}\n\n"

    @staticmethod
    def done() -> str:
        return "data: [DONE]\n\n"


def embeddings_to_openai(resp: EmbeddingResponse) -> dict[str, Any]:
    return {
        "object": "list",
        "model": resp.model,
        "data": [
            {"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(resp.vectors)
        ],
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "total_tokens": resp.usage.prompt_tokens,
        },
    }


def new_request_id() -> str:
    return uuid.uuid4().hex
