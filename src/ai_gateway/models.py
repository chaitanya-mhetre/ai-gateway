"""Provider-neutral internal data model.

Every adapter translates *to* and *from* these types, so routing, caching, metering and the
public API never see provider-specific JSON.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]
FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "error"]


class ToolCall(BaseModel):
    """A completed tool call. `arguments` is a JSON string (OpenAI convention)."""

    id: str
    name: str
    arguments: str = "{}"


class Message(BaseModel):
    role: Role
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None


class ToolSpec(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    tools: list[ToolSpec] = Field(default_factory=list)
    tool_choice: str | dict[str, Any] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: list[str] = Field(default_factory=list)
    stream: bool = False
    user: str | None = None

    @property
    def system_prompt(self) -> str | None:
        parts = [m.content for m in self.messages if m.role == "system" and m.content]
        return "\n\n".join(parts) if parts else None

    @property
    def non_system_messages(self) -> list[Message]:
        return [m for m in self.messages if m.role != "system"]


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    estimated: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatResponse(BaseModel):
    id: str
    provider: str
    model: str
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: FinishReason = "stop"
    usage: Usage = Field(default_factory=Usage)


class ToolCallDelta(BaseModel):
    """A fragment of a tool call inside a stream (OpenAI-style: id/name first, then argument text)."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


class StreamChunk(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCallDelta] = Field(default_factory=list)
    finish_reason: FinishReason | None = None
    usage: Usage | None = None


class EmbeddingRequest(BaseModel):
    model: str
    input: list[str]


class EmbeddingResponse(BaseModel):
    provider: str
    model: str
    vectors: list[list[float]]
    usage: Usage = Field(default_factory=Usage)
