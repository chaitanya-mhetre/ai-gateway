"""Anthropic Messages API adapter.

Wire format: POST {base}/v1/messages with `x-api-key` and `anthropic-version` headers.
Differences from OpenAI that this adapter hides:
- `system` is a top-level field, not a message.
- `max_tokens` is required.
- Content is a list of typed blocks: `text`, `tool_use` (assistant → tool call), and
  `tool_result` (sent back inside a *user* message).
- Roles must alternate user/assistant, so consecutive same-role messages are merged.
- Streaming uses named SSE events: message_start, content_block_start/delta/stop,
  message_delta (stop_reason + output tokens), message_stop, ping, error.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ai_gateway.errors import HTTP_5XX, HTTP_429, INVALID_REQUEST, MALFORMED_RESPONSE, ProviderError
from ai_gateway.models import (
    ChatRequest,
    ChatResponse,
    FinishReason,
    StreamChunk,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from ai_gateway.providers.base import HttpProvider, iter_sse, loads_or_error

API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 1024

_STOP: dict[str, FinishReason] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
    "pause_turn": "stop",
}


def _parse_args(arguments: str) -> Any:
    try:
        return json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {"_raw": arguments}


def _tool_choice(choice: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if choice is None:
        return None
    if choice == "auto":
        return {"type": "auto"}
    if choice == "none":
        return {"type": "none"}
    if choice == "required":
        return {"type": "any"}
    if isinstance(choice, dict):
        name = (choice.get("function") or {}).get("name")
        if name:
            return {"type": "tool", "name": name}
    return None


def build_payload(req: ChatRequest, model: str, *, stream: bool) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []

    def append(role: str, blocks: list[dict[str, Any]]) -> None:
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(blocks)  # merge to keep roles alternating
        else:
            messages.append({"role": role, "content": blocks})

    for m in req.non_system_messages:
        if m.role == "tool":
            append(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id or "",
                        "content": m.content or "",
                    }
                ],
            )
        elif m.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.content:
                blocks.append({"type": "text", "text": m.content})
            for tc in m.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": _parse_args(tc.arguments),
                    }
                )
            append("assistant", blocks)
        else:
            append("user", [{"type": "text", "text": m.content or ""}])

    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": req.max_tokens or DEFAULT_MAX_TOKENS,
        "messages": messages,
    }
    if req.system_prompt:
        payload["system"] = req.system_prompt
    if req.tools:
        payload["tools"] = [
            {"name": t.name, "description": t.description or "", "input_schema": t.parameters}
            for t in req.tools
        ]
        choice = _tool_choice(req.tool_choice)
        if choice:
            payload["tool_choice"] = choice
    if req.temperature is not None:
        payload["temperature"] = min(req.temperature, 1.0)  # Anthropic range is 0..1
    if req.top_p is not None:
        payload["top_p"] = req.top_p
    if req.stop:
        payload["stop_sequences"] = req.stop
    if stream:
        payload["stream"] = True
    return payload


def parse_usage(raw: dict[str, Any] | None) -> Usage:
    """Anthropic's `input_tokens` EXCLUDES cache reads/writes; we report the OpenAI-style total."""
    raw = raw or {}
    cache_read = int(raw.get("cache_read_input_tokens") or 0)
    cache_write = int(raw.get("cache_creation_input_tokens") or 0)
    return Usage(
        prompt_tokens=int(raw.get("input_tokens") or 0) + cache_read + cache_write,
        completion_tokens=int(raw.get("output_tokens") or 0),
        cached_tokens=cache_read,
    )


class AnthropicProvider(HttpProvider):
    def auth_headers(self) -> dict[str, str]:
        headers = {"anthropic-version": API_VERSION}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def chat(self, req: ChatRequest, model: str, timeout: float) -> ChatResponse:
        data = await self._post_json(
            "/v1/messages", build_payload(req, model, stream=False), timeout
        )
        blocks = data.get("content")
        if not isinstance(blocks, list):
            raise ProviderError(
                self.name, "missing content blocks", reason=MALFORMED_RESPONSE, retryable=True
            )
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        tool_calls = [
            ToolCall(
                id=b["id"],
                name=b["name"],
                arguments=json.dumps(b.get("input") or {}, separators=(",", ":")),
            )
            for b in blocks
            if b.get("type") == "tool_use"
        ]
        return ChatResponse(
            id=str(data.get("id", "")),
            provider=self.name,
            model=str(data.get("model", model)),
            content=text or None,
            tool_calls=tool_calls,
            finish_reason=_STOP.get(str(data.get("stop_reason")), "stop"),
            usage=parse_usage(data.get("usage")),
        )

    async def stream(
        self, req: ChatRequest, model: str, timeout: float
    ) -> AsyncIterator[StreamChunk]:
        lines = self._stream_lines("/v1/messages", build_payload(req, model, stream=True), timeout)
        usage = Usage()
        tool_index_by_block: dict[int, int] = {}  # Anthropic block index → OpenAI tool index
        async for event, data in iter_sse(lines):
            if event == "ping":
                continue
            obj = loads_or_error(self.name, data)
            kind = event or obj.get("type")
            if kind == "message_start":
                usage = parse_usage((obj.get("message") or {}).get("usage"))
            elif kind == "content_block_start":
                block = obj.get("content_block") or {}
                if block.get("type") == "tool_use":
                    idx = len(tool_index_by_block)
                    tool_index_by_block[int(obj.get("index", 0))] = idx
                    yield StreamChunk(
                        tool_calls=[
                            ToolCallDelta(index=idx, id=block.get("id"), name=block.get("name"))
                        ]
                    )
            elif kind == "content_block_delta":
                delta = obj.get("delta") or {}
                if delta.get("type") == "text_delta":
                    yield StreamChunk(content=delta.get("text", ""))
                elif delta.get("type") == "input_json_delta":
                    idx = tool_index_by_block.get(int(obj.get("index", 0)), 0)
                    yield StreamChunk(
                        tool_calls=[
                            ToolCallDelta(index=idx, arguments=delta.get("partial_json", ""))
                        ]
                    )
            elif kind == "message_delta":
                out = int((obj.get("usage") or {}).get("output_tokens") or 0)
                usage = usage.model_copy(update={"completion_tokens": out})
                stop = (obj.get("delta") or {}).get("stop_reason")
                yield StreamChunk(finish_reason=_STOP.get(str(stop), "stop"), usage=usage)
            elif kind == "message_stop":
                return
            elif kind == "error":
                err = obj.get("error") or {}
                etype = str(err.get("type", ""))
                if etype == "overloaded_error" or etype == "api_error":
                    reason, retryable = HTTP_5XX, True
                elif etype == "rate_limit_error":
                    reason, retryable = HTTP_429, True
                else:
                    reason, retryable = INVALID_REQUEST, False
                raise ProviderError(
                    self.name,
                    f"stream error {etype}: {err.get('message')}",
                    reason=reason,
                    retryable=retryable,
                )
