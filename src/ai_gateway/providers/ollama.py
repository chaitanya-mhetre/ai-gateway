"""Ollama adapter (local models; free, so ideal as the last-resort fallback and for demos).

Wire format: POST {base}/api/chat. Streaming is newline-delimited JSON (NDJSON), not SSE: each line
is `{"message": {...}, "done": false}`, and the final line has `done: true` plus token counts.
Tool-call arguments are JSON *objects* and have no ids, so ids are generated.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ai_gateway.errors import MALFORMED_RESPONSE, ProviderError
from ai_gateway.models import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    FinishReason,
    StreamChunk,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from ai_gateway.providers.base import HttpProvider, loads_or_error
from ai_gateway.providers.ids import new_tool_call_id


def _parse_args(arguments: str) -> Any:
    try:
        return json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {}


def build_payload(req: ChatRequest, model: str, *, stream: bool) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    for m in req.messages:
        msg: dict[str, Any] = {"role": m.role, "content": m.content or ""}
        if m.tool_calls:
            msg["tool_calls"] = [
                {"function": {"name": tc.name, "arguments": _parse_args(tc.arguments)}}
                for tc in m.tool_calls
            ]
        if m.role == "tool" and m.name:
            msg["tool_name"] = m.name
        messages.append(msg)
    options: dict[str, Any] = {}
    if req.temperature is not None:
        options["temperature"] = req.temperature
    if req.top_p is not None:
        options["top_p"] = req.top_p
    if req.max_tokens is not None:
        options["num_predict"] = req.max_tokens
    if req.stop:
        options["stop"] = req.stop
    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
    if options:
        payload["options"] = options
    if req.tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.parameters,
                },
            }
            for t in req.tools
        ]
    return payload


def _usage(obj: dict[str, Any]) -> Usage:
    return Usage(
        prompt_tokens=int(obj.get("prompt_eval_count") or 0),
        completion_tokens=int(obj.get("eval_count") or 0),
    )


def _finish(obj: dict[str, Any], has_tools: bool) -> FinishReason:
    if has_tools:
        return "tool_calls"
    return "length" if obj.get("done_reason") == "length" else "stop"


def _tool_calls(msg: dict[str, Any]) -> list[ToolCall]:
    return [
        ToolCall(
            id=new_tool_call_id(),
            name=tc["function"]["name"],
            arguments=json.dumps(tc["function"].get("arguments") or {}, separators=(",", ":")),
        )
        for tc in msg.get("tool_calls") or []
    ]


class OllamaProvider(HttpProvider):
    async def chat(self, req: ChatRequest, model: str, timeout: float) -> ChatResponse:
        data = await self._post_json("/api/chat", build_payload(req, model, stream=False), timeout)
        msg = data.get("message")
        if not isinstance(msg, dict):
            raise ProviderError(
                self.name, "missing message", reason=MALFORMED_RESPONSE, retryable=True
            )
        calls = _tool_calls(msg)
        return ChatResponse(
            id=f"ollama-{data.get('created_at', '')}",
            provider=self.name,
            model=str(data.get("model", model)),
            content=msg.get("content") or None,
            tool_calls=calls,
            finish_reason=_finish(data, bool(calls)),
            usage=_usage(data),
        )

    async def stream(
        self, req: ChatRequest, model: str, timeout: float
    ) -> AsyncIterator[StreamChunk]:
        tool_count = 0
        async for line in self._stream_lines(
            "/api/chat", build_payload(req, model, stream=True), timeout
        ):
            if not line.strip():
                continue
            obj = loads_or_error(self.name, line)
            if obj.get("error"):
                raise ProviderError(
                    self.name, str(obj["error"]), reason=MALFORMED_RESPONSE, retryable=True
                )
            msg = obj.get("message") or {}
            if msg.get("content"):
                yield StreamChunk(content=msg["content"])
            for tc in _tool_calls(msg):
                yield StreamChunk(
                    tool_calls=[
                        ToolCallDelta(
                            index=tool_count, id=tc.id, name=tc.name, arguments=tc.arguments
                        )
                    ]
                )
                tool_count += 1
            if obj.get("done"):
                yield StreamChunk(finish_reason=_finish(obj, tool_count > 0), usage=_usage(obj))
                return

    async def embed(self, req: EmbeddingRequest, model: str, timeout: float) -> EmbeddingResponse:
        data = await self._post_json("/api/embed", {"model": model, "input": req.input}, timeout)
        vectors = [list(map(float, v)) for v in data.get("embeddings") or []]
        return EmbeddingResponse(
            provider=self.name,
            model=model,
            vectors=vectors,
            usage=Usage(prompt_tokens=int(data.get("prompt_eval_count") or 0)),
        )
