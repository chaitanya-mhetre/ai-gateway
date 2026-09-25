"""OpenAI Chat Completions adapter (also works for OpenAI-compatible servers, e.g. vLLM, OpenRouter).

Wire format: POST {base}/chat/completions, `Authorization: Bearer <key>`.
Streaming: SSE `data: {chunk}` lines, terminated by `data: [DONE]`. With
`stream_options.include_usage`, the final chunk has `choices: []` and a `usage` object.
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
    Message,
    StreamChunk,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from ai_gateway.providers.base import HttpProvider, iter_sse, loads_or_error

_FINISH: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "content_filter": "content_filter",
}


def to_openai_message(m: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": m.role, "content": m.content}
    if m.name:
        out["name"] = m.name
    if m.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": tc.arguments},
            }
            for tc in m.tool_calls
        ]
    if m.tool_call_id:
        out["tool_call_id"] = m.tool_call_id
    return out


def build_payload(req: ChatRequest, model: str, *, stream: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [to_openai_message(m) for m in req.messages],
    }
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
        if req.tool_choice is not None:
            payload["tool_choice"] = req.tool_choice
    if req.temperature is not None:
        payload["temperature"] = req.temperature
    if req.top_p is not None:
        payload["top_p"] = req.top_p
    if req.max_tokens is not None:
        # Newer OpenAI models reject `max_tokens` in favour of `max_completion_tokens`.
        payload["max_completion_tokens"] = req.max_tokens
    if req.stop:
        payload["stop"] = req.stop
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return payload


def parse_usage(raw: dict[str, Any] | None) -> Usage:
    if not raw:
        return Usage()
    details = raw.get("prompt_tokens_details") or {}
    return Usage(
        prompt_tokens=int(raw.get("prompt_tokens") or 0),
        completion_tokens=int(raw.get("completion_tokens") or 0),
        cached_tokens=int(details.get("cached_tokens") or 0),
    )


class OpenAIProvider(HttpProvider):
    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def chat(self, req: ChatRequest, model: str, timeout: float) -> ChatResponse:
        data = await self._post_json(
            "/chat/completions", build_payload(req, model, stream=False), timeout
        )
        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                self.name, "missing choices[0].message", reason=MALFORMED_RESPONSE, retryable=True
            ) from exc
        tool_calls = [
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=tc["function"].get("arguments") or "{}",
            )
            for tc in msg.get("tool_calls") or []
        ]
        return ChatResponse(
            id=str(data.get("id", "")),
            provider=self.name,
            model=str(data.get("model", model)),
            content=msg.get("content"),
            tool_calls=tool_calls,
            finish_reason=_FINISH.get(choice.get("finish_reason") or "stop", "stop"),
            usage=parse_usage(data.get("usage")),
        )

    async def stream(
        self, req: ChatRequest, model: str, timeout: float
    ) -> AsyncIterator[StreamChunk]:
        lines = self._stream_lines(
            "/chat/completions", build_payload(req, model, stream=True), timeout
        )
        async for _event, data in iter_sse(lines):
            if data.strip() == "[DONE]":
                return
            obj = loads_or_error(self.name, data)
            usage = parse_usage(obj["usage"]) if obj.get("usage") else None
            choices = obj.get("choices") or []
            if not choices:
                if usage:
                    yield StreamChunk(usage=usage)
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            deltas = [
                ToolCallDelta(
                    index=int(tc.get("index", 0)),
                    id=tc.get("id"),
                    name=(tc.get("function") or {}).get("name"),
                    arguments=(tc.get("function") or {}).get("arguments") or "",
                )
                for tc in delta.get("tool_calls") or []
            ]
            finish = choice.get("finish_reason")
            yield StreamChunk(
                content=delta.get("content"),
                tool_calls=deltas,
                finish_reason=_FINISH.get(finish, "stop") if finish else None,
                usage=usage,
            )

    async def embed(self, req: EmbeddingRequest, model: str, timeout: float) -> EmbeddingResponse:
        data = await self._post_json("/embeddings", {"model": model, "input": req.input}, timeout)
        items = sorted(data.get("data") or [], key=lambda d: int(d.get("index", 0)))
        vectors = [list(map(float, d["embedding"])) for d in items]
        usage = data.get("usage") or {}
        return EmbeddingResponse(
            provider=self.name,
            model=model,
            vectors=vectors,
            usage=Usage(prompt_tokens=int(usage.get("prompt_tokens") or 0)),
        )


def dumps_args(args: Any) -> str:
    """Normalise tool-call arguments (dict from Anthropic/Gemini/Ollama) to a JSON string."""
    if isinstance(args, str):
        return args
    return json.dumps(args or {}, separators=(",", ":"))
