"""Google Gemini adapter (Generative Language API, `generateContent`).

Wire format: POST {base}/models/{model}:generateContent with the `x-goog-api-key` header.
Streaming:   POST {base}/models/{model}:streamGenerateContent?alt=sse, where each SSE `data:` is a
             partial GenerateContentResponse.
Differences this adapter hides:
- Roles are `user` and `model`; the system prompt goes in `systemInstruction`.
- Tool calls are `functionCall` parts (args as an object) and results are `functionResponse` parts,
  which reference the function *name*, not a call id, so we keep an id→name map.
- Older responses don't give tool calls an id, so we generate one.
- Blocked prompts come back with no candidates and `promptFeedback.blockReason`.

Note: Google also offers a newer "Interactions" API. This adapter targets the long-standing
generateContent surface; see docs/providers.md.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from ai_gateway.errors import MALFORMED_RESPONSE, REFUSAL, ProviderError
from ai_gateway.metering.tokens import estimate_text_tokens
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
from ai_gateway.providers.base import HttpProvider, iter_sse, loads_or_error
from ai_gateway.providers.ids import new_tool_call_id

_FINISH: dict[str, FinishReason] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "RECITATION": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
}


def _parse_args(arguments: str) -> Any:
    try:
        return json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {"_raw": arguments}


def _tool_config(choice: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if choice is None:
        return None
    if choice == "auto":
        mode: dict[str, Any] = {"mode": "AUTO"}
    elif choice == "none":
        mode = {"mode": "NONE"}
    elif choice == "required":
        mode = {"mode": "ANY"}
    elif isinstance(choice, dict) and (choice.get("function") or {}).get("name"):
        mode = {"mode": "ANY", "allowedFunctionNames": [choice["function"]["name"]]}
    else:
        return None
    return {"functionCallingConfig": mode}


def build_payload(req: ChatRequest) -> dict[str, Any]:
    contents: list[dict[str, Any]] = []
    call_names: dict[str, str] = {}

    def append(role: str, parts: list[dict[str, Any]]) -> None:
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": role, "parts": parts})

    for m in req.non_system_messages:
        if m.role == "assistant":
            parts: list[dict[str, Any]] = []
            if m.content:
                parts.append({"text": m.content})
            for tc in m.tool_calls:
                call_names[tc.id] = tc.name
                parts.append({"functionCall": {"name": tc.name, "args": _parse_args(tc.arguments)}})
            append("model", parts)
        elif m.role == "tool":
            name = call_names.get(m.tool_call_id or "", m.name or "tool")
            append(
                "user",
                [{"functionResponse": {"name": name, "response": {"content": m.content or ""}}}],
            )
        else:
            append("user", [{"text": m.content or ""}])

    payload: dict[str, Any] = {"contents": contents}
    if req.system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": req.system_prompt}]}
    if req.tools:
        payload["tools"] = [
            {
                "functionDeclarations": [
                    {"name": t.name, "description": t.description or "", "parameters": t.parameters}
                    for t in req.tools
                ]
            }
        ]
        cfg = _tool_config(req.tool_choice)
        if cfg:
            payload["toolConfig"] = cfg
    gen: dict[str, Any] = {}
    if req.temperature is not None:
        gen["temperature"] = req.temperature
    if req.top_p is not None:
        gen["topP"] = req.top_p
    if req.max_tokens is not None:
        gen["maxOutputTokens"] = req.max_tokens
    if req.stop:
        gen["stopSequences"] = req.stop
    if gen:
        payload["generationConfig"] = gen
    return payload


def parse_usage(raw: dict[str, Any] | None) -> Usage:
    raw = raw or {}
    return Usage(
        prompt_tokens=int(raw.get("promptTokenCount") or 0),
        completion_tokens=int(raw.get("candidatesTokenCount") or 0),
        cached_tokens=int(raw.get("cachedContentTokenCount") or 0),
    )


class GeminiProvider(HttpProvider):
    def auth_headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.api_key} if self.api_key else {}

    def _candidate(self, data: dict[str, Any]) -> dict[str, Any] | None:
        candidates = data.get("candidates") or []
        if not candidates:
            block = (data.get("promptFeedback") or {}).get("blockReason")
            if block:
                raise ProviderError(
                    self.name, f"prompt blocked: {block}", reason=REFUSAL, retryable=False
                )
            return None
        first = candidates[0]
        return first if isinstance(first, dict) else None

    async def chat(self, req: ChatRequest, model: str, timeout: float) -> ChatResponse:
        data = await self._post_json(
            f"/models/{model}:generateContent", build_payload(req), timeout
        )
        cand = self._candidate(data)
        if cand is None:
            raise ProviderError(
                self.name, "no candidates", reason=MALFORMED_RESPONSE, retryable=True
            )
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if "text" in p and not p.get("thought"))
        tool_calls = [
            ToolCall(
                id=p["functionCall"].get("id") or new_tool_call_id(),
                name=p["functionCall"]["name"],
                arguments=json.dumps(p["functionCall"].get("args") or {}, separators=(",", ":")),
            )
            for p in parts
            if "functionCall" in p
        ]
        finish: FinishReason = (
            "tool_calls" if tool_calls else _FINISH.get(str(cand.get("finishReason")), "stop")
        )
        return ChatResponse(
            id=str(data.get("responseId", "")),
            provider=self.name,
            model=str(data.get("modelVersion", model)),
            content=text or None,
            tool_calls=tool_calls,
            finish_reason=finish,
            usage=parse_usage(data.get("usageMetadata")),
        )

    async def stream(
        self, req: ChatRequest, model: str, timeout: float
    ) -> AsyncIterator[StreamChunk]:
        lines = self._stream_lines(
            f"/models/{model}:streamGenerateContent",
            build_payload(req),
            timeout,
            params={"alt": "sse"},
        )
        usage: Usage | None = None
        finish: FinishReason | None = None
        tool_count = 0
        async for _event, data in iter_sse(lines):
            obj = loads_or_error(self.name, data)
            if obj.get("usageMetadata"):
                usage = parse_usage(obj["usageMetadata"])  # cumulative; the last one wins
            cand = self._candidate(obj)
            if cand is None:
                continue
            for p in (cand.get("content") or {}).get("parts") or []:
                if "text" in p and not p.get("thought"):
                    yield StreamChunk(content=p["text"])
                elif "functionCall" in p:
                    fc = p["functionCall"]
                    # Gemini delivers each function call complete, in a single chunk.
                    yield StreamChunk(
                        tool_calls=[
                            ToolCallDelta(
                                index=tool_count,
                                id=fc.get("id") or new_tool_call_id(),
                                name=fc["name"],
                                arguments=json.dumps(fc.get("args") or {}, separators=(",", ":")),
                            )
                        ]
                    )
                    tool_count += 1
            if cand.get("finishReason"):
                finish = (
                    "tool_calls" if tool_count else _FINISH.get(str(cand["finishReason"]), "stop")
                )
        yield StreamChunk(finish_reason=finish or "stop", usage=usage)

    async def embed(self, req: EmbeddingRequest, model: str, timeout: float) -> EmbeddingResponse:
        payload = {
            "requests": [
                {"model": f"models/{model}", "content": {"parts": [{"text": t}]}} for t in req.input
            ]
        }
        data = await self._post_json(f"/models/{model}:batchEmbedContents", payload, timeout)
        vectors = [list(map(float, e.get("values") or [])) for e in data.get("embeddings") or []]
        # batchEmbedContents doesn't return token usage, so this is estimated and flagged as such.
        usage = Usage(prompt_tokens=sum(estimate_text_tokens(t) for t in req.input), estimated=True)
        return EmbeddingResponse(provider=self.name, model=model, vectors=vectors, usage=usage)
