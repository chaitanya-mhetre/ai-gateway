"""Provider interface + shared HTTP plumbing.

Adapters do NOT use vendor SDKs on purpose: we want full control over timeouts, streaming and
error classification, and to understand the raw wire formats.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

import httpx

from ai_gateway.errors import (
    CONNECTION,
    MALFORMED_RESPONSE,
    TIMEOUT,
    ProviderError,
    classify_http_error,
)
from ai_gateway.models import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    StreamChunk,
)


@runtime_checkable
class Provider(Protocol):
    """What the executor needs from any provider. `timeout` is the remaining deadline in seconds."""

    name: str

    async def chat(self, req: ChatRequest, model: str, timeout: float) -> ChatResponse: ...

    def stream(
        self, req: ChatRequest, model: str, timeout: float
    ) -> AsyncIterator[StreamChunk]: ...

    async def embed(
        self, req: EmbeddingRequest, model: str, timeout: float
    ) -> EmbeddingResponse: ...

    async def aclose(self) -> None: ...


class HttpProvider:
    """Base class: pooled httpx client + error mapping. Subclasses implement translation."""

    name: str

    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        # One pooled client per provider: TCP/TLS connections are reused across requests.
        self._client = client or httpx.AsyncClient(
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50)
        )

    # --- subclass hooks -----------------------------------------------------------------------
    def auth_headers(self) -> dict[str, str]:
        return {}

    async def embed(self, req: EmbeddingRequest, model: str, timeout: float) -> EmbeddingResponse:
        raise ProviderError(
            self.name, "embeddings not supported", reason="invalid_request", retryable=False
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    # --- HTTP helpers -------------------------------------------------------------------------
    def _timeout(self, timeout: float) -> httpx.Timeout:
        return httpx.Timeout(timeout, connect=min(timeout, 5.0))

    async def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        timeout: float,
        params: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            # httpx timeouts are per-operation; asyncio.timeout enforces the *total* deadline.
            async with asyncio.timeout(timeout):
                resp = await self._client.post(
                    url,
                    json=payload,
                    headers=self.auth_headers(),
                    params=params,
                    timeout=self._timeout(timeout),
                )
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise ProviderError(
                self.name, f"timed out after {timeout:.1f}s", reason=TIMEOUT, retryable=True
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                self.name, f"connection error: {exc}", reason=CONNECTION, retryable=True
            ) from exc
        if resp.status_code >= 400:
            raise classify_http_error(self.name, resp.status_code, resp.text, dict(resp.headers))
        try:
            data = resp.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(
                self.name, "non-JSON response", reason=MALFORMED_RESPONSE, retryable=True
            ) from exc
        if not isinstance(data, dict):
            raise ProviderError(
                self.name, "unexpected JSON shape", reason=MALFORMED_RESPONSE, retryable=True
            )
        return data

    async def _stream_lines(
        self,
        path: str,
        payload: dict[str, Any],
        timeout: float,
        params: dict[str, str] | None = None,
    ) -> AsyncIterator[str]:
        """Yield raw response lines. Errors before the first line raise ProviderError."""
        url = f"{self.base_url}{path}"
        try:
            async with self._client.stream(
                "POST",
                url,
                json=payload,
                headers=self.auth_headers(),
                params=params,
                timeout=self._timeout(timeout),
            ) as resp:
                if resp.status_code >= 400:
                    body = (await resp.aread()).decode(errors="replace")
                    raise classify_http_error(self.name, resp.status_code, body, dict(resp.headers))
                async for line in resp.aiter_lines():
                    yield line
        except httpx.TimeoutException as exc:
            raise ProviderError(
                self.name, "stream timed out", reason=TIMEOUT, retryable=True
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                self.name, f"stream connection error: {exc}", reason=CONNECTION, retryable=True
            ) from exc


async def iter_sse(lines: AsyncIterator[str]) -> AsyncIterator[tuple[str | None, str]]:
    """Minimal Server-Sent Events parser: yields (event, data) per event block.

    SSE format: `event: <name>` and one or more `data: <text>` lines, terminated by a blank line.
    Lines starting with ':' are comments (keep-alives).
    """
    event: str | None = None
    data_lines: list[str] = []
    async for raw in lines:
        line = raw.rstrip("\r")
        if line == "":
            if data_lines:
                yield event, "\n".join(data_lines)
            event, data_lines = None, []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            event = value
        elif field == "data":
            data_lines.append(value)
    if data_lines:
        yield event, "\n".join(data_lines)


def loads_or_error(provider: str, data: str) -> dict[str, Any]:
    try:
        obj = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            provider,
            f"malformed stream chunk: {data[:80]}",
            reason=MALFORMED_RESPONSE,
            retryable=True,
        ) from exc
    if not isinstance(obj, dict):
        raise ProviderError(
            provider, "malformed stream chunk", reason=MALFORMED_RESPONSE, retryable=True
        )
    return obj
