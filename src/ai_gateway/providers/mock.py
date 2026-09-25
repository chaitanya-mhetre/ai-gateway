"""A scriptable fake provider for tests, fault injection and load tests.

No network, no API keys. Behaviour per call is taken from `script` (FIFO); once the script is
exhausted every call succeeds. That makes failure scenarios deterministic:

    MockProvider("primary", script=[timeout(), http_5xx(), None])  # fail, fail, then succeed
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import math
from collections import deque
from collections.abc import AsyncIterator, Iterable

from ai_gateway.errors import (
    HTTP_5XX,
    HTTP_429,
    INVALID_REQUEST,
    STREAM_INTERRUPTED,
    TIMEOUT,
    ProviderError,
)
from ai_gateway.metering.tokens import estimate_text_tokens
from ai_gateway.models import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    StreamChunk,
    ToolCall,
    ToolCallDelta,
    Usage,
)

Outcome = ProviderError | None
_ids = itertools.count(1)


class MockProvider:
    def __init__(
        self,
        name: str = "mock",
        *,
        reply: str = "Hello from the mock provider.",
        latency: float = 0.0,
        script: Iterable[Outcome] = (),
        tool_call: ToolCall | None = None,
        stream_fail_after: int | None = None,
        chunk_size: int = 8,
        embedding_dim: int = 16,
    ) -> None:
        self.name = name
        self.reply = reply
        self.latency = latency
        self.script: deque[Outcome] = deque(script)
        self.tool_call = tool_call
        self.stream_fail_after = stream_fail_after
        self.chunk_size = chunk_size
        self.embedding_dim = embedding_dim
        self.calls = 0
        self.models_seen: list[str] = []

    def _next_outcome(self) -> Outcome:
        self.calls += 1
        return self.script.popleft() if self.script else None

    async def _simulate_latency(self, timeout: float) -> None:
        if self.latency > timeout:
            await asyncio.sleep(timeout)
            raise ProviderError(self.name, "mock timeout", reason=TIMEOUT, retryable=True)
        if self.latency:
            await asyncio.sleep(self.latency)

    def _usage(self, req: ChatRequest, completion: str) -> Usage:
        prompt = sum(estimate_text_tokens(m.content or "") for m in req.messages)
        return Usage(prompt_tokens=prompt, completion_tokens=estimate_text_tokens(completion))

    async def chat(self, req: ChatRequest, model: str, timeout: float) -> ChatResponse:
        self.models_seen.append(model)
        outcome = self._next_outcome()
        await self._simulate_latency(timeout)
        if outcome is not None:
            raise outcome
        if self.tool_call and req.tools:
            return ChatResponse(
                id=f"mock-{next(_ids)}",
                provider=self.name,
                model=model,
                content=None,
                tool_calls=[self.tool_call],
                finish_reason="tool_calls",
                usage=self._usage(req, ""),
            )
        return ChatResponse(
            id=f"mock-{next(_ids)}",
            provider=self.name,
            model=model,
            content=self.reply,
            finish_reason="stop",
            usage=self._usage(req, self.reply),
        )

    async def stream(
        self, req: ChatRequest, model: str, timeout: float
    ) -> AsyncIterator[StreamChunk]:
        self.models_seen.append(model)
        outcome = self._next_outcome()
        await self._simulate_latency(timeout)
        if outcome is not None:
            raise outcome
        if self.tool_call and req.tools:
            tc = self.tool_call
            yield StreamChunk(tool_calls=[ToolCallDelta(index=0, id=tc.id, name=tc.name)])
            yield StreamChunk(tool_calls=[ToolCallDelta(index=0, arguments=tc.arguments)])
            yield StreamChunk(finish_reason="tool_calls", usage=self._usage(req, ""))
            return
        pieces = [
            self.reply[i : i + self.chunk_size] for i in range(0, len(self.reply), self.chunk_size)
        ]
        for n, piece in enumerate(pieces):
            if self.stream_fail_after is not None and n >= self.stream_fail_after:
                raise ProviderError(
                    self.name, "mock stream broke", reason=STREAM_INTERRUPTED, retryable=True
                )
            yield StreamChunk(content=piece)
            await asyncio.sleep(0)
        yield StreamChunk(finish_reason="stop", usage=self._usage(req, self.reply))

    async def embed(self, req: EmbeddingRequest, model: str, timeout: float) -> EmbeddingResponse:
        outcome = self._next_outcome()
        await self._simulate_latency(timeout)
        if outcome is not None:
            raise outcome
        vectors = [fake_embedding(text, self.embedding_dim) for text in req.input]
        usage = Usage(prompt_tokens=sum(estimate_text_tokens(t) for t in req.input))
        return EmbeddingResponse(provider=self.name, model=model, vectors=vectors, usage=usage)

    async def aclose(self) -> None:
        return None


def fake_embedding(text: str, dim: int = 16) -> list[float]:
    """Deterministic bag-of-words hash embedding: similar word sets → similar vectors.

    Good enough to exercise the semantic cache without a real model.
    """
    vec = [0.0] * dim
    for word in text.lower().split():
        h = int.from_bytes(hashlib.sha256(word.strip(".,!?").encode()).digest()[:4], "big")
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


# Convenience constructors for scripts ----------------------------------------------------------
def timeout(provider: str = "mock") -> ProviderError:
    return ProviderError(provider, "scripted timeout", reason=TIMEOUT, retryable=True)


def http_5xx(provider: str = "mock", status: int = 503) -> ProviderError:
    return ProviderError(
        provider, f"scripted {status}", reason=HTTP_5XX, retryable=True, status=status
    )


def http_429(provider: str = "mock", retry_after: float | None = None) -> ProviderError:
    return ProviderError(
        provider,
        "scripted 429",
        reason=HTTP_429,
        retryable=True,
        status=429,
        retry_after=retry_after,
    )


def bad_request(provider: str = "mock") -> ProviderError:
    return ProviderError(
        provider, "scripted 400", reason=INVALID_REQUEST, retryable=False, status=400
    )
