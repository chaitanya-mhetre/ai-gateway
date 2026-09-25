"""Executor: attempts the planned targets with timeouts, retries, circuit breaking and fallback.

Rules, in order of precedence:
1. One overall deadline per request. Every attempt gets only the *remaining* budget, so retries can
   never make a request exceed its deadline (this prevents retry storms from piling up latency).
   A non-streaming attempt is additionally capped by its provider's `timeout_ms`, so one hung
   provider can't spend the whole deadline and leave nothing for fallback (streams use
   `first_token_timeout_ms` for the same purpose).
2. A target whose circuit is open is skipped instantly (recorded as `circuit_open`).
3. Retryable errors (timeout, 5xx, 429, connection) are retried on the same target, up to
   `max_attempts_per_target`, with jittered backoff that honours `Retry-After`.
4. After a target is exhausted, we fall back to the next target only if the error's reason is in
   the alias's `fallback_on` list. A 400 "your request is invalid" stops immediately: another
   provider would reject it too.
5. Streaming: fallback is allowed ONLY before the first chunk reaches the client. After that, a
   failure becomes an in-band error event. Splicing two models' answers would be wrong.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import TypeVar

from opentelemetry.trace import Status, StatusCode

from ai_gateway import errors as err
from ai_gateway.config import AliasConfig, TargetConfig
from ai_gateway.errors import AllTargetsFailedError, ProviderError
from ai_gateway.models import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    StreamChunk,
)
from ai_gateway.observability.tracing import tracer
from ai_gateway.providers.base import Provider
from ai_gateway.routing.backoff import backoff_delay
from ai_gateway.routing.breaker import CircuitBreaker
from ai_gateway.routing.health import LatencyTracker

T = TypeVar("T")

# Failures that say "this provider is unhealthy" (they count against its circuit breaker).
# Client errors (invalid request, context too long, refusal) mean the provider is up and answering.
HEALTH_FAILURES = frozenset(
    {
        err.TIMEOUT,
        err.HTTP_5XX,
        err.HTTP_429,
        err.CONNECTION,
        err.MALFORMED_RESPONSE,
        err.STREAM_INTERRUPTED,
        err.PROVIDER_AUTH,
    }
)


class ExecutionListener:
    """Hooks for metrics/tracing. The default implementation does nothing."""

    def on_attempt(
        self, provider: str, model: str, ok: bool, reason: str | None, seconds: float
    ) -> None: ...
    def on_retry(self, provider: str, reason: str) -> None: ...
    def on_fallback(
        self, alias: str, from_provider: str, to_provider: str, reason: str
    ) -> None: ...
    def on_ttft(self, provider: str, seconds: float) -> None: ...
    def on_stream_interrupted(self, provider: str) -> None: ...


@dataclass
class ExecInfo:
    alias: str
    provider: str = ""
    model: str = ""
    attempts: int = 0
    fallback_used: bool = False
    ttft_s: float | None = None
    provider_time_s: float = 0.0  # time spent waiting on providers (for the overhead metric)
    backoff_time_s: float = 0.0
    errors: list[ProviderError] = field(default_factory=list)


async def _aclose(it: AsyncIterator[StreamChunk]) -> None:
    close = getattr(it, "aclose", None)
    if close is not None:
        await close()


class Executor:
    def __init__(
        self,
        providers: dict[str, Provider],
        breaker: CircuitBreaker,
        latency: LatencyTracker,
        *,
        listener: ExecutionListener | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
        attempt_timeouts: dict[str, float] | None = None,
    ) -> None:
        self.providers = providers
        self.attempt_timeouts = attempt_timeouts or {}
        self.breaker = breaker
        self.latency = latency
        self.listener = listener or ExecutionListener()
        self.sleep = sleep
        self.clock = clock
        self.rng = rng or random.Random()

    # --- shared attempt loop ----------------------------------------------------------------
    async def _run(
        self,
        alias_name: str,
        alias: AliasConfig,
        targets: list[TargetConfig],
        deadline_s: float,
        call: Callable[[Provider, TargetConfig, float], Awaitable[T]],
    ) -> tuple[T, ExecInfo]:
        info = ExecInfo(alias=alias_name)
        deadline = self.clock() + deadline_s
        last_failure: tuple[str, str] | None = None  # (provider, reason) of the previous target

        for target in targets:
            provider = self.providers.get(target.provider)
            if provider is None:
                continue
            if not await self.breaker.allow(target.provider):
                open_error = ProviderError(
                    target.provider, "circuit open", reason=err.CIRCUIT_OPEN, retryable=False
                )
                info.errors.append(open_error)
                last_failure = (target.provider, err.CIRCUIT_OPEN)
                if err.CIRCUIT_OPEN not in alias.fallback_on:
                    break
                continue
            if last_failure is not None:
                info.fallback_used = True
                self.listener.on_fallback(
                    alias_name, last_failure[0], target.provider, last_failure[1]
                )

            error: ProviderError | None = None
            for attempt in range(alias.retry.max_attempts_per_target):
                remaining = deadline - self.clock()
                if remaining <= 0:
                    info.errors.append(
                        ProviderError(
                            target.provider,
                            "gateway deadline exceeded",
                            reason=err.TIMEOUT,
                            retryable=False,
                        )
                    )
                    raise AllTargetsFailedError(alias_name, info.errors)
                info.attempts += 1
                started = self.clock()
                try:
                    with tracer.start_as_current_span(
                        "gateway.attempt",
                        attributes={
                            "gateway.provider": target.provider,
                            "gateway.model": target.model,
                            "gateway.attempt": info.attempts,
                        },
                    ) as span:
                        try:
                            result = await call(provider, target, remaining)
                        except ProviderError as exc:
                            span.set_attribute("gateway.error_reason", exc.reason)
                            span.set_status(Status(StatusCode.ERROR, exc.reason))
                            raise
                except ProviderError as exc:
                    elapsed = self.clock() - started
                    info.provider_time_s += elapsed
                    error = exc
                    info.errors.append(exc)
                    self.listener.on_attempt(
                        target.provider, target.model, False, exc.reason, elapsed
                    )
                    if exc.reason in HEALTH_FAILURES:
                        await self.breaker.record_failure(target.provider)
                    else:
                        await self.breaker.record_success(target.provider)  # it answered; it's up
                    if not exc.retryable or attempt == alias.retry.max_attempts_per_target - 1:
                        break
                    delay = backoff_delay(
                        attempt,
                        alias.retry.backoff_ms,
                        jitter=alias.retry.jitter,
                        retry_after=exc.retry_after,
                        rng=self.rng,
                    )
                    if delay >= deadline - self.clock():
                        break  # waiting would blow the deadline; try the next target instead
                    self.listener.on_retry(target.provider, exc.reason)
                    await self.sleep(delay)
                    info.backoff_time_s += delay
                    continue
                elapsed = self.clock() - started
                info.provider_time_s += elapsed
                self.listener.on_attempt(target.provider, target.model, True, None, elapsed)
                self.latency.observe(target.key, elapsed)
                await self.breaker.record_success(target.provider)
                info.provider, info.model = target.provider, target.model
                return result, info

            assert error is not None
            if error.reason not in alias.fallback_on:
                break
            last_failure = (target.provider, error.reason)

        raise AllTargetsFailedError(alias_name, info.errors)

    def _attempt_budget(self, provider: str, remaining: float) -> float:
        """Seconds one non-streaming attempt may take: the provider cap, never past the deadline."""
        cap = self.attempt_timeouts.get(provider)
        return remaining if cap is None else min(cap, remaining)

    # --- public entry points ------------------------------------------------------------------
    async def chat(
        self,
        alias_name: str,
        alias: AliasConfig,
        targets: list[TargetConfig],
        req: ChatRequest,
        deadline_s: float,
    ) -> tuple[ChatResponse, ExecInfo]:
        async def call(p: Provider, t: TargetConfig, remaining: float) -> ChatResponse:
            return await p.chat(req, t.model, self._attempt_budget(t.provider, remaining))

        return await self._run(alias_name, alias, targets, deadline_s, call)

    async def embed(
        self,
        alias_name: str,
        alias: AliasConfig,
        targets: list[TargetConfig],
        req: EmbeddingRequest,
        deadline_s: float,
    ) -> tuple[EmbeddingResponse, ExecInfo]:
        async def call(p: Provider, t: TargetConfig, remaining: float) -> EmbeddingResponse:
            return await p.embed(req, t.model, self._attempt_budget(t.provider, remaining))

        return await self._run(alias_name, alias, targets, deadline_s, call)

    async def stream(
        self,
        alias_name: str,
        alias: AliasConfig,
        targets: list[TargetConfig],
        req: ChatRequest,
        deadline_s: float,
    ) -> tuple[AsyncIterator[StreamChunk], ExecInfo]:
        """An "attempt" succeeds when the FIRST chunk arrives. Only then do we commit to a target."""
        first_token_timeout = alias.first_token_timeout_ms / 1000

        async def call(
            p: Provider, t: TargetConfig, remaining: float
        ) -> tuple[StreamChunk, AsyncIterator[StreamChunk], float]:
            it = p.stream(req, t.model, remaining).__aiter__()
            started = self.clock()
            try:
                first = await asyncio.wait_for(
                    anext(it), timeout=min(first_token_timeout, remaining)
                )
            except StopAsyncIteration:
                first = StreamChunk(finish_reason="stop")
            except TimeoutError as exc:
                await _aclose(it)
                raise ProviderError(
                    t.provider, "no first token in time", reason=err.TIMEOUT, retryable=True
                ) from exc
            except BaseException:
                await _aclose(it)
                raise
            return first, it, self.clock() - started

        (first, it, ttft), info = await self._run(alias_name, alias, targets, deadline_s, call)
        info.ttft_s = ttft
        self.listener.on_ttft(info.provider, ttft)
        return self._continue(first, it, info, idle_timeout=first_token_timeout), info

    async def _continue(
        self,
        first: StreamChunk,
        it: AsyncIterator[StreamChunk],
        info: ExecInfo,
        *,
        idle_timeout: float,
    ) -> AsyncIterator[StreamChunk]:
        try:
            yield first
            if (
                first.finish_reason is not None
                and first.content is None
                and not first.tool_calls
                and first.usage is None
            ):
                return
            while True:
                try:
                    chunk = await asyncio.wait_for(anext(it), timeout=idle_timeout)
                except StopAsyncIteration:
                    return
                except (TimeoutError, ProviderError) as exc:
                    # Committed to this provider: no fallback now, surface an in-band error instead.
                    self.listener.on_stream_interrupted(info.provider)
                    await self.breaker.record_failure(info.provider)
                    raise ProviderError(
                        info.provider,
                        f"stream interrupted: {exc}",
                        reason=err.STREAM_INTERRUPTED,
                        retryable=False,
                    ) from exc
                yield chunk
        finally:
            await _aclose(it)
