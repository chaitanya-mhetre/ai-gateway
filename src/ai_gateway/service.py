"""Per-request orchestration around the core Gateway.

    principal ─▶ Guard.admit (budget, RPM, TPM reserve)
              ─▶ Gateway (route + execute with retries/fallback)
              ─▶ cost from the price table ─▶ Guard.settle (TPM reconcile, budget)
              ─▶ usage event (async sink) + Prometheus metrics + trace span

Kept separate from `app.py` so the whole flow is testable without HTTP.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from decimal import Decimal

from ai_gateway.auth import Principal
from ai_gateway.errors import AllTargetsFailedError, GatewayError, RateLimitedError
from ai_gateway.gateway import CallMeta, Gateway, RequestOptions
from ai_gateway.limits import Guard, Reservation
from ai_gateway.metering.pricing import PriceTable
from ai_gateway.metering.tokens import estimate_request_tokens, estimate_text_tokens
from ai_gateway.metering.usage import UsageRecord, UsageSink
from ai_gateway.models import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    StreamChunk,
    Usage,
)
from ai_gateway.observability.metrics import Metrics
from ai_gateway.observability.tracing import tracer
from ai_gateway.routing.breaker import STATE_GAUGE_VALUE


class GatewayService:
    def __init__(
        self,
        gateway: Gateway,
        guard: Guard,
        *,
        prices: PriceTable,
        sink: UsageSink,
        metrics: Metrics,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.gateway = gateway
        self.guard = guard
        self.prices = prices
        self.sink = sink
        self.metrics = metrics
        self.clock = clock

    # --- helpers --------------------------------------------------------------------------------
    async def _admit(self, principal: Principal, alias: str, estimate: int) -> Reservation:
        principal.check_alias(alias)
        try:
            return await self.guard.admit(principal, estimate)
        except RateLimitedError as exc:
            self.metrics.rate_limited.labels(principal.key_prefix, exc.error_type).inc()
            raise

    async def _refresh_circuit_gauges(self, providers: set[str]) -> None:
        for name in providers:
            if name:
                state = await self.gateway.breaker.state(name)
                self.metrics.circuit.labels(name).set(STATE_GAUGE_VALUE[state])

    async def _record(
        self,
        *,
        request_id: str,
        principal: Principal,
        alias: str,
        meta: CallMeta | None,
        usage: Usage,
        cost: Decimal | None,
        started: float,
        stream: bool,
        status_code: int = 200,
        error_type: str | None = None,
    ) -> None:
        elapsed = self.clock() - started
        provider = meta.provider if meta else ""
        model = meta.model if meta else ""
        status = "ok" if status_code < 400 else error_type or "error"
        m = self.metrics
        m.requests.labels(alias, provider or "none", model or "none", status).inc()
        m.duration.labels(alias, provider or "none").observe(elapsed)
        if (
            meta is not None
            and provider
            and status_code < 400
            and not stream
            and meta.cache == "miss"
        ):
            m.overhead.observe(max(elapsed - meta.provider_time_s, 0.0))
        if provider:
            m.tokens.labels(provider, model, "prompt").inc(usage.prompt_tokens)
            m.tokens.labels(provider, model, "completion").inc(usage.completion_tokens)
            m.tokens.labels(provider, model, "cached").inc(usage.cached_tokens)
            if cost:
                m.cost.labels(principal.project_id or "anonymous", provider, model).inc(float(cost))
        await self._refresh_circuit_gauges({provider, *(meta.error_providers if meta else [])})
        await self.sink.publish(
            UsageRecord(
                request_id=request_id,
                project_id=principal.project_id,
                api_key_id=principal.key_id,
                alias=alias,
                provider=provider,
                model=model,
                attempt_count=meta.attempts if meta else 0,
                fallback_used=meta.fallback_used if meta else False,
                cache=meta.cache if meta else "miss",
                stream=stream,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cached_tokens=usage.cached_tokens,
                tokens_estimated=usage.estimated,
                est_cost_usd=cost,
                latency_ms=round(elapsed * 1000),
                ttft_ms=round(meta.ttft_ms) if meta and meta.ttft_ms is not None else None,
                status_code=status_code,
                error_type=error_type,
            )
        )

    @staticmethod
    def _failure_meta(alias: str, exc: GatewayError) -> CallMeta | None:
        if isinstance(exc, AllTargetsFailedError):
            return CallMeta(
                alias=alias,
                attempts=len(exc.errors),
                errors=[e.reason for e in exc.errors],
                error_providers=[e.provider for e in exc.errors],
            )
        return None

    async def _fail(
        self,
        exc: GatewayError,
        reservation: Reservation,
        *,
        request_id: str,
        principal: Principal,
        alias: str,
        started: float,
        stream: bool,
    ) -> None:
        await self.guard.settle(reservation, 0, Decimal(0))  # refund the whole reservation
        await self._record(
            request_id=request_id,
            principal=principal,
            alias=alias,
            meta=self._failure_meta(alias, exc),
            usage=Usage(),
            cost=None,
            started=started,
            stream=stream,
            status_code=exc.status_code,
            error_type=exc.error_type,
        )

    # --- entry points ---------------------------------------------------------------------------
    async def chat(
        self, principal: Principal, req: ChatRequest, opts: RequestOptions, request_id: str
    ) -> tuple[ChatResponse, CallMeta]:
        started = self.clock()
        with tracer.start_as_current_span(
            "gateway.request", attributes={"gateway.alias": req.model, "gateway.stream": False}
        ) as span:
            reservation = await self._admit(principal, req.model, estimate_request_tokens(req))
            try:
                resp, meta = await self.gateway.chat(req, opts)
            except GatewayError as exc:
                await self._fail(
                    exc,
                    reservation,
                    request_id=request_id,
                    principal=principal,
                    alias=req.model,
                    started=started,
                    stream=False,
                )
                raise
            cost = self.prices.cost(meta.provider, meta.model, resp.usage)
            await self.guard.settle(reservation, resp.usage.total_tokens, cost or Decimal(0))
            await self._record(
                request_id=request_id,
                principal=principal,
                alias=req.model,
                meta=meta,
                usage=resp.usage,
                cost=cost,
                started=started,
                stream=False,
            )
            span.set_attributes(
                {
                    "gateway.provider": meta.provider,
                    "gateway.model": meta.model,
                    "gateway.attempts": meta.attempts,
                    "gateway.prompt_tokens": resp.usage.prompt_tokens,
                    "gateway.completion_tokens": resp.usage.completion_tokens,
                }
            )
            return resp, meta

    async def stream(
        self, principal: Principal, req: ChatRequest, opts: RequestOptions, request_id: str
    ) -> tuple[AsyncIterator[StreamChunk], CallMeta]:
        started = self.clock()
        reservation = await self._admit(principal, req.model, estimate_request_tokens(req))
        try:
            chunks, meta = await self.gateway.stream(req, opts)
        except GatewayError as exc:
            await self._fail(
                exc,
                reservation,
                request_id=request_id,
                principal=principal,
                alias=req.model,
                started=started,
                stream=True,
            )
            raise

        async def metered() -> AsyncIterator[StreamChunk]:
            usage: Usage | None = None
            text_chars = 0
            error: GatewayError | None = None
            try:
                async for chunk in chunks:
                    if chunk.usage is not None:
                        usage = chunk.usage
                    text_chars += len(chunk.content or "")
                    yield chunk
            except GatewayError as exc:
                error = exc
                raise
            finally:
                if usage is None:  # provider sent no usage: estimate, never silently charge zero
                    usage = Usage(
                        completion_tokens=estimate_text_tokens("x" * text_chars), estimated=True
                    )
                cost = self.prices.cost(meta.provider, meta.model, usage)
                await self.guard.settle(reservation, usage.total_tokens, cost or Decimal(0))
                await self._record(
                    request_id=request_id,
                    principal=principal,
                    alias=req.model,
                    meta=meta,
                    usage=usage,
                    cost=cost,
                    started=started,
                    stream=True,
                    status_code=200 if error is None else 502,
                    error_type=None if error is None else error.error_type,
                )

        return metered(), meta

    async def embed(
        self, principal: Principal, req: EmbeddingRequest, opts: RequestOptions, request_id: str
    ) -> tuple[EmbeddingResponse, CallMeta]:
        started = self.clock()
        reservation = await self._admit(
            principal, req.model, sum(estimate_text_tokens(t) for t in req.input)
        )
        try:
            resp, meta = await self.gateway.embed(req, opts)
        except GatewayError as exc:
            await self._fail(
                exc,
                reservation,
                request_id=request_id,
                principal=principal,
                alias=req.model,
                started=started,
                stream=False,
            )
            raise
        cost = self.prices.cost(meta.provider, meta.model, resp.usage)
        await self.guard.settle(reservation, resp.usage.total_tokens, cost or Decimal(0))
        await self._record(
            request_id=request_id,
            principal=principal,
            alias=req.model,
            meta=meta,
            usage=resp.usage,
            cost=cost,
            started=started,
            stream=False,
        )
        return resp, meta
