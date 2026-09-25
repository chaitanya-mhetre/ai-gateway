"""Per-request orchestration around the core Gateway: admission control and settlement.

    principal ─▶ Guard.admit (budget, RPM, TPM reserve) ─▶ Gateway (route + execute)
              ─▶ Guard.settle (TPM reconcile, budget increment)

Kept separate from `app.py` so the whole flow is testable without HTTP.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from decimal import Decimal

from ai_gateway.auth import Principal
from ai_gateway.gateway import CallMeta, Gateway, RequestOptions
from ai_gateway.limits import Guard
from ai_gateway.metering.tokens import estimate_request_tokens, estimate_text_tokens
from ai_gateway.models import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    StreamChunk,
    Usage,
)


class GatewayService:
    def __init__(self, gateway: Gateway, guard: Guard) -> None:
        self.gateway = gateway
        self.guard = guard

    async def chat(
        self, principal: Principal, req: ChatRequest, opts: RequestOptions
    ) -> tuple[ChatResponse, CallMeta]:
        principal.check_alias(req.model)
        reservation = await self.guard.admit(principal, estimate_request_tokens(req))
        try:
            resp, meta = await self.gateway.chat(req, opts)
        except BaseException:
            await self.guard.settle(reservation, 0, Decimal(0))  # refund the whole reservation
            raise
        await self.guard.settle(reservation, resp.usage.total_tokens, Decimal(0))
        return resp, meta

    async def stream(
        self, principal: Principal, req: ChatRequest, opts: RequestOptions
    ) -> tuple[AsyncIterator[StreamChunk], CallMeta]:
        principal.check_alias(req.model)
        reservation = await self.guard.admit(principal, estimate_request_tokens(req))
        try:
            chunks, meta = await self.gateway.stream(req, opts)
        except BaseException:
            await self.guard.settle(reservation, 0, Decimal(0))
            raise

        async def settled() -> AsyncIterator[StreamChunk]:
            usage: Usage | None = None
            text_chars = 0
            try:
                async for chunk in chunks:
                    if chunk.usage is not None:
                        usage = chunk.usage
                    text_chars += len(chunk.content or "")
                    yield chunk
            finally:
                if usage is None:  # provider sent no usage: estimate, never silently charge zero
                    usage = Usage(
                        completion_tokens=estimate_text_tokens("x" * text_chars), estimated=True
                    )
                await self.guard.settle(reservation, usage.total_tokens, Decimal(0))

        return settled(), meta

    async def embed(
        self, principal: Principal, req: EmbeddingRequest, opts: RequestOptions
    ) -> tuple[EmbeddingResponse, CallMeta]:
        principal.check_alias(req.model)
        estimate = sum(estimate_text_tokens(t) for t in req.input)
        reservation = await self.guard.admit(principal, estimate)
        try:
            resp, meta = await self.gateway.embed(req, opts)
        except BaseException:
            await self.guard.settle(reservation, 0, Decimal(0))
            raise
        await self.guard.settle(reservation, resp.usage.total_tokens, Decimal(0))
        return resp, meta
