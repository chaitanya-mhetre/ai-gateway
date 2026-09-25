"""The request pipeline, independent of HTTP (so it can be unit-tested without FastAPI).

resolve alias → plan targets (router) → execute with retries/fallback (executor)
"""

from __future__ import annotations

import random
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from ai_gateway.config import AliasConfig, GatewayConfig, PolicyName, TargetConfig
from ai_gateway.errors import AllTargetsFailedError, ModelNotFoundError
from ai_gateway.metering.tokens import estimate_prompt_tokens
from ai_gateway.models import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    StreamChunk,
)
from ai_gateway.providers.base import Provider
from ai_gateway.routing.breaker import CircuitBreaker, InMemoryBreaker
from ai_gateway.routing.executor import ExecInfo, ExecutionListener, Executor
from ai_gateway.routing.health import LatencyTracker
from ai_gateway.routing.router import PriceLookup, RequestFeatures, plan


@dataclass(frozen=True)
class RequestOptions:
    """Per-request knobs, from the `X-Gateway-*` headers."""

    policy: PolicyName | None = None
    timeout_ms: int | None = None
    cache: str | None = None  # bypass | exact | semantic


@dataclass
class CallMeta:
    """What happened while serving one request; becomes response headers + a usage event."""

    alias: str
    provider: str = ""
    model: str = ""
    attempts: int = 0
    fallback_used: bool = False
    cache: str = "miss"
    similarity: float | None = None
    ttft_ms: float | None = None
    provider_time_s: float = 0.0
    errors: list[str] = field(default_factory=list)
    error_providers: list[str] = field(default_factory=list)

    @classmethod
    def from_exec(cls, info: ExecInfo) -> CallMeta:
        return cls(
            alias=info.alias,
            provider=info.provider,
            model=info.model,
            attempts=info.attempts,
            fallback_used=info.fallback_used,
            ttft_ms=info.ttft_s * 1000 if info.ttft_s is not None else None,
            provider_time_s=info.provider_time_s,
            errors=[e.reason for e in info.errors],
            error_providers=[e.provider for e in info.errors],
        )


class Gateway:
    def __init__(
        self,
        config: GatewayConfig,
        providers: dict[str, Provider],
        *,
        breaker: CircuitBreaker | None = None,
        listener: ExecutionListener | None = None,
        price_of: PriceLookup | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.config = config
        self.providers = providers
        self.latency = LatencyTracker()
        self.breaker: CircuitBreaker = breaker or InMemoryBreaker(config.breaker)
        self.price_of = price_of
        self.listener = listener
        self.rng = rng or random.Random()
        self.executor = Executor(
            providers,
            self.breaker,
            self.latency,
            listener=listener,
            rng=self.rng,
            attempt_timeouts={name: pc.timeout_ms / 1000 for name, pc in config.providers.items()},
        )

    # --- resolution -----------------------------------------------------------------------------
    def resolve(self, model: str) -> tuple[str, AliasConfig]:
        """`model` is an alias (`chat-default`) or an explicit `provider:model` (single target)."""
        if model in self.config.aliases:
            return model, self.config.aliases[model]
        provider, sep, concrete = model.partition(":")
        if sep and provider in self.providers and concrete:
            return model, AliasConfig(targets=[TargetConfig(provider=provider, model=concrete)])
        raise ModelNotFoundError(f"unknown model or alias '{model}'")

    def _deadline_s(self, alias: AliasConfig, opts: RequestOptions) -> float:
        ms = (
            alias.deadline_ms
            if opts.timeout_ms is None
            else min(opts.timeout_ms, alias.deadline_ms)
        )
        return max(ms, 1) / 1000

    def _plan(
        self, alias_name: str, alias: AliasConfig, req: ChatRequest, opts: RequestOptions
    ) -> list[TargetConfig]:
        features = RequestFeatures(
            needs_tools=bool(req.tools),
            stream=req.stream,
            estimated_prompt_tokens=estimate_prompt_tokens(req),
        )
        targets = plan(
            alias,
            features,
            latency=self.latency,
            price_of=self.price_of,
            policy_override=opts.policy,
            rng=self.rng,
        )
        if not targets:
            raise AllTargetsFailedError(alias_name, [])
        return targets

    # --- entry points ---------------------------------------------------------------------------
    async def chat(
        self, req: ChatRequest, opts: RequestOptions | None = None
    ) -> tuple[ChatResponse, CallMeta]:
        opts = opts or RequestOptions()
        alias_name, alias = self.resolve(req.model)
        targets = self._plan(alias_name, alias, req, opts)
        resp, info = await self.executor.chat(
            alias_name, alias, targets, req, self._deadline_s(alias, opts)
        )
        return resp, CallMeta.from_exec(info)

    async def stream(
        self, req: ChatRequest, opts: RequestOptions | None = None
    ) -> tuple[AsyncIterator[StreamChunk], CallMeta]:
        opts = opts or RequestOptions()
        alias_name, alias = self.resolve(req.model)
        targets = self._plan(alias_name, alias, req, opts)
        chunks, info = await self.executor.stream(
            alias_name, alias, targets, req, self._deadline_s(alias, opts)
        )
        return chunks, CallMeta.from_exec(info)

    async def embed(
        self, req: EmbeddingRequest, opts: RequestOptions | None = None
    ) -> tuple[EmbeddingResponse, CallMeta]:
        opts = opts or RequestOptions()
        alias_name, alias = self.resolve(req.model)
        resp, info = await self.executor.embed(
            alias_name, alias, list(alias.targets), req, self._deadline_s(alias, opts)
        )
        return resp, CallMeta.from_exec(info)

    async def aclose(self) -> None:
        for p in self.providers.values():
            await p.aclose()
