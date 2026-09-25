"""The request pipeline, independent of HTTP (so it can be unit-tested without FastAPI)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from ai_gateway.config import GatewayConfig, TargetConfig
from ai_gateway.errors import ModelNotFoundError
from ai_gateway.models import (
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    StreamChunk,
)
from ai_gateway.providers.base import Provider


@dataclass
class CallMeta:
    """What happened while serving one request; becomes response headers + a usage event."""

    alias: str
    provider: str = ""
    model: str = ""
    attempts: int = 0
    fallback_used: bool = False
    cache: str = "miss"
    ttft_ms: float | None = None
    errors: list[str] = field(default_factory=list)


class Gateway:
    def __init__(self, config: GatewayConfig, providers: dict[str, Provider]) -> None:
        self.config = config
        self.providers = providers

    def resolve_targets(self, model: str) -> list[TargetConfig]:
        """`model` is an alias (`chat-default`) or an explicit `provider:model`."""
        if model in self.config.aliases:
            return sorted(self.config.aliases[model].targets, key=lambda t: t.priority)
        provider, sep, concrete = model.partition(":")
        if sep and provider in self.providers and concrete:
            return [TargetConfig(provider=provider, model=concrete)]
        raise ModelNotFoundError(f"unknown model or alias '{model}'")

    async def chat(self, req: ChatRequest) -> tuple[ChatResponse, CallMeta]:
        target = self.resolve_targets(req.model)[0]
        meta = CallMeta(alias=req.model, provider=target.provider, model=target.model, attempts=1)
        resp = await self.providers[target.provider].chat(req, target.model, 30.0)
        return resp, meta

    async def stream(self, req: ChatRequest) -> tuple[AsyncIterator[StreamChunk], CallMeta]:
        target = self.resolve_targets(req.model)[0]
        meta = CallMeta(alias=req.model, provider=target.provider, model=target.model, attempts=1)
        return self.providers[target.provider].stream(req, target.model, 30.0), meta

    async def embed(self, req: EmbeddingRequest) -> tuple[EmbeddingResponse, CallMeta]:
        target = self.resolve_targets(req.model)[0]
        meta = CallMeta(alias=req.model, provider=target.provider, model=target.model, attempts=1)
        return await self.providers[target.provider].embed(req, target.model, 30.0), meta

    async def aclose(self) -> None:
        for p in self.providers.values():
            await p.aclose()
