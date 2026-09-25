"""FastAPI application: the HTTP surface. The orchestration lives in `service.py` and `gateway.py`."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from redis.asyncio import Redis

from ai_gateway import admin
from ai_gateway.auth import ANONYMOUS, KeyAuthenticator, Principal
from ai_gateway.config import GatewayConfig, Settings
from ai_gateway.db import Database
from ai_gateway.errors import GatewayError, InvalidRequestError
from ai_gateway.gateway import CallMeta, Gateway, RequestOptions
from ai_gateway.limits import (
    BudgetStore,
    Guard,
    InMemoryBudgetStore,
    InMemoryLimiter,
    Limiter,
    RedisBudgetStore,
    RedisLimiter,
)
from ai_gateway.openai_compat import (
    ChunkEncoder,
    OpenAIChatRequest,
    OpenAIEmbeddingRequest,
    embedding_to_internal,
    embeddings_to_openai,
    new_request_id,
    response_to_openai,
    to_internal,
)
from ai_gateway.providers.base import Provider
from ai_gateway.redis_client import make_redis
from ai_gateway.registry import build_providers
from ai_gateway.routing.breaker import CircuitBreaker, InMemoryBreaker, RedisBreaker
from ai_gateway.service import GatewayService

_POLICIES = {"priority", "weighted", "cost", "latency"}
_CACHE_MODES = {"bypass", "exact", "semantic"}


def parse_options(request: Request) -> RequestOptions:
    """Read the optional `X-Gateway-*` request headers."""
    policy = request.headers.get("x-gateway-route-policy")
    if policy is not None and policy not in _POLICIES:
        raise InvalidRequestError(f"X-Gateway-Route-Policy must be one of {sorted(_POLICIES)}")
    cache = request.headers.get("x-gateway-cache")
    if cache is not None and cache not in _CACHE_MODES:
        raise InvalidRequestError(f"X-Gateway-Cache must be one of {sorted(_CACHE_MODES)}")
    timeout_raw = request.headers.get("x-gateway-timeout-ms")
    timeout_ms: int | None = None
    if timeout_raw is not None:
        if not timeout_raw.isdigit() or int(timeout_raw) <= 0:
            raise InvalidRequestError("X-Gateway-Timeout-Ms must be a positive integer")
        timeout_ms = int(timeout_raw)
    return RequestOptions(policy=policy, timeout_ms=timeout_ms, cache=cache)  # type: ignore[arg-type]


def _meta_headers(meta: CallMeta, request_id: str) -> dict[str, str]:
    return {
        "X-Request-Id": request_id,
        "X-Gateway-Provider": meta.provider,
        "X-Gateway-Model": meta.model,
        "X-Gateway-Attempts": str(meta.attempts),
        "X-Gateway-Cache": meta.cache,
    }


def create_app(
    settings: Settings | None = None,
    *,
    config: GatewayConfig | None = None,
    providers: dict[str, Provider] | None = None,
    redis: Redis | None = None,
) -> FastAPI:
    settings = settings or Settings()
    config = config or GatewayConfig.load(settings.config_path)
    if redis is None and settings.redis_url:
        redis = make_redis(settings.redis_url)

    # Shared state lives in Redis when available (multi-replica safe), else in memory.
    breaker: CircuitBreaker = (
        RedisBreaker(redis, config.breaker) if redis else InMemoryBreaker(config.breaker)
    )
    limiter: Limiter = RedisLimiter(redis) if redis else InMemoryLimiter()
    budgets: BudgetStore = RedisBudgetStore(redis) if redis else InMemoryBudgetStore()

    gateway = Gateway(
        config, providers if providers is not None else build_providers(config), breaker=breaker
    )
    db = Database(settings.database_url)
    authenticator = KeyAuthenticator(db, settings.key_pepper, cache_ttl_s=settings.key_cache_ttl_s)
    service = GatewayService(gateway, Guard(limiter, budgets))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await db.create_all()
        yield
        await gateway.aclose()
        await db.dispose()
        if redis is not None:
            await redis.aclose()

    app = FastAPI(title="ai-gateway", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.gateway = gateway
    app.state.db = db
    app.state.authenticator = authenticator
    app.state.service = service
    app.include_router(admin.router)

    async def principal_for(request: Request) -> Principal:
        if not settings.auth_enabled:
            return ANONYMOUS
        return await authenticator.authenticate(request.headers.get("authorization"))

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next: Any) -> Any:
        length = request.headers.get("content-length")
        if length is not None and length.isdigit() and int(length) > settings.max_body_bytes:
            err = InvalidRequestError(f"request body exceeds {settings.max_body_bytes} bytes")
            return JSONResponse(err.to_openai(), status_code=413)
        return await call_next(request)

    @app.exception_handler(GatewayError)
    async def _gateway_error(_: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(exc.to_openai(), status_code=exc.status_code, headers=exc.headers())

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        msg = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        return JSONResponse(
            {"error": {"message": msg, "type": "invalid_request_error", "code": None}},
            status_code=400,
        )

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(
        body: OpenAIChatRequest, request: Request
    ) -> JSONResponse | StreamingResponse:
        request_id = new_request_id()
        principal = await principal_for(request)
        opts = parse_options(request)
        req = to_internal(body, max_tokens_cap=settings.max_tokens_cap)
        if req.stream:
            chunks, meta = await service.stream(principal, req, opts)
            encoder = ChunkEncoder(request_id, meta.model or req.model)

            async def sse() -> AsyncIterator[str]:
                try:
                    async for chunk in chunks:
                        for event in encoder.encode(chunk):
                            yield event
                except GatewayError as exc:
                    # Headers are already sent; the only honest thing left is an in-band error event.
                    yield ChunkEncoder.error(exc.message, exc.error_type)
                yield ChunkEncoder.done()

            return StreamingResponse(
                sse(), media_type="text/event-stream", headers=_meta_headers(meta, request_id)
            )
        resp, meta = await service.chat(principal, req, opts)
        return JSONResponse(
            response_to_openai(resp, request_id), headers=_meta_headers(meta, request_id)
        )

    @app.post("/v1/embeddings")
    async def embeddings(body: OpenAIEmbeddingRequest, request: Request) -> JSONResponse:
        request_id = new_request_id()
        principal = await principal_for(request)
        resp, meta = await service.embed(
            principal, embedding_to_internal(body), parse_options(request)
        )
        return JSONResponse(embeddings_to_openai(resp), headers=_meta_headers(meta, request_id))

    @app.get("/v1/models")
    async def list_models(request: Request) -> dict[str, Any]:
        principal = await principal_for(request)
        visible = [
            a
            for a in config.aliases
            if principal.allowed_aliases is None or a in principal.allowed_aliases
        ]
        return {
            "object": "list",
            "data": [{"id": a, "object": "model", "owned_by": "gateway"} for a in visible],
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> dict[str, Any]:
        return {"status": "ready", "providers": sorted(gateway.providers)}

    return app
