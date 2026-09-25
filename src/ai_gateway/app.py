"""FastAPI application: the HTTP surface. Business logic lives in `gateway.py`."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from ai_gateway.config import GatewayConfig, Settings
from ai_gateway.errors import GatewayError
from ai_gateway.gateway import CallMeta, Gateway
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
from ai_gateway.registry import build_providers


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
) -> FastAPI:
    settings = settings or Settings()
    config = config or GatewayConfig.load(settings.config_path)
    gateway = Gateway(config, providers if providers is not None else build_providers(config))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await gateway.aclose()

    app = FastAPI(title="ai-gateway", version="0.1.0", lifespan=lifespan)
    app.state.gateway = gateway
    app.state.settings = settings

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
    async def chat_completions(body: OpenAIChatRequest) -> JSONResponse | StreamingResponse:
        request_id = new_request_id()
        req = to_internal(body, max_tokens_cap=settings.max_tokens_cap)
        if req.stream:
            chunks, meta = await gateway.stream(req)
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
        resp, meta = await gateway.chat(req)
        return JSONResponse(
            response_to_openai(resp, request_id), headers=_meta_headers(meta, request_id)
        )

    @app.post("/v1/embeddings")
    async def embeddings(body: OpenAIEmbeddingRequest) -> JSONResponse:
        request_id = new_request_id()
        resp, meta = await gateway.embed(embedding_to_internal(body))
        return JSONResponse(embeddings_to_openai(resp), headers=_meta_headers(meta, request_id))

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        data = [{"id": alias, "object": "model", "owned_by": "gateway"} for alias in config.aliases]
        return {"object": "list", "data": data}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready() -> dict[str, Any]:
        return {"status": "ready", "providers": sorted(gateway.providers)}

    return app
