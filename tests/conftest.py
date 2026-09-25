from __future__ import annotations

from collections.abc import AsyncIterator, Callable

import httpx
import pytest

from ai_gateway.app import create_app
from ai_gateway.config import GatewayConfig, Settings
from ai_gateway.providers.base import Provider
from ai_gateway.providers.mock import MockProvider


def make_config(**aliases: object) -> GatewayConfig:
    raw: dict[str, object] = {
        "providers": {
            "primary": {"type": "mock"},
            "secondary": {"type": "mock"},
            "local": {"type": "mock"},
        },
        "aliases": aliases
        or {
            "chat-default": {
                "targets": [
                    {"provider": "primary", "model": "p-model", "priority": 1},
                    {"provider": "secondary", "model": "s-model", "priority": 2},
                ],
                "retry": {"max_attempts_per_target": 1, "backoff_ms": [0], "jitter": False},
            },
            "embed-default": {
                "embedding": True,
                "targets": [{"provider": "local", "model": "embed-model"}],
            },
        },
    }
    return GatewayConfig.model_validate(raw)


@pytest.fixture
def settings() -> Settings:
    return Settings(auth_enabled=False, redis_url=None, database_url="sqlite+aiosqlite:///:memory:")


@pytest.fixture
def providers() -> dict[str, Provider]:
    return {
        "primary": MockProvider("primary"),
        "secondary": MockProvider("secondary"),
        "local": MockProvider("local"),
    }


AppClientFactory = Callable[..., httpx.AsyncClient]


@pytest.fixture
async def client(
    settings: Settings, providers: dict[str, Provider]
) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(settings, config=make_config(), providers=providers)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://gw") as c:
        yield c
