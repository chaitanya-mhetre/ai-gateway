"""Build provider adapters from config. Secrets are read from env vars named in the config."""

from __future__ import annotations

import os

from ai_gateway.config import GatewayConfig, ProviderConfig
from ai_gateway.providers.base import Provider
from ai_gateway.providers.mock import MockProvider
from ai_gateway.providers.openai import OpenAIProvider

DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
}


def build_provider(name: str, cfg: ProviderConfig) -> Provider:
    api_key = os.environ.get(cfg.api_key_env) if cfg.api_key_env else None
    base_url = cfg.base_url or DEFAULT_BASE_URLS.get(cfg.type, "")
    if cfg.type == "mock":
        return MockProvider(name, latency=cfg.mock.latency_ms / 1000, reply=cfg.mock.reply)
    if cfg.type == "openai":
        return OpenAIProvider(name, base_url, api_key)
    raise ValueError(f"provider type '{cfg.type}' not supported yet")


def build_providers(config: GatewayConfig) -> dict[str, Provider]:
    return {
        name: build_provider(name, cfg) for name, cfg in config.providers.items() if cfg.enabled
    }
