"""Build provider adapters from config. Secrets are read from env vars named in the config."""

from __future__ import annotations

import os

from ai_gateway.config import GatewayConfig, ProviderConfig
from ai_gateway.providers.anthropic import AnthropicProvider
from ai_gateway.providers.base import Provider
from ai_gateway.providers.gemini import GeminiProvider
from ai_gateway.providers.mock import MockProvider
from ai_gateway.providers.ollama import OllamaProvider
from ai_gateway.providers.openai import OpenAIProvider

DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "ollama": "http://localhost:11434",
}


def build_provider(name: str, cfg: ProviderConfig) -> Provider:
    api_key = os.environ.get(cfg.api_key_env) if cfg.api_key_env else None
    base_url = cfg.base_url or DEFAULT_BASE_URLS.get(cfg.type, "")
    if cfg.type == "mock":
        return MockProvider(name, latency=cfg.mock.latency_ms / 1000, reply=cfg.mock.reply)
    if cfg.type == "openai":
        return OpenAIProvider(name, base_url, api_key)
    if cfg.type == "anthropic":
        return AnthropicProvider(name, base_url, api_key)
    if cfg.type == "gemini":
        return GeminiProvider(name, base_url, api_key)
    return OllamaProvider(name, base_url, api_key)


def build_providers(config: GatewayConfig) -> dict[str, Provider]:
    return {
        name: build_provider(name, cfg) for name, cfg in config.providers.items() if cfg.enabled
    }
