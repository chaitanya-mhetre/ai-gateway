"""Configuration: process settings (env vars) + routing config (YAML)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_gateway import errors

ProviderType = Literal["openai", "anthropic", "gemini", "ollama", "mock"]
PolicyName = Literal["priority", "weighted", "cost", "latency"]


class Settings(BaseSettings):
    """Process-level settings, read from environment variables prefixed `GATEWAY_`."""

    model_config = SettingsConfigDict(env_prefix="GATEWAY_", env_file=".env", extra="ignore")

    config_path: Path = Path("config/gateway.yaml")
    redis_url: str | None = None  # None → in-memory state (single replica only)
    database_url: str = "sqlite+aiosqlite:///./gateway.db"
    key_pepper: str = "change-me-pepper"
    auth_enabled: bool = True
    otel_enabled: bool = False
    log_prompts: bool = False  # prompt/response logging is OFF by default
    max_body_bytes: int = 1_000_000
    max_tokens_cap: int = 16_384
    key_cache_ttl_s: float = 30.0
    # Run `alembic upgrade head` on startup. Convenient for dev/tests; in production run
    # `ai-gateway migrate` as a separate release step and set this to false.
    auto_migrate: bool = True


class MockOptions(BaseModel):
    latency_ms: int = 0
    reply: str = "Hello from the mock provider."


class ProviderConfig(BaseModel):
    type: ProviderType
    base_url: str | None = None
    api_key_env: str | None = (
        None  # name of the env var holding the secret; never the secret itself
    )
    # Max duration of ONE non-streaming attempt (capped by the request's remaining deadline).
    # Keep it well below the alias `deadline_ms` so a hung provider leaves time to fall back.
    timeout_ms: int = Field(default=30_000, ge=1)
    enabled: bool = True
    mock: MockOptions = Field(default_factory=MockOptions)


class TargetConfig(BaseModel):
    provider: str
    model: str
    priority: int = 1
    weight: int = 1
    max_context: int = 128_000
    supports_tools: bool = True
    supports_stream: bool = True

    @property
    def key(self) -> str:
        return f"{self.provider}:{self.model}"


class RetryConfig(BaseModel):
    max_attempts_per_target: int = Field(default=2, ge=1, le=5)
    backoff_ms: list[int] = Field(default_factory=lambda: [200, 800])
    jitter: bool = True


class Requirements(BaseModel):
    supports_tools: bool | None = None
    min_context: int | None = None


class AliasConfig(BaseModel):
    policy: PolicyName = "priority"
    targets: list[TargetConfig]
    fallback_on: list[str] = Field(
        default_factory=lambda: [
            errors.TIMEOUT,
            errors.HTTP_5XX,
            errors.HTTP_429,
            errors.CONNECTION,
            errors.CIRCUIT_OPEN,
            errors.CONTEXT_LENGTH,
            errors.PROVIDER_AUTH,
            errors.MALFORMED_RESPONSE,
            errors.STREAM_INTERRUPTED,
        ]
    )
    retry: RetryConfig = Field(default_factory=RetryConfig)
    deadline_ms: int = 30_000
    first_token_timeout_ms: int = 10_000
    requirements: Requirements = Field(default_factory=Requirements)
    embedding: bool = False


class BreakerConfig(BaseModel):
    window_s: float = 30.0
    min_requests: int = 10
    failure_ratio: float = 0.5
    cooldown_s: float = 15.0


class CacheConfig(BaseModel):
    exact_enabled: bool = True
    exact_ttl_s: int = 3600
    semantic_enabled: bool = False
    semantic_threshold: float = 0.95
    semantic_embedding_alias: str = "embed-default"


class GatewayConfig(BaseModel):
    providers: dict[str, ProviderConfig]
    aliases: dict[str, AliasConfig] = Field(default_factory=dict)
    breaker: BreakerConfig = Field(default_factory=BreakerConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    price_table_path: Path | None = None

    @model_validator(mode="after")
    def _targets_reference_known_providers(self) -> GatewayConfig:
        for alias, cfg in self.aliases.items():
            for t in cfg.targets:
                if t.provider not in self.providers:
                    raise ValueError(f"alias '{alias}' references unknown provider '{t.provider}'")
        return self

    @classmethod
    def load(cls, path: Path) -> GatewayConfig:
        raw = yaml.safe_load(path.read_text())
        cfg = cls.model_validate(raw)
        if cfg.price_table_path is not None and not cfg.price_table_path.is_absolute():
            cfg.price_table_path = (path.parent / cfg.price_table_path).resolve()
        return cfg
