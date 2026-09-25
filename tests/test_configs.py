from __future__ import annotations

from pathlib import Path

import pytest

from ai_gateway.config import GatewayConfig

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "path", ["config/gateway.yaml", "config/gateway.example.yaml", "bench/gateway.bench.yaml"]
)
def test_shipped_configs_are_valid(path: str) -> None:
    GatewayConfig.load(ROOT / path)


def test_unknown_provider_reference_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        GatewayConfig.model_validate(
            {
                "providers": {"a": {"type": "mock"}},
                "aliases": {"x": {"targets": [{"provider": "b", "model": "m"}]}},
            }
        )
