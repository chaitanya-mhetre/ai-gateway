from __future__ import annotations

import random
from collections import Counter

from ai_gateway.config import AliasConfig, TargetConfig
from ai_gateway.routing.health import LatencyTracker
from ai_gateway.routing.router import RequestFeatures, eligible, plan


def alias(policy: str = "priority", **req: object) -> AliasConfig:
    return AliasConfig.model_validate(
        {
            "policy": policy,
            "requirements": req,
            "targets": [
                {
                    "provider": "a",
                    "model": "big",
                    "priority": 2,
                    "weight": 1,
                    "max_context": 200_000,
                },
                {
                    "provider": "b",
                    "model": "cheap",
                    "priority": 1,
                    "weight": 3,
                    "max_context": 8_000,
                    "supports_tools": False,
                },
                {
                    "provider": "c",
                    "model": "local",
                    "priority": 3,
                    "weight": 1,
                    "max_context": 32_000,
                    "supports_stream": False,
                },
            ],
        }
    )


def providers(targets: list[TargetConfig]) -> list[str]:
    return [t.provider for t in targets]


def test_eligibility_filters_capabilities_and_context() -> None:
    a = alias()
    assert providers(eligible(a, RequestFeatures(needs_tools=True))) == ["a", "c"]
    assert providers(eligible(a, RequestFeatures(stream=True))) == ["a", "b"]
    assert providers(eligible(a, RequestFeatures(estimated_prompt_tokens=10_000))) == ["a", "c"]
    assert providers(eligible(alias(min_context=16_000), RequestFeatures())) == ["a", "c"]


def test_priority_policy() -> None:
    assert providers(plan(alias(), RequestFeatures())) == ["b", "a", "c"]


def test_policy_override_wins() -> None:
    lat = LatencyTracker()
    lat.observe("a:big", 0.1)
    lat.observe("b:cheap", 0.9)
    lat.observe("c:local", 0.5)
    assert providers(plan(alias(), RequestFeatures(), latency=lat, policy_override="latency")) == [
        "a",
        "c",
        "b",
    ]


def test_latency_policy_samples_unknown_targets_first() -> None:
    lat = LatencyTracker()
    lat.observe("a:big", 0.2)
    assert providers(plan(alias("latency"), RequestFeatures(), latency=lat)) == ["b", "c", "a"]


def test_cost_policy_puts_unknown_price_last() -> None:
    prices = {("a", "big"): 10.0, ("b", "cheap"): 0.5}
    ordered = plan(alias("cost"), RequestFeatures(), price_of=lambda p, m: prices.get((p, m)))
    assert providers(ordered) == ["b", "a", "c"]


def test_weighted_policy_is_proportional() -> None:
    rng = random.Random(7)
    firsts = Counter(
        plan(alias("weighted"), RequestFeatures(), rng=rng)[0].provider for _ in range(5000)
    )
    # weights 1:3:1 -> b first about 60% of the time
    assert 0.55 < firsts["b"] / 5000 < 0.65
    assert 0.15 < firsts["a"] / 5000 < 0.25
