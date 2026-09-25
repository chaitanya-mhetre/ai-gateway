"""Router: decides WHICH targets may serve a request and in WHAT ORDER.

It is a pure function of (alias config, request features, observed health, prices). No I/O, which
makes every policy trivially unit-testable. Actually *attempting* the targets is the executor's job.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from ai_gateway.config import AliasConfig, PolicyName, TargetConfig
from ai_gateway.routing.health import LatencyTracker

PriceLookup = Callable[[str, str], float | None]  # (provider, model) -> blended $ per 1M tokens


@dataclass(frozen=True)
class RequestFeatures:
    needs_tools: bool = False
    stream: bool = False
    estimated_prompt_tokens: int = 0


def eligible(alias: AliasConfig, features: RequestFeatures) -> list[TargetConfig]:
    """Drop targets that *cannot* serve this request (missing capability, context too small)."""
    req = alias.requirements
    out: list[TargetConfig] = []
    for t in alias.targets:
        if (features.needs_tools or req.supports_tools) and not t.supports_tools:
            continue
        if features.stream and not t.supports_stream:
            continue
        if req.min_context is not None and t.max_context < req.min_context:
            continue
        if features.estimated_prompt_tokens > t.max_context:
            continue
        out.append(t)
    return out


def _weighted_order(targets: list[TargetConfig], rng: random.Random) -> list[TargetConfig]:
    """Weighted random order without replacement (the first pick is proportional to weight)."""
    pool = list(targets)
    ordered: list[TargetConfig] = []
    while pool:
        total = sum(max(t.weight, 0) for t in pool)
        if total <= 0:
            ordered.extend(pool)
            break
        r = rng.uniform(0, total)
        acc = 0.0
        for i, t in enumerate(pool):
            acc += max(t.weight, 0)
            if r <= acc:
                ordered.append(pool.pop(i))
                break
    return ordered


def plan(
    alias: AliasConfig,
    features: RequestFeatures,
    *,
    latency: LatencyTracker | None = None,
    price_of: PriceLookup | None = None,
    policy_override: PolicyName | None = None,
    rng: random.Random | None = None,
) -> list[TargetConfig]:
    """Return eligible targets ordered by the alias policy (or a per-request override)."""
    targets = eligible(alias, features)
    policy = policy_override or alias.policy
    rng = rng or random.Random()
    by_priority = sorted(targets, key=lambda t: t.priority)

    if policy == "priority":
        return by_priority
    if policy == "weighted":
        return _weighted_order(targets, rng)
    if policy == "cost":
        # Unknown price sorts last: we don't pretend an unpriced model is free.
        def cost_key(t: TargetConfig) -> tuple[int, float, int]:
            price = price_of(t.provider, t.model) if price_of else None
            return (1, 0.0, t.priority) if price is None else (0, price, t.priority)

        return sorted(targets, key=cost_key)
    if policy == "latency":
        # Targets with no observations sort first, so they get sampled once and earn a real EWMA.
        def latency_key(t: TargetConfig) -> tuple[float, int]:
            observed = latency.get(t.key) if latency else None
            return (-1.0 if observed is None else observed, t.priority)

        return sorted(targets, key=latency_key)
    raise ValueError(f"unknown policy {policy}")  # pragma: no cover - guarded by config Literal
