from __future__ import annotations

import random

import pytest

from ai_gateway.routing.backoff import backoff_delay
from ai_gateway.routing.health import LatencyTracker


def test_no_jitter_follows_schedule_and_repeats_last() -> None:
    sched = [100, 400]
    assert [backoff_delay(i, sched, jitter=False) for i in range(4)] == [0.1, 0.4, 0.4, 0.4]


def test_full_jitter_stays_within_bounds() -> None:
    rng = random.Random(42)
    delays = [backoff_delay(1, [100, 800], rng=rng) for _ in range(500)]
    assert all(0 <= d <= 0.8 for d in delays)
    assert max(delays) - min(delays) > 0.5  # actually spread out


def test_retry_after_is_a_floor() -> None:
    assert backoff_delay(0, [100], jitter=False, retry_after=2.0) == 2.0
    assert backoff_delay(0, [5000], jitter=False, retry_after=2.0) == 5.0


def test_ewma() -> None:
    t = LatencyTracker(alpha=0.5)
    t.observe("a", 1.0)
    t.observe("a", 3.0)
    assert t.get("a") == pytest.approx(2.0)
    assert t.get("missing") is None
    with pytest.raises(ValueError):
        LatencyTracker(alpha=0)
