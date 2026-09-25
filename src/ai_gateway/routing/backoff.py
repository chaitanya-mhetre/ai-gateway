"""Retry delays: exponential backoff with *full jitter*.

Full jitter (delay = uniform(0, base)) spreads retries from many clients over time, so they don't
hit a recovering provider at the same instant (the "thundering herd" problem).
See: AWS Architecture Blog, "Exponential Backoff and Jitter".
"""

from __future__ import annotations

import random


def backoff_delay(
    attempt: int,
    schedule_ms: list[int],
    *,
    jitter: bool = True,
    retry_after: float | None = None,
    rng: random.Random | None = None,
) -> float:
    """Seconds to wait before retry number `attempt` (0-based).

    `schedule_ms` gives the base delay per attempt; attempts beyond its end reuse the last value.
    A provider's `Retry-After` is a floor: we never retry *sooner* than the provider asked.
    """
    base = schedule_ms[min(attempt, len(schedule_ms) - 1)] / 1000 if schedule_ms else 0.0
    delay = (rng or random).uniform(0, base) if jitter else base
    if retry_after is not None:
        delay = max(delay, retry_after)
    return delay
