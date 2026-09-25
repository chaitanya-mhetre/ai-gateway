"""Per-target latency tracking with an exponentially weighted moving average (EWMA).

EWMA gives recent observations more weight (`alpha`) than old ones, with O(1) memory, which is
what a latency-aware router needs: "how fast is this target *lately*?"
"""

from __future__ import annotations


class LatencyTracker:
    def __init__(self, alpha: float = 0.2) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._ewma: dict[str, float] = {}

    def observe(self, key: str, seconds: float) -> None:
        prev = self._ewma.get(key)
        self._ewma[key] = (
            seconds if prev is None else self.alpha * seconds + (1 - self.alpha) * prev
        )

    def get(self, key: str) -> float | None:
        return self._ewma.get(key)

    def snapshot(self) -> dict[str, float]:
        return dict(self._ewma)
