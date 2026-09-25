"""Versioned price table → estimated cost.

Prices are CONFIG, entered by hand from each provider's pricing page, with the source URL and the
date checked recorded next to every entry. The gateway only ever reports *estimated* cost. If a
model has no price entry, cost is reported as unknown (None). It is never silently treated as free.

Versioning: several entries may exist for one model with different `effective_from` dates. The
entry in force on the request date is used, so historical costs stay correct after price changes.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from ai_gateway.models import Usage

MTOK = Decimal(1_000_000)


class PriceEntry(BaseModel):
    provider: str
    model: str  # exact name, or a prefix ending in "*" (e.g. "mock-*")
    effective_from: date
    input_per_mtok: Decimal = Field(ge=0)
    output_per_mtok: Decimal = Field(ge=0)
    cached_input_per_mtok: Decimal | None = Field(default=None, ge=0)
    source: str  # URL of the pricing page, or "FAKE" for mock providers
    checked_on: date | None = None

    def matches(self, provider: str, model: str) -> bool:
        if provider != self.provider:
            return False
        if self.model.endswith("*"):
            return model.startswith(self.model[:-1])
        return model == self.model


class PriceTable:
    def __init__(self, entries: list[PriceEntry]) -> None:
        # Most specific first (exact beats wildcard), then newest effective date first.
        self.entries = sorted(
            entries, key=lambda e: (e.model.endswith("*"), -e.effective_from.toordinal())
        )

    @classmethod
    def load(cls, path: Path | None) -> PriceTable:
        if path is None or not path.exists():
            return cls([])
        raw = yaml.safe_load(path.read_text()) or {}
        return cls([PriceEntry.model_validate(e) for e in raw.get("prices", [])])

    def lookup(self, provider: str, model: str, on: date | None = None) -> PriceEntry | None:
        on = on or date.today()
        for e in self.entries:
            if e.matches(provider, model) and e.effective_from <= on:
                return e
        return None

    def cost(
        self, provider: str, model: str, usage: Usage, on: date | None = None
    ) -> Decimal | None:
        entry = self.lookup(provider, model, on)
        if entry is None:
            return None
        cached = min(usage.cached_tokens, usage.prompt_tokens)
        uncached = usage.prompt_tokens - cached
        cached_rate = (
            entry.cached_input_per_mtok
            if entry.cached_input_per_mtok is not None
            else entry.input_per_mtok
        )
        total = (
            Decimal(uncached) * entry.input_per_mtok
            + Decimal(cached) * cached_rate
            + Decimal(usage.completion_tokens) * entry.output_per_mtok
        ) / MTOK
        return total.quantize(Decimal("0.00000001"))

    def blended_price(self, provider: str, model: str) -> float | None:
        """A single number for the cost-aware router: $/1M tokens assuming a 3:1 input:output mix."""
        entry = self.lookup(provider, model)
        if entry is None:
            return None
        return float((entry.input_per_mtok * 3 + entry.output_per_mtok) / 4)
