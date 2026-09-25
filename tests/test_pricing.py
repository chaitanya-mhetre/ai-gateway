from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

from ai_gateway.metering.pricing import PriceEntry, PriceTable
from ai_gateway.models import Usage


def entry(**kw: object) -> PriceEntry:
    base: dict[str, object] = {
        "provider": "p",
        "model": "m",
        "effective_from": date(2026, 1, 1),
        "input_per_mtok": Decimal("2"),
        "output_per_mtok": Decimal("8"),
        "source": "FAKE",
    }
    base.update(kw)
    return PriceEntry.model_validate(base)


def test_cost_formula_with_cached_input() -> None:
    table = PriceTable([entry(cached_input_per_mtok=Decimal("0.5"))])
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=500_000, cached_tokens=400_000)
    # 600k uncached * $2 + 400k cached * $0.5 + 500k output * $8 = 1.2 + 0.2 + 4.0
    assert table.cost("p", "m", usage, on=date(2026, 6, 1)) == Decimal("5.40000000")


def test_cached_tokens_fall_back_to_input_price() -> None:
    table = PriceTable([entry()])
    assert table.cost("p", "m", Usage(prompt_tokens=1_000_000, cached_tokens=1_000_000)) == Decimal(
        "2.00000000"
    )


def test_versioned_prices_use_the_entry_in_force() -> None:
    table = PriceTable(
        [entry(), entry(effective_from=date(2026, 7, 1), input_per_mtok=Decimal("1"))]
    )
    u = Usage(prompt_tokens=1_000_000)
    assert table.cost("p", "m", u, on=date(2026, 6, 30)) == Decimal("2.00000000")
    assert table.cost("p", "m", u, on=date(2026, 7, 1)) == Decimal("1.00000000")


def test_exact_match_beats_wildcard() -> None:
    table = PriceTable([entry(model="m*", input_per_mtok=Decimal("9")), entry(model="m-large")])
    assert table.lookup("p", "m-large") is not None
    assert table.lookup("p", "m-large").input_per_mtok == Decimal("2")  # type: ignore[union-attr]
    assert table.lookup("p", "m-small").input_per_mtok == Decimal("9")  # type: ignore[union-attr]


def test_unknown_model_cost_is_none_not_zero() -> None:
    assert PriceTable([entry()]).cost("p", "other", Usage(prompt_tokens=10)) is None
    assert PriceTable([entry()]).blended_price("q", "m") is None


def test_blended_price_assumes_3_to_1_mix() -> None:
    assert PriceTable([entry()]).blended_price("p", "m") == (2 * 3 + 8) / 4


def test_load_repo_price_file() -> None:
    table = PriceTable.load(Path(__file__).parents[1] / "config" / "prices.yaml")
    assert table.lookup("mock-primary", "mock-large") is not None
    assert all(e.source for e in table.entries)
