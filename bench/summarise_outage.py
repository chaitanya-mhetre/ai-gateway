"""Print a Markdown table from every bench/results/outage-*.json.

The timeline columns are recomputed from each run's raw per-request samples with the current
`bench.outage.timeline`, so older result files stay comparable if that function changes.

    uv run python -m bench.summarise_outage
"""

from __future__ import annotations

import json
from pathlib import Path

from bench.outage import Sample, timeline

RESULTS = Path(__file__).resolve().parent / "results"


def fmt(v: object, unit: str = "") -> str:
    return "n/a" if v is None else f"{v}{unit}"


def main() -> None:
    print(
        "| Run | Scenario | Breaker | Outage error rate | Failed / tried primary | "
        "First fallback | Circuit open | Outage p50 / p95 / max | Primary back |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for path in sorted(RESULTS.glob("outage-*.json")):
        run = json.loads(path.read_text())
        p = run["params"]
        samples = [Sample(**s) for s in run["samples"]]
        outage_at = p["before"]
        tl = timeline(samples, outage_at, outage_at + p["outage"])
        o = run["phases"]["outage"]
        print(
            f"| {run['label'] or path.stem} | {run['scenario']} | {run['breaker']} "
            f"| {o['error_rate']:.1%} ({o['requests']} req) "
            f"| {tl['failed_requests_during_outage']} / {tl['requests_that_tried_primary_during_outage']} "
            f"| {fmt(tl['first_fallback_success_after_s'], ' s')} "
            f"| {fmt(tl['circuit_open_after_s'], ' s')} "
            f"| {fmt(o['p50_ms'])} / {fmt(o['p95_ms'])} / {fmt(o['max_ms'])} ms "
            f"| {fmt(tl['primary_serving_again_after_restore_s'], ' s')} |"
        )


if __name__ == "__main__":
    main()
