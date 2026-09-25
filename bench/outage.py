"""Provider-outage benchmark: what clients see when the primary provider fails mid-load.

Setup (all local processes): two mock upstreams (`primary`, `secondary`, fixed latency) behind one
gateway alias `chat-default` with priority fallback primary → secondary. A closed-loop load
generator keeps `--concurrency` requests in flight for the whole run:

    healthy (--before s)  →  primary fails (--outage s)  →  primary back (--after s)

Scenarios (`--scenario`):
- `kill`: SIGKILL the primary process (connection refused), restart it for the recovery phase.
- `hang`: primary accepts connections but never answers (a stuck provider). Costs a timeout per try.
- `5xx`:  primary answers 503 immediately.

`--breaker off` sets `min_requests` so high that the circuit never opens, to compare with it on.

Each request records its start time, latency, status and the `X-Gateway-Provider` / `-Attempts`
headers. Results go to bench/results/outage-*.json and are summarised in docs/benchmarks.md.

    uv run python -m bench.outage --scenario kill --breaker on
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml

from bench.overhead import cpu_model, pct, wait_healthy

ROOT = Path(__file__).resolve().parents[1]
PRIMARY, SECONDARY, GATEWAY = 59111, 59112, 59113
BODY = {"model": "chat-default", "messages": [{"role": "user", "content": "ping"}]}
UV = [sys.executable, "-m", "uvicorn", "--log-level", "warning", "--no-access-log"]


@dataclass
class Sample:
    t: float  # request start, seconds since the load started
    latency_ms: float
    status: int
    provider: str
    attempts: int


def gateway_config(breaker_on: bool, provider_timeout_ms: int, deadline_ms: int) -> dict[str, Any]:
    upstream = {"type": "openai", "timeout_ms": provider_timeout_ms}
    return {
        "providers": {
            "primary": {**upstream, "base_url": f"http://127.0.0.1:{PRIMARY}/v1"},
            "secondary": {**upstream, "base_url": f"http://127.0.0.1:{SECONDARY}/v1"},
        },
        "aliases": {
            "chat-default": {
                "targets": [
                    {"provider": "primary", "model": "mock-model", "priority": 1},
                    {"provider": "secondary", "model": "mock-model", "priority": 2},
                ],
                # One try per target, so every extra attempt in the results is a fallback.
                "retry": {"max_attempts_per_target": 1},
                "deadline_ms": deadline_ms,
            }
        },
        "breaker": {
            "window_s": 10,
            "min_requests": 5 if breaker_on else 10**9,
            "failure_ratio": 0.5,
            "cooldown_s": 5,
        },
        "cache": {"exact_enabled": False},
    }


def start_upstream(port: int, latency_ms: int) -> subprocess.Popen[bytes]:
    env = {**os.environ, "MOCK_LATENCY_MS": str(latency_ms)}
    return subprocess.Popen(
        [*UV, "bench.mock_upstream:app", "--port", str(port)], cwd=ROOT, env=env
    )


async def set_fault(port: int, mode: str) -> None:
    async with httpx.AsyncClient() as c:
        (await c.post(f"http://127.0.0.1:{port}/_fault", json={"mode": mode})).raise_for_status()


async def load(
    url: str, concurrency: int, duration_s: float, t0: float, timeout_s: float
) -> list[Sample]:
    samples: list[Sample] = []
    stop_at = t0 + duration_s
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=timeout_s) as client:

        async def worker() -> None:
            while (start := time.perf_counter()) < stop_at:
                try:
                    r = await client.post(url, json=BODY)
                    status = r.status_code
                    provider = r.headers.get("x-gateway-provider", "")
                    attempts = int(r.headers.get("x-gateway-attempts", "0") or 0)
                except httpx.HTTPError:
                    status, provider, attempts = 0, "", 0
                samples.append(
                    Sample(
                        t=round(start - t0, 3),
                        latency_ms=round((time.perf_counter() - start) * 1000, 2),
                        status=status,
                        provider=provider,
                        attempts=attempts,
                    )
                )

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    return samples


def summarise(samples: list[Sample]) -> dict[str, Any]:
    if not samples:
        return {"requests": 0}
    ok = [s for s in samples if s.status == 200]
    lat = [s.latency_ms for s in ok]
    return {
        "requests": len(samples),
        "error_rate": round(1 - len(ok) / len(samples), 4),
        "served_by_secondary": round(
            sum(s.provider == "secondary" for s in ok) / max(len(ok), 1), 4
        ),
        "paid_a_failed_primary_try": round(sum(s.attempts >= 2 for s in ok) / max(len(ok), 1), 4),
        "p50_ms": round(pct(lat, 50), 1) if lat else None,
        "p95_ms": round(pct(lat, 95), 1) if lat else None,
        "p99_ms": round(pct(lat, 99), 1) if lat else None,
        "mean_ms": round(statistics.fmean(lat), 1) if lat else None,
        "max_ms": round(max(lat), 1) if lat else None,
    }


def timeline(samples: list[Sample], outage_at: float, restore_at: float) -> dict[str, Any]:
    """The moments that matter, derived from per-request data (times are request start times)."""
    during = [s for s in samples if outage_at <= s.t < restore_at]
    after = [s for s in samples if s.t >= restore_at]
    first_fallback = next(
        (s for s in during if s.status == 200 and s.provider == "secondary"), None
    )
    # Served by the secondary in ONE attempt = the primary was skipped, i.e. its circuit was open.
    first_skip = next(
        (s for s in during if s.status == 200 and s.provider == "secondary" and s.attempts == 1),
        None,
    )
    first_primary_back = next((s for s in after if s.provider == "primary"), None)

    def since(s: Sample | None, origin: float) -> float | None:
        return round(s.t - origin, 3) if s else None

    return {
        "first_fallback_success_after_s": since(first_fallback, outage_at),
        "circuit_open_after_s": since(first_skip, outage_at),
        "failed_requests_during_outage": sum(s.status != 200 for s in during),
        # Requests that tried the dead primary before falling back (breaker closed or half-open probe).
        "requests_that_tried_primary_during_outage": sum(
            s.attempts >= 2 or s.status != 200 for s in during
        ),
        "primary_serving_again_after_restore_s": since(first_primary_back, restore_at),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["kill", "hang", "5xx"], default="kill")
    ap.add_argument("--breaker", choices=["on", "off"], default="on")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--before", type=float, default=15)
    ap.add_argument("--outage", type=float, default=30)
    ap.add_argument("--after", type=float, default=15)
    ap.add_argument("--latency-ms", type=int, default=200)
    ap.add_argument("--provider-timeout-ms", type=int, default=1000)
    ap.add_argument("--deadline-ms", type=int, default=5000)
    ap.add_argument("--label", default="", help="free text stored with the result")
    args = ap.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="gw-outage-"))
    cfg_path = workdir / "gateway.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            gateway_config(args.breaker == "on", args.provider_timeout_ms, args.deadline_ms)
        )
    )
    gw_env = {
        **os.environ,
        "GATEWAY_CONFIG_PATH": str(cfg_path),
        "GATEWAY_DATABASE_URL": f"sqlite+aiosqlite:///{workdir / 'gateway.db'}",
        "GATEWAY_AUTH_ENABLED": "false",
    }
    gw_env.pop("GATEWAY_REDIS_URL", None)

    primary = start_upstream(PRIMARY, args.latency_ms)
    secondary = start_upstream(SECONDARY, args.latency_ms)
    gateway = subprocess.Popen(
        [*UV, "ai_gateway.app:create_app", "--factory", "--port", str(GATEWAY)],
        cwd=ROOT,
        env=gw_env,
    )
    procs = [primary, secondary, gateway]
    try:
        for port in (PRIMARY, SECONDARY, GATEWAY):
            await wait_healthy(f"http://127.0.0.1:{port}/health")

        outage_at, restore_at = args.before, args.before + args.outage
        total = restore_at + args.after
        t0 = time.perf_counter()

        async def inject() -> None:
            nonlocal primary
            await asyncio.sleep(outage_at - (time.perf_counter() - t0))
            if args.scenario == "kill":
                primary.send_signal(signal.SIGKILL)
                primary.wait()
            else:
                await set_fault(PRIMARY, args.scenario)
            await asyncio.sleep(restore_at - (time.perf_counter() - t0))
            if args.scenario == "kill":
                primary = start_upstream(PRIMARY, args.latency_ms)
                procs.append(primary)
            else:
                await set_fault(PRIMARY, "ok")

        client_timeout = args.deadline_ms / 1000 + 5
        samples, _ = await asyncio.gather(
            load(
                f"http://127.0.0.1:{GATEWAY}/v1/chat/completions",
                args.concurrency,
                total,
                t0,
                client_timeout,
            ),
            inject(),
        )
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()

    samples.sort(key=lambda s: s.t)
    phases = {
        "before": summarise([s for s in samples if s.t < outage_at]),
        "outage": summarise([s for s in samples if outage_at <= s.t < restore_at]),
        "after": summarise([s for s in samples if s.t >= restore_at]),
    }
    result = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": args.label,
        "scenario": args.scenario,
        "breaker": args.breaker,
        "params": {
            k: v for k, v in vars(args).items() if k not in {"scenario", "breaker", "label"}
        },
        "environment": {
            "cpu": cpu_model(),
            "cores": os.cpu_count(),
            "os": platform.platform(),
            "python": platform.python_version(),
        },
        "phases": phases,
        "timeline": timeline(samples, outage_at, restore_at),
        "status_counts": {
            str(code): sum(s.status == code for s in samples)
            for code in sorted({s.status for s in samples})
        },
        "samples": [asdict(s) for s in samples],
    }
    stamp = int(time.time())
    out = ROOT / "bench" / "results" / f"outage-{args.scenario}-breaker_{args.breaker}-{stamp}.json"
    out.write_text(json.dumps(result, indent=1))
    printable = {k: v for k, v in result.items() if k != "samples"}
    print(json.dumps(printable, indent=2))
    print(f"\nsaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
