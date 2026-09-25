"""Measure the latency the gateway adds on top of a provider call.

Method: the same request mix goes (a) directly to a mock upstream with a fixed latency and
(b) through the gateway to that upstream. The overhead is (b) - (a) at matching percentiles.
Both servers run locally as separate uvicorn processes (1 worker each).

    uv run python bench/overhead.py --requests 2000 --concurrency 50 --mode memory
    uv run python bench/overhead.py --mode redis   # needs `make up` (Redis on :56382, auth on)

Results are appended to bench/results/ as JSON and summarised in docs/benchmarks.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM, GATEWAY = 59101, 59102
BODY = {"model": "chat-default", "messages": [{"role": "user", "content": "ping"}]}


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1))))
    return ordered[k]


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


async def wait_healthy(url: str, timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient() as c:
        while time.monotonic() < deadline:
            try:
                if (await c.get(url)).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(0.2)
    raise RuntimeError(f"{url} never became healthy")


async def run_load(
    url: str, headers: dict[str, str], n: int, concurrency: int, body: dict[str, Any]
) -> dict[str, Any]:
    latencies: list[float] = []
    errors = 0
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=30) as client:

        async def one() -> None:
            nonlocal errors
            async with sem:
                t0 = time.perf_counter()
                r = await client.post(url, json=body, headers=headers)
                dt = time.perf_counter() - t0
                if r.status_code == 200:
                    latencies.append(dt)
                else:
                    errors += 1

        # warm-up (connections, JIT-free but caches/pools)
        await asyncio.gather(*(one() for _ in range(min(100, n))))
        latencies.clear()
        errors = 0
        t0 = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(n)))
        wall = time.perf_counter() - t0
    return {
        "requests": n,
        "errors": errors,
        "wall_s": round(wall, 3),
        "rps": round(len(latencies) / wall, 1),
        "p50_ms": round(pct(latencies, 50) * 1000, 2),
        "p95_ms": round(pct(latencies, 95) * 1000, 2),
        "p99_ms": round(pct(latencies, 99) * 1000, 2),
        "mean_ms": round(statistics.fmean(latencies) * 1000, 2),
    }


def create_admin(gw_env: dict[str, str]) -> str:
    """Create an owner admin in the bench DB via the CLI (it also runs migrations); return its token."""
    out = subprocess.run(
        [sys.executable, "-m", "ai_gateway.cli", "create-admin", "--email", "bench@example.com"],
        cwd=ROOT,
        env=gw_env,
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip().splitlines()[-1]


async def create_key(base: str, admin_token: str) -> str:
    h = {"Authorization": f"Bearer {admin_token}"}
    async with httpx.AsyncClient(base_url=base) as c:
        t = (
            await c.post("/admin/v1/tenants", json={"name": f"bench-{time.time_ns()}"}, headers=h)
        ).json()
        p = (
            await c.post(
                "/admin/v1/projects", json={"tenant_id": t["id"], "name": "bench"}, headers=h
            )
        ).json()
        k = (
            await c.post(
                "/admin/v1/keys",
                json={
                    "project_id": p["id"],
                    "name": "bench",
                    "rpm_limit": 10_000_000,
                    "tpm_limit": 100_000_000,
                },
                headers=h,
            )
        ).json()
        return str(k["key"])


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--latency-ms", type=int, default=200)
    ap.add_argument("--mode", choices=["memory", "redis"], default="memory")
    args = ap.parse_args()

    env = {**os.environ, "MOCK_LATENCY_MS": str(args.latency_ms)}
    db_file = Path(tempfile.mkdtemp(prefix="gw-bench-")) / "gateway.db"
    gw_env = {
        **os.environ,
        "GATEWAY_CONFIG_PATH": str(ROOT / "bench" / "gateway.bench.yaml"),
        # A file (not :memory:) so the admin created below is visible to the gateway process.
        "GATEWAY_DATABASE_URL": f"sqlite+aiosqlite:///{db_file}",
        "GATEWAY_AUTH_ENABLED": "true" if args.mode == "redis" else "false",
        "GATEWAY_REDIS_URL": "redis://localhost:56382/1" if args.mode == "redis" else "",
    }
    gw_env.pop("GATEWAY_ADMIN_TOKEN", None)
    if args.mode == "memory":
        gw_env.pop("GATEWAY_REDIS_URL")
    admin_token = create_admin(gw_env)
    uv = [sys.executable, "-m", "uvicorn", "--log-level", "warning", "--no-access-log"]
    procs = [
        subprocess.Popen(
            [*uv, "bench.mock_upstream:app", "--port", str(UPSTREAM)], cwd=ROOT, env=env
        ),
        subprocess.Popen(
            [*uv, "ai_gateway.app:create_app", "--factory", "--port", str(GATEWAY)],
            cwd=ROOT,
            env=gw_env,
        ),
    ]
    try:
        await wait_healthy(f"http://127.0.0.1:{UPSTREAM}/health")
        await wait_healthy(f"http://127.0.0.1:{GATEWAY}/health")
        headers: dict[str, str] = {}
        if args.mode == "redis":
            headers["Authorization"] = (
                f"Bearer {await create_key(f'http://127.0.0.1:{GATEWAY}', admin_token)}"
            )
        direct = await run_load(
            f"http://127.0.0.1:{UPSTREAM}/v1/chat/completions",
            {},
            args.requests,
            args.concurrency,
            {**BODY, "model": "mock-model"},
        )
        via = await run_load(
            f"http://127.0.0.1:{GATEWAY}/v1/chat/completions",
            headers,
            args.requests,
            args.concurrency,
            BODY,
        )
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(timeout=10)

    result = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": args.mode,
        "concurrency": args.concurrency,
        "upstream_latency_ms": args.latency_ms,
        "environment": {
            "cpu": cpu_model(),
            "cores": os.cpu_count(),
            "os": platform.platform(),
            "python": platform.python_version(),
        },
        "direct": direct,
        "via_gateway": via,
        "overhead_ms": {
            k: round(via[k] - direct[k], 2) for k in ("p50_ms", "p95_ms", "p99_ms", "mean_ms")
        },
    }
    out = ROOT / "bench" / "results" / f"overhead-{args.mode}-{int(time.time())}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nsaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
