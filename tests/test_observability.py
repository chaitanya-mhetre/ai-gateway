from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from ai_gateway.app import create_app
from ai_gateway.config import Settings
from ai_gateway.metering.usage import InProcessSink
from ai_gateway.providers.base import Provider
from ai_gateway.providers.mock import MockProvider, http_5xx
from tests.conftest import app_client, make_config

ADMIN = {"Authorization": "Bearer admin-secret"}
CHAT = {"model": "chat-default", "messages": [{"role": "user", "content": "hello there"}]}

PRICES = """
prices:
  - {provider: primary,   model: "p-*", effective_from: 2026-01-01, input_per_mtok: 1000000, output_per_mtok: 1000000, source: FAKE}
  - {provider: secondary, model: "s-*", effective_from: 2026-01-01, input_per_mtok: 1000000, output_per_mtok: 1000000, source: FAKE}
"""


@pytest.fixture
async def gw(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    (tmp_path / "prices.yaml").write_text(PRICES)
    config = make_config()
    config.price_table_path = tmp_path / "prices.yaml"
    providers: dict[str, Provider] = {
        "primary": MockProvider("primary", script=[http_5xx()]),  # first call fails over
        "secondary": MockProvider("secondary"),
        "local": MockProvider("local"),
    }
    settings = Settings(
        auth_enabled=True,
        admin_token="admin-secret",
        key_pepper="pep",
        redis_url=None,
        database_url="sqlite+aiosqlite:///:memory:",
    )
    async with app_client(create_app(settings, config=config, providers=providers)) as c:
        yield c


async def key_for(c: httpx.AsyncClient) -> tuple[str, str]:
    t = (await c.post("/admin/v1/tenants", json={"name": "acme"}, headers=ADMIN)).json()
    p = (
        await c.post("/admin/v1/projects", json={"tenant_id": t["id"], "name": "x"}, headers=ADMIN)
    ).json()
    k = (
        await c.post("/admin/v1/keys", json={"project_id": p["id"], "name": "k"}, headers=ADMIN)
    ).json()
    return k["key"], p["id"]


def metric_value(text: str, name: str, **labels: str) -> float:
    for line in text.splitlines():
        if line.startswith(name) and all(f'{k}="{v}"' in line for k, v in labels.items()):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


async def test_metrics_after_fallback(gw: httpx.AsyncClient) -> None:
    key, _ = await key_for(gw)
    auth = {"Authorization": f"Bearer {key}"}
    assert (await gw.post("/v1/chat/completions", json=CHAT, headers=auth)).status_code == 200
    assert (await gw.post("/v1/chat/completions", json=CHAT, headers=auth)).status_code == 200
    text = (await gw.get("/metrics")).text
    assert (
        metric_value(
            text, "gateway_requests_total", alias="chat-default", provider="secondary", status="ok"
        )
        == 1
    )
    assert (
        metric_value(
            text, "gateway_requests_total", alias="chat-default", provider="primary", status="ok"
        )
        == 1
    )
    assert (
        metric_value(
            text,
            "gateway_fallbacks_total",
            from_provider="primary",
            to_provider="secondary",
            reason="http_5xx",
        )
        == 1
    )
    assert metric_value(text, "gateway_tokens_total", provider="primary", type="prompt") > 0
    assert metric_value(text, "gateway_cost_usd_total", provider="primary") > 0
    assert metric_value(text, "gateway_overhead_seconds_count") == 2
    assert metric_value(text, "gateway_circuit_state", provider="primary") == 0


async def test_usage_api_groups_and_costs(gw: httpx.AsyncClient) -> None:
    key, project_id = await key_for(gw)
    auth = {"Authorization": f"Bearer {key}"}
    for _ in range(3):
        await gw.post("/v1/chat/completions", json=CHAT, headers=auth)
    sink = gw._transport.app.state.sink  # type: ignore[attr-defined]
    assert isinstance(sink, InProcessSink)
    await sink.flush()
    by_model: list[dict[str, Any]] = (
        await gw.get(f"/admin/v1/usage?project_id={project_id}&group_by=model", headers=ADMIN)
    ).json()
    totals = {(r["provider"], r["model"]): r for r in by_model}
    assert totals[("secondary", "s-model")]["requests"] == 1
    assert totals[("primary", "p-model")]["requests"] == 2
    assert totals[("secondary", "s-model")]["fallbacks"] == 1
    assert float(totals[("primary", "p-model")]["est_cost_usd"]) > 0
    by_day = (
        await gw.get(f"/admin/v1/usage?project_id={project_id}&group_by=day", headers=ADMIN)
    ).json()
    assert by_day[0]["requests"] == 3 and "day" in by_day[0]
    prices = (await gw.get("/admin/v1/prices", headers=ADMIN)).json()
    assert prices[0]["source"] == "FAKE"


async def test_rate_limited_metric(gw: httpx.AsyncClient) -> None:
    t = (await gw.post("/admin/v1/tenants", json={"name": "rl"}, headers=ADMIN)).json()
    p = (
        await gw.post("/admin/v1/projects", json={"tenant_id": t["id"], "name": "x"}, headers=ADMIN)
    ).json()
    k = (
        await gw.post(
            "/admin/v1/keys",
            json={"project_id": p["id"], "name": "k", "rpm_limit": 1},
            headers=ADMIN,
        )
    ).json()
    auth = {"Authorization": f"Bearer {k['key']}"}
    await gw.post("/v1/chat/completions", json=CHAT, headers=auth)
    assert (await gw.post("/v1/chat/completions", json=CHAT, headers=auth)).status_code == 429
    text = (await gw.get("/metrics")).text
    assert metric_value(text, "gateway_rate_limited_total", key_prefix=k["prefix"]) == 1


async def test_trace_spans_per_attempt_without_prompt_content() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    providers: dict[str, Provider] = {
        "primary": MockProvider("primary", script=[http_5xx()]),
        "secondary": MockProvider("secondary"),
        "local": MockProvider("local"),
    }
    settings = Settings(
        auth_enabled=False, redis_url=None, database_url="sqlite+aiosqlite:///:memory:"
    )
    async with app_client(create_app(settings, config=make_config(), providers=providers)) as c:
        await c.post("/v1/chat/completions", json=CHAT)
    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    if "gateway.request" not in names:
        pytest.skip("a global TracerProvider was already installed by another test run")
    attempts = [s for s in spans if s.name == "gateway.attempt"]
    assert [a.attributes["gateway.provider"] for a in attempts if a.attributes] == [
        "primary",
        "secondary",
    ]
    for s in spans:
        assert "hello there" not in str(s.attributes)
