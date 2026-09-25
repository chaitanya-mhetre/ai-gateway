"""Fault-injection tests: the mock provider is scripted to fail in specific ways."""

from __future__ import annotations

import pytest

from ai_gateway import errors as err
from ai_gateway.config import AliasConfig, BreakerConfig
from ai_gateway.errors import AllTargetsFailedError, ProviderError
from ai_gateway.models import ChatRequest, Message, StreamChunk
from ai_gateway.providers.base import Provider
from ai_gateway.providers.mock import MockProvider, bad_request, http_5xx, http_429, timeout
from ai_gateway.routing.breaker import InMemoryBreaker
from ai_gateway.routing.executor import ExecutionListener, Executor
from ai_gateway.routing.health import LatencyTracker


class Recorder(ExecutionListener):
    def __init__(self) -> None:
        self.retries: list[tuple[str, str]] = []
        self.fallbacks: list[tuple[str, str, str]] = []
        self.interrupted: list[str] = []

    def on_retry(self, provider: str, reason: str) -> None:
        self.retries.append((provider, reason))

    def on_fallback(self, alias: str, from_provider: str, to_provider: str, reason: str) -> None:
        self.fallbacks.append((from_provider, to_provider, reason))

    def on_stream_interrupted(self, provider: str) -> None:
        self.interrupted.append(provider)


class Sleeps:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def alias_cfg(attempts: int = 2, **kw: object) -> AliasConfig:
    return AliasConfig.model_validate(
        {
            "targets": [
                {"provider": "primary", "model": "p"},
                {"provider": "secondary", "model": "s", "priority": 2},
            ],
            "retry": {
                "max_attempts_per_target": attempts,
                "backoff_ms": [100, 400],
                "jitter": False,
            },
            **kw,
        }
    )


REQ = ChatRequest(model="chat", messages=[Message(role="user", content="hi")])


def make(
    primary: MockProvider,
    secondary: MockProvider | None = None,
    breaker: InMemoryBreaker | None = None,
) -> tuple[Executor, Recorder, Sleeps, dict[str, Provider]]:
    providers: dict[str, Provider] = {
        "primary": primary,
        "secondary": secondary or MockProvider("secondary"),
    }
    rec, sleeps = Recorder(), Sleeps()
    ex = Executor(
        providers,
        breaker or InMemoryBreaker(BreakerConfig()),
        LatencyTracker(),
        listener=rec,
        sleep=sleeps,
    )
    return ex, rec, sleeps, providers


async def test_retry_on_same_target_then_success() -> None:
    ex, rec, sleeps, _ = make(MockProvider("primary", script=[http_5xx()]))
    a = alias_cfg()
    resp, info = await ex.chat("chat", a, a.targets, REQ, 5)
    assert resp.provider == "primary"
    assert info.attempts == 2 and not info.fallback_used
    assert rec.retries == [("primary", err.HTTP_5XX)]
    assert sleeps.delays == [0.1]


async def test_fallback_after_retries_exhausted() -> None:
    primary = MockProvider("primary", script=[timeout(), timeout()])
    ex, rec, _, _ = make(primary)
    a = alias_cfg()
    resp, info = await ex.chat("chat", a, a.targets, REQ, 5)
    assert resp.provider == "secondary"
    assert info.fallback_used and info.attempts == 3
    assert rec.fallbacks == [("primary", "secondary", err.TIMEOUT)]
    assert [e.reason for e in info.errors] == [err.TIMEOUT, err.TIMEOUT]


async def test_invalid_request_is_not_retried_or_failed_over() -> None:
    primary = MockProvider("primary", script=[bad_request()])
    secondary = MockProvider("secondary")
    ex, _, _, _ = make(primary, secondary)
    a = alias_cfg()
    with pytest.raises(AllTargetsFailedError) as exc:
        await ex.chat("chat", a, a.targets, REQ, 5)
    assert primary.calls == 1 and secondary.calls == 0
    assert exc.value.status_code == 400


async def test_retry_after_floor_is_honoured() -> None:
    ex, _, sleeps, _ = make(MockProvider("primary", script=[http_429(retry_after=1.5)]))
    a = alias_cfg()
    await ex.chat("chat", a, a.targets, REQ, 5)
    assert sleeps.delays == [1.5]


async def test_retry_after_longer_than_deadline_falls_back_immediately() -> None:
    ex, _, sleeps, _ = make(MockProvider("primary", script=[http_429(retry_after=60)]))
    a = alias_cfg()
    resp, _ = await ex.chat("chat", a, a.targets, REQ, 5)
    assert resp.provider == "secondary"
    assert sleeps.delays == []  # never waited 60 s


async def test_open_circuit_skips_provider_without_calling_it() -> None:
    breaker = InMemoryBreaker(BreakerConfig(min_requests=1, failure_ratio=0.1, cooldown_s=60))
    await breaker.record_failure("primary")
    primary = MockProvider("primary")
    ex, rec, _, _ = make(primary, breaker=breaker)
    a = alias_cfg()
    resp, info = await ex.chat("chat", a, a.targets, REQ, 5)
    assert primary.calls == 0 and resp.provider == "secondary"
    assert info.errors[0].reason == err.CIRCUIT_OPEN
    assert rec.fallbacks == [("primary", "secondary", err.CIRCUIT_OPEN)]


async def test_repeated_failures_open_the_circuit() -> None:
    breaker = InMemoryBreaker(BreakerConfig(min_requests=3, failure_ratio=0.5, cooldown_s=60))
    primary = MockProvider("primary", script=[http_5xx()] * 3)
    ex, _, _, _ = make(primary, breaker=breaker)
    a = alias_cfg(attempts=1)
    for _ in range(3):
        await ex.chat("chat", a, a.targets, REQ, 5)
    assert not await breaker.allow("primary")
    await ex.chat("chat", a, a.targets, REQ, 5)
    assert primary.calls == 3  # the 4th request never touched the failing provider


async def test_overall_deadline_bounds_total_time() -> None:
    slow = MockProvider("primary", latency=0.2)
    slow2 = MockProvider("secondary", latency=0.2)
    ex, _, _, _ = make(slow, slow2)
    a = alias_cfg(attempts=1)
    with pytest.raises(AllTargetsFailedError) as exc:
        await ex.chat("chat", a, a.targets, REQ, 0.05)
    assert all(e.reason == err.TIMEOUT for e in exc.value.errors)


async def test_fallback_disabled_for_reason_stops() -> None:
    primary = MockProvider("primary", script=[http_5xx()])
    ex, _, _, _ = make(primary)
    a = alias_cfg(attempts=1, fallback_on=[err.TIMEOUT])
    with pytest.raises(AllTargetsFailedError):
        await ex.chat("chat", a, a.targets, REQ, 5)


# --- streaming ------------------------------------------------------------------------------


async def collect(it: object) -> list[StreamChunk]:
    return [c async for c in it]  # type: ignore[attr-defined]


async def test_stream_falls_back_before_first_chunk() -> None:
    primary = MockProvider("primary", script=[http_5xx()])
    secondary = MockProvider("secondary", reply="from secondary")
    ex, rec, _, _ = make(primary, secondary)
    a = alias_cfg(attempts=1)
    it, info = await ex.stream("chat", a, a.targets, REQ.model_copy(update={"stream": True}), 5)
    text = "".join(c.content or "" for c in await collect(it))
    assert text == "from secondary"
    assert info.provider == "secondary" and info.ttft_s is not None
    assert rec.fallbacks[0][2] == err.HTTP_5XX


async def test_stream_first_token_timeout_triggers_fallback() -> None:
    primary = MockProvider("primary", latency=0.3)
    ex, _, _, _ = make(primary)
    a = alias_cfg(attempts=1, first_token_timeout_ms=50)
    it, info = await ex.stream("chat", a, a.targets, REQ.model_copy(update={"stream": True}), 5)
    await collect(it)
    assert info.provider == "secondary"
    assert info.errors[0].reason == err.TIMEOUT


async def test_mid_stream_failure_does_not_fall_back() -> None:
    primary = MockProvider("primary", reply="abcdefghijklmnop", chunk_size=4, stream_fail_after=2)
    secondary = MockProvider("secondary")
    ex, rec, _, _ = make(primary, secondary)
    a = alias_cfg(attempts=1)
    it, _ = await ex.stream("chat", a, a.targets, REQ.model_copy(update={"stream": True}), 5)
    received: list[str] = []
    with pytest.raises(ProviderError) as exc:
        async for c in it:
            received.append(c.content or "")
    assert "".join(received) == "abcdefgh"  # the client got 2 chunks, then an error, never a splice
    assert exc.value.reason == err.STREAM_INTERRUPTED
    assert secondary.calls == 0
    assert rec.interrupted == ["primary"]
