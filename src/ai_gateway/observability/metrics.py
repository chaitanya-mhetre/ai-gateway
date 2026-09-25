"""Prometheus metrics.

Metric types used (see docs/LEARNING_GUIDE.md):
- Counter   : only goes up (requests, tokens, cost). Query rates with `rate()`.
- Histogram : buckets of observations (latency). Query percentiles with `histogram_quantile()`.
- Gauge     : goes up and down (circuit state).

Label cardinality is deliberately bounded: providers/models/aliases come from config, and the
per-key metric uses the key *prefix* (never the secret). No user text ever becomes a label.

A dedicated CollectorRegistry per app instance keeps tests isolated (no global state).
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from ai_gateway.routing.executor import ExecutionListener

LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64)
OVERHEAD_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25)


class Metrics(ExecutionListener):
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        r = self.registry = registry or CollectorRegistry()
        self.requests = Counter(
            "gateway_requests",
            "Requests served",
            ["alias", "provider", "model", "status"],
            registry=r,
        )
        self.duration = Histogram(
            "gateway_request_duration_seconds",
            "End-to-end latency",
            ["alias", "provider"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.ttft = Histogram(
            "gateway_ttft_seconds",
            "Time to first streamed token",
            ["provider"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.tokens = Counter(
            "gateway_tokens", "Tokens processed", ["provider", "model", "type"], registry=r
        )
        self.cost = Counter(
            "gateway_cost_usd",
            "Estimated cost in USD",
            ["project", "provider", "model"],
            registry=r,
        )
        self.fallbacks = Counter(
            "gateway_fallbacks",
            "Fallbacks between targets",
            ["alias", "from_provider", "to_provider", "reason"],
            registry=r,
        )
        self.retries = Counter(
            "gateway_retries", "Retries on the same target", ["provider", "reason"], registry=r
        )
        self.attempts = Counter(
            "gateway_provider_attempts", "Provider attempts", ["provider", "outcome"], registry=r
        )
        self.circuit = Gauge(
            "gateway_circuit_state", "0=closed 1=half_open 2=open", ["provider"], registry=r
        )
        self.cache = Counter(
            "gateway_cache_requests", "Cache lookups", ["type", "result"], registry=r
        )
        self.rate_limited = Counter(
            "gateway_rate_limited",
            "Requests rejected by limits",
            ["key_prefix", "limit"],
            registry=r,
        )
        self.stream_interrupted = Counter(
            "gateway_stream_interrupted",
            "Streams broken after first token",
            ["provider"],
            registry=r,
        )
        self.overhead = Histogram(
            "gateway_overhead_seconds",
            "Time spent in the gateway itself (total - provider time)",
            buckets=OVERHEAD_BUCKETS,
            registry=r,
        )

    # ExecutionListener hooks ---------------------------------------------------------------------
    def on_attempt(
        self, provider: str, model: str, ok: bool, reason: str | None, seconds: float
    ) -> None:
        self.attempts.labels(provider, "ok" if ok else (reason or "error")).inc()

    def on_retry(self, provider: str, reason: str) -> None:
        self.retries.labels(provider, reason).inc()

    def on_fallback(self, alias: str, from_provider: str, to_provider: str, reason: str) -> None:
        self.fallbacks.labels(alias, from_provider, to_provider, reason).inc()

    def on_ttft(self, provider: str, seconds: float) -> None:
        self.ttft.labels(provider).observe(seconds)

    def on_stream_interrupted(self, provider: str) -> None:
        self.stream_interrupted.labels(provider).inc()
