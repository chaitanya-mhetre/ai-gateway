"""OpenTelemetry tracing.

Spans: `gateway.request` → `gateway.attempt` (one per provider attempt). Attributes are
provider/model/tokens/outcome. **Prompt and response content are never recorded.**

When tracing is disabled, the OTel API's default no-op tracer is used, which costs ~nothing.
"""

from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

tracer = trace.get_tracer("ai_gateway")


def setup_tracing(exporter: SpanExporter | None = None) -> TracerProvider:
    """Install a global TracerProvider. Uses OTLP/HTTP (`OTEL_EXPORTER_OTLP_ENDPOINT`) by default."""
    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": os.environ.get("OTEL_SERVICE_NAME", "ai-gateway")}
        )
    )
    if exporter is None:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return provider
