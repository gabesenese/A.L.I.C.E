from __future__ import annotations

import os

_ENABLE_OTEL = str(os.getenv("ALICE_ENABLE_OTEL", "false")).lower() in {
    "1",
    "true",
    "yes",
    "on",
}

try:
    if not _ENABLE_OTEL:
        raise RuntimeError("OTEL disabled")
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource(attributes={SERVICE_NAME: "alice"})

    trace_provider = TracerProvider(resource=resource)
    trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(trace_provider)

    metric_reader = PeriodicExportingMetricReader(OTLPMetricExporter())
    metrics_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(metrics_provider)

    tracer = trace.get_tracer("alice")
    meter = metrics.get_meter("alice")

    turn_latency = meter.create_histogram(
        "alice.turn.latency_ms",
        description="End-to-end turn latency",
        unit="ms",
    )
    turn_counter = meter.create_counter(
        "alice.turns.total",
        description="Total turns processed",
    )
except Exception:  # pragma: no cover - fallback when telemetry deps are optional

    class _NoopSpan:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class _NoopTracer:
        def start_as_current_span(self, name: str):
            _ = name
            return _NoopSpan()

    class _NoopMetric:
        def add(self, value: int):
            _ = value

        def record(self, value: float):
            _ = value

    tracer = _NoopTracer()
    turn_latency = _NoopMetric()
    turn_counter = _NoopMetric()
