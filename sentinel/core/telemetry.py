"""OpenTelemetry setup, trace providers, and span management for Sentinel."""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

from sentinel.core.config import Settings
from sentinel.core.logging import get_logger

logger = get_logger(__name__)

_tracer_provider: TracerProvider | None = None
_in_memory_exporter: InMemorySpanExporter | None = None


def _register_provider(provider: TracerProvider) -> None:
    """Register the provider globally with OpenTelemetry, allowing reset in test runs."""
    trace._TRACER_PROVIDER = provider


def setup_telemetry(settings: Settings) -> TracerProvider:
    """Initialize OpenTelemetry TracerProvider with OTLP exporter and resource metadata."""
    global _tracer_provider

    if not settings.otel_enabled:
        logger.info("OpenTelemetry tracing is disabled by configuration")
        provider = TracerProvider()
        _register_provider(provider)
        _tracer_provider = provider
        return provider

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.version": "0.1.0",
            "deployment.environment": settings.environment,
        }
    )

    provider = TracerProvider(resource=resource)

    if settings.otel_exporter_otlp_endpoint:
        logger.info(
            "Configuring OTLP trace exporter",
            extra={"endpoint": settings.otel_exporter_otlp_endpoint},
        )
        exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
        span_processor = BatchSpanProcessor(exporter)
        provider.add_span_processor(span_processor)

    _register_provider(provider)
    _tracer_provider = provider
    return provider


def setup_in_memory_telemetry() -> tuple[TracerProvider, InMemorySpanExporter]:
    """Configure an in-memory tracer provider for testing without network dependencies."""
    global _tracer_provider, _in_memory_exporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource.create(
            {"service.name": "sentinel-test", "deployment.environment": "test"}
        )
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    _register_provider(provider)

    _tracer_provider = provider
    _in_memory_exporter = exporter
    return provider, exporter


def get_tracer(name: str = "sentinel") -> Tracer:
    """Retrieve an OpenTelemetry tracer instance."""
    return trace.get_tracer(name)


def get_current_trace_id() -> str | None:
    """Return the current active trace ID as a 32-character hex string, or None."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.is_valid and ctx.trace_id != 0:
        return f"{ctx.trace_id:032x}"
    return None


def get_current_span_id() -> str | None:
    """Return the current active span ID as a 16-character hex string, or None."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.is_valid and ctx.span_id != 0:
        return f"{ctx.span_id:016x}"
    return None


def shutdown_telemetry() -> None:
    """Shut down active tracer provider and flush remaining spans."""
    global _tracer_provider, _in_memory_exporter
    if _tracer_provider is not None:
        try:
            _tracer_provider.shutdown()
        except Exception as exc:
            logger.warning("Error shutting down telemetry provider", extra={"error": str(exc)})
        finally:
            _tracer_provider = None
            _in_memory_exporter = None
            trace._TRACER_PROVIDER = None
