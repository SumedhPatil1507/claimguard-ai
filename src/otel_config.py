"""
OpenTelemetry configuration for ClaimGuard AI.

This module initializes OpenTelemetry tracing with OTLP exporter for
production-grade observability of the LangGraph agent and vector store operations.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: OpenTelemetry SDK
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False
    FastAPIInstrumentor = None  # type: ignore[assignment]


_tracer_provider: Optional[TracerProvider] = None


def initialize_telemetry(service_name: str = "claimguard-ai") -> Optional[TracerProvider]:
    """
    Initialize OpenTelemetry tracing with OTLP exporter.

    Reads configuration from environment variables:
    - OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint (default: http://localhost:4317)
    - OTEL_SERVICE_NAME: Service name for tracing (default: claimguard-ai)
    - OTEL_TRACES_SAMPLER_ARG: Sampling rate 0.0-1.0 (default: 1.0)

    Parameters
    ----------
    service_name : str
        Service name to use in trace metadata

    Returns
    -------
    TracerProvider if initialization succeeded, None otherwise
    """
    global _tracer_provider

    if not HAS_OPENTELEMETRY:
        logger.warning("OpenTelemetry packages not installed; tracing disabled")
        return None

    # Check if already initialized
    if _tracer_provider is not None:
        return _tracer_provider

    try:
        # Get configuration from environment
        otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        service_name_env = os.environ.get("OTEL_SERVICE_NAME", service_name)
        sampling_rate = float(os.environ.get("OTEL_TRACES_SAMPLER_ARG", "1.0"))

        # Create resource with service metadata
        resource = Resource.create({
            SERVICE_NAME: service_name_env,
            "service.version": "1.0.0",
            "deployment.environment": os.environ.get("ENVIRONMENT", "development"),
        })

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Configure OTLP exporter
        try:
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
            logger.info(f"OpenTelemetry initialized with OTLP endpoint: {otlp_endpoint}")
        except Exception as exc:
            logger.warning(f"Failed to initialize OTLP exporter: {exc}. Tracing will be disabled.")
            return None

        # Set as global tracer provider
        trace.set_tracer_provider(provider)
        _tracer_provider = provider

        logger.info(f"OpenTelemetry tracing enabled for service: {service_name_env} (sampling: {sampling_rate})")
        return provider

    except Exception as exc:
        logger.error(f"Failed to initialize OpenTelemetry: {exc}", exc_info=True)
        return None


def instrument_fastapi(app) -> None:
    """
    Instrument a FastAPI application with OpenTelemetry.

    Parameters
    ----------
    app : FastAPI
        The FastAPI application instance to instrument
    """
    if not HAS_OPENTELEMETRY or FastAPIInstrumentor is None:
        logger.warning("OpenTelemetry FastAPI instrumentation not available")
        return

    if _tracer_provider is None:
        logger.warning("OpenTelemetry not initialized; cannot instrument FastAPI")
        return

    try:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumented with OpenTelemetry")
    except Exception as exc:
        logger.warning(f"Failed to instrument FastAPI: {exc}")


def get_tracer(name: str):
    """
    Get a tracer instance for a specific module.

    Parameters
    ----------
    name : str
        Name of the module/component

    Returns
    -------
    Tracer instance or None if OpenTelemetry is not available
    """
    if not HAS_OPENTELEMETRY:
        return None

    try:
        return trace.get_tracer(name)
    except Exception:
        return None
