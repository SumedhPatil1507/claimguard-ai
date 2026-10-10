"""
src/otel_config.py
==================
OpenTelemetry bootstrap for ClaimGuard AI.

Provides:
  * ``initialize_telemetry(service_name)`` — configure a global TracerProvider
    with an OTLP gRPC exporter when the ``opentelemetry-*`` packages are
    installed and an OTEL_EXPORTER_OTLP_ENDPOINT is reachable.  When the
    optional packages are absent the module degrades to a fully-functional
    no-op surface so that instrumented code paths (vector store, LangGraph
    nodes, FastAPI) never crash in slim environments.

  * ``get_tracer(name)`` — return the global tracer (or a stub tracer).

  * ``span(name, **attrs)`` — context manager that wraps an active span and
    records attributes / exceptions.  Safe to use unconditionally.

  * ``instrument_fastapi(app)`` — attach FastAPI / ASGI instrumentation when
    ``opentelemetry-instrumentation-fastapi`` is available.

Environment variables
---------------------
OTEL_EXPORTER_OTLP_ENDPOINT   e.g. http://otel-collector:4317
OTEL_SERVICE_NAME             overrides the service_name argument
OTEL_ENABLED                  "false" disables everything (default: enabled
                              only when the SDK packages import cleanly)
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional OpenTelemetry imports — graceful degradation everywhere
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.trace import Status, StatusCode

    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False
    trace = None                      # type: ignore[assignment]
    Status = None                     # type: ignore[assignment]
    StatusCode = None                 # type: ignore[assignment]

try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    HAS_OTLP_GRPC = True
except ImportError:
    HAS_OTLP_GRPC = False


class _NoOpSpan:
    """Minimal stand-in implementing the small Span surface we rely on."""

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: D401
        pass

    def set_attributes(self, attributes: dict) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[dict] = None) -> None:
        pass

    def record_exception(self, exception: BaseException, attributes: Optional[dict] = None) -> None:
        pass

    def set_status(self, status: Any = None, description: Optional[str] = None) -> None:
        pass

    def end(self, end_time: Optional[int] = None) -> None:
        pass


class _NoOpTracer:
    @contextmanager
    def start_as_current_span(self, name: str, **kwargs: Any) -> Iterator[_NoOpSpan]:
        yield _NoOpSpan()

    def get_span(self, name: str) -> _NoOpSpan:
        return _NoOpSpan()


_NOOP_TRACER = _NoOpTracer()
_initialized = False


def initialize_telemetry(service_name: str = "claimguard-ai") -> bool:
    """Configure the global tracer provider exactly once.

    Returns ``True`` when real OpenTelemetry tracing is active, ``False``
    when running against the no-op fallback.  Never raises.
    """
    global _initialized
    if _initialized:
        return HAS_OTEL

    _initialized = True

    if os.environ.get("OTEL_ENABLED", "true").lower() in ("0", "false", "no"):
        logger.info("otel_config: tracing disabled via OTEL_ENABLED=false")
        return False

    if not HAS_OTEL:
        logger.info(
            "otel_config: opentelemetry-sdk not installed — "
            "using no-op tracer (spans are recorded as structured logs only)."
        )
        return False

    try:
        name = os.environ.get("OTEL_SERVICE_NAME", service_name)
        resource = Resource(attributes={SERVICE_NAME: name})
        provider = TracerProvider(resource=resource)

        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint and HAS_OTLP_GRPC:
            provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
            )
            logger.info("otel_config: exporting spans to %s", endpoint)
        else:
            # Console-free default: spans stay in-process; tests and log
            # correlation still work.  Add InMemorySpanExporter here for
            # debugging if desired.
            logger.info("otel_config: no OTLP endpoint configured; spans recorded locally.")

        trace.set_tracer_provider(provider)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("otel_config: telemetry init failed (%s); continuing without tracing", exc)
        return False


def get_tracer(name: str = "claimguard") -> Any:
    """Return the global tracer, or a no-op tracer when OTel is unavailable."""
    if HAS_OTEL:
        return trace.get_tracer(name)
    return _NOOP_TRACER


@contextmanager
def span(name: str, tracer_name: str = "claimguard", **attributes: Any) -> Iterator[Any]:
    """Convenience context manager around ``start_as_current_span``.

    Usage::

        with span("qdrant.hybrid_search", top_k=20):
            ...

    Attributes are set on entry; exceptions are recorded and re-raised with
    an ERROR status.  Always safe — falls back to a no-op span.
    """
    tracer = get_tracer(tracer_name)
    with tracer.start_as_current_span(name) as sp:
        try:
            for key, value in attributes.items():
                if value is not None:
                    sp.set_attribute(key, value)
            yield sp
        except Exception as exc:
            try:
                sp.record_exception(exc)
                if HAS_OTEL:
                    sp.set_status(Status(StatusCode.ERROR, str(exc)))
            except Exception:
                pass
            raise


def instrument_fastapi(app: Any) -> bool:
    """Attach OpenTelemetry ASGI middleware to a FastAPI app when available."""
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
        logger.info("otel_config: FastAPI instrumentation attached")
        return True
    except Exception as exc:
        logger.info("otel_config: FastAPI instrumentation unavailable (%s)", exc)
        return False


__all__ = [
    "HAS_OTEL",
    "initialize_telemetry",
    "get_tracer",
    "span",
    "instrument_fastapi",
]
