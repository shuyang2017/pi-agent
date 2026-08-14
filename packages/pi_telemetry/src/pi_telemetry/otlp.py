"""OpenTelemetry OTLP telemetry adapter (optional, lazy-imported).

This adapter maps the vendor-neutral :class:`TelemetryContext` contract onto the
OpenTelemetry SDK tracer API. OpenTelemetry is imported lazily so that the rest of the
library (and the default no-op path) never requires it to be installed. Construct via
:func:`create_otlp_telemetry_context`; if OpenTelemetry is unavailable the factory
raises ``ImportError`` with an actionable message.

Defaults across Pi: applications start with :data:`NOOP_TELEMETRY_CONTEXT` and only
opt into OTLP export when configured.
"""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .contracts import SpanAttributes, SpanOptions, TelemetryContext, TelemetrySpan


def _to_otel_value(value):
    # OpenTelemetry accepts str/bool/int/float and sequences thereof; our
    # AttributeValue union already matches, so pass through (dropping None upstream).
    return value


class _OtelSpan:
    """Wraps an OpenTelemetry span as a :class:`TelemetrySpan`."""

    def __init__(self, context: "OtlpTelemetryContext", otel_span: Any) -> None:
        self._context = context
        self._otel_span = otel_span
        self.explicit = False

    def start_span(self, options: SpanOptions, callback) -> Awaitable[Any]:
        return self._context.start_span(options, callback)

    def add_event(self, name: str, attributes: Optional[SpanAttributes] = None) -> None:
        attrs = self._clean(attributes)
        if attrs is not None:
            self._otel_span.add_event(name, attrs)

    def set_attributes(self, attributes: SpanAttributes) -> None:
        attrs = self._clean(attributes)
        if attrs is None:
            return
        for key, value in attrs.items():
            self._otel_span.set_attribute(key, _to_otel_value(value))

    def set_status(self, status: Dict[str, Any]) -> None:
        from opentelemetry.trace import Status, StatusCode

        self.explicit = True
        if status.get("status") == "ok":
            self._otel_span.set_status(Status(StatusCode.OK))
        else:
            error = status.get("error") or {}
            message = error.get("message") or error.get("name") or "error"
            self._otel_span.set_status(Status(StatusCode.ERROR, message))
            self._otel_span.record_exception(Exception(message))

    def _clean(self, attributes: Optional[SpanAttributes]) -> Optional[Dict[str, Any]]:
        if not attributes:
            return None
        cleaned: Dict[str, Any] = {}
        for key, value in attributes.items():
            if value is not None:
                cleaned[key] = value
        return cleaned


class OtlpTelemetryContext:
    """A :class:`TelemetryContext` backed by an OpenTelemetry tracer."""

    def __init__(self, tracer: Any, exporter: Any = None, provider: Any = None) -> None:
        self._tracer = tracer
        self._exporter = exporter
        self._provider = provider

    async def start_span(self, options: SpanOptions, callback: Callable) -> Awaitable[Any]:
        from opentelemetry.trace import StatusCode

        name = options["name"]
        attributes = options.get("attributes") or {}
        with self._tracer.start_as_current_span(name) as otel_span:
            cleaned = {k: v for k, v in attributes.items() if v is not None}
            if cleaned:
                otel_span.set_attributes(cleaned)
            wrapper = _OtelSpan(self, otel_span)
            try:
                result = callback(wrapper)
                if inspect.isawaitable(result):
                    result = await result
                return result
            except Exception as error:
                if not wrapper.explicit:
                    otel_span.set_status(StatusCode.ERROR, str(error) or type(error).__name__)
                    otel_span.record_exception(error)
                raise

    def get_spans(self) -> Optional[List[Any]]:
        """Best-effort snapshot when an in-memory exporter was supplied.

        Returns ``None`` when telemetry is exported remotely (no local buffer).
        """
        exporter = self._exporter
        if exporter is None or not hasattr(exporter, "get_finished_spans"):
            return None
        finished = exporter.get_finished_spans()
        id_map: Dict[int, int] = {}
        out: List[Any] = []
        for index, span in enumerate(finished, start=1):
            our_id = index
            id_map[span.context.span_id] = our_id
        for span in finished:
            parent_id = None
            if span.parent is not None:
                parent_id = id_map.get(span.parent.span_id)
            status = span.status
            status_dict: Dict[str, Any] = {
                "status": "ok" if status.status_code.name == "OK" else "error"
            }
            if status.status_code.name != "OK" and status.description:
                status_dict["error"] = {
                    "name": "OpenTelemetry",
                    "message": status.description,
                }
            events = tuple(
                {"name": ev.name, "attributes": dict(ev.attributes or {})}
                for ev in span.events
            )
            out.append(
                {
                    "id": id_map[span.context.span_id],
                    "parent_id": parent_id,
                    "name": span.name,
                    "attributes": dict(span.attributes or {}),
                    "events": events,
                    "status": status_dict,
                    "settled": True,
                }
            )
        return out


def create_otlp_telemetry_context(
    *,
    service_name: str = "pi",
    endpoint: Optional[str] = None,
    exporter: Any = None,
    tracer_provider: Any = None,
    span_processor: Any = None,
) -> OtlpTelemetryContext:
    """Create an OTLP-backed telemetry context.

    Requires the optional ``opentelemetry-*`` dependencies. Provide one of
    ``endpoint``, ``exporter``, or ``tracer_provider``. With no configuration this
    raises ``ValueError``.
    """
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            SimpleSpanProcessor,
        )
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "OpenTelemetry is not installed. Install the OTLP extra with "
            "`pip install pi-telemetry[otlp]` (or `opentelemetry-api`, "
            "`opentelemetry-sdk`, `opentelemetry-exporter-otlp`)."
        ) from exc

    if tracer_provider is None:
        if endpoint is None and exporter is None:
            raise ValueError(
                "Provide one of `endpoint`, `exporter`, or `tracer_provider` to "
                "create an OTLP telemetry context."
            )
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        if exporter is None:
            exporter = OTLPSpanExporter(endpoint=endpoint)
        processor = span_processor or (
            SimpleSpanProcessor(exporter)
            if endpoint is None
            else BatchSpanProcessor(exporter)
        )
        provider.add_span_processor(processor)
        tracer = provider.get_tracer(service_name)
    else:
        provider = tracer_provider
        tracer = provider.get_tracer(service_name)

    return OtlpTelemetryContext(tracer=tracer, exporter=exporter, provider=provider)
