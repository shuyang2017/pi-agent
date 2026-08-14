"""Pi telemetry — vendor-neutral, pluggable, OTLP-ready.

Public surface mirrors upstream ``@earendil-works/pi-telemetry``:
* contracts (``TelemetryContext`` / ``TelemetrySpan`` protocols + value types),
* ``NOOP_TELEMETRY_CONTEXT`` (default, records nothing but runs callbacks),
* ``InMemoryTelemetryContext`` (reference recording adapter),
* schema helpers (``define_telemetry_schema`` / ``create_typed_span_starter``),
* ``OtlpTelemetryContext`` + ``create_otlp_telemetry_context`` (optional, lazy).
"""

from __future__ import annotations

from .contracts import (
    AttributeValue,
    SpanAttributes,
    SpanOptions,
    SpanStatus,
    TelemetryContext,
    TelemetrySpan,
)
from .memory import (
    InMemoryTelemetryContext,
    RecordedTelemetryEvent,
    RecordedTelemetrySpan,
)
from .noop import NOOP_TELEMETRY_CONTEXT
from .otlp import OtlpTelemetryContext, create_otlp_telemetry_context
from .schema import (
    TelemetryAttributeDefinition,
    TelemetryAttributeMetadata,
    TelemetryEventAttributeDefinition,
    TelemetryEventDefinition,
    TelemetryParentDefinition,
    TelemetrySchemaDefinition,
    TelemetrySpanDefinition,
    TelemetryStartAttributeDefinition,
    create_typed_span_starter,
    define_telemetry_schema,
)

__all__ = [
    "AttributeValue",
    "SpanAttributes",
    "SpanOptions",
    "SpanStatus",
    "TelemetryContext",
    "TelemetrySpan",
    "NOOP_TELEMETRY_CONTEXT",
    "InMemoryTelemetryContext",
    "RecordedTelemetryEvent",
    "RecordedTelemetrySpan",
    "define_telemetry_schema",
    "create_typed_span_starter",
    "TelemetrySchemaDefinition",
    "TelemetrySpanDefinition",
    "TelemetryAttributeMetadata",
    "TelemetryAttributeDefinition",
    "TelemetryStartAttributeDefinition",
    "TelemetryEventAttributeDefinition",
    "TelemetryEventDefinition",
    "TelemetryParentDefinition",
    "OtlpTelemetryContext",
    "create_otlp_telemetry_context",
]
