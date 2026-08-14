"""Telemetry span-schema definitions and a typed span starter.

The upstream package encodes span schemas as a heavily type-level construct used only
for compile-time inference (required vs. optional attributes, allowed event names,
parent constraints). Python has no equivalent compile-time machinery, so this module:

* provides dataclasses that mirror the upstream schema shape (so schemas remain
  inspectable, documentable data),
* exposes :func:`define_telemetry_schema` as an identity helper (the upstream
  ``defineTelemetrySchema`` simply returns its argument), and
* exposes :func:`create_typed_span_starter`, which binds one or more schemas to a
  concrete :class:`TelemetryContext`. At runtime the starter is a thin delegating
  wrapper -- no validation is performed (mirrors the upstream note: "Schema values are
  used only for type inference; no runtime schema validation is performed.").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Union

from .contracts import SpanAttributes, SpanOptions, TelemetryContext

AttributeType = Literal["string", "number", "boolean", "string[]", "number[]", "boolean[]"]
ParentKind = Literal["any", "root_or_external", "spans"]


@dataclass
class TelemetryAttributeMetadata:
    description: str
    sensitive: bool = False
    cardinality: Optional[Literal["low", "high"]] = None


@dataclass
class TelemetryAttributeDefinition(TelemetryAttributeMetadata):
    type: AttributeType = "string"
    values: Optional[List[Any]] = None
    examples: Optional[List[Any]] = None
    element_values: Optional[List[Any]] = None


@dataclass
class TelemetryStartAttributeDefinition(TelemetryAttributeDefinition):
    required: bool = False


@dataclass
class TelemetryEventAttributeDefinition(TelemetryAttributeDefinition):
    required: bool = False


@dataclass
class TelemetryEventDefinition:
    description: str
    attributes: Dict[str, TelemetryEventAttributeDefinition] = field(default_factory=dict)


@dataclass
class TelemetryParentDefinition:
    kind: ParentKind = "any"
    spans: List[str] = field(default_factory=list)


@dataclass
class TelemetrySpanDefinition:
    description: str
    parents: TelemetryParentDefinition = field(default_factory=TelemetryParentDefinition)
    start_attributes: Dict[str, TelemetryStartAttributeDefinition] = field(default_factory=dict)
    end_attributes: Dict[str, TelemetryAttributeDefinition] = field(default_factory=dict)
    events: Dict[str, TelemetryEventDefinition] = field(default_factory=dict)
    status_default: str = "ok"
    status_error_when: str = ""


@dataclass
class TelemetrySchemaDefinition:
    version: int
    spans: Dict[str, TelemetrySpanDefinition] = field(default_factory=dict)


def define_telemetry_schema(schema: TelemetrySchemaDefinition) -> TelemetrySchemaDefinition:
    """Identity helper for serializable telemetry schema data."""
    return schema


def validate_span_start(
    schema: TelemetrySchemaDefinition,
    name: str,
    attributes: Optional[SpanAttributes],
) -> None:
    """Best-effort, opt-in validation of required start attributes for a span.

    Raises ``ValueError`` if ``name`` is unknown or a required start attribute is
    missing. This is *not* invoked by :func:`create_typed_span_starter`; call it
    yourself if you want runtime enforcement.
    """
    definition = schema.spans.get(name)
    if definition is None:
        raise ValueError(f"Unknown span name: {name!r}")
    provided = attributes or {}
    missing = [
        attr_name
        for attr_name, attr_def in definition.start_attributes.items()
        if attr_def.required and attr_name not in provided
    ]
    if missing:
        raise ValueError(
            f"Span {name!r} missing required start attributes: {', '.join(missing)}"
        )


def create_typed_span_starter(
    context: TelemetryContext,
    schemas: Union[TelemetrySchemaDefinition, List[TelemetrySchemaDefinition]],
):
    """Bind a concrete :class:`TelemetryContext` to one or more schemas.

    Returns a ``start_span(name, attributes, callback)`` callable that delegates to
    ``context.start_span``. Schemas are carried only for documentation/introspection;
    no runtime validation is performed.
    """
    schema_list = schemas if isinstance(schemas, list) else [schemas]

    async def start_span(
        name: str,
        attributes: Optional[SpanAttributes],
        callback: Callable,
    ) -> Awaitable[Any]:
        options: SpanOptions = {"name": name}
        if attributes:
            options["attributes"] = attributes
        return await context.start_span(options, callback)

    start_span.schemas = schema_list  # type: ignore[attr-defined]
    return start_span
