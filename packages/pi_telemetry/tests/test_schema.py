import pytest

from pi_telemetry import (
    InMemoryTelemetryContext,
    TelemetrySchemaDefinition,
    TelemetrySpanDefinition,
    TelemetryStartAttributeDefinition,
    create_typed_span_starter,
    define_telemetry_schema,
)
from pi_telemetry.schema import validate_span_start


def _sample_schema():
    return define_telemetry_schema(
        TelemetrySchemaDefinition(
            version=1,
            spans={
                "agent.turn": TelemetrySpanDefinition(
                    description="A single agent turn.",
                    start_attributes={
                        "model": TelemetryStartAttributeDefinition(
                            description="Model id", required=True
                        ),
                        "session": TelemetryStartAttributeDefinition(
                            description="Session id", required=False
                        ),
                    },
                )
            },
        )
    )


def test_define_telemetry_schema_is_identity():
    schema = _sample_schema()
    assert define_telemetry_schema(schema) is schema


async def test_typed_span_starter_delegates_to_context():
    ctx = InMemoryTelemetryContext()
    starter = create_typed_span_starter(ctx, _sample_schema())

    result = await starter("agent.turn", {"model": "claude", "session": "s1"}, lambda s: "done")
    assert result == "done"
    spans = ctx.get_spans()
    assert len(spans) == 1
    assert spans[0].name == "agent.turn"
    assert spans[0].attributes["model"] == "claude"


def test_validate_span_start_enforces_required():
    schema = _sample_schema()
    # Missing required "model".
    with pytest.raises(ValueError):
        validate_span_start(schema, "agent.turn", {"session": "s1"})
    # Present is fine.
    validate_span_start(schema, "agent.turn", {"model": "claude"})
    # Unknown span name rejected.
    with pytest.raises(ValueError):
        validate_span_start(schema, "nope", {})
