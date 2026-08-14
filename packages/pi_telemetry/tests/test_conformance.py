import pytest

from pi_telemetry import InMemoryTelemetryContext
from pi_telemetry.testing import (
    TelemetryAdapterFixture,
    create_telemetry_adapter_conformance,
)

# The in-memory reference adapter is the canonical conformance target: it records
# spans and exposes a normalized `get_spans()` snapshot.


async def _memory_fixture():
    ctx = InMemoryTelemetryContext()
    return TelemetryAdapterFixture(context=ctx, get_spans=ctx.get_spans)


MEMORY_CASES = create_telemetry_adapter_conformance(_memory_fixture)


@pytest.mark.parametrize(
    "case", MEMORY_CASES, ids=lambda c: f"{c.group}: {c.name}"
)
async def test_conformance_memory(case):
    await case.run()
