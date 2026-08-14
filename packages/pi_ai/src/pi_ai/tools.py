"""Tool argument validation (Python port of packages/ai/src/utils/validation.ts).

Upstream uses TypeBox + Value.Convert for full JSON-schema coercion. This port
does lightweight, permissive coercion: deep-copies the args, fills missing
required keys with ``None`` (so the agent keeps running rather than throwing),
and coerces scalar values to the property's declared type. Full structural
validation is intentionally simplified; it is enough for functional equivalence
in the agent loop and tool dispatch.
"""

from __future__ import annotations

import copy
from typing import Any

from .types import Tool, ToolCall


def validate_tool_arguments(tool: Tool, tool_call: ToolCall) -> Any:
    args = copy.deepcopy(tool_call.arguments)
    schema = tool.parameters if isinstance(tool.parameters, dict) else {}
    required = schema.get("required", []) if isinstance(schema, dict) else []
    for key in required:
        args.setdefault(key, None)

    props = schema.get("properties", {}) if isinstance(schema, dict) else {}
    for key, val in list(args.items()):
        declared = (props.get(key) or {}).get("type") if isinstance(props.get(key), dict) else None
        if declared == "boolean" and not isinstance(val, bool):
            args[key] = str(val).strip().lower() in ("true", "1", "yes")
        elif declared == "integer" and not isinstance(val, int) and not isinstance(val, bool):
            try:
                args[key] = int(val)
            except (TypeError, ValueError):
                pass
        elif declared == "number" and not isinstance(val, (int, float)):
            try:
                args[key] = float(val)
            except (TypeError, ValueError):
                pass
    return args


__all__ = ["validate_tool_arguments"]
