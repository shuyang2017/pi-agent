"""Streaming-tolerant JSON parsing (Python port of packages/ai/src/utils/json-parse.ts).

Four-layer fallback used while accumulating streamed tool-call arguments:

    parseStreamingJson(json):
      try: return parseJsonWithRepair(json)         # JSON.parse, else repair+parse
      except: try: return partialParse(json)        # tolerant partial-JSON parse
      except: try: return partialParse(repairJson(json))
      except: return {}

Mirrors the upstream exactly. ``partialParse`` comes from the ``partialjson``
package; if it is unavailable we degrade gracefully to ``{}`` (the same value
the final ``except`` layer returns).
"""

from __future__ import annotations

import json
from typing import Any, Dict, TypeVar

T = TypeVar("T")

try:  # pragma: no cover - depends on environment
    from partialjson import JSONParser

    _partial_parser = JSONParser(strict=False)

    def _partial_parse(s: str) -> Any:
        return _partial_parser.parse(s)

    _HAS_PARTIALJSON = True
except Exception:  # pragma: no cover
    _HAS_PARTIALJSON = False

    def _partial_parse(_s: str) -> Any:  # type: ignore
        raise ValueError("partialjson unavailable")


_VALID_JSON_ESCAPES = set(['"', "\\", "/", "b", "f", "n", "r", "t", "u"])


def _is_control_character(char: str) -> bool:
    cp = ord(char)
    return 0x00 <= cp <= 0x1F


def _escape_control_character(char: str) -> str:
    if char == "\b":
        return "\\b"
    if char == "\f":
        return "\\f"
    if char == "\n":
        return "\\n"
    if char == "\r":
        return "\\r"
    if char == "\t":
        return "\\t"
    cp = ord(char)
    return f"\\u{cp:04x}"


def repair_json(json_str: str) -> str:
    """Repairs malformed JSON string literals (control chars, invalid escapes)."""
    repaired: list[str] = []
    in_string = False

    i = 0
    n = len(json_str)
    while i < n:
        char = json_str[i]

        if not in_string:
            repaired.append(char)
            if char == '"':
                in_string = True
            i += 1
            continue

        if char == '"':
            repaired.append(char)
            in_string = False
            i += 1
            continue

        if char == "\\":
            nxt = json_str[i + 1] if i + 1 < n else None
            if nxt is None:
                repaired.append("\\\\")
                i += 1
                continue
            if nxt == "u":
                unicode_digits = json_str[i + 2 : i + 6]
                if len(unicode_digits) == 4 and all(c in "0123456789abcdefABCDEF" for c in unicode_digits):
                    repaired.append(f"\\u{unicode_digits}")
                    i += 6
                    continue
            if nxt in _VALID_JSON_ESCAPES:
                repaired.append(f"\\{nxt}")
                i += 2
                continue
            repaired.append("\\\\")
            i += 1
            continue

        repaired.append(_escape_control_character(char) if _is_control_character(char) else char)
        i += 1

    return "".join(repaired)


def parse_json_with_repair(json_str: str) -> Any:
    try:
        return json.loads(json_str)
    except Exception:
        repaired = repair_json(json_str)
        if repaired != json_str:
            return json.loads(repaired)
        raise


def parse_streaming_json(partial_json: Any) -> Dict[str, Any]:
    """Always returns a dict, even for incomplete JSON (mirrors upstream)."""
    if not partial_json or (isinstance(partial_json, str) and partial_json.strip() == ""):
        return {}

    try:
        return parse_json_with_repair(partial_json)
    except Exception:
        try:
            result = _partial_parse(partial_json)
            return result if result is not None else {}
        except Exception:
            try:
                result = _partial_parse(repair_json(partial_json))
                return result if result is not None else {}
            except Exception:
                return {}


__all__ = ["repair_json", "parse_json_with_repair", "parse_streaming_json", "_HAS_PARTIALJSON"]
