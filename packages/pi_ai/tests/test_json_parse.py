from pi_ai.utils.json_parse import parse_json_with_repair, parse_streaming_json, repair_json


def test_repair_json_escapes_control_chars():
    bad = '{"a": "line1\nline2"}'  # raw newline inside string is invalid JSON
    # A literal newline (not escaped) inside a JSON string is invalid; repair should escape it.
    repaired = repair_json('{"a": "line1\nline2"}')
    assert '"line1\\nline2"' in repaired
    assert parse_json_with_repair(repaired) == {"a": "line1\nline2"}


def test_repair_json_doubles_backslash_before_invalid_escape():
    # "\x" is not a valid escape; repair should turn it into "\\x"
    repaired = repair_json('{"a": "\\x"}')
    assert parse_json_with_repair(repaired) == {"a": "\\x"}


def test_parse_streaming_json_complete():
    assert parse_streaming_json('{"a": 1}') == {"a": 1}


def test_parse_streaming_json_empty():
    assert parse_streaming_json("") == {}
    assert parse_streaming_json("   ") == {}
    assert parse_streaming_json(None) == {}


def test_parse_streaming_json_partial_tool_args():
    # Incomplete JSON with a trailing comma and unterminated string -> tolerant parse.
    partial = '{"pattern": "TODO", "path": "./sr'  # truncated
    result = parse_streaming_json(partial)
    # partialjson should recover at least the complete key.
    assert result.get("pattern") == "TODO"


def test_parse_streaming_json_four_layer_fallback_returns_dict():
    # Something completely unparsable still returns a dict (layer 4).
    assert isinstance(parse_streaming_json("{not json at all"), dict)
