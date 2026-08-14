import pytest

from pi_ai.api.anthropic_messages import map_stop_reason


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("end_turn", "stop"),
        ("max_tokens", "length"),
        ("tool_use", "toolUse"),
        ("pause_turn", "stop"),
        ("stop_sequence", "stop"),
        ("refusal", "error"),
        ("sensitive", "error"),
    ],
)
def test_map_stop_reason_known(reason, expected):
    res = map_stop_reason(reason)
    assert res.stopReason == expected


def test_map_stop_reason_refusal_carries_message():
    res = map_stop_reason("refusal", {"explanation": "nope"})
    assert res.stopReason == "error"
    assert res.errorMessage == "nope"


def test_map_stop_reason_unknown_raises():
    with pytest.raises(ValueError):
        map_stop_reason("something_new")
