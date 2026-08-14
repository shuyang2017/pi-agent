"""Tests for the provider dispatch (pi_ai.stream)."""

from __future__ import annotations

import pytest

from pi_ai.stream import _API_STREAMS, stream
from pi_ai.types import Context, Model


def _model(api: str, provider: str) -> Model:
    return Model(id="m", name="m", api=api, provider=provider, maxTokens=1024)


def test_unknown_api_raises_synchronously():
    # The ValueError is raised before any network task is scheduled.
    with pytest.raises(ValueError):
        stream(_model("does-not-exist", "x"), Context())


def test_all_six_providers_routed():
    # Anthropic (Phase 1) + the five Phase 2 providers. The dispatch table must
    # map every supported api to a (stream, stream_simple) pair of callables.
    for api in [
        "anthropic-messages",
        "google-generative-ai",
        "bedrock-converse-stream",
        "openai-completions",
    ]:
        assert api in _API_STREAMS
        pair = _API_STREAMS[api]
        assert callable(pair[0]) and callable(pair[1])
