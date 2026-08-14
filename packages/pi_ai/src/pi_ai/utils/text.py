"""Small text helpers (Python port of packages/ai/src/utils/sanitize-unicode.ts)."""

from __future__ import annotations

import re

_SURROGATE_RE = re.compile(
    r"[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]"
)


def sanitize_surrogates(text: str) -> str:
    """Remove unpaired Unicode surrogate characters that break JSON serialization."""
    return _SURROGATE_RE.sub("", text)


__all__ = ["sanitize_surrogates"]
