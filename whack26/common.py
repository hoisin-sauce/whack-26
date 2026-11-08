"""Small shared helpers that are not specific to one pipeline stage."""

from __future__ import annotations

import re
from datetime import datetime, timezone

_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;\"']+"),
    re.compile(r"(?i)(x-goog-api-key\s*[:=]\s*)[^\s,;\"']+"),
    re.compile(r"(?i)([?&](?:api_key|key|token|access_token)=)[^&\s]+"),
)


def redact_secrets(text: str) -> str:
    """Remove credentials from text that may be logged or raised."""
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(lambda match: match.group(1) + "[redacted]", redacted)
    return redacted


def utc_now_iso() -> str:
    """Return an RFC 3339 UTC timestamp with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
