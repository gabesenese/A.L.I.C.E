"""Learning-data redaction utilities.

Protects persisted training artifacts by masking common sensitive data
before it is written to disk.
"""

from __future__ import annotations

import re
from typing import Any

_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?<![A-Za-z0-9_\-])(?:\+?\d{1,3}[\s\-.]?)?(?:\(?\d{3}\)?[\s\-.])\d{3}[\s\-.]\d{4}(?![A-Za-z0-9_\-])"
)
_API_KEY_RE = re.compile(
    r"\b(?:sk-[a-zA-Z0-9]{16,}|xox[baprs]-[a-zA-Z0-9-]{10,}|ghp_[a-zA-Z0-9]{20,}|AIza[0-9A-Za-z\-_]{20,})\b"
)
_BEARER_RE = re.compile(r"\bBearer\s+[A-Za-z0-9._\-+/=]{16,}\b", re.IGNORECASE)
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9._-]+\.[A-Za-z0-9._-]+\b")
_IP_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)


def redact_text(text: str) -> str:
    """Mask common sensitive patterns while retaining semantic structure."""
    if not isinstance(text, str) or not text:
        return text

    out = text
    out = _EMAIL_RE.sub("[REDACTED_EMAIL]", out)
    out = _PHONE_RE.sub("[REDACTED_PHONE]", out)
    out = _API_KEY_RE.sub("[REDACTED_API_KEY]", out)
    out = _BEARER_RE.sub("[REDACTED_BEARER_TOKEN]", out)
    out = _JWT_RE.sub("[REDACTED_JWT]", out)
    out = _IP_RE.sub("[REDACTED_IP]", out)
    return out


def sanitize_for_learning(value: Any) -> Any:
    """Recursively sanitize values before persisting learning artifacts."""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        sanitized = {}
        for key, val in value.items():
            key_str = str(key)
            if key_str.lower() in {
                "password",
                "passphrase",
                "token",
                "access_token",
                "refresh_token",
                "authorization",
                "auth_header",
                "api_key",
                "secret",
                "private_key",
                "cookie",
                "session",
                "session_id",
            }:
                sanitized[key] = "[REDACTED_SECRET]"
            else:
                sanitized[key] = sanitize_for_learning(val)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_learning(item) for item in value]
    if hasattr(value, "__dict__"):
        return sanitize_for_learning(vars(value))
    return redact_text(str(value))
