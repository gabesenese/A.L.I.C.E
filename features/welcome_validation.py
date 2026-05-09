from __future__ import annotations

from typing import List


_BANNED_TOKENS = (
    "how can i help",
    "how may i assist",
    "i'm here to help",
    "i am here to help",
    "assist",
    "support",
    "anything you need",
    "whatever you need",
    "need help",
    "give me",
    "tell me",
    "share",
    "point me",
    "want",
    "do you want",
    "i will",
    "i'll",
    "execution plan",
    "critical path",
    "handoff",
    "open loops",
    "deep work",
    "minimal and high-value",
    "map the shortest path",
    "systems online",
    "systems steady",
    "quiet mode",
    "neural",
    "memory cores",
    "activating",
    "initializing",
    "cry for help",
    "bad for sleep",
    "denial",
    "dead",
    "barely",
    "terrible timing",
    "responsible people",
)


def validate_startup_greeting(text: str, *, require_name_placeholder: bool = False) -> List[str]:
    reasons: List[str] = []
    raw = str(text or "")
    low = raw.lower()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]

    if "?" in raw:
        reasons.append("contains_question_mark")
    if require_name_placeholder and "{name}" not in raw:
        reasons.append("missing_name_placeholder")
    if not lines:
        reasons.append("empty")
    if len(lines) > 3:
        reasons.append("too_many_lines")
    if any(len(line) > 90 for line in lines):
        reasons.append("line_too_long")

    for token in _BANNED_TOKENS:
        if token in low:
            reasons.append(f"banned_phrase:{token}")

    return reasons


def is_valid_startup_greeting(text: str, *, require_name_placeholder: bool = False) -> bool:
    return not validate_startup_greeting(
        text, require_name_placeholder=require_name_placeholder
    )
