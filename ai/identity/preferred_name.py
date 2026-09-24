"""What he wants to be called, learned from saying so.

"call me Gabe" was stored nowhere, and the persona was built once, at start-up,
around a fixed default name, so she went on using the default.
"""

from __future__ import annotations

import re
from typing import Optional

_NAME_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|so|hey|actually|also|and)[,\s]+)?(?:please\s+)?(?:(?:you\s+can|just|from\s+now\s+on,?)\s+)?"
    r"(?:call\s+me|i\s+go\s+by|my\s+name(?:'s|\s+is))\s+"
    r"(?P<name>[a-z][a-z'-]{0,20}(?:\s+(?!please\b|instead\b|from\b|now\b)[a-z][a-z'-]{0,20})?)"
    r"(?:\s+(?:please|from\s+now\s+on|instead))?[.!]*$",
    re.IGNORECASE,
)
# "call me later", "call me back": not a name.
_NOT_A_NAME = {
    "later",
    "back",
    "tomorrow",
    "tonight",
    "sometime",
    "maybe",
    "anytime",
    "now",
    "soon",
    "crazy",
    "lazy",
    "old",
    "old fashioned",
    "whatever",
    "anything",
    "a",
    "an",
    "the",
}


def stated_name(text: str) -> Optional[str]:
    """The name in "call me Gabe", "I go by Gabe", "my name is Gabriel", if any."""
    match = _NAME_RE.match(" ".join(str(text or "").split()))
    if not match:
        return None
    name = match.group("name").strip()
    if name.lower() in _NOT_A_NAME or name.split()[0].lower() in _NOT_A_NAME:
        return None
    return name.title() if name.islower() else name


def preferred_name() -> str:
    try:
        from ai.identity.user_identity import load_identity

        return str(load_identity().name or "").strip()
    except Exception:
        return ""


def remember_name(name: str) -> None:
    name = str(name or "").strip()
    if not name:
        return
    try:
        from ai.identity.user_identity import load_identity, save_identity

        identity = load_identity()
        if identity.name != name:
            identity.name = name
            save_identity(identity)
    except Exception:
        return
