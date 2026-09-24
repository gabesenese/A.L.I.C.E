"""Short follow-ups that only make sense next to the turn before.

"what about tomorrow?" after the agenda, "and eggs" after adding milk to a
list, "actually make it 6" after setting a reminder. Each went to the model on
its own, which had nothing to act with: no agenda for tomorrow, no list, and
no way to move a reminder. The NLP uses these to keep the turn with the plugin
that did the last one, and the plugins use them to finish it.
"""

from __future__ import annotations

import re
from typing import Optional

# "what about tomorrow?", "and tomorrow?", "how about this week"
DAY_FOLLOWUP_RE = re.compile(
    r"^(?:and\s+|so\s+)?(?:(?:what|how)\s+about\s+)?(?:for\s+)?(?:tomorrow|today|tonight|this\s+week)\s*\??$",
    re.IGNORECASE,
)

_NOT_AN_ITEM = r"(?:what|how|why|when|where|who|is|are|can|could|do|does|did|then|now|so|that|it|thanks?|thank)\b"
# "and eggs", "also bread and butter", "eggs too"
_MORE_ITEMS_RE = re.compile(
    r"^(?:and|also|plus)\s+(?!" + _NOT_AN_ITEM + r")(?P<items>[\w' ,-]{1,60}?)(?:\s+too)?[.!]?$"
    r"|^(?!(?:and|also|plus)\b|" + _NOT_AN_ITEM + r")(?P<items_too>[\w' ,-]{1,40}?)\s+(?:too|as\s+well)[.!]?$",
    re.IGNORECASE,
)

# "actually make it 6", "make that 6pm", "change it to tomorrow at 9", "move it to 7"
RESCHEDULE_RE = re.compile(
    r"^(?:(?:actually|no|sorry|wait|oh|hmm)[,\s]+)*(?:can\s+you\s+)?"
    r"(?:make\s+(?:it|that)|change\s+(?:it|that)\s+to|move\s+(?:it|that)(?:\s+to)?|push\s+(?:it|that)(?:\s+back)?\s+to"
    r"|set\s+it\s+(?:for|to))\s+(?P<when>.+?)[.!?]*$",
    re.IGNORECASE,
)


def more_items(text: str) -> Optional[str]:
    """The items in "and eggs" or "eggs too", or None when it is not that."""
    match = _MORE_ITEMS_RE.match(" ".join(str(text or "").split()))
    if not match:
        return None
    items = (match.group("items") or match.group("items_too") or "").strip(" ,")
    return items or None


# "snooze", "snooze for 10 minutes", "snooze it 5 min" once a reminder has gone off.
SNOOZE_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|ugh|hmm)[,\s]+)?snooze(?:\s+(?:it|that))?(?:\s+(?:for\s+)?(?P<amount>\d+|five|ten|fifteen|twenty|thirty)"
    r"\s*(?:min(?:ute)?s?|m)?)?[.!]*$",
    re.IGNORECASE,
)


# The answer to "When should I remind you to call mom?": "at 5", "tomorrow morning".
TIME_ANSWER_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|um+|uh+|hmm+|let'?s\s+say|say)[,\s]+)*"
    r"(?:(?:at|in|by|around|on|this|next|tomorrow|tonight|today|noon|midnight)\b"
    r"|\d{1,2}(?::\d{2})?\s*(?:am|pm|a\.?m\.?|p\.?m\.?)?(?=\W|$)"
    r"|(?:mon|tues|wednes|thurs|fri|satur|sun)day\b)"
    r"[\w\s:.,']{0,40}$",
    re.IGNORECASE,
)
