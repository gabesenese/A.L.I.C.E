"""Where he lives, learned from what he says, for the weather and for context.

Asked "What city should I check the weather for?", "Toronto" went to the model
as conversation, and the next weather question asked for the city again. "I
live in Toronto" was not kept anywhere the weather could use it.
"""

from __future__ import annotations

import re
from typing import Optional

# Replies to "what city?" that are not a city.
_NOT_A_PLACE = {
    "yes",
    "yeah",
    "yep",
    "no",
    "nope",
    "ok",
    "okay",
    "thanks",
    "thank you",
    "never mind",
    "nevermind",
    "forget it",
    "cancel",
    "stop",
    "idk",
    "dunno",
    "not sure",
    "why",
    "what",
    "huh",
    "here",
    "home",
    "there",
    "anywhere",
    "nowhere",
    "later",
    "skip",
    "cool",
    "nice",
    "sure",
}
_PLACE = r"(?P<place>[a-z][a-z.'-]*(?:\s+[a-z][a-z.'-]*){0,3}(?:,\s*[a-z][a-z .'-]{1,30})?)"
_ANSWER_RE = re.compile(
    r"^(?:(?:it'?s|in|i'?m\s+in|i\s+live\s+in|i'?m\s+based\s+in)\s+)?" + _PLACE + r"[.!]?$", re.IGNORECASE
)
_HOME_RE = re.compile(
    r"\b(?:i\s+live\s+in|i'?m\s+based\s+in|i\s+(?:just\s+)?moved\s+to|i'?m\s+(?:now\s+)?living\s+in|"
    r"my\s+(?:home\s+)?(?:city|town)\s+is)\s+" + _PLACE,
    re.IGNORECASE,
)
# Where a stated city ends: "I moved to Berlin last year".
_TRAILING = re.compile(
    r"\s+(?:last|this|next|in|for|since|because|with|and|but|now|recently|already|a|about|these|so)\b.*$",
    re.IGNORECASE,
)


# A reply with a verb or a pronoun in it is a sentence, not a city.
_SENTENCE_WORDS = {
    "is",
    "are",
    "was",
    "were",
    "be",
    "it",
    "i",
    "you",
    "we",
    "they",
    "do",
    "does",
    "did",
    "can",
    "will",
    "my",
    "your",
    "what",
    "how",
    "why",
    "when",
    "not",
    "weather",
    "raining",
    "snowing",
    "sunny",
    "cold",
    "hot",
}


def _tidy(place: str) -> Optional[str]:
    place = _TRAILING.sub("", " ".join(str(place or "").split())).strip(" .,")
    if not place or place.lower() in _NOT_A_PLACE:
        return None
    if _SENTENCE_WORDS & set(re.findall(r"[a-z']+", place.lower())):
        return None
    return place.title() if place.islower() else place


def place_answer(text: str) -> Optional[str]:
    """The city in a reply to "what city?", or None when the reply is not one."""
    match = _ANSWER_RE.match(" ".join(str(text or "").split()))
    return _tidy(match.group("place")) if match else None


def stated_home(text: str) -> Optional[str]:
    """The city in "I live in Toronto", "I moved to Berlin last year", if he said one."""
    match = _HOME_RE.search(str(text or ""))
    return _tidy(match.group("place")) if match else None


def home_location() -> str:
    try:
        from ai.identity.user_identity import load_identity

        return str(load_identity().home_location or "").strip()
    except Exception:
        return ""


def remember_home(place: str) -> None:
    place = str(place or "").strip()
    if not place:
        return
    try:
        from ai.identity.user_identity import load_identity, save_identity

        identity = load_identity()
        if identity.home_location != place:
            identity.home_location = place
            save_identity(identity)
    except Exception:
        return
