"""What "forget ..." asks her to forget, and which memories are about it.

Deletion cannot be undone, so it goes by the words he used, not by semantic
similarity: asked to forget his favourite colour, recall at 0.35 similarity
also returns his favourite food. And "forget it" means "never mind", not
"delete something".
"""

from __future__ import annotations

import re
from typing import Optional

_FORGET_RE = re.compile(
    r"^(?:(?:ok(?:ay)?|so|and|actually|also)[,\s]+)?(?:please\s+)?(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
    r"(?:forget|stop\s+remembering|erase)\s+"
    r"(?:about\s+|that\s+(?=\S+\s)|what\s+i\s+(?:told|said\s+to)\s+you\s+about\s+|what\s+you\s+know\s+about\s+)?"
    r"(?P<topic>.+?)[\s.!?]*$",
    re.IGNORECASE,
)
# "forget it", "never mind, forget that": nothing to delete.
_IDIOMS = {
    "it",
    "that",
    "this",
    "it then",
    "that then",
    "about it",
    "about that",
    "all that",
    "all of that",
    "everything",
    "i said anything",
    "i asked",
    "what i said",
    "the whole thing",
}
_STOPWORDS = {
    "a",
    "an",
    "the",
    "my",
    "me",
    "i",
    "you",
    "your",
    "is",
    "are",
    "was",
    "that",
    "this",
    "about",
    "what",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "it",
    "said",
    "told",
    "user",
}
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def forget_topic(text: str) -> Optional[str]:
    """The thing a "forget ..." request names, or None when it names nothing."""
    match = _FORGET_RE.match(str(text or "").strip())
    if not match:
        return None
    topic = " ".join(match.group("topic").split())
    if topic.lower().strip(" ,.") in _IDIOMS or topic.lower().startswith(("it ", "that ")):
        return None
    return topic


def _words(text: str) -> set:
    words = set()
    for word in _WORD_RE.findall(str(text or "").lower()):
        word = word[:-2] if word.endswith("'s") else word
        if word and word not in _STOPWORDS:
            words.add(word)
    return words


def is_about(content: str, topic: str) -> bool:
    """Whether a memory says something about ``topic``: every word he used for it is in it."""
    wanted = _words(topic)
    return bool(wanted) and wanted <= _words(content)


def as_his(topic: str) -> str:
    """Said back to him: "my favorite color" becomes "your favorite color"."""
    return re.sub(r"\bmy\b", "your", str(topic or ""), flags=re.IGNORECASE)
