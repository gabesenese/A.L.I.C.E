"""Trims the padding that makes a reply read as machine written.

A retry rule used to force any answer under fifty words into a three or four
sentence "take", which trained the surface toward essays: a compliment, a restated
question, a paragraph of generic commentary, then an offer to go deeper. This
removes that scaffolding and leaves the substance.
"""

from __future__ import annotations

import re
from typing import List

DEFAULT_MAX_SENTENCES = 6

_FILLER_OPENINGS = (
    r"your enthusiasm is palpable",
    r"i love your enthusiasm",
    r"that'?s? (?:a )?(?:great|excellent|good|interesting|fascinating) (?:question|point|idea)",
    r"what (?:a|an) (?:great|excellent|interesting|fascinating) (?:question|point|idea)",
    r"i'?m (?:really |so )?(?:glad|excited|happy) (?:you|to)",
    r"(?:but )?let'?s dive (?:deeper|right in|into)",
    r"let'?s (?:unpack|explore) (?:this|that)",
    r"i'?d be happy to",
    r"thanks for (?:sharing|asking)",
    r"it'?s worth noting that",
)

# Standalone interjections, only stripped when they are the whole opening sentence.
_FILLER_INTERJECTIONS = (
    r"absolutely",
    r"certainly",
    r"of course",
    r"sure thing",
    r"great",
    r"perfect",
    r"got it",
)

_FILLER_CLOSINGS = (
    r"let me know if (?:you|there)",
    r"(?:feel free to|don'?t hesitate to) (?:ask|reach out|let me know)",
    r"i'?m here (?:to help|if you)",
    r"would you like (?:me to|to explore)",
    r"hope (?:this|that) helps",
    r"what (?:do you think|are your thoughts)",
)

_OPENING_RE = re.compile(
    r"^\W*(?:"
    + r"(?:" + "|".join(_FILLER_OPENINGS) + r")\b[^.!?]*[.!?]"
    + r"|(?:" + "|".join(_FILLER_INTERJECTIONS) + r")\s*[.!,]"
    + r")\s*",
    re.IGNORECASE,
)
_CLOSING_RE = re.compile(r"^\W*(?:" + "|".join(_FILLER_CLOSINGS) + r")\b", re.IGNORECASE)


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if s.strip()]


def strip_filler_opening(text: str) -> str:
    """Drop a leading compliment or throat-clearing sentence, but never everything."""
    cleaned = str(text or "").strip()
    for _ in range(2):
        candidate = _OPENING_RE.sub("", cleaned, count=1).strip()
        if not candidate or candidate == cleaned:
            break
        cleaned = candidate
    return cleaned or str(text or "").strip()


def strip_filler_closing(text: str) -> str:
    sentences = split_sentences(text)
    while len(sentences) > 1 and _CLOSING_RE.match(sentences[-1]):
        sentences.pop()
    return " ".join(sentences) if sentences else str(text or "").strip()


def limit_sentences(text: str, max_sentences: int = DEFAULT_MAX_SENTENCES) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences) if sentences else str(text or "").strip()
    return " ".join(sentences[:max_sentences])


_EXECUTION_CLAIMS = re.compile(
    r"\b(?:"
    r"tests?\s+(?:are\s+)?(?:pass(?:ed|ing)?|fail(?:ed|ing)?)"
    r"|all\s+tests?\s+\w+"
    r"|no\s+(?:test\s+)?(?:failures|errors)"
    r"|\d+\s+tests?\s+(?:ran|passed|failed)"
    r"|test\s+suite\s+\w+"
    r"|build\s+(?:succeeded|passed|failed)"
    r"|suite\s+is\s+green"
    r")\b",
    re.IGNORECASE,
)


def claims_execution_happened(text: str) -> bool:
    return bool(_EXECUTION_CLAIMS.search(str(text or "")))


_RUN_REQUEST = re.compile(r"^\s*(?:run|execute|please run|go ahead and run|kick off)\b", re.IGNORECASE)

# Questions about the state of a build or test run. Answering these from the model
# rather than from a command is guessing, and it guesses confidently.
_EXECUTION_QUESTION = re.compile(
    r"\b(?:tests?|test\s+suite|build|ci|pipeline|lint(?:er)?|coverage)\b.{0,40}?"
    r"\b(?:pass(?:ing|ed)?|fail(?:ing|ed)?|green|red|broken|clean|ok|working|succeed(?:ing|ed)?)\b",
    re.IGNORECASE,
)


def asks_to_run_something(user_input: str) -> bool:
    text = str(user_input or "")
    return bool(_RUN_REQUEST.match(text) or _EXECUTION_QUESTION.search(text))


def guard_unverified_execution_claims(text: str, ran_command: bool = False, user_input: str = "") -> str:
    """Refuse to report the result of work that was never done.

    Asked to run the tests without actually running them, the model reported passing
    tests, a duration, and a flaky test fixed "last week", then on another turn
    invented failing test names and assertion errors. Chasing each phrasing is a
    losing game, so an explicit request to run something is also gated: if no command
    executed, there is no result to report, whatever the wording.
    """
    if ran_command:
        return text
    if asks_to_run_something(user_input) or claims_execution_happened(text):
        return "I haven't actually run that yet. Want me to run it now?"
    return text


def apply_response_discipline(text: str, max_sentences: int = DEFAULT_MAX_SENTENCES) -> str:
    """Remove padding and cap length at a sentence boundary, never mid sentence."""
    original = str(text or "").strip()
    if not original:
        return ""
    cleaned = strip_filler_opening(original)
    cleaned = strip_filler_closing(cleaned)
    cleaned = limit_sentences(cleaned, max_sentences=max_sentences)
    return cleaned.strip() or original
