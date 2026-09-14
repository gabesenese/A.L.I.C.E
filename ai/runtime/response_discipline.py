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

# Servile sign-offs that carry no content. "What do you think?" is deliberately
# not here: it is a real question, and stripping it removed the reciprocity that
# makes an exchange two-way rather than a lookup returning a value. An offer to
# go and do something ("would you like me to…") does stay, because Alice can
# simply go and do it.
_FILLER_CLOSINGS = (
    r"let me know if (?:you|there)",
    r"(?:feel free to|don'?t hesitate to) (?:ask|reach out|let me know)",
    r"i'?m here (?:to help|if you)",
    r"would you like (?:me to|to explore)",
    r"hope (?:this|that) helps",
)

_OPENING_RE = re.compile(
    r"^\W*(?:"
    + r"(?:"
    + "|".join(_FILLER_OPENINGS)
    + r")\b[^.!?]*[.!?]"
    + r"|(?:"
    + "|".join(_FILLER_INTERJECTIONS)
    + r")\s*[.!,]"
    + r")\s*",
    re.IGNORECASE,
)
_CLOSING_RE = re.compile(r"^\W*(?:" + "|".join(_FILLER_CLOSINGS) + r")\b", re.IGNORECASE)


# Nine labelled exchanges in the system prompt is a strong format prime, and a
# local 8B will sometimes answer in the shape it was shown — prefixing its reply
# with "Alice:", occasionally inventing a "Gabriel:" turn after it, and in the
# tool loop echoing the bracketed act line the exemplars use to show a tool call.
# The prompt bans all three in words, which is the kind of negative an 8B mostly
# honours, but "mostly" is visible when it slips. This is the belt to that
# braces: cheap, exact, and applied where the text leaves the model.
_SPEAKER_LABEL_RE = re.compile(r"^\s*(?:alice|assistant)\s*:\s*", re.IGNORECASE)
_HANDOFF_RE = re.compile(r"\n\s*(?:gabriel|user)\s*:.*\Z", re.IGNORECASE | re.DOTALL)
_ACT_LINE_RE = re.compile(
    r"^[ \t]*\((?:you |i )?(?:call|calls|called|check|run)[^\n]*\)[ \t]*$\n?", re.IGNORECASE | re.MULTILINE
)


def strip_speaker_label(text: str) -> str:
    """Remove transcript scaffolding the model copied out of its own examples.

    Never returns empty for non-empty input. An empty reply reads to the caller as
    "the model had nothing to say" and triggers a fallback, so a turn that is
    somehow *all* scaffolding is shown as it came rather than erased.
    """
    original = str(text or "").strip()
    if not original:
        return ""
    cleaned = _ACT_LINE_RE.sub("", original).strip()
    cleaned = _SPEAKER_LABEL_RE.sub("", cleaned, count=1).strip()
    # A hallucinated next turn from the user: everything from that label on is the
    # model writing his side of the conversation, so it is never part of the reply.
    cleaned = _HANDOFF_RE.sub("", cleaned).strip()
    return cleaned or original


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


def apply_response_discipline(text: str, max_sentences: int = DEFAULT_MAX_SENTENCES) -> str:
    """Remove padding and cap length at a sentence boundary, never mid sentence."""
    original = str(text or "").strip()
    if not original:
        return ""
    cleaned = strip_speaker_label(original)
    cleaned = strip_filler_opening(cleaned)
    cleaned = strip_filler_closing(cleaned)
    cleaned = limit_sentences(cleaned, max_sentences=max_sentences)
    return cleaned.strip() or original
