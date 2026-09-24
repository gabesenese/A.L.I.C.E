"""Trims the padding that makes a reply read as machine written.

A retry rule used to force any answer under fifty words into a three or four
sentence "take", which trained the surface toward essays: a compliment, a restated
question, a paragraph of generic commentary, then an offer to go deeper. This
removes that scaffolding and leaves the substance.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

DEFAULT_MAX_SENTENCES = 6

_FILLER_OPENINGS = (
    r"your enthusiasm is palpable",
    r"i love your enthusiasm",
    r"that'?s? (?:a )?(?:great|excellent|good|interesting|fascinating) (?:question|point|idea)",
    r"what (?:a|an) (?:great|excellent|interesting|fascinating) (?:question|point|idea)",
    r"(?:great|excellent|good|interesting|fascinating) question",
    r"i'?m (?:really |so )?(?:glad|excited|happy) (?:you|to)",
    r"(?:but )?let'?s dive (?:deeper|right in|into)",
    r"let'?s (?:unpack|explore) (?:this|that)",
    r"i'?d be happy to",
    r"thanks for (?:sharing|asking)",
)

# Throat-clearing that leads straight into the point, so only the words go.
_LEAD_INS = (r"it'?s worth noting that",)

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
# go and do something ("would you like me to…") is still stripped, because Alice
# can simply go and do it.
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
    + r")\b(?P<tail>[^.!?]*)[.!?]"
    + r"|(?:"
    + "|".join(_FILLER_INTERJECTIONS)
    + r")\s*[.!,]"
    + r"|(?:"
    + "|".join(_LEAD_INS)
    + r")\b"
    + r")\s*",
    re.IGNORECASE,
)
# Where a courtesy ends and the sentence carrying the point begins.
_CLAUSE_BREAK_RE = re.compile(r"[,;:\u2013\u2014]|\s-\s")
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


# "As an AI language model, ..." is a generic chatbot's opening, not her voice. The
# clause must start a sentence and end at a comma or colon, or run straight into
# "I", so a sentence about language models is left alone: "As a language model
# grows, its loss falls" does not match, and neither does "such as an airline".
_AI_DISCLAIMER_RE = re.compile(
    r"(^|(?<=[.!?])[ \t]+)"
    r"as an? (?:ai|artificial intelligence|(?:large )?language model)\b"
    r"(?: (?:language )?model| assistant| chatbot)?"
    r"(?:[ \t]*[,:][ \t]*|[ \t]+(?=i\b))"
    r"(\w?)",
    re.IGNORECASE | re.MULTILINE,
)


def strip_ai_disclaimer(text: str) -> str:
    """Remove a self-disclaimer clause and keep the rest of the sentence.

    Only the clause goes. The words on their own are usually the subject of the
    question ("how does a language model work?"), not a disclaimer.
    """
    return _AI_DISCLAIMER_RE.sub(lambda m: m.group(1) + m.group(2).upper(), str(text or ""))


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


# Where a sentence can end: terminal punctuation, any closing quote or bracket, then
# whitespace. A full stop after a bare number or a common abbreviation is not an
# ending. Counting the "1." of a numbered list as a sentence is how a capped reply
# promised "three things" and then stopped at the numeral.
_SENTENCE_END_RE = re.compile(r"[.!?]+[\"')\]]*(?=\s)")
_NOT_AN_ENDING = {"e.g", "i.e", "vs", "cf", "mr", "mrs", "ms", "dr"}

# A list item or a code fence makes a reply structured rather than prose.
_LIST_LINE_RE = re.compile(r"^[ \t]*(?:[-*+•]|\d{1,3}[.)])[ \t]+\S", re.MULTILINE)


def _sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Offsets of each sentence, so callers cut the original text instead of
    rejoining pieces. Rejoining with spaces is what flattened every line break."""
    spans: List[Tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(text):
        if match.group(0) == ".":
            words = text[start : match.start()].split()
            last_word = words[-1].lstrip("(\"'[").lower() if words else ""
            if last_word.isdigit() or last_word in _NOT_AN_ENDING:
                continue
        if text[start : match.end()].strip():
            spans.append((start, match.end()))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _is_structured(text: str) -> bool:
    return "```" in text or bool(_LIST_LINE_RE.search(text))


def split_sentences(text: str) -> List[str]:
    content = str(text or "").strip()
    return [content[start:end].strip() for start, end in _sentence_spans(content)]


def capitalise_start(text: str) -> str:
    """Capitalise a reply that now starts mid-sentence, leaving identifiers alone."""
    first = text.split(" ", 1)[0]
    if first.isalpha() and first.islower():
        return text[:1].upper() + text[1:]
    return text


def _drop_opening(match: re.Match) -> str:
    # "That's a good point, but the cache still needs a lock." carries its point
    # after the courtesy, and dropping the whole sentence dropped the point: the
    # stripper turned "It's worth noting that the cache is per-process." into
    # nothing, and "Thanks for asking, it was the lock all along." into nothing.
    tail = match.group("tail") or ""
    brk = _CLAUSE_BREAK_RE.search(tail)
    if brk and re.search(r"\w", tail[brk.end() :]):
        return tail[brk.end() :].strip() + match.group(0).rstrip()[-1] + " "
    return ""


def strip_filler_opening(text: str) -> str:
    """Drop a leading compliment or throat-clearing, but never the point, and never everything."""
    cleaned = str(text or "").strip()
    for _ in range(2):
        candidate = capitalise_start(_OPENING_RE.sub(_drop_opening, cleaned, count=1).strip())
        if not candidate or candidate == cleaned:
            break
        cleaned = candidate
    return cleaned or str(text or "").strip()


def strip_filler_closing(text: str) -> str:
    content = str(text or "").strip()
    spans = _sentence_spans(content)
    while len(spans) > 1 and _CLOSING_RE.match(content[spans[-1][0] : spans[-1][1]].strip()):
        spans.pop()
    return content[: spans[-1][1]].rstrip() if spans else content


def limit_sentences(text: str, max_sentences: int = DEFAULT_MAX_SENTENCES) -> str:
    """Cap prose at a sentence boundary, keeping the reply's own line breaks.

    The cap is for rambling prose. A reply with a list or a code block is
    structured, and there is no place to cut it that keeps what it promised, so it
    is left whole.
    """
    content = str(text or "").strip()
    if _is_structured(content):
        return content
    spans = _sentence_spans(content)
    if len(spans) <= max_sentences:
        return content
    return content[: spans[max(1, max_sentences) - 1][1]].rstrip()


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


# A request for depth. The cap is for rambling, and an explanation someone asked
# for is not rambling: at five sentences "walk me through the memory layer" lost
# its last steps, which is usually where the point is.
_DEPTH_REQUEST_RE = re.compile(
    r"\b(?:explain|elaborate|in (?:more )?detail|in depth|walk me through|step by step|tell me more"
    r"|go deeper|break (?:it|this|that) down|the long version|how (?:does|do) .+ work)\b",
    re.IGNORECASE,
)


def asks_for_depth(user_input: str) -> bool:
    """Whether the user asked for an explanation rather than an answer."""
    return bool(_DEPTH_REQUEST_RE.search(str(user_input or "")))


def apply_response_discipline(text: str, max_sentences: Optional[int] = DEFAULT_MAX_SENTENCES) -> str:
    """Remove padding and cap length at a sentence boundary, never mid sentence.

    ``max_sentences=None`` removes the padding and leaves the length alone.
    """
    original = str(text or "").strip()
    if not original:
        return ""
    cleaned = strip_speaker_label(original)
    cleaned = strip_filler_opening(cleaned)
    cleaned = strip_filler_closing(cleaned)
    if max_sentences is not None:
        cleaned = limit_sentences(cleaned, max_sentences=max_sentences)
    return cleaned.strip() or original
