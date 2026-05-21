"""Post-turn opinion extraction — zero LLM calls.

Scans ALICE's response text for opinionated language using compiled regex
patterns. Extracts (topic, stance) pairs and upserts them into IdentityStore.
Runs only on successful turns and relevant intents.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# Intents that don't produce useful opinions — skip extraction
_SKIP_INTENTS = {
    "weather", "greeting", "notes", "calendar", "email",
    "search", "maps", "document", "unknown",
}

_OPINION_PATTERNS = [
    re.compile(
        r"\b(this|that|it)\s+(is|feels|seems)\s+(a\s+)?"
        r"(bad|good|wrong|right|fragile|solid|clean|messy|risky|correct|incorrect)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(i'?d?\s+(?:avoid|prefer|recommend|not\s+recommend|skip|go\s+with))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(the\s+(?:problem|issue|gap|risk|downside|trouble)\s+(?:is|here\s+is|with\s+this\s+is))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(\w[\w\s]{2,25})\s+is\s+(better|worse|cleaner|more\s+fragile|more\s+robust)\s+than\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(this\s+approach|this\s+pattern|this\s+design|this\s+structure)\s+"
        r"(works|doesn'?t\s+work|is\s+|has\s+a\s+)\b",
        re.IGNORECASE,
    ),
]

# Compiled sentence splitter
_SENTENCE_SEP = re.compile(r"[.\n]+")


def _split_sentences(text: str) -> List[str]:
    raw = _SENTENCE_SEP.split(text)
    return [s.strip() for s in raw if s.strip()]


def _is_question(sentence: str) -> bool:
    return sentence.strip().endswith("?")


def _intent_prefix(intent: str) -> str:
    return str(intent or "").split(":")[0].strip()[:20]


def extract_opinions(response_text: str, intent: str) -> List[Tuple[str, str]]:
    """Return up to 2 (topic, stance) pairs extracted from response_text."""
    if len(response_text.strip()) < 40:
        return []

    prefix = _intent_prefix(intent)
    results: List[Tuple[str, str]] = []

    for sentence in _split_sentences(response_text):
        if len(sentence) < 15 or _is_question(sentence):
            continue
        for pattern in _OPINION_PATTERNS:
            m = pattern.search(sentence)
            if m:
                # Topic: intent prefix + first significant phrase of sentence
                words = sentence.split()
                phrase = " ".join(words[:6]).rstrip(",:;")
                topic = f"{prefix}: {phrase}"[:60] if prefix else phrase[:60]
                stance = sentence[:160]
                results.append((topic.lower().strip(), stance))
                break  # one pattern match per sentence
        if len(results) >= 2:
            break

    return results


def extract_and_record_opinions(
    alice_response: str,
    intent: str,
    success: bool,
) -> None:
    """Extract opinions from a successful ALICE response and persist them."""
    if not success:
        return

    intent_root = str(intent or "").split(":")[0].lower()
    if intent_root in _SKIP_INTENTS:
        return

    pairs = extract_opinions(alice_response, intent)
    if not pairs:
        return

    try:
        from ai.identity.alice_identity import record_opinion
        for topic, stance in pairs:
            record_opinion(topic, stance, source="auto")
    except Exception:
        pass
