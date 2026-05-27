"""Prompt complexity heuristics for multi-LLM routing."""

from __future__ import annotations

import re

_PLANNING_RE = re.compile(r"\b(plan|roadmap|strategy|design|architecture|step[- ]by[- ]step)\b", re.IGNORECASE)
_REASONING_RE = re.compile(r"\b(why|tradeoff|compare|analyze|root cause|diagnose|evaluate)\b", re.IGNORECASE)
_MULTI_RE = re.compile(r"\b(and|then|also|after that|plus)\b", re.IGNORECASE)


def score_prompt(prompt: str) -> int:
    """Return a bounded complexity score in [0, 10]."""
    text = str(prompt or "").strip()
    if not text:
        return 0

    score = 0
    length = len(text)
    if length > 120:
        score += 2
    if length > 260:
        score += 2
    if length > 500:
        score += 1

    if _PLANNING_RE.search(text):
        score += 3
    if _REASONING_RE.search(text):
        score += 2

    multi_hits = len(_MULTI_RE.findall(text))
    if multi_hits >= 2:
        score += 2
    elif multi_hits == 1:
        score += 1

    # More question marks and clauses usually imply more reasoning work.
    score += min(2, text.count("?") // 2)

    return max(0, min(10, score))
