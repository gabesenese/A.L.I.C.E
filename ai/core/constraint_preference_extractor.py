"""Extract response constraints and output-style preferences from user text."""

from __future__ import annotations

import re
from typing import Any, Dict


# Asked for as a shape, not mentioned as a thing: "as a list" or "in bullet
# points", never "my shopping list", "the playlist" or "is the build stable?".
# Matched as substrings, those turned "Cleared your shopping list. It had eggs
# and bread." into two bullet points.
_TABLE_RE = re.compile(r"\b(?:tabular|(?:as|in(?:to)?)\s+a\s+table|table\s+form(?:at)?)\b")
_BULLETS_RE = re.compile(
    r"\bbullet(?:s|ed)?\b|\bbullet\s+points?\b|\b(?:as|in(?:to)?)\s+(?:a\s+)?(?:bulleted\s+|numbered\s+)?list\b"
    r"|\b(?:list|point)\s+form\b|\blist\s+(?:them|those|these|it)(?:\s+out)?\b"
)
_NARRATIVE_RE = re.compile(r"\b(?:as|in)\s+(?:a\s+)?(?:narrative|paragraphs?|prose|story)\b")


class ConstraintPreferenceExtractor:
    def extract(self, text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
        lower = raw.lower()

        format_pref = "default"
        if _TABLE_RE.search(lower):
            format_pref = "table"
        elif _BULLETS_RE.search(lower):
            format_pref = "bullet_points"
        elif _NARRATIVE_RE.search(lower):
            format_pref = "narrative"

        detail = "normal"
        if any(k in lower for k in ("quick", "quickly", "short", "brief", "tldr", "summary")):
            detail = "concise"
        if any(k in lower for k in ("detailed", "deep", "in-depth", "thorough", "step-by-step")):
            detail = "detailed"

        constraints = []
        if "no code" in lower:
            constraints.append("no_code")
        if "with code" in lower:
            constraints.append("include_code")
        if "examples" in lower or "example" in lower:
            constraints.append("include_examples")

        max_words = None
        m = re.search(r"\b(?:under|within|max(?:imum)?|at most)\s+(\d{1,4})\s+words\b", lower)
        if m:
            try:
                max_words = int(m.group(1))
            except Exception:
                max_words = None

        return {
            "format": format_pref,
            "detail": detail,
            "constraints": constraints,
            "max_words": max_words,
        }
