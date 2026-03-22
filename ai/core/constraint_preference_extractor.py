"""Extract response constraints and output-style preferences from user text."""

from __future__ import annotations

import re
from typing import Any, Dict


class ConstraintPreferenceExtractor:
    def extract(self, text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
        lower = raw.lower()

        format_pref = "default"
        if any(k in lower for k in ("table", "tabular")):
            format_pref = "table"
        elif any(k in lower for k in ("bullet", "bullets", "bullet points", "list")):
            format_pref = "bullet_points"
        elif any(k in lower for k in ("narrative", "paragraph", "story")):
            format_pref = "narrative"

        detail = "normal"
        if any(
            k in lower
            for k in ("quick", "quickly", "short", "brief", "tldr", "summary")
        ):
            detail = "concise"
        if any(
            k in lower
            for k in ("detailed", "deep", "in-depth", "thorough", "step-by-step")
        ):
            detail = "detailed"

        constraints = []
        if "no code" in lower:
            constraints.append("no_code")
        if "with code" in lower:
            constraints.append("include_code")
        if "examples" in lower or "example" in lower:
            constraints.append("include_examples")

        max_words = None
        m = re.search(
            r"\b(?:under|within|max(?:imum)?|at most)\s+(\d{1,4})\s+words\b", lower
        )
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
