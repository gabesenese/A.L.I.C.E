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
        if re.search(
            r"(?:\bno\s+api(?:s)?\b|\bwithout\s+api(?:s)?\b|don't\s+want\s+to\s+use\s+any\s+api(?:s)?|do\s+not\s+want\s+to\s+use\s+any\s+api(?:s)?)",
            lower,
        ):
            constraints.append("no_external_apis")
        if re.search(r"\blocal[-\s]?first\b|\boffline\b", lower):
            constraints.append("local_first")
        if "unique architecture" in lower or "make alice unique" in lower:
            constraints.append("unique_architecture")
        if "frameworks as references" in lower or "stepping stone" in lower:
            constraints.append("frameworks_as_reference_only")
        if "not a bunch of frameworks bundled together" in lower or "bundled framework" in lower:
            constraints.append("avoid_bundled_framework_feel")
        if "maximum value, minimal effort" in lower:
            constraints.append("maximum_value_minimal_effort")
        if "advanced engineering" in lower and "lean" in lower:
            constraints.append("advanced_engineering_lean_implementation")

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
