"""Adaptive response style control (verbosity + format constraints)."""

from __future__ import annotations

import re
from typing import Any, Dict


class AdaptiveResponseStyle:
    def derive_style(self, *, intent: str, sentiment: Dict[str, Any] | None, preferences: Dict[str, Any] | None) -> Dict[str, Any]:
        prefs = dict(preferences or {})
        style = {
            "verbosity": "normal",
            "format": prefs.get("format", "paragraph"),
            "detail": prefs.get("detail", "balanced"),
        }
        if prefs.get("detail") == "concise":
            style["verbosity"] = "brief"
        elif prefs.get("detail") == "detailed":
            style["verbosity"] = "expanded"

        mood = str((sentiment or {}).get("category") or "").lower()
        if mood in {"negative", "frustrated", "angry"}:
            style["verbosity"] = "brief"

        if str(intent or "").startswith("technical:") and style["verbosity"] == "normal":
            style["verbosity"] = "expanded"
        return style

    def apply_constraints(self, response: str, preferences: Dict[str, Any] | None) -> str:
        prefs = dict(preferences or {})
        out = str(response or "")
        max_words = int(prefs.get("max_words", 0) or 0)

        if prefs.get("format") == "bullet_points" and "\n- " not in out and out:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", out) if s.strip()]
            if len(sentences) >= 2:
                out = "\n".join(f"- {s}" for s in sentences[:6])

        if max_words > 0:
            words = out.split()
            if len(words) > max_words:
                out = " ".join(words[:max_words]).rstrip(" .") + "..."

        return out
