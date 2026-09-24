"""Adaptive response style control (verbosity + format constraints)."""

from __future__ import annotations

import re
from typing import Any, Dict, List


# Filler phrases that add length without content — removed during intelligent shortening
_FILLER_PATTERNS = [
    # Only as an opener: "make sure" lost its "sure", and "Absolutely." answering
    # a yes-or-no question was the whole answer.
    re.compile(r"^(?:Of course|Certainly|Absolutely|Sure)[,!]\s*", re.IGNORECASE),
    re.compile(r"\bGreat question[!.]?\s*", re.IGNORECASE),
    re.compile(r"\bI(?:'d| would) be happy to help[.!]?\s*", re.IGNORECASE),
    re.compile(r"\bI'm glad you asked[.!]?\s*", re.IGNORECASE),
    re.compile(r"\bAs an AI[^.]*?\.\s*", re.IGNORECASE),
    re.compile(r"\bIt'?s worth (?:noting|mentioning) that\s*", re.IGNORECASE),
    re.compile(r"\bIt is important to note that\s*", re.IGNORECASE),
    re.compile(r"\bIn summary[,:]?\s*", re.IGNORECASE),
    re.compile(r"\bTo summarize[,:]?\s*", re.IGNORECASE),
    re.compile(r"\bIn conclusion[,:]?\s*", re.IGNORECASE),
    re.compile(r"\bTo conclude[,:]?\s*", re.IGNORECASE),
    re.compile(r"\bHope (?:this|that) helps[.!]?\s*$", re.IGNORECASE),
    re.compile(r"\bLet me know if you (?:have|need) (?:any|more)[^.]*[.!]?\s*$", re.IGNORECASE),
    re.compile(r"\bDo you (?:have|need) any (?:other|more) questions[?!]?\s*$", re.IGNORECASE),
]


def _strip_fillers(text: str) -> str:
    out = text
    for pat in _FILLER_PATTERNS:
        out = pat.sub("", out)
    out = out.strip()
    if out != text.strip() and out[:1].islower():
        out = out[0].upper() + out[1:]
    return out


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _intelligent_shorten(text: str, max_words: int) -> str:
    """Shorten `text` to `max_words` without hard truncation.

    Strategy:
    1. Strip filler phrases first (often saves 10-20 words at no content cost).
    2. If still over, drop trailing sentences from the least-important end.
    3. If a single remaining sentence is still over, truncate at a clause boundary.
    4. Last resort: hard word-count truncation with ellipsis.
    """
    # Step 1: strip fillers
    out = _strip_fillers(text)
    if len(out.split()) <= max_words:
        return out

    # Step 2: drop trailing sentences
    sentences = _split_sentences(out)
    while sentences and len(" ".join(sentences).split()) > max_words:
        sentences.pop()
    if sentences:
        candidate = " ".join(sentences)
        if len(candidate.split()) <= max_words:
            return candidate

    # Step 3: truncate at clause boundary (comma or semicolon)
    if sentences:
        first = sentences[0] if sentences else out
        clause_cut = re.split(r"[,;]\s*", first)
        rebuilt = ""
        for clause in clause_cut:
            trial = (rebuilt + ", " + clause).strip(", ") if rebuilt else clause
            if len(trial.split()) <= max_words:
                rebuilt = trial
            else:
                break
        if rebuilt and len(rebuilt.split()) <= max_words:
            return rebuilt + ("." if not rebuilt.endswith((".", "!", "?")) else "")

    # Step 4: hard truncation
    words = out.split()
    return " ".join(words[:max_words]).rstrip(" .,;") + "..."


class AdaptiveResponseStyle:
    def derive_style(
        self,
        *,
        intent: str,
        sentiment: Dict[str, Any] | None,
        preferences: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
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

        # Strip filler whenever response_length prefers brief or a max_words is set
        response_length = str(prefs.get("response_length") or "").lower()
        max_words = int(prefs.get("max_words", 0) or 0)
        if response_length == "brief" or max_words > 0:
            out = _strip_fillers(out)

        if prefs.get("format") == "bullet_points" and "\n- " not in out and out:
            sentences = _split_sentences(out)
            # Every sentence is kept: this used to stop at six and drop the rest.
            # Two sentences read better as they are than as two bullets.
            if len(sentences) >= 3:
                out = "\n".join(f"- {s}" for s in sentences)

        if max_words > 0 and len(out.split()) > max_words:
            out = _intelligent_shorten(out, max_words)

        return out
