"""Conversation style mirror: auto-calibrate response verbosity by mirroring user style.

Tracks the last N user message word counts and question density. When user
messages are consistently short, automatically applies a brief constraint so
Alice doesn't flood a terse user with long prose.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Dict, Optional


_QUESTION_RE = re.compile(r"\?")
_SENTENCE_RE = re.compile(r"[.!?]+")


class ConversationStyleMirror:
    """Per-session message-length tracker with auto-constraint derivation."""

    _WINDOW = 5          # number of recent messages to consider
    _BRIEF_THRESHOLD = 15   # avg user words below this → apply brief constraint
    _TERSE_THRESHOLD = 8    # avg user words below this → apply very brief constraint
    _MULTI_Q_THRESHOLD = 2  # questions per message above this → prefer bullets

    def __init__(self) -> None:
        # user_id → deque of (word_count, question_count)
        self._history: Dict[str, deque] = {}

    def record_user_message(self, user_id: str, text: str) -> None:
        uid = str(user_id or "default")
        words = len(str(text or "").split())
        questions = len(_QUESTION_RE.findall(str(text or "")))
        q = self._history.setdefault(uid, deque(maxlen=self._WINDOW))
        q.append((words, questions))

    def derive_auto_constraints(self, user_id: str) -> Dict[str, Any]:
        """Return a constraints dict for AdaptiveResponseStyle based on user history.

        Returns empty dict when there isn't enough data to act on.
        """
        uid = str(user_id or "default")
        history = list(self._history.get(uid, []))
        if len(history) < 2:
            return {}

        avg_words = sum(w for w, _ in history) / len(history)
        avg_questions = sum(q for _, q in history) / len(history)

        constraints: Dict[str, Any] = {}

        if avg_words <= self._TERSE_THRESHOLD:
            constraints["max_words"] = 40
            constraints["response_length"] = "brief"
        elif avg_words <= self._BRIEF_THRESHOLD:
            constraints["max_words"] = 80
            constraints["response_length"] = "brief"

        if avg_questions >= self._MULTI_Q_THRESHOLD:
            constraints["format"] = "bullet_points"

        return constraints

    def stats(self, user_id: str) -> Dict[str, float]:
        uid = str(user_id or "default")
        history = list(self._history.get(uid, []))
        if not history:
            return {}
        avg_words = sum(w for w, _ in history) / len(history)
        avg_q = sum(q for _, q in history) / len(history)
        return {"avg_words": round(avg_words, 1), "avg_questions": round(avg_q, 2), "samples": len(history)}


_mirror: ConversationStyleMirror | None = None


def get_style_mirror() -> ConversationStyleMirror:
    global _mirror
    if _mirror is None:
        _mirror = ConversationStyleMirror()
    return _mirror
