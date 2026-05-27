"""Conversation style mirror: auto-calibrate response verbosity by mirroring user style.

Tracks the last N user message word counts and question density. When user
messages are consistently short, automatically applies a brief constraint so
Alice doesn't flood a terse user with long prose.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Dict


_QUESTION_RE = re.compile(r"\?")
_SENTENCE_RE = re.compile(r"[.!?]+")


class ConversationStyleMirror:
    """Per-session message-length tracker with auto-constraint derivation."""

    _WINDOW = 5  # number of recent messages to consider
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

        Only derives format hints (e.g. bullet points for multi-question messages).
        Word-count mirroring is intentionally removed — response length should come
        from the content, not from matching the user's message brevity.
        """
        uid = str(user_id or "default")
        history = list(self._history.get(uid, []))
        if len(history) < 2:
            return {}

        avg_questions = sum(q for _, q in history) / len(history)

        constraints: Dict[str, Any] = {}

        if avg_questions >= self._MULTI_Q_THRESHOLD:
            constraints["format"] = "bullet_points"

        return constraints

    def stats(self, user_id: str) -> Dict[str, float]:
        uid = str(user_id or "default")
        history = list(self._history.get(uid, []))
        if not history:
            return {}
        avg_q = sum(q for _, q in history) / len(history)
        return {
            "avg_questions": round(avg_q, 2),
            "samples": len(history),
        }


_mirror: ConversationStyleMirror | None = None


def get_style_mirror() -> ConversationStyleMirror:
    global _mirror
    if _mirror is None:
        _mirror = ConversationStyleMirror()
    return _mirror
