"""Clarification feedback loop: boosts routing confidence when a clarification is resolved.

When Alice asks a clarifying question and the user answers, we store the
(original_input, question, answer) triple and compute a per-session confidence
boost for the inferred intent.  The boost decays over 10 turns and feeds into
ConfidenceFusion._behavioral_prior().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ClarificationRecord:
    original_input: str
    question: str
    answer: str
    intent: str
    turn_recorded: int
    boost: float = 0.10  # confidence boost applied after resolution


class ClarificationFeedbackLoop:
    """Session-scoped store of clarification Q&A with confidence boost decay."""

    _DECAY_TURNS = 10   # boost halves every N turns
    _MAX_BOOST = 0.15
    _MIN_BOOST = 0.02

    def __init__(self) -> None:
        # user_id → list of records
        self._records: Dict[str, List[ClarificationRecord]] = {}
        self._current_turn: Dict[str, int] = {}

    def tick(self, user_id: str) -> None:
        """Advance turn counter for a user (call once per turn)."""
        uid = str(user_id or "default")
        self._current_turn[uid] = self._current_turn.get(uid, 0) + 1

    def record_clarification(
        self,
        *,
        user_id: str,
        original_input: str,
        question: str,
        answer: str,
        intent: str,
    ) -> None:
        uid = str(user_id or "default")
        turn = self._current_turn.get(uid, 0)
        rec = ClarificationRecord(
            original_input=str(original_input or "")[:200],
            question=str(question or "")[:200],
            answer=str(answer or "")[:200],
            intent=str(intent or ""),
            turn_recorded=turn,
        )
        self._records.setdefault(uid, []).append(rec)
        # Keep last 20 records per user
        self._records[uid] = self._records[uid][-20:]

    def get_confidence_boost(self, user_id: str, intent: str) -> float:
        """Return the accumulated decayed boost for this intent, or 0.0."""
        uid = str(user_id or "default")
        current = self._current_turn.get(uid, 0)
        intent_prefix = str(intent or "").split(":")[0].lower()
        total = 0.0
        for rec in self._records.get(uid, []):
            rec_prefix = str(rec.intent or "").split(":")[0].lower()
            if rec_prefix != intent_prefix:
                continue
            age = max(0, current - rec.turn_recorded)
            # Exponential decay: halves every _DECAY_TURNS turns
            decay = 0.5 ** (age / max(1, self._DECAY_TURNS))
            total += rec.boost * decay
        return round(min(self._MAX_BOOST, max(0.0, total)), 4)

    def clear(self, user_id: str) -> None:
        uid = str(user_id or "default")
        self._records.pop(uid, None)
        self._current_turn.pop(uid, None)


_loop: ClarificationFeedbackLoop | None = None


def get_clarification_feedback_loop() -> ClarificationFeedbackLoop:
    global _loop
    if _loop is None:
        _loop = ClarificationFeedbackLoop()
    return _loop
