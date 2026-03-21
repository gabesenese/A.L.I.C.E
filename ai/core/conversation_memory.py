"""Lightweight persistent conversation memory for short-horizon context carryover."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from collections import deque
from typing import Any, Deque, Dict, List, Optional


@dataclass
class ConversationTurn:
    user_input: str
    intent: str
    response: str = ""
    context_extracted: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class ConversationMemory:
    """Stores recent turns and offers compact contextual augmentation hints."""

    def __init__(self, max_turns: int = 20) -> None:
        self._turns: Deque[ConversationTurn] = deque(maxlen=max(5, int(max_turns or 20)))

    def add_turn(
        self,
        *,
        user_input: str,
        intent: str,
        response: str = "",
        context_extracted: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._turns.append(
            ConversationTurn(
                user_input=str(user_input or "").strip(),
                intent=str(intent or "conversation:general").strip(),
                response=str(response or "").strip(),
                context_extracted=dict(context_extracted or {}),
            )
        )

    def recent(self, limit: int = 5) -> List[ConversationTurn]:
        if limit <= 0:
            return []
        return list(self._turns)[-limit:]

    def latest_topic(self) -> str:
        for turn in reversed(self._turns):
            topic = str((turn.context_extracted or {}).get("topic") or "").strip()
            if topic:
                return topic
            words = [w for w in re.findall(r"[a-zA-Z0-9_]+", turn.user_input.lower()) if len(w) > 3]
            if words:
                return " ".join(words[:4])
        return ""

    def build_contextual_input(self, user_input: str) -> str:
        """Augment vague follow-ups with compact previous-topic context."""
        text = str(user_input or "").strip()
        if not text:
            return text

        lower = text.lower()
        pronounish = bool(re.search(r"\b(it|that|this|them|those|these|there)\b", lower))
        vague_short = len(lower.split()) <= 8 and (
            lower.startswith("can you help")
            or lower.startswith("help me")
            or "with that" in lower
            or "continue" in lower
        )
        if not (pronounish or vague_short):
            return text

        topic = self.latest_topic()
        if not topic:
            return text

        return f"Previous context topic: {topic}. Current user request: {text}"

    def snapshot(self) -> Dict[str, Any]:
        rows = []
        for t in self._turns:
            rows.append(
                {
                    "user_input": t.user_input,
                    "intent": t.intent,
                    "response": t.response,
                    "context_extracted": t.context_extracted,
                    "created_at": t.created_at,
                }
            )
        return {
            "turn_count": len(rows),
            "latest_topic": self.latest_topic(),
            "turns": rows,
        }
