"""Lightweight episodic memory for turn-level recall and retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class Episode:
    timestamp: str
    user_input: str
    intent: str
    response: str
    entities: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "user_input": self.user_input,
            "intent": self.intent,
            "response": self.response,
            "entities": dict(self.entities),
        }


class EpisodicMemoryEngine:
    def __init__(self, max_episodes: int = 300) -> None:
        self.max_episodes = max(10, int(max_episodes or 300))
        self._episodes: List[Episode] = []

    def add_episode(
        self,
        *,
        user_input: str,
        intent: str,
        response: str,
        entities: Dict[str, Any] | None = None,
    ) -> None:
        self._episodes.append(
            Episode(
                timestamp=datetime.now().isoformat(),
                user_input=str(user_input or ""),
                intent=str(intent or "conversation:general"),
                response=str(response or ""),
                entities=dict(entities or {}),
            )
        )
        if len(self._episodes) > self.max_episodes:
            self._episodes = self._episodes[-self.max_episodes :]

    def recall_recent(self, limit: int = 3) -> List[Dict[str, Any]]:
        return [e.as_dict() for e in self._episodes[-max(1, int(limit or 3)) :]]

    def recall_similar(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        q_tokens = set(str(query or "").lower().split())
        if not q_tokens:
            return self.recall_recent(limit=limit)

        ranked: List[tuple[float, Episode]] = []
        for ep in self._episodes:
            text = f"{ep.user_input} {ep.intent}"
            tokens = set(text.lower().split())
            overlap = len(q_tokens & tokens)
            if overlap <= 0:
                continue
            score = overlap / max(1, len(q_tokens))
            ranked.append((score, ep))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [ep.as_dict() for _, ep in ranked[: max(1, int(limit or 3))]]
