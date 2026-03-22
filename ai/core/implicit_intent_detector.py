"""Implicit intent detector for underspecified user phrasing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class ImplicitIntentMatch:
    intent: str
    confidence: float
    reason: str
    evidence: str

    def as_dict(self) -> Dict[str, str | float]:
        return {
            "intent": self.intent,
            "confidence": float(self.confidence),
            "reason": self.reason,
            "evidence": self.evidence,
        }


class ImplicitIntentDetector:
    """Infer likely intents when explicit action phrases are missing."""

    def detect(self, text: str, *, recent_topic: str = "") -> List[ImplicitIntentMatch]:
        lower = (text or "").strip().lower()
        if not lower:
            return []

        matches: List[ImplicitIntentMatch] = []

        if re.search(
            r"\b(code|script|app|program).{0,24}\b(crash|crashing|error|exception|traceback|line\s*\d+)\b",
            lower,
        ):
            matches.append(
                ImplicitIntentMatch(
                    intent="conversation:question",
                    confidence=0.78,
                    reason="implicit_debugging_request",
                    evidence="code failure language",
                )
            )

        if re.search(
            r"\b(?:been|it's been)\s+\d+\s+(?:hour|hours)\b.*\b(since i ate|since eating|without eating)\b",
            lower,
        ):
            matches.append(
                ImplicitIntentMatch(
                    intent="reminder:set",
                    confidence=0.80,
                    reason="implicit_wellbeing_reminder",
                    evidence="elapsed-time since meal pattern",
                )
            )

        if re.search(r"\bnot sure\b.*\b(investing|stocks?|portfolio|market)\b", lower):
            matches.append(
                ImplicitIntentMatch(
                    intent="conversation:question",
                    confidence=0.73,
                    reason="implicit_analysis_request",
                    evidence="uncertainty + finance domain",
                )
            )

        if re.search(r"\bmaybe\b.*\b(weather|forecast|rain|temperature)\b", lower):
            matches.append(
                ImplicitIntentMatch(
                    intent="weather:current",
                    confidence=0.82,
                    reason="soft_weather_request",
                    evidence="hedged request with weather terms",
                )
            )

        if (
            recent_topic
            and len(lower.split()) <= 8
            and re.search(r"\b(can you help|continue|with that|do that)\b", lower)
        ):
            matches.append(
                ImplicitIntentMatch(
                    intent="conversation:question",
                    confidence=0.70,
                    reason="topic_continuation",
                    evidence=f"short follow-up referencing prior topic: {recent_topic[:40]}",
                )
            )

        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches

    def best(
        self, text: str, *, recent_topic: str = ""
    ) -> Optional[ImplicitIntentMatch]:
        matches = self.detect(text, recent_topic=recent_topic)
        return matches[0] if matches else None
