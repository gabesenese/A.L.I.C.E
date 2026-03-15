"""
Executive reflection loop.

Scores outcome quality per turn and suggests routing/confidence adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import re


@dataclass
class ReflectionResult:
    was_correct: bool
    was_relevant: bool
    success_score: float
    confidence_adjustment: float
    routing_adjustments: Dict[str, float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "was_correct": self.was_correct,
            "was_relevant": self.was_relevant,
            "success_score": self.success_score,
            "confidence_adjustment": self.confidence_adjustment,
            "routing_adjustments": dict(self.routing_adjustments),
        }


class ReflectionEngine:
    """Lightweight post-response reflection for adaptive control."""

    def reflect(
        self,
        *,
        user_input: str,
        intent: str,
        response: str,
        route: str,
        gate_accepted: bool,
        decision_scores: Dict[str, float],
        prior_confidence: float,
        quality_metrics: Dict[str, Any] | None = None,
        failure_type: str = "none",
    ) -> ReflectionResult:
        response = (response or "").strip()
        quality_metrics = quality_metrics or {}
        relevance = self._relevance_score(user_input, response)
        topic_adherence = float(
            quality_metrics.get("topic_adherence", relevance) or relevance
        )
        alignment = float(quality_metrics.get("alignment", 1.0) or 1.0)
        adherence = float(
            quality_metrics.get(
                "adherence",
                (0.55 * topic_adherence) + (0.45 * alignment),
            )
            or 0.0
        )
        is_relevant = relevance >= 0.20
        correct_like = gate_accepted and is_relevant and len(response) > 8

        success_score = (
            (0.45 if gate_accepted else 0.0)
            + (0.30 * relevance)
            + (0.15 * topic_adherence)
            + (0.10 * adherence)
        )
        success_score = max(0.0, min(1.0, success_score))

        confidence_adjustment = (
            0.08
            if success_score >= 0.70
            else (-0.10 if success_score < 0.45 else -0.02)
        )

        routing_adjustments: Dict[str, float] = {}
        route_key = route if route in ("llm", "tools", "search", "memory") else "llm"
        if success_score >= 0.70:
            routing_adjustments[route_key] = 0.03
            routing_adjustments["clarify"] = -0.01
        elif success_score < 0.45:
            routing_adjustments[route_key] = -0.05
            routing_adjustments["clarify"] = 0.03

        # Failure-type specific corrections target the subsystem most likely at fault.
        if failure_type == "routing_mistake":
            routing_adjustments["tools"] = routing_adjustments.get("tools", 0.0) + 0.03
            routing_adjustments["llm"] = routing_adjustments.get("llm", 0.0) - 0.03
        elif failure_type == "topic_drift":
            routing_adjustments["clarify"] = (
                routing_adjustments.get("clarify", 0.0) + 0.03
            )
        elif failure_type in ("weak_knowledge", "overgeneralization"):
            routing_adjustments["search"] = (
                routing_adjustments.get("search", 0.0) + 0.02
            )
            routing_adjustments["llm"] = routing_adjustments.get("llm", 0.0) - 0.02

        # If decision scores were very close, favor clarification slightly next turn.
        ranked = sorted((decision_scores or {}).values(), reverse=True)
        if len(ranked) >= 2 and (ranked[0] - ranked[1]) < 0.10:
            routing_adjustments["clarify"] = (
                routing_adjustments.get("clarify", 0.0) + 0.02
            )

        return ReflectionResult(
            was_correct=bool(correct_like),
            was_relevant=bool(is_relevant),
            success_score=float(success_score),
            confidence_adjustment=float(confidence_adjustment),
            routing_adjustments=routing_adjustments,
        )

    def _relevance_score(self, user_input: str, response: str) -> float:
        user_tokens = set(self._tokens(user_input))
        resp_tokens = set(self._tokens(response))
        if not user_tokens:
            return 0.0
        overlap = len(user_tokens.intersection(resp_tokens))
        return overlap / max(len(user_tokens), 1)

    def _tokens(self, text: str) -> List[str]:
        return re.findall(r"[a-z0-9']+", (text or "").lower())


_reflection_engine: ReflectionEngine | None = None


def get_reflection_engine() -> ReflectionEngine:
    global _reflection_engine
    if _reflection_engine is None:
        _reflection_engine = ReflectionEngine()
    return _reflection_engine
