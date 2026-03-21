"""Multi-step reasoning planner for compound and high-complexity turns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ReasoningStep:
    step_id: str
    intent: str
    text: str
    confidence: float
    status: str = "planned"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "intent": self.intent,
            "text": self.text,
            "confidence": round(float(self.confidence), 3),
            "status": self.status,
        }


@dataclass
class ReasoningPlan:
    plan_id: str
    complexity: str
    steps: List[ReasoningStep] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "complexity": self.complexity,
            "steps": [s.as_dict() for s in self.steps],
            "step_count": len(self.steps),
        }


class MultiStepReasoningEngine:
    """Builds and tracks a minimal turn-level reasoning plan."""

    def __init__(self) -> None:
        self._counter = 0

    def plan_turn(
        self,
        *,
        user_input: str,
        primary_intent: str,
        primary_confidence: float,
        secondary_intents: List[Dict[str, Any]] | None = None,
    ) -> ReasoningPlan:
        self._counter += 1
        secondary_intents = secondary_intents or []
        complexity = "multi_step" if secondary_intents else "single_step"
        steps: List[ReasoningStep] = [
            ReasoningStep(
                step_id=f"{self._counter}-1",
                intent=str(primary_intent or "conversation:general"),
                text=str(user_input or "").strip()[:160],
                confidence=float(primary_confidence or 0.0),
                status="planned",
            )
        ]

        for idx, item in enumerate(secondary_intents, start=2):
            steps.append(
                ReasoningStep(
                    step_id=f"{self._counter}-{idx}",
                    intent=str(item.get("intent") or "conversation:general"),
                    text=str(item.get("text") or "").strip()[:160],
                    confidence=float(item.get("confidence", 0.0) or 0.0),
                    status="planned",
                )
            )

        return ReasoningPlan(
            plan_id=f"turn-{self._counter}",
            complexity=complexity,
            steps=steps,
        )

    def summarize_outcomes(self, outcomes: List[Dict[str, Any]]) -> str:
        if not outcomes:
            return ""
        ok = sum(1 for item in outcomes if item.get("success"))
        total = len(outcomes)
        labels = ", ".join(str(item.get("intent", "unknown")) for item in outcomes[:3])
        return f"Follow-up actions: {ok}/{total} succeeded ({labels})."
