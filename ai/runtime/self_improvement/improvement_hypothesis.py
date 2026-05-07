from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

from ai.runtime.self_improvement.behavior_event import BehaviorEvent
from ai.runtime.self_improvement.failure_classifier import FailureClassification


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ImprovementHypothesis:
    hypothesis_id: str
    event_id: str
    failure_kind: str
    hypothesis: str
    rationale: str
    target_layer: str
    likely_files: List[str] = field(default_factory=list)
    expected_behavior: str = ""
    risks: List[str] = field(default_factory=list)
    confidence: float = 0.0
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_hypothesis(
    event: BehaviorEvent,
    classification: FailureClassification,
    project_state: Dict[str, Any] | None = None,
) -> ImprovementHypothesis:
    low = " ".join(
        [event.symptom, event.actual_behavior, event.expected_behavior]
    ).lower()
    state = dict(project_state or {})
    objective = str(state.get("active_objective") or "").strip()
    expected_behavior = str(event.expected_behavior or "").strip()
    if not expected_behavior:
        expected_behavior = "Behavior should be grounded, safe, and objective-driven."

    hypothesis_text = (
        "Runtime behavior likely diverged due to policy/routing mismatch and needs "
        "targeted bounded updates."
    )
    risks = ["regression in adjacent runtime behaviors"]

    if classification.failure_kind == "greeting_tone":
        hypothesis_text = (
            "Greeting surface policy likely overcorrected toward safety-short responses; "
            "it should stay continuity-safe while restoring warm companion presence."
        )
        risks = ["reintroducing assistant-like prompts", "continuity guard regressions"]
    elif classification.failure_kind == "continuity_claim":
        hypothesis_text = (
            "Continuity filtering is too permissive for unsupported prior-topic claims "
            "and should enforce stricter grounded-claim validation."
        )
        risks = ["over-filtering valid same-session references"]
    elif classification.failure_kind == "routing":
        hypothesis_text = (
            "Routing arbitration/evidence scoring likely underweights objective-grounded "
            "safe routes and overuses clarification or weak intents."
        )
        risks = ["route selection churn", "intent mapping regressions"]
    elif classification.failure_kind == "memory":
        hypothesis_text = (
            "Memory selection/guarding likely blends stale recall with active state; "
            "project-state-first retrieval should be enforced."
        )
        risks = ["loss of useful recall context"]
    elif classification.failure_kind == "local_execution":
        hypothesis_text = (
            "Local execution target resolution and fallback handling are likely missing "
            "safe target disambiguation."
        )
        risks = ["more clarification loops", "tool fallback regressions"]
    elif classification.failure_kind == "response_grounding":
        hypothesis_text = (
            "Response momentum/verification layers likely allow unsupported execution claims; "
            "grounding checks should gate final text."
        )
        risks = ["overly terse responses"]
    elif classification.failure_kind == "passive_response":
        hypothesis_text = (
            "Response shaping likely defaults to passive assistant phrasing and should "
            "favor objective momentum summaries."
        )
        risks = ["tone instability"]
    elif classification.failure_kind == "runtime_error":
        hypothesis_text = (
            "Runtime lifecycle path likely lacks safe optional-component guards causing "
            "startup/shutdown exceptions."
        )
        risks = ["partial subsystem startup states"]

    if "machine learning last time" in low:
        hypothesis_text = (
            "Unsupported continuity phrasing indicates stale memory leakage; continuity guard "
            "must reject ungrounded prior-topic claims."
        )

    rationale = str(classification.reason or "classified from runtime evidence")
    if objective:
        rationale = f"{rationale}; active objective={objective}"

    return ImprovementHypothesis(
        hypothesis_id=str(uuid4()),
        event_id=str(event.event_id),
        failure_kind=str(classification.failure_kind or "unknown"),
        hypothesis=hypothesis_text,
        rationale=rationale,
        target_layer=str(classification.likely_layer or "runtime"),
        likely_files=list(classification.likely_files or []),
        expected_behavior=expected_behavior,
        risks=risks,
        confidence=float(classification.confidence or 0.5),
    )

