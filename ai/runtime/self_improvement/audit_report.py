from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict
from uuid import uuid4

from ai.runtime.self_improvement.behavior_event import BehaviorEvent
from ai.runtime.self_improvement.evaluation_plan import EvaluationPlan
from ai.runtime.self_improvement.failure_classifier import FailureClassification
from ai.runtime.self_improvement.improvement_hypothesis import ImprovementHypothesis
from ai.runtime.self_improvement.patch_plan import PatchPlan


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AuditReport:
    report_id: str
    event: Dict[str, Any]
    classification: Dict[str, Any]
    hypothesis: Dict[str, Any]
    patch_plan: Dict[str, Any]
    evaluation_plan: Dict[str, Any]
    recommendation: str
    approval_required: bool
    risk_level: str
    generated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_audit_report(
    *,
    event: BehaviorEvent,
    classification: FailureClassification,
    hypothesis: ImprovementHypothesis,
    patch_plan: PatchPlan,
    evaluation_plan: EvaluationPlan,
) -> AuditReport:
    risk_level = "medium"
    if str(event.severity or "").lower() in {"high", "critical"}:
        risk_level = "high"
    recommendation = (
        f"Failure classified as {classification.failure_kind}. "
        f"Apply patch plan on {len(patch_plan.target_files)} target file(s), "
        "run evaluation plan, and require explicit approval before source edits."
    )
    return AuditReport(
        report_id=str(uuid4()),
        event=event.to_dict(),
        classification=classification.to_dict(),
        hypothesis=hypothesis.to_dict(),
        patch_plan=patch_plan.to_dict(),
        evaluation_plan=evaluation_plan.to_dict(),
        recommendation=recommendation,
        approval_required=bool(patch_plan.requires_approval),
        risk_level=risk_level,
        generated_at=_now_iso(),
    )
