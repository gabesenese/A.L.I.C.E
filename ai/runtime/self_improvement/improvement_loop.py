from __future__ import annotations

import os
from typing import Any, Dict

from ai.memory.project_memory import record_improvement_audit, update_project_state
from ai.runtime.self_improvement.audit_report import AuditReport, build_audit_report
from ai.runtime.self_improvement.behavior_event import (
    BehaviorEvent,
    create_behavior_event,
)
from ai.runtime.self_improvement.codex_brief import build_codex_brief
from ai.runtime.self_improvement.evaluation_plan import (
    EvaluationPlan,
    build_evaluation_plan,
)
from ai.runtime.self_improvement.failure_classifier import (
    FailureClassification,
    classify_failure,
)
from ai.runtime.self_improvement.improvement_hypothesis import (
    ImprovementHypothesis,
    build_hypothesis,
)
from ai.runtime.self_improvement.patch_plan import PatchPlan, build_patch_plan
from ai.runtime.self_improvement.storage import (
    read_audit_reports,
    read_behavior_events,
    record_audit_report,
    record_behavior_event,
    record_evaluation_plan,
    record_hypothesis,
    record_patch_plan,
)


class ImprovementLoop:
    def __init__(self, user_id: str = "default") -> None:
        self.user_id = str(user_id or "default")

    def observe_event(self, **kwargs: Any) -> BehaviorEvent:
        event = create_behavior_event(user_id=self.user_id, **kwargs)
        record_behavior_event(event)
        return event

    def classify(self, event: BehaviorEvent) -> FailureClassification:
        return classify_failure(event)

    def hypothesize(
        self, event: BehaviorEvent, classification: FailureClassification
    ) -> ImprovementHypothesis:
        return build_hypothesis(event, classification)

    def plan_patch(self, hypothesis: ImprovementHypothesis) -> PatchPlan:
        return build_patch_plan(hypothesis)

    def plan_evaluation(self, patch_plan: PatchPlan) -> EvaluationPlan:
        return build_evaluation_plan(patch_plan)

    def build_audit_report(
        self,
        *,
        event: BehaviorEvent,
        classification: FailureClassification,
        hypothesis: ImprovementHypothesis,
        patch_plan: PatchPlan,
        evaluation_plan: EvaluationPlan,
    ) -> AuditReport:
        return build_audit_report(
            event=event,
            classification=classification,
            hypothesis=hypothesis,
            patch_plan=patch_plan,
            evaluation_plan=evaluation_plan,
        )

    def run_audit_from_event(self, event: BehaviorEvent) -> AuditReport:
        self._ensure_behavior_event_recorded(event)
        classification = self.classify(event)
        hypothesis = self.hypothesize(event, classification)
        patch_plan = self.plan_patch(hypothesis)
        evaluation_plan = self.plan_evaluation(patch_plan)
        report = self.build_audit_report(
            event=event,
            classification=classification,
            hypothesis=hypothesis,
            patch_plan=patch_plan,
            evaluation_plan=evaluation_plan,
        )

        record_hypothesis(hypothesis.to_dict())
        record_patch_plan(patch_plan.to_dict())
        record_evaluation_plan(evaluation_plan.to_dict())
        record_audit_report(report.to_dict())

        update_project_state(
            {
                "last_failure": f"{classification.failure_kind}: {event.symptom or event.actual_behavior}",
                "next_recommended_action": "Review audit report and approve patch plan before source edits.",
            },
            user_id=self.user_id,
        )
        record_improvement_audit(
            {
                "event_id": event.event_id,
                "hypothesis_id": hypothesis.hypothesis_id,
                "patch_plan_id": patch_plan.plan_id,
                "audit_report_id": report.report_id,
                "failure_kind": classification.failure_kind,
            },
            user_id=self.user_id,
        )
        return report

    def _ensure_behavior_event_recorded(self, event: BehaviorEvent) -> None:
        rows = read_behavior_events(limit=500)
        event_id = str(event.event_id or "")
        if not event_id:
            return
        if any(str(row.get("event_id") or "") == event_id for row in rows):
            return
        record_behavior_event(event)

    def build_codex_brief(self, audit_report: AuditReport) -> str:
        return build_codex_brief(audit_report)

    def record_outcome(self, outcome: Dict[str, Any]) -> None:
        summary = dict(outcome or {})
        update_project_state(
            {
                "last_success": str(
                    summary.get("result") or "self_improvement_outcome_recorded"
                )
            },
            user_id=self.user_id,
        )

    def maybe_auto_audit(self, event: BehaviorEvent) -> AuditReport | None:
        enabled = str(os.getenv("ALICE_SELF_IMPROVEMENT_AUTO_AUDIT", "false")).lower()
        if enabled not in {"1", "true", "yes", "on"}:
            return None
        return self.run_audit_from_event(event)

    def latest_event(self) -> Dict[str, Any]:
        rows = read_behavior_events(limit=1)
        return rows[-1] if rows else {}

    def latest_report(self) -> Dict[str, Any]:
        rows = read_audit_reports(limit=1)
        return rows[-1] if rows else {}

    def pending_status(self) -> Dict[str, Any]:
        events = read_behavior_events(limit=500)
        reports = read_audit_reports(limit=500)
        audited_event_ids = {
            str((row.get("event") or {}).get("event_id") or "")
            for row in reports
            if isinstance(row, dict)
        }
        pending_events = [
            row
            for row in events
            if str(row.get("event_id") or "") not in audited_event_ids
        ]
        return {
            "pending_event_count": len(pending_events),
            "audit_report_count": len(reports),
            "latest_event_id": str(events[-1].get("event_id") or "") if events else "",
            "latest_report_id": str(reports[-1].get("report_id") or "")
            if reports
            else "",
            "pending_event_ids": [
                str(row.get("event_id") or "") for row in pending_events[-10:]
            ],
        }
