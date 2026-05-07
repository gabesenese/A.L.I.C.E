from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

from ai.memory.project_memory import ProjectMemoryState, update_project_state


@dataclass
class OperatorState:
    active_mode: str = "general"
    active_objective: str = ""
    current_focus: str = ""
    awaiting_target: bool = False
    last_route: str = ""
    last_intent: str = ""
    last_inspected_file: str = ""
    last_failure: str = ""
    last_success: str = ""
    known_blockers: List[str] = field(default_factory=list)
    files_inspected: List[str] = field(default_factory=list)
    current_plan: List[str] = field(default_factory=list)
    current_step: str = ""
    last_user_correction: str = ""
    active_task_id: str = ""
    next_recommended_action: str = ""
    suggested_next_files: List[str] = field(default_factory=list)
    active_file_candidates: List[str] = field(default_factory=list)
    files_changed: List[str] = field(default_factory=list)
    tests_run: List[Dict[str, Any]] = field(default_factory=list)
    last_test_failure: str = ""
    user_corrections: List[str] = field(default_factory=list)
    design_constraints: List[str] = field(default_factory=list)
    last_self_improvement_event_id: str = ""
    last_hypothesis_id: str = ""
    last_patch_plan_id: str = ""
    last_audit_report_id: str = ""
    self_improvement_status: str = ""
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_mode": self.active_mode,
            "active_objective": self.active_objective,
            "current_focus": self.current_focus,
            "awaiting_target": bool(self.awaiting_target),
            "last_route": self.last_route,
            "last_intent": self.last_intent,
            "last_inspected_file": self.last_inspected_file,
            "last_failure": self.last_failure,
            "last_success": self.last_success,
            "known_blockers": list(self.known_blockers or []),
            "files_inspected": list(self.files_inspected or []),
            "current_plan": list(self.current_plan or []),
            "current_step": self.current_step,
            "last_user_correction": self.last_user_correction,
            "active_task_id": self.active_task_id,
            "next_recommended_action": self.next_recommended_action,
            "suggested_next_files": list(self.suggested_next_files or []),
            "active_file_candidates": list(self.active_file_candidates or []),
            "files_changed": list(self.files_changed or []),
            "tests_run": list(self.tests_run or []),
            "last_test_failure": self.last_test_failure,
            "user_corrections": list(self.user_corrections or []),
            "design_constraints": list(self.design_constraints or []),
            "last_self_improvement_event_id": self.last_self_improvement_event_id,
            "last_hypothesis_id": self.last_hypothesis_id,
            "last_patch_plan_id": self.last_patch_plan_id,
            "last_audit_report_id": self.last_audit_report_id,
            "self_improvement_status": self.self_improvement_status,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "OperatorState":
        data = dict(payload or {})
        return cls(
            active_mode=str(data.get("active_mode") or "general"),
            active_objective=str(data.get("active_objective") or ""),
            current_focus=str(data.get("current_focus") or ""),
            awaiting_target=bool(data.get("awaiting_target")),
            last_route=str(data.get("last_route") or ""),
            last_intent=str(data.get("last_intent") or ""),
            last_inspected_file=str(data.get("last_inspected_file") or ""),
            last_failure=str(data.get("last_failure") or ""),
            last_success=str(data.get("last_success") or ""),
            known_blockers=list(data.get("known_blockers") or []),
            files_inspected=list(data.get("files_inspected") or []),
            current_plan=list(data.get("current_plan") or []),
            current_step=str(data.get("current_step") or ""),
            last_user_correction=str(data.get("last_user_correction") or ""),
            active_task_id=str(data.get("active_task_id") or ""),
            next_recommended_action=str(data.get("next_recommended_action") or ""),
            suggested_next_files=list(data.get("suggested_next_files") or []),
            active_file_candidates=list(data.get("active_file_candidates") or []),
            files_changed=list(data.get("files_changed") or []),
            tests_run=list(data.get("tests_run") or []),
            last_test_failure=str(data.get("last_test_failure") or ""),
            user_corrections=list(data.get("user_corrections") or []),
            design_constraints=list(data.get("design_constraints") or []),
            last_self_improvement_event_id=str(
                data.get("last_self_improvement_event_id") or ""
            ),
            last_hypothesis_id=str(data.get("last_hypothesis_id") or ""),
            last_patch_plan_id=str(data.get("last_patch_plan_id") or ""),
            last_audit_report_id=str(data.get("last_audit_report_id") or ""),
            self_improvement_status=str(data.get("self_improvement_status") or ""),
            updated_at=str(
                data.get("updated_at") or datetime.now(timezone.utc).isoformat()
            ),
        )


def update_operator_state(
    existing: Dict[str, Any] | None, updates: Dict[str, Any]
) -> Dict[str, Any]:
    state = OperatorState.from_dict(existing)
    for key, value in dict(updates or {}).items():
        if hasattr(state, key):
            if key in {
                "known_blockers",
                "files_inspected",
                "current_plan",
                "suggested_next_files",
                "active_file_candidates",
                "files_changed",
                "tests_run",
                "user_corrections",
                "design_constraints",
            }:
                if isinstance(value, list):
                    current = list(getattr(state, key) or [])
                    merged = current + [v for v in value if v not in current]
                    setattr(state, key, merged)
                else:
                    setattr(state, key, value)
            else:
                setattr(state, key, value)
    state.updated_at = datetime.now(timezone.utc).isoformat()
    return state.to_dict()


def sync_operator_state_with_project_memory(
    operator_state: Dict[str, Any] | None,
    project_memory_state: ProjectMemoryState,
) -> Dict[str, Any]:
    state = OperatorState.from_dict(operator_state)
    pm = project_memory_state

    if pm.active_objective and not state.active_objective:
        state.active_objective = pm.active_objective
    if pm.current_focus and not state.current_focus:
        state.current_focus = pm.current_focus

    # Durable fields should always be hydrated from project memory.
    state.last_failure = str(pm.last_failure or state.last_failure)
    state.last_success = str(pm.last_success or state.last_success)
    state.known_blockers = list(pm.known_blockers or state.known_blockers)
    state.files_inspected = list(pm.files_inspected or state.files_inspected)
    state.files_changed = list(pm.files_changed or state.files_changed)
    state.tests_run = list(pm.tests_run or state.tests_run)
    state.last_test_failure = str(pm.last_test_failure or state.last_test_failure)
    state.current_plan = list(pm.current_plan or state.current_plan)
    state.current_step = str(pm.current_step or state.current_step)
    state.next_recommended_action = str(
        pm.next_recommended_action or state.next_recommended_action
    )
    state.last_self_improvement_event_id = str(
        getattr(pm, "last_self_improvement_event_id", "")
        or state.last_self_improvement_event_id
    )
    state.last_hypothesis_id = str(
        getattr(pm, "last_hypothesis_id", "") or state.last_hypothesis_id
    )
    state.last_patch_plan_id = str(
        getattr(pm, "last_patch_plan_id", "") or state.last_patch_plan_id
    )
    state.last_audit_report_id = str(
        getattr(pm, "last_audit_report_id", "") or state.last_audit_report_id
    )
    state.self_improvement_status = str(
        getattr(pm, "self_improvement_status", "") or state.self_improvement_status
    )
    state.user_corrections = list(pm.user_corrections or state.user_corrections)
    state.design_constraints = list(pm.design_constraints or state.design_constraints)
    state.updated_at = datetime.now(timezone.utc).isoformat()
    return state.to_dict()


def commit_operator_state_to_project_memory(
    operator_state: Dict[str, Any] | None, user_id: str = "default"
) -> None:
    state = OperatorState.from_dict(operator_state)
    update_project_state(
        {
            "active_objective": state.active_objective,
            "current_focus": state.current_focus,
            "last_failure": state.last_failure,
            "last_success": state.last_success,
            "known_blockers": list(state.known_blockers or []),
            "files_inspected": list(state.files_inspected or []),
            "files_changed": list(state.files_changed or []),
            "tests_run": list(state.tests_run or []),
            "last_test_failure": state.last_test_failure,
            "current_plan": list(state.current_plan or []),
            "current_step": state.current_step,
            "next_recommended_action": state.next_recommended_action,
            "user_corrections": list(state.user_corrections or []),
            "design_constraints": list(state.design_constraints or []),
            "last_self_improvement_event_id": state.last_self_improvement_event_id,
            "last_hypothesis_id": state.last_hypothesis_id,
            "last_patch_plan_id": state.last_patch_plan_id,
            "last_audit_report_id": state.last_audit_report_id,
            "self_improvement_status": state.self_improvement_status,
        },
        user_id=user_id,
    )
