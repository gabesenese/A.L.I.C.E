from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


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
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

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
            updated_at=str(data.get("updated_at") or datetime.now(timezone.utc).isoformat()),
        )


def update_operator_state(existing: Dict[str, Any] | None, updates: Dict[str, Any]) -> Dict[str, Any]:
    state = OperatorState.from_dict(existing)
    for key, value in dict(updates or {}).items():
        if hasattr(state, key):
            if key in {"known_blockers", "files_inspected", "current_plan", "suggested_next_files", "active_file_candidates"}:
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
