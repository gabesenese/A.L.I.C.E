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
    next_recommended_action: str = ""
    suggested_next_files: List[str] = field(default_factory=list)
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
            "next_recommended_action": self.next_recommended_action,
            "suggested_next_files": list(self.suggested_next_files or []),
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
            next_recommended_action=str(data.get("next_recommended_action") or ""),
            suggested_next_files=list(data.get("suggested_next_files") or []),
            updated_at=str(data.get("updated_at") or datetime.now(timezone.utc).isoformat()),
        )


def update_operator_state(existing: Dict[str, Any] | None, updates: Dict[str, Any]) -> Dict[str, Any]:
    state = OperatorState.from_dict(existing)
    for key, value in dict(updates or {}).items():
        if hasattr(state, key):
            setattr(state, key, value)
    state.updated_at = datetime.now(timezone.utc).isoformat()
    return state.to_dict()
