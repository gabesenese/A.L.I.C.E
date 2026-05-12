from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


COMPANION_STATE_PATH = Path("data/companion_state.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CompanionState:
    user_id: str = "default"
    active_topic: str = ""
    active_goal: str = ""
    user_context: Dict[str, Any] = field(default_factory=dict)
    energy_signal: str = ""
    mood_signal: str = ""
    time_context: str = ""
    open_loops: list[str] = field(default_factory=list)
    design_constraints: list[str] = field(default_factory=list)
    last_user_request: str = ""
    last_actual_request: str = ""
    last_action_name: str = ""
    last_action_result: Dict[str, Any] = field(default_factory=dict)
    next_safe_action: Dict[str, Any] = field(default_factory=dict)
    privacy_flags: Dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "CompanionState":
        data = dict(payload or {})
        return cls(
            user_id=str(data.get("user_id") or "default"),
            active_topic=str(data.get("active_topic") or ""),
            active_goal=str(data.get("active_goal") or ""),
            user_context=dict(data.get("user_context") or {}),
            energy_signal=str(data.get("energy_signal") or ""),
            mood_signal=str(data.get("mood_signal") or ""),
            time_context=str(data.get("time_context") or ""),
            open_loops=list(data.get("open_loops") or []),
            design_constraints=list(data.get("design_constraints") or []),
            last_user_request=str(data.get("last_user_request") or ""),
            last_actual_request=str(data.get("last_actual_request") or ""),
            last_action_name=str(data.get("last_action_name") or ""),
            last_action_result=dict(data.get("last_action_result") or {}),
            next_safe_action=dict(data.get("next_safe_action") or {}),
            privacy_flags=dict(data.get("privacy_flags") or {}),
            updated_at=str(data.get("updated_at") or _now_iso()),
        )


def _read_all() -> Dict[str, Any]:
    if not COMPANION_STATE_PATH.exists():
        return {}
    try:
        return json.loads(COMPANION_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_all(payload: Dict[str, Any]) -> None:
    COMPANION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    COMPANION_STATE_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )


def load_companion_state(user_id: str = "default") -> CompanionState:
    data = _read_all()
    state = CompanionState.from_dict(data.get(str(user_id)) or {})
    state.user_id = str(user_id or "default")
    return state


def save_companion_state(state: CompanionState, user_id: str = "default") -> None:
    data = _read_all()
    state.user_id = str(user_id or state.user_id or "default")
    state.updated_at = _now_iso()
    data[str(state.user_id)] = state.to_dict()
    _write_all(data)


def update_companion_state(
    updates: Dict[str, Any], user_id: str = "default"
) -> CompanionState:
    state = load_companion_state(user_id=user_id)
    for key, value in dict(updates or {}).items():
        if not hasattr(state, key):
            continue
        if key in {"open_loops", "design_constraints"} and isinstance(value, list):
            current = list(getattr(state, key) or [])
            for item in value:
                if item not in current:
                    current.append(item)
            setattr(state, key, current)
            continue
        setattr(state, key, value)
    save_companion_state(state, user_id=user_id)
    return state


def sync_from_operator_project_state(
    operator_state: Dict[str, Any] | None,
    project_memory: Dict[str, Any] | None,
    user_id: str = "default",
) -> CompanionState:
    op = dict(operator_state or {})
    pm = dict(project_memory or {})
    updates = {
        "active_goal": str(op.get("active_objective") or pm.get("active_objective") or ""),
        "active_topic": str(op.get("current_focus") or pm.get("current_focus") or ""),
        "open_loops": list(pm.get("known_blockers") or op.get("known_blockers") or []),
        "design_constraints": list(pm.get("design_constraints") or op.get("design_constraints") or []),
        "next_safe_action": dict(
            op.get("last_recommended_action") or pm.get("last_recommended_action") or {}
        ),
    }
    return update_companion_state(updates, user_id=user_id)
