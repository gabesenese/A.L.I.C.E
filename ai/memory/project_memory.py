from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_MEMORY_PATH = Path("data/project_memory.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_design_constraints() -> List[str]:
    return [
        "no voice for now",
        "keep foundation minimal",
        "avoid hardcoded fallback messages",
        "do not use fictional or external character names",
        "prioritize routing, memory, local execution, operator loop, runtime cleanup",
        "do not fake memory or tool or file access",
        "greetings should be grounded and companion-like without fake continuity",
    ]


@dataclass
class ProjectMemoryState:
    active_objective: str = ""
    current_focus: str = ""
    last_failure: str = ""
    last_success: str = ""
    known_blockers: List[str] = field(default_factory=list)
    files_inspected: List[str] = field(default_factory=list)
    files_changed: List[str] = field(default_factory=list)
    tests_run: List[Dict[str, Any]] = field(default_factory=list)
    last_test_failure: str = ""
    current_plan: List[str] = field(default_factory=list)
    current_step: str = ""
    next_recommended_action: str = ""
    user_corrections: List[str] = field(default_factory=list)
    design_constraints: List[str] = field(default_factory=_default_design_constraints)
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "ProjectMemoryState":
        data = dict(payload or {})
        state = cls(
            active_objective=str(data.get("active_objective") or ""),
            current_focus=str(data.get("current_focus") or ""),
            last_failure=str(data.get("last_failure") or ""),
            last_success=str(data.get("last_success") or ""),
            known_blockers=list(data.get("known_blockers") or []),
            files_inspected=list(data.get("files_inspected") or []),
            files_changed=list(data.get("files_changed") or []),
            tests_run=list(data.get("tests_run") or []),
            last_test_failure=str(data.get("last_test_failure") or ""),
            current_plan=list(data.get("current_plan") or []),
            current_step=str(data.get("current_step") or ""),
            next_recommended_action=str(data.get("next_recommended_action") or ""),
            user_corrections=list(data.get("user_corrections") or []),
            design_constraints=list(
                data.get("design_constraints") or _default_design_constraints()
            ),
            updated_at=str(data.get("updated_at") or _now_iso()),
        )
        # Ensure required design constraints always exist.
        for constraint in _default_design_constraints():
            if constraint not in state.design_constraints:
                state.design_constraints.append(constraint)
        return state


def _read_all() -> Dict[str, Any]:
    if not PROJECT_MEMORY_PATH.exists():
        return {}
    try:
        return json.loads(PROJECT_MEMORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_all(payload: Dict[str, Any]) -> None:
    PROJECT_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROJECT_MEMORY_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8"
    )


def load_project_state(user_id: str = "default") -> ProjectMemoryState:
    data = _read_all()
    return ProjectMemoryState.from_dict(data.get(str(user_id)) or {})


def save_project_state(state: ProjectMemoryState, user_id: str = "default") -> None:
    data = _read_all()
    state.updated_at = _now_iso()
    data[str(user_id)] = state.to_dict()
    _write_all(data)


def update_project_state(
    updates: Dict[str, Any], user_id: str = "default"
) -> ProjectMemoryState:
    state = load_project_state(user_id)
    for key, value in dict(updates or {}).items():
        if not hasattr(state, key):
            continue
        if key in {
            "known_blockers",
            "files_inspected",
            "files_changed",
            "current_plan",
            "user_corrections",
            "design_constraints",
        }:
            if isinstance(value, list):
                current = list(getattr(state, key) or [])
                for item in value:
                    if item not in current:
                        current.append(item)
                setattr(state, key, current)
            elif isinstance(value, str) and value:
                current = list(getattr(state, key) or [])
                if value not in current:
                    current.append(value)
                setattr(state, key, current)
        elif key == "tests_run" and isinstance(value, list):
            current = list(state.tests_run or [])
            current.extend(value)
            state.tests_run = current
        else:
            setattr(state, key, value)
    save_project_state(state, user_id=user_id)
    return state


def record_failure(
    kind: str, detail: str, evidence: Dict[str, Any] | None = None, user_id: str = "default"
) -> ProjectMemoryState:
    payload = {
        "last_failure": f"{kind}: {detail}".strip(": "),
        "known_blockers": [detail] if detail else [],
    }
    if evidence:
        payload["known_blockers"] = [
            f"{detail} | evidence={json.dumps(evidence, ensure_ascii=True)}"
        ]
    return update_project_state(payload, user_id=user_id)


def record_success(
    kind: str, detail: str, evidence: Dict[str, Any] | None = None, user_id: str = "default"
) -> ProjectMemoryState:
    msg = f"{kind}: {detail}".strip(": ")
    if evidence:
        msg = f"{msg} | evidence={json.dumps(evidence, ensure_ascii=True)}"
    return update_project_state({"last_success": msg}, user_id=user_id)


def record_file_inspected(path: str, user_id: str = "default") -> ProjectMemoryState:
    return update_project_state({"files_inspected": [str(path or "")]}, user_id=user_id)


def record_file_changed(path: str, user_id: str = "default") -> ProjectMemoryState:
    return update_project_state({"files_changed": [str(path or "")]}, user_id=user_id)


def record_test_run(
    command: str, success: bool, detail: str = "", user_id: str = "default"
) -> ProjectMemoryState:
    entry = {
        "command": str(command or ""),
        "success": bool(success),
        "detail": str(detail or ""),
        "at": _now_iso(),
    }
    updates: Dict[str, Any] = {"tests_run": [entry]}
    if not success:
        updates["last_test_failure"] = str(detail or command or "test_failure")
    return update_project_state(updates, user_id=user_id)


def record_user_correction(text: str, user_id: str = "default") -> ProjectMemoryState:
    return update_project_state({"user_corrections": [str(text or "")]}, user_id=user_id)


def get_next_context_summary(user_id: str = "default") -> Dict[str, Any]:
    state = load_project_state(user_id)
    return {
        "active_objective": state.active_objective,
        "current_focus": state.current_focus,
        "last_failure": state.last_failure,
        "last_success": state.last_success,
        "known_blockers": list(state.known_blockers or [])[:5],
        "last_inspected_file": (state.files_inspected[-1] if state.files_inspected else ""),
        "next_recommended_action": state.next_recommended_action,
        "updated_at": state.updated_at,
    }
