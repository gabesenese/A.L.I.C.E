"""Persistent world-state memory for operator-aware execution."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


class WorldStateMemory:
    def __init__(self, storage_path: str = "data/world_state.json") -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._state: Dict[str, Any] = {
            "active_task": None,
            "last_tool": None,
            "last_successful_target": None,
            "pending_approvals": [],
            "unresolved_ambiguity": [],
            "pending_clarification": {},
            "current_narrowing_question": "",
            "selected_object_reference": "",
            "paused_autonomy_reason": "",
            "last_recovery_outcome": "",
            "last_trigger_decision": {},
            "workflow_chain": [],
            "environment": {},
            "updated_at": time.time(),
        }
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._state.update(data)
        except Exception:
            return

    def _save(self) -> None:
        self._state["updated_at"] = time.time()
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self._state, f, indent=2)
        except Exception:
            return

    def snapshot(self) -> Dict[str, Any]:
        return dict(self._state)

    def set_environment_state(
        self,
        key: str,
        data: Dict[str, Any],
        *,
        captured_at: Any = None,
    ) -> None:
        env = self._state.get("environment")
        if not isinstance(env, dict):
            env = {}
            self._state["environment"] = env

        entry = {
            "data": dict(data or {}),
            "captured_at": captured_at,
            "updated_at": time.time(),
        }
        env[str(key)] = entry
        self._save()

    def get_environment_state(self, key: str) -> Dict[str, Any]:
        env = self._state.get("environment")
        if not isinstance(env, dict):
            return {}
        payload = env.get(str(key))
        return dict(payload) if isinstance(payload, dict) else {}

    def set_pending_clarification(self, payload: Dict[str, Any]) -> None:
        self._state["pending_clarification"] = dict(payload or {})
        question = str((payload or {}).get("last_narrowing_question") or "").strip()
        if question:
            self._state["current_narrowing_question"] = question
        self._save()

    def clear_pending_clarification(self) -> None:
        self._state["pending_clarification"] = {}
        self._state["current_narrowing_question"] = ""
        self._save()

    def set_selected_object_reference(self, value: str) -> None:
        self._state["selected_object_reference"] = str(value or "")
        self._save()

    def set_autonomy_pause_reason(self, reason: str) -> None:
        self._state["paused_autonomy_reason"] = str(reason or "")
        self._save()

    def set_last_recovery_outcome(self, outcome: str) -> None:
        self._state["last_recovery_outcome"] = str(outcome or "")
        self._save()

    def set_last_trigger_decision(self, decision: Dict[str, Any]) -> None:
        self._state["last_trigger_decision"] = dict(decision or {})
        self._save()

    def update_from_action(self, request: Any, result: Any) -> None:
        req_goal = str(getattr(request, "goal", "") or "").strip() or None
        req_plugin = str(getattr(request, "plugin", "") or "").strip() or None
        req_action = str(getattr(request, "action", "") or "").strip() or None

        self._state["active_task"] = req_goal
        self._state["last_tool"] = {
            "plugin": req_plugin,
            "action": req_action,
            "success": bool(getattr(result, "success", False)),
            "status": str(getattr(result, "status", "unknown")),
            "confidence": float(getattr(result, "confidence", 0.0) or 0.0),
        }

        if bool(getattr(result, "success", False)):
            target = (getattr(request, "target_spec", {}) or {}).get("target")
            if not target:
                params = getattr(request, "params", {}) or {}
                target = (
                    params.get("target")
                    or params.get("path")
                    or params.get("note_id")
                    or params.get("title")
                )
            if target:
                self._state["last_successful_target"] = target

        ambiguity = list(getattr(result, "ambiguity_flags", []) or [])
        if ambiguity:
            self._state["unresolved_ambiguity"] = ambiguity
            self._state["pending_clarification"] = {
                "slot_type": "goal_linked_clarification",
                "parent_goal": req_goal or "",
                "expected_answer_shape": "short_disambiguation_or_selection",
                "last_narrowing_question": "Which exact target should I use?",
                "ambiguity_flags": ambiguity,
            }
            self._state["current_narrowing_question"] = (
                "Which exact target should I use?"
            )
        elif self._state.get("unresolved_ambiguity"):
            self._state["unresolved_ambiguity"] = []
            self._state["pending_clarification"] = {}
            self._state["current_narrowing_question"] = ""

        recovery_path = str(getattr(result, "recovery_path", "") or "").strip()
        if recovery_path:
            self._state["last_recovery_outcome"] = recovery_path

        state_updates = getattr(result, "state_updates", {}) or {}
        rollback = (
            state_updates.get("rollback") if isinstance(state_updates, dict) else {}
        )
        if isinstance(rollback, dict):
            rollback_status = str(rollback.get("status") or "").strip()
            if rollback_status:
                self._state["last_recovery_outcome"] = rollback_status

        chain = self._state.get("workflow_chain") or []
        chain.append(
            {
                "goal": req_goal,
                "plugin": req_plugin,
                "action": req_action,
                "status": str(getattr(result, "status", "unknown")),
                "at": time.time(),
            }
        )
        self._state["workflow_chain"] = chain[-15:]
        self._save()


_world_state_memory: WorldStateMemory | None = None


def get_world_state_memory(
    storage_path: str = "data/world_state.json",
) -> WorldStateMemory:
    global _world_state_memory
    if _world_state_memory is None:
        _world_state_memory = WorldStateMemory(storage_path=storage_path)
    return _world_state_memory
