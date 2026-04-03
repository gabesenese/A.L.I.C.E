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
        elif self._state.get("unresolved_ambiguity"):
            self._state["unresolved_ambiguity"] = []

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
