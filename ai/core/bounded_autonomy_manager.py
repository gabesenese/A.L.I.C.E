"""Bounded autonomy loop registry with permission and audit controls."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class AutonomyLoop:
    name: str
    permission_level: str
    scope: str
    stop_conditions: List[str]
    confidence_threshold: float
    enabled: bool = False


@dataclass
class LoopTriggerEvent:
    loop: str
    severity: str
    reason: str
    recommended_action: str
    confidence: float
    timestamp: float


class BoundedAutonomyManager:
    def __init__(self, storage_path: str = "data/autonomy_loops.json") -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.loops: Dict[str, AutonomyLoop] = {}
        self.audit_log: List[Dict[str, Any]] = []
        self.trigger_events: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for row in data.get("loops", []):
                loop = AutonomyLoop(**row)
                self.loops[loop.name] = loop
            self.audit_log = data.get("audit_log", [])[-200:]
            self.trigger_events = data.get("trigger_events", [])[-200:]
        except Exception:
            return

    def _save(self) -> None:
        try:
            payload = {
                "loops": [asdict(loop) for loop in self.loops.values()],
                "audit_log": self.audit_log[-200:],
                "trigger_events": self.trigger_events[-200:],
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            return

    def register_loop(self, loop: AutonomyLoop) -> None:
        self.loops[loop.name] = loop
        self._audit("register", loop.name, {"scope": loop.scope, "permission": loop.permission_level})
        self._save()

    def set_enabled(self, name: str, enabled: bool, actor: str = "operator") -> bool:
        loop = self.loops.get(name)
        if not loop:
            return False
        loop.enabled = bool(enabled)
        self._audit("enable" if enabled else "disable", name, {"actor": actor})
        self._save()
        return True

    def _audit(self, event: str, name: str, data: Dict[str, Any]) -> None:
        self.audit_log.append({"event": event, "loop": name, "data": data, "at": time.time()})
        self.audit_log = self.audit_log[-200:]

    def evaluate_triggers(
        self,
        *,
        world_state: Dict[str, Any],
        goal_summary: Dict[str, Any],
        journal_summary: Dict[str, Any],
        execution_state: Dict[str, Any],
    ) -> List[LoopTriggerEvent]:
        events: List[LoopTriggerEvent] = []
        now = time.time()

        for loop in self.loops.values():
            if not loop.enabled:
                continue

            if loop.name == "goal_health":
                active_goals = int(goal_summary.get("active_goals", 0) or 0)
                unresolved_ambiguity = len((world_state or {}).get("unresolved_ambiguity") or [])
                if active_goals > 0 and unresolved_ambiguity > 0:
                    events.append(
                        LoopTriggerEvent(
                            loop=loop.name,
                            severity="medium",
                            reason="active_goals_with_unresolved_ambiguity",
                            recommended_action="ask_clarification_then_replan",
                            confidence=max(loop.confidence_threshold, 0.72),
                            timestamp=now,
                        )
                    )

            elif loop.name == "repo_failure_watch":
                failed = int(journal_summary.get("failed", 0) or 0)
                success = int(journal_summary.get("success", 0) or 0)
                if failed >= 3 and failed > success:
                    events.append(
                        LoopTriggerEvent(
                            loop=loop.name,
                            severity="high",
                            reason="failure_rate_spike",
                            recommended_action="pause_autonomy_and_escalate",
                            confidence=max(loop.confidence_threshold, 0.81),
                            timestamp=now,
                        )
                    )

            elif loop.name == "approval_guard":
                pending_approvals = len((world_state or {}).get("pending_approvals") or [])
                if pending_approvals > 0:
                    events.append(
                        LoopTriggerEvent(
                            loop=loop.name,
                            severity="low",
                            reason="pending_approvals_waiting",
                            recommended_action="request_operator_decision",
                            confidence=max(loop.confidence_threshold, 0.66),
                            timestamp=now,
                        )
                    )

        if events:
            for ev in events:
                self.trigger_events.append(asdict(ev))
            self.trigger_events = self.trigger_events[-200:]
            self._save()

        return events

    def status(self) -> Dict[str, Any]:
        enabled = [name for name, loop in self.loops.items() if loop.enabled]
        return {
            "total_loops": len(self.loops),
            "enabled_loops": enabled,
            "disabled_loops": [name for name in self.loops.keys() if name not in enabled],
            "recent_audit": self.audit_log[-10:],
            "recent_triggers": self.trigger_events[-10:],
        }


_bounded_autonomy_manager: BoundedAutonomyManager | None = None


def get_bounded_autonomy_manager(storage_path: str = "data/autonomy_loops.json") -> BoundedAutonomyManager:
    global _bounded_autonomy_manager
    if _bounded_autonomy_manager is None:
        _bounded_autonomy_manager = BoundedAutonomyManager(storage_path=storage_path)
    return _bounded_autonomy_manager
