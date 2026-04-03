"""Compensating rollback executor for plugin-backed actions."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class RollbackOutcome:
    attempted: bool
    success: bool
    status: str
    rollback_intent: str = ""
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["details"] = dict(self.details or {})
        return payload


@dataclass
class RollbackPlan:
    applicable: bool
    rollback_intent: str = ""
    rollback_entities: Dict[str, Any] = None
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "applicable": bool(self.applicable),
            "rollback_intent": self.rollback_intent,
            "rollback_entities": dict(self.rollback_entities or {}),
            "reason": self.reason,
        }


class RollbackExecutor:
    """Runs best-effort compensating actions per plugin/action class."""

    def plan(self, *, request: Any, result: Any) -> RollbackPlan:
        rollback_intent, rollback_entities = self._build_compensating_call(
            request, result
        )
        if not rollback_intent:
            return RollbackPlan(False, reason="no_compensation_mapping")
        return RollbackPlan(
            True,
            rollback_intent=rollback_intent,
            rollback_entities=rollback_entities,
            reason="compensating_action_available",
        )

    def execute(
        self,
        *,
        plugin_manager: Any,
        request: Any,
        result: Any,
        dry_run: bool = False,
    ) -> RollbackOutcome:
        if plugin_manager is None and not dry_run:
            return RollbackOutcome(
                False,
                False,
                "rollback_unavailable",
                details={"reason": "missing_plugin_manager"},
            )

        plan = self.plan(request=request, result=result)
        if not plan.applicable:
            return RollbackOutcome(
                False, False, "rollback_not_applicable", details={"reason": plan.reason}
            )

        rollback_intent = plan.rollback_intent
        rollback_entities = plan.rollback_entities or {}

        if dry_run:
            return RollbackOutcome(
                attempted=False,
                success=True,
                status="rollback_preview",
                rollback_intent=rollback_intent,
                details={
                    "reason": "dry_run",
                    "entities": rollback_entities,
                },
            )

        try:
            raw_query = f"rollback {getattr(request, 'plugin', 'unknown')}:{getattr(request, 'action', 'unknown')}"
            rollback_result = plugin_manager.execute_for_intent(
                rollback_intent,
                raw_query,
                rollback_entities,
                {
                    "rollback": True,
                    "source_intent": getattr(request, "source_intent", ""),
                },
            )
            ok = (
                bool((rollback_result or {}).get("success", False))
                if isinstance(rollback_result, dict)
                else False
            )
            return RollbackOutcome(
                attempted=True,
                success=ok,
                status="rollback_success" if ok else "rollback_failed",
                rollback_intent=rollback_intent,
                details={
                    "entities": rollback_entities,
                    "result": (
                        rollback_result
                        if isinstance(rollback_result, dict)
                        else {"raw": str(rollback_result)}
                    ),
                },
            )
        except Exception as exc:
            return RollbackOutcome(
                attempted=True,
                success=False,
                status="rollback_failed",
                rollback_intent=rollback_intent,
                details={"error": str(exc), "entities": rollback_entities},
            )

    def _build_compensating_call(
        self, request: Any, result: Any
    ) -> tuple[str, Dict[str, Any]]:
        plugin = self._normalize_plugin(str(getattr(request, "plugin", "") or ""))
        action = str(getattr(request, "action", "") or "").strip().lower()

        if action not in {"create", "append", "update"}:
            return "", {}

        data = (
            getattr(result, "data", {})
            if isinstance(getattr(result, "data", {}), dict)
            else {}
        )
        params = (
            getattr(request, "params", {})
            if isinstance(getattr(request, "params", {}), dict)
            else {}
        )

        if plugin == "notes":
            target = (
                data.get("note_id")
                or data.get("id")
                or data.get("title")
                or params.get("title")
            )
            if target:
                return "notes:delete", {
                    "target": target,
                    "note_id": data.get("note_id") or target,
                }

        if plugin == "file_operations":
            filepath = (
                data.get("filepath")
                or params.get("path")
                or params.get("filename")
                or params.get("target")
            )
            if filepath:
                filename = Path(str(filepath)).name
                return "file_operations:delete", {
                    "filename": filename,
                    "target": filename,
                }

        if plugin == "calendar":
            target = (
                data.get("event_id")
                or data.get("id")
                or params.get("title")
                or params.get("target")
            )
            if target:
                return "calendar:delete", {
                    "target": target,
                    "event_id": data.get("event_id") or target,
                }

        if plugin == "memory":
            target = data.get("id") or params.get("target") or params.get("memory_id")
            if target:
                return "memory:delete", {"target": target, "memory_id": target}

        if plugin == "document":
            target = data.get("document_id") or data.get("id") or params.get("target")
            if target:
                return "document:delete", {"target": target, "document_id": target}

        return "", {}

    def _normalize_plugin(self, name: str) -> str:
        n = name.strip().lower().replace("plugin", "").replace(" ", "_")
        n = n.strip("_")
        aliases = {
            "notes": "notes",
            "notes_": "notes",
            "file_operations": "file_operations",
            "file_operations_": "file_operations",
            "file_operations__": "file_operations",
            "file_operationss": "file_operations",
            "calendar": "calendar",
            "memory": "memory",
            "document": "document",
        }
        if "file" in n and "operation" in n:
            return "file_operations"
        return aliases.get(n, n)


_rollback_executor: Optional[RollbackExecutor] = None


def get_rollback_executor() -> RollbackExecutor:
    global _rollback_executor
    if _rollback_executor is None:
        _rollback_executor = RollbackExecutor()
    return _rollback_executor
