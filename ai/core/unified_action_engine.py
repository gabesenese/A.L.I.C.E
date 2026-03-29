"""Unified action execution fabric for Foundation 3 (Action Sovereignty)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ai.core.tool_executor import ToolExecutor, get_tool_executor


@dataclass
class ActionRequest:
    goal: str
    plugin: str
    action: str
    params: dict
    source_intent: str
    confidence: float
    requires_confirmation: bool = False


@dataclass
class ActionResult:
    success: bool
    status: str
    plugin: str
    action: str
    goal_satisfied: bool
    data: dict
    error: str | None = None
    retryable: bool = False
    confidence: float = 1.0
    side_effects: list[str] = field(default_factory=list)
    state_updates: dict = field(default_factory=dict)

    def to_plugin_result(self) -> Dict[str, Any]:
        """Compatibility shape for existing post-plugin response flow."""
        verification = self.state_updates.get("verification", {})
        message = self.error or ""
        return {
            "success": self.success,
            "status": self.status,
            "plugin": self.plugin,
            "action": self.action,
            "data": self.data if isinstance(self.data, dict) else {},
            "message": message,
            "response": message,
            "retryable": self.retryable,
            "confidence": self.confidence,
            "side_effects": list(self.side_effects or []),
            "verification": verification,
        }


class UnifiedActionEngine:
    """Single action authority for plugin dispatch, verification, and state updates."""

    def __init__(self, tool_executor: Optional[ToolExecutor] = None) -> None:
        self.tool_executor = tool_executor or get_tool_executor()
        self.plugin_manager: Any = None

    def bind_plugin_manager(self, plugin_manager: Any) -> None:
        self.plugin_manager = plugin_manager

    def execute(self, request: ActionRequest) -> ActionResult:
        if request.requires_confirmation:
            return ActionResult(
                success=False,
                status="requires_confirmation",
                plugin=request.plugin,
                action=request.action,
                goal_satisfied=False,
                data={},
                error="This action requires explicit confirmation before execution.",
                retryable=True,
                confidence=max(0.0, min(1.0, float(request.confidence or 0.0))),
                side_effects=[],
                state_updates=self._build_state_updates(
                    request=request,
                    tool_result={},
                    verified=False,
                    goal_satisfied=False,
                    handled=False,
                ),
            )

        if self.plugin_manager is None:
            return ActionResult(
                success=False,
                status="engine_not_ready",
                plugin=request.plugin,
                action=request.action,
                goal_satisfied=False,
                data={},
                error="Action engine plugin manager is not initialized.",
                retryable=True,
                confidence=0.0,
                side_effects=[],
                state_updates=self._build_state_updates(
                    request=request,
                    tool_result={},
                    verified=False,
                    goal_satisfied=False,
                    handled=False,
                ),
            )

        intent = self._compose_intent(request)
        raw_query = request.params.get("_raw_query") if isinstance(request.params, dict) else None
        tool_outcome = self.tool_executor.execute(
            plugin_manager=self.plugin_manager,
            intent=intent,
            query=str(raw_query or request.goal or ""),
            entities=request.params or {},
            context={
                "goal": request.goal,
                "source_intent": request.source_intent,
                "intent_confidence": request.confidence,
                "action_engine": "unified",
            },
        )

        result_dict = tool_outcome.result or {}
        status = str(result_dict.get("status") or ("success" if result_dict.get("success") else "failed"))
        plugin = str(result_dict.get("plugin") or request.plugin or "unknown")
        action = str(result_dict.get("action") or request.action or "unknown")
        success = bool(result_dict.get("success", False))
        verified = bool((result_dict.get("verification") or {}).get("accepted", False))
        goal_satisfied = bool(success and verified)

        state_updates = self._build_state_updates(
            request=request,
            tool_result=result_dict,
            verified=verified,
            goal_satisfied=goal_satisfied,
            handled=bool(tool_outcome.handled),
        )

        confidence = self._coerce_confidence(
            result_dict.get("confidence", request.confidence or (1.0 if success else 0.2))
        )

        return ActionResult(
            success=success,
            status=status,
            plugin=plugin,
            action=action,
            goal_satisfied=goal_satisfied,
            data=result_dict.get("data") if isinstance(result_dict.get("data"), dict) else {},
            error=str(result_dict.get("message") or result_dict.get("response") or "") or None,
            retryable=bool(result_dict.get("retryable", not success)),
            confidence=confidence,
            side_effects=list(result_dict.get("side_effects") or []),
            state_updates=state_updates,
        )

    def apply_state_updates(self, assistant: Any, result: ActionResult) -> None:
        updates = result.state_updates if isinstance(result.state_updates, dict) else {}
        if not updates:
            return

        try:
            if hasattr(assistant, "_internal_reasoning_state") and isinstance(assistant._internal_reasoning_state, dict):
                assistant._internal_reasoning_state.update(updates)
                verification = updates.get("verification")
                if isinstance(verification, dict):
                    assistant._internal_reasoning_state["tool_verification"] = verification
        except Exception:
            return

    def _compose_intent(self, request: ActionRequest) -> str:
        source_intent = str(request.source_intent or "").strip()
        if ":" in source_intent:
            return source_intent
        plugin = str(request.plugin or "unknown").strip().lower() or "unknown"
        action = str(request.action or "unknown").strip().lower() or "unknown"
        return f"{plugin}:{action}"

    def _coerce_confidence(self, value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 0.0

    def _build_state_updates(
        self,
        *,
        request: ActionRequest,
        tool_result: Dict[str, Any],
        verified: bool,
        goal_satisfied: bool,
        handled: bool,
    ) -> Dict[str, Any]:
        return {
            "verification": dict(tool_result.get("verification") or {}),
            "last_action_request": {
                "goal": request.goal,
                "plugin": request.plugin,
                "action": request.action,
                "source_intent": request.source_intent,
                "confidence": request.confidence,
                "requires_confirmation": request.requires_confirmation,
            },
            "last_action_result": {
                "handled": handled,
                "success": bool(tool_result.get("success", False)),
                "status": str(tool_result.get("status", "unknown")),
                "goal_satisfied": goal_satisfied,
                "plugin": str(tool_result.get("plugin") or request.plugin or "unknown"),
                "action": str(tool_result.get("action") or request.action or "unknown"),
            },
            "goal_satisfied": goal_satisfied,
        }


_unified_action_engine: Optional[UnifiedActionEngine] = None


def get_unified_action_engine() -> UnifiedActionEngine:
    global _unified_action_engine
    if _unified_action_engine is None:
        _unified_action_engine = UnifiedActionEngine()
    return _unified_action_engine
