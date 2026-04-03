"""Unified action execution fabric for Foundation 3 (Action Sovereignty)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ai.core.execution_journal import ExecutionJournal, get_execution_journal
from ai.core.goal_action_verifier import GoalActionVerifier, get_goal_action_verifier
from ai.core.rollback_executor import RollbackExecutor, get_rollback_executor
from ai.core.tool_executor import ToolExecutor, get_tool_executor
from ai.core.world_state_memory import WorldStateMemory, get_world_state_memory
from ai.planning.action_planner import ActionPlanner, get_action_planner


@dataclass
class ActionRequest:
    goal: str
    plugin: str
    action: str
    params: dict
    source_intent: str
    confidence: float
    requires_confirmation: bool = False
    expected_outcome: str = ""
    target_spec: dict = field(default_factory=dict)
    risk_level: str = "medium"
    retry_budget: int = 1
    rollback_policy: str = "none"
    plan_steps: list[str] = field(default_factory=list)


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
    verification_report: dict = field(default_factory=dict)
    ambiguity_flags: list[str] = field(default_factory=list)
    recovery_path: str | None = None
    retry_count: int = 0

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
            "verification_report": dict(self.verification_report or {}),
            "ambiguity_flags": list(self.ambiguity_flags or []),
            "recovery_path": self.recovery_path,
            "retry_count": int(self.retry_count),
        }


class UnifiedActionEngine:
    """Single action authority for plugin dispatch, verification, and state updates."""

    def __init__(
        self,
        tool_executor: Optional[ToolExecutor] = None,
        goal_verifier: Optional[GoalActionVerifier] = None,
        world_state_memory: Optional[WorldStateMemory] = None,
        execution_journal: Optional[ExecutionJournal] = None,
        planner: Optional[ActionPlanner] = None,
        rollback_executor: Optional[RollbackExecutor] = None,
    ) -> None:
        self.tool_executor = tool_executor or get_tool_executor()
        self.goal_verifier = goal_verifier or get_goal_action_verifier()
        self.world_state_memory = world_state_memory or get_world_state_memory()
        self.execution_journal = execution_journal or get_execution_journal()
        self.planner = planner or get_action_planner()
        self.rollback_executor = rollback_executor or get_rollback_executor()
        self.plugin_manager: Any = None

    def bind_plugin_manager(self, plugin_manager: Any) -> None:
        self.plugin_manager = plugin_manager

    def execute(self, request: ActionRequest) -> ActionResult:
        request = self._normalize_request(request)

        if request.requires_confirmation:
            result = ActionResult(
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
                recovery_path="await_confirmation",
            )
            self._record_journal("blocked", request, result)
            return result

        if self.plugin_manager is None:
            result = ActionResult(
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
                recovery_path="retry",
            )
            self._record_journal("engine_not_ready", request, result)
            return result

        if self._policy_requires_clarification(request):
            result = ActionResult(
                success=False,
                status="policy_blocked",
                plugin=request.plugin,
                action=request.action,
                goal_satisfied=False,
                data={},
                error="High-risk action needs explicit confirmation and clearer target.",
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
                ambiguity_flags=["policy_confirmation_required"],
                recovery_path="clarify_then_continue",
            )
            self._record_journal("policy_blocked", request, result)
            return result

        intent = self._compose_intent(request)
        raw_query = (
            request.params.get("_raw_query")
            if isinstance(request.params, dict)
            else None
        )
        max_attempts = max(1, min(4, int(request.retry_budget) + 1))
        plan_steps = self.planner.decompose(request)
        last_result: Optional[ActionResult] = None

        for attempt_idx in range(max_attempts):
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
                    "expected_outcome": request.expected_outcome,
                    "target_spec": request.target_spec,
                    "risk_level": request.risk_level,
                    "rollback_policy": request.rollback_policy,
                    "plan_steps": plan_steps,
                    "attempt": attempt_idx + 1,
                },
            )

            result_dict = tool_outcome.result or {}
            status = str(
                result_dict.get("status")
                or ("success" if result_dict.get("success") else "failed")
            )
            plugin = str(result_dict.get("plugin") or request.plugin or "unknown")
            action = str(result_dict.get("action") or request.action or "unknown")
            success = bool(result_dict.get("success", False))
            verified = bool(
                (result_dict.get("verification") or {}).get("accepted", False)
            )

            goal_report = self.goal_verifier.verify(request, result_dict).to_dict()
            goal_satisfied = bool(
                success and verified and goal_report.get("goal_satisfied", False)
            )

            state_updates = self._build_state_updates(
                request=request,
                tool_result=result_dict,
                verified=verified,
                goal_satisfied=goal_satisfied,
                handled=bool(tool_outcome.handled),
            )

            confidence = self._coerce_confidence(
                result_dict.get(
                    "confidence", request.confidence or (1.0 if success else 0.2)
                )
            )
            recovery_path = self.planner.recovery_path(
                retry_count=attempt_idx,
                retry_budget=request.retry_budget,
                partial_success=bool(goal_report.get("partial_success", False)),
            )

            last_result = ActionResult(
                success=success,
                status=status,
                plugin=plugin,
                action=action,
                goal_satisfied=goal_satisfied,
                data=(
                    result_dict.get("data")
                    if isinstance(result_dict.get("data"), dict)
                    else {}
                ),
                error=str(
                    result_dict.get("message") or result_dict.get("response") or ""
                )
                or None,
                retryable=bool(result_dict.get("retryable", not success)),
                confidence=confidence,
                side_effects=list(result_dict.get("side_effects") or []),
                state_updates=state_updates,
                verification_report=goal_report,
                ambiguity_flags=list(goal_report.get("ambiguity_flags") or []),
                recovery_path=recovery_path,
                retry_count=attempt_idx,
            )

            self._record_journal("attempt", request, last_result)

            should_retry = self._should_retry(request, last_result, attempt_idx)
            if should_retry:
                self._record_journal("retry", request, last_result)
                continue

            break

        if last_result is None:
            last_result = ActionResult(
                success=False,
                status="failed",
                plugin=request.plugin,
                action=request.action,
                goal_satisfied=False,
                data={},
                error="Action execution failed unexpectedly.",
                retryable=True,
                confidence=0.0,
                recovery_path="retry",
            )

        rollback_outcome = self._maybe_rollback(request, last_result)
        if rollback_outcome:
            last_result.state_updates["rollback"] = rollback_outcome
            last_result.recovery_path = (
                "rollback_success"
                if rollback_outcome.get("success", False)
                else "rollback_failed_escalate"
            )

        self.world_state_memory.update_from_action(request, last_result)
        return last_result

    def apply_state_updates(self, assistant: Any, result: ActionResult) -> None:
        updates = result.state_updates if isinstance(result.state_updates, dict) else {}
        if not updates:
            return

        try:
            if hasattr(assistant, "_internal_reasoning_state") and isinstance(
                assistant._internal_reasoning_state, dict
            ):
                assistant._internal_reasoning_state.update(updates)
                verification = updates.get("verification")
                if isinstance(verification, dict):
                    assistant._internal_reasoning_state["tool_verification"] = (
                        verification
                    )
                if isinstance(result.verification_report, dict):
                    assistant._internal_reasoning_state["goal_verification"] = dict(
                        result.verification_report
                    )
                if result.ambiguity_flags:
                    assistant._internal_reasoning_state["action_ambiguity"] = list(
                        result.ambiguity_flags
                    )
                if result.recovery_path:
                    assistant._internal_reasoning_state["action_recovery_path"] = (
                        result.recovery_path
                    )

            if (
                hasattr(assistant, "world_state_memory")
                and assistant.world_state_memory is not None
            ):
                assistant._internal_reasoning_state["world_state"] = (
                    assistant.world_state_memory.snapshot()
                )
            else:
                assistant._internal_reasoning_state["world_state"] = (
                    self.world_state_memory.snapshot()
                )
        except Exception:
            return

    def _normalize_request(self, request: ActionRequest) -> ActionRequest:
        request.goal = str(request.goal or "").strip()
        request.plugin = str(request.plugin or "unknown").strip().lower() or "unknown"
        request.action = str(request.action or "unknown").strip().lower() or "unknown"
        request.params = request.params or {}
        request.source_intent = str(request.source_intent or "").strip()
        request.confidence = self._coerce_confidence(request.confidence)
        request.expected_outcome = str(
            request.expected_outcome or request.goal or ""
        ).strip()
        request.target_spec = request.target_spec or {}
        request.risk_level = str(request.risk_level or "medium").strip().lower()
        request.retry_budget = max(0, min(3, int(request.retry_budget or 0)))
        request.rollback_policy = str(request.rollback_policy or "none").strip().lower()
        request.plan_steps = list(request.plan_steps or [])
        return request

    def _policy_requires_clarification(self, request: ActionRequest) -> bool:
        if request.risk_level not in {"high", "critical"}:
            return False

        if request.confidence >= 0.8 and self._has_explicit_approval(request):
            return False

        # Allow read-only actions with medium confidence.
        if request.action in {"read", "list", "search"} and request.confidence >= 0.6:
            return False
        return True

    def _has_explicit_approval(self, request: ActionRequest) -> bool:
        params = request.params or {}
        if bool(params.get("_approval_token")):
            return True
        raw_query = str(params.get("_raw_query") or "").lower()
        return any(
            token in raw_query
            for token in ("confirm", "approved", "go ahead", "yes do it")
        )

    def _should_retry(
        self, request: ActionRequest, result: ActionResult, attempt_idx: int
    ) -> bool:
        if attempt_idx >= request.retry_budget:
            return False
        if not result.retryable:
            return False
        if request.risk_level in {"high", "critical"}:
            return False
        if result.goal_satisfied:
            return False
        if result.ambiguity_flags:
            return False
        return True

    def _record_journal(
        self, event: str, request: ActionRequest, result: ActionResult
    ) -> None:
        self.execution_journal.record(
            {
                "event": event,
                "goal": request.goal,
                "plugin": request.plugin,
                "action": request.action,
                "risk_level": request.risk_level,
                "retry_budget": request.retry_budget,
                "rollback_policy": request.rollback_policy,
                "success": result.success,
                "status": result.status,
                "goal_satisfied": result.goal_satisfied,
                "confidence": result.confidence,
                "retry_count": result.retry_count,
                "recovery_path": result.recovery_path,
                "ambiguity_flags": list(result.ambiguity_flags or []),
            }
        )

    def _maybe_rollback(
        self, request: ActionRequest, result: ActionResult
    ) -> Optional[Dict[str, Any]]:
        policy = str(request.rollback_policy or "none").lower()
        if policy not in {"auto", "best_effort", "immediate"}:
            return None

        if result.goal_satisfied:
            return None

        if not result.side_effects and not result.verification_report.get(
            "partial_success", False
        ):
            return None

        preview_only = bool((request.params or {}).get("_rollback_preview_only", False))
        preview = self.rollback_executor.execute(
            plugin_manager=self.plugin_manager,
            request=request,
            result=result,
            dry_run=True,
        ).to_dict()

        if preview_only:
            self.execution_journal.record(
                {
                    "event": "rollback_preview",
                    "goal": request.goal,
                    "plugin": request.plugin,
                    "action": request.action,
                    "success": True,
                    "status": preview.get("status", "rollback_preview"),
                    "goal_satisfied": result.goal_satisfied,
                }
            )
            return preview

        if request.risk_level in {
            "high",
            "critical",
        } and not self._has_rollback_approval(request):
            blocked = {
                "attempted": False,
                "success": False,
                "status": "rollback_confirmation_required",
                "rollback_intent": preview.get("rollback_intent", ""),
                "details": {
                    "reason": "high_risk_requires_operator_approval",
                    "preview": preview,
                },
            }
            self.execution_journal.record(
                {
                    "event": "rollback_blocked",
                    "goal": request.goal,
                    "plugin": request.plugin,
                    "action": request.action,
                    "success": False,
                    "status": blocked.get("status"),
                    "goal_satisfied": result.goal_satisfied,
                }
            )
            return blocked

        outcome = self.rollback_executor.execute(
            plugin_manager=self.plugin_manager,
            request=request,
            result=result,
        ).to_dict()

        self.execution_journal.record(
            {
                "event": "rollback",
                "goal": request.goal,
                "plugin": request.plugin,
                "action": request.action,
                "success": bool(outcome.get("success", False)),
                "status": outcome.get("status", "rollback_unknown"),
                "goal_satisfied": result.goal_satisfied,
            }
        )
        return outcome

    def _has_rollback_approval(self, request: ActionRequest) -> bool:
        params = request.params or {}
        if bool(params.get("_rollback_approval_token", False)):
            return True
        raw_query = str(params.get("_raw_query") or "").lower()
        return any(
            token in raw_query
            for token in ("confirm rollback", "approve rollback", "yes rollback")
        )

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
                "expected_outcome": request.expected_outcome,
                "target_spec": dict(request.target_spec or {}),
                "risk_level": request.risk_level,
                "retry_budget": request.retry_budget,
                "rollback_policy": request.rollback_policy,
            },
            "last_action_result": {
                "handled": handled,
                "success": bool(tool_result.get("success", False)),
                "status": str(tool_result.get("status", "unknown")),
                "goal_satisfied": goal_satisfied,
                "plugin": str(tool_result.get("plugin") or request.plugin or "unknown"),
                "action": str(tool_result.get("action") or request.action or "unknown"),
                "verified": bool(verified),
            },
            "goal_satisfied": goal_satisfied,
        }


_unified_action_engine: Optional[UnifiedActionEngine] = None


def get_unified_action_engine() -> UnifiedActionEngine:
    global _unified_action_engine
    if _unified_action_engine is None:
        _unified_action_engine = UnifiedActionEngine()
    return _unified_action_engine
