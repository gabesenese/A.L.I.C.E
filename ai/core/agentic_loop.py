"""Core autonomous-perception-reasoning-execution loop for A.L.I.C.E."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ai.core.foundation_layers import FoundationLayers
from ai.core.live_state_service import LiveStateService, get_live_state_service
from ai.core.unified_action_engine import ActionRequest, UnifiedActionEngine


@dataclass
class AgentCycleInput:
    user_input: str
    intent: str
    confidence: float
    entities: Dict[str, Any] = field(default_factory=dict)
    long_horizon_goal: str = ""
    world_state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentCycleResult:
    perception: Dict[str, Any]
    reasoning: Dict[str, Any]
    goal_plan: Dict[str, Any]
    decision: Dict[str, Any]
    execution: Dict[str, Any]
    learning: Dict[str, Any]
    orchestration: Dict[str, Any]


class AgenticLoop:
    """
    Implements the canonical agentic flow:
    perception -> reasoning -> goal setting -> decision -> execution -> learning -> orchestration.
    """

    def __init__(
        self,
        *,
        foundation_layers: Optional[FoundationLayers] = None,
        live_state: Optional[LiveStateService] = None,
        action_engine: Optional[UnifiedActionEngine] = None,
    ) -> None:
        self.foundation_layers = foundation_layers or FoundationLayers()
        self.live_state = live_state or get_live_state_service()
        self.action_engine = action_engine or UnifiedActionEngine()

    def run_cycle(self, payload: AgentCycleInput) -> AgentCycleResult:
        perception = self._perception(payload)
        reasoning = self._reasoning(payload, perception)
        goal_plan = self._goal_setting(payload, reasoning)
        decision = self._decision(payload, goal_plan)
        execution = self._execution(payload, decision)
        learning = self._learning(payload, execution)
        orchestration = self._orchestration(payload, goal_plan, execution)
        return AgentCycleResult(
            perception=perception,
            reasoning=reasoning,
            goal_plan=goal_plan,
            decision=decision,
            execution=execution,
            learning=learning,
            orchestration=orchestration,
        )

    def _perception(self, payload: AgentCycleInput) -> Dict[str, Any]:
        world = dict(payload.world_state or {})
        return {
            "intent": payload.intent,
            "confidence": float(payload.confidence or 0.0),
            "entities": dict(payload.entities or {}),
            "world": world,
        }

    def _reasoning(self, payload: AgentCycleInput, perception: Dict[str, Any]) -> Dict[str, Any]:
        policy = self.foundation_layers.clarification_policy(
            plugin_scores={payload.intent.split(":", 1)[0]: max(0.0, payload.confidence)},
            confidence=float(payload.confidence or 0.0),
        )
        return {
            "needs_clarification": bool(policy.get("needs_clarification")),
            "clarification_prompt": policy.get("prompt", ""),
            "risk_level": self._infer_risk(payload.intent, payload.entities),
        }

    def _goal_setting(self, payload: AgentCycleInput, reasoning: Dict[str, Any]) -> Dict[str, Any]:
        goal = str(payload.long_horizon_goal or payload.user_input or "").strip()
        milestones = []
        if goal:
            milestones = [
                "confirm_scope",
                "execute_primary_action",
                "verify_outcome",
            ]
        return {
            "goal": goal,
            "milestones": milestones,
            "autonomy_mode": "supervised" if reasoning["needs_clarification"] else "autonomous",
        }

    def _decision(self, payload: AgentCycleInput, goal_plan: Dict[str, Any]) -> Dict[str, Any]:
        plugin, action = self._plugin_action_from_intent(payload.intent)
        return {
            "plugin": plugin,
            "action": action,
            "should_execute": bool(plugin and action and goal_plan.get("goal")),
        }

    def _execution(self, payload: AgentCycleInput, decision: Dict[str, Any]) -> Dict[str, Any]:
        if not decision.get("should_execute"):
            return {"status": "skipped", "reason": "insufficient_goal_or_decision"}

        request = ActionRequest(
            goal=payload.long_horizon_goal or payload.user_input,
            plugin=decision["plugin"],
            action=decision["action"],
            params=dict(payload.entities or {}),
            source_intent=payload.intent,
            confidence=float(payload.confidence or 0.0),
            risk_level=self._infer_risk(payload.intent, payload.entities),
            target_spec={"target": (payload.entities or {}).get("target")},
            retry_budget=1,
        )
        result = self.action_engine.execute(request)
        return {
            "status": result.status,
            "success": bool(result.success),
            "goal_satisfied": bool(result.goal_satisfied),
            "recovery_path": result.recovery_path,
            "state_updates": dict(result.state_updates or {}),
        }

    def _learning(self, payload: AgentCycleInput, execution: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "feedback_signal": "positive" if execution.get("success") else "negative",
            "adjustment_hint": (
                "lower_clarification_threshold"
                if execution.get("status") == "policy_blocked"
                else "none"
            ),
        }

    def _orchestration(
        self,
        payload: AgentCycleInput,
        goal_plan: Dict[str, Any],
        execution: Dict[str, Any],
    ) -> Dict[str, Any]:
        milestones = list(goal_plan.get("milestones") or [])
        next_milestone = None
        if milestones:
            next_milestone = milestones[1] if execution.get("success") and len(milestones) > 1 else milestones[0]
        return {
            "goal": goal_plan.get("goal", ""),
            "next_milestone": next_milestone,
            "execution_status": execution.get("status"),
        }

    @staticmethod
    def _plugin_action_from_intent(intent: str) -> tuple[str, str]:
        raw = str(intent or "")
        if ":" not in raw:
            return "", ""
        plugin, action = raw.split(":", 1)
        return plugin.strip().lower(), action.strip().lower()

    @staticmethod
    def _infer_risk(intent: str, entities: Dict[str, Any]) -> str:
        action = str(intent or "").split(":", 1)[-1].lower()
        destructive = {"delete", "remove", "shutdown", "reboot", "format"}
        if action in destructive:
            return "high"
        if entities and entities.get("target") in (None, "", "unknown"):
            return "medium"
        return "low"
