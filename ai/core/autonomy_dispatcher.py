"""Tiny rule-based autonomy dispatcher for bounded initiative."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

OUTCOME_ACT_AUTOMATICALLY = "act_automatically"
OUTCOME_ASK_USER = "ask_user"
OUTCOME_LOG_SILENTLY = "log_silently"
OUTCOME_ESCALATE_AND_STOP = "escalate_and_stop"


@dataclass
class AutonomyDispatchDecision:
    trigger_reason: str
    severity: str
    outcome: str
    affected_goal_id: str
    next_goal_action: str
    should_notify_user: bool
    operator_message: str
    pause_autonomy: bool
    escalate: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger_reason": self.trigger_reason,
            "severity": self.severity,
            "outcome": self.outcome,
            "affected_goal_id": self.affected_goal_id,
            "next_goal_action": self.next_goal_action,
            "should_notify_user": self.should_notify_user,
            "operator_message": self.operator_message,
            "pause_autonomy": self.pause_autonomy,
            "escalate": self.escalate,
        }


class TinyAutonomyDispatcher:
    """Maps trusted trigger reasons to bounded policy outcomes."""

    _TRUSTED_TRIGGER_RULES: Dict[str, Dict[str, str]] = {
        "pending_approvals_waiting": {
            "outcome": OUTCOME_ASK_USER,
            "next_goal_action": "clarify",
            "message": "I have a pending approval decision for an active operation. Reply with operator approve <approval_id> or operator reject <approval_id>.",
        },
        "active_goals_with_unresolved_ambiguity": {
            "outcome": OUTCOME_ASK_USER,
            "next_goal_action": "clarify",
            "message": "I found an unresolved ambiguity tied to the active goal. Which exact target should I use?",
        },
        "failure_rate_spike": {
            "outcome": OUTCOME_ESCALATE_AND_STOP,
            "next_goal_action": "pause",
            "message": "I paused autonomous execution due to repeated failures and need operator direction.",
        },
        "rollback_success": {
            "outcome": OUTCOME_ACT_AUTOMATICALLY,
            "next_goal_action": "continue",
            "message": "",
        },
        "rollback_failure": {
            "outcome": OUTCOME_ESCALATE_AND_STOP,
            "next_goal_action": "pause",
            "message": "Rollback did not complete safely. I paused autonomous execution and need operator guidance.",
        },
    }

    def can_act_without_user(self, trigger_reason: str, severity: str) -> str:
        """Return one of four policy outcomes for a trigger."""
        reason = str(trigger_reason or "").strip().lower()
        sev = str(severity or "").strip().lower()

        rule = self._TRUSTED_TRIGGER_RULES.get(reason)
        if not rule:
            return OUTCOME_LOG_SILENTLY

        # High severity always escalates and stops in this bounded policy.
        if sev == "high":
            return OUTCOME_ESCALATE_AND_STOP

        return str(rule.get("outcome") or OUTCOME_LOG_SILENTLY)

    def dispatch(
        self,
        *,
        event: Dict[str, Any],
        goal_summary: Optional[Dict[str, Any]] = None,
        active_goal_id: str = "",
        ask_history: Optional[Dict[str, float]] = None,
    ) -> AutonomyDispatchDecision:
        reason = str((event or {}).get("reason") or "").strip().lower()
        severity = str((event or {}).get("severity") or "low").strip().lower()
        outcome = self.can_act_without_user(reason, severity)
        rule = self._TRUSTED_TRIGGER_RULES.get(reason, {})

        goal_id = self._resolve_goal_id(
            active_goal_id=active_goal_id, goal_summary=goal_summary
        )
        next_goal_action = str(
            rule.get("next_goal_action")
            or self._default_goal_action_for_outcome(outcome)
        )
        message = str(rule.get("message") or "")

        should_notify_user = self._should_notify(
            reason=reason,
            severity=severity,
            outcome=outcome,
            goal_id=goal_id,
            ask_history=ask_history,
        )

        return AutonomyDispatchDecision(
            trigger_reason=reason,
            severity=severity,
            outcome=outcome,
            affected_goal_id=goal_id,
            next_goal_action=next_goal_action,
            should_notify_user=should_notify_user,
            operator_message=message if should_notify_user else "",
            pause_autonomy=(outcome == OUTCOME_ESCALATE_AND_STOP),
            escalate=(outcome == OUTCOME_ESCALATE_AND_STOP),
        )

    def _resolve_goal_id(
        self,
        *,
        active_goal_id: str,
        goal_summary: Optional[Dict[str, Any]],
    ) -> str:
        if active_goal_id:
            return str(active_goal_id)

        summary = goal_summary or {}
        goals = summary.get("goals") or []
        if goals and isinstance(goals[0], dict):
            return str(goals[0].get("goal_id") or "")
        return ""

    def _default_goal_action_for_outcome(self, outcome: str) -> str:
        if outcome == OUTCOME_ACT_AUTOMATICALLY:
            return "continue"
        if outcome == OUTCOME_ASK_USER:
            return "clarify"
        if outcome == OUTCOME_ESCALATE_AND_STOP:
            return "pause"
        return "continue"

    def _should_notify(
        self,
        *,
        reason: str,
        severity: str,
        outcome: str,
        goal_id: str,
        ask_history: Optional[Dict[str, float]],
    ) -> bool:
        # Proactive messaging policy:
        # low -> log only, medium -> ask once briefly, high -> pause + escalate immediately.
        if severity == "low":
            return False

        if severity == "high":
            return outcome == OUTCOME_ESCALATE_AND_STOP

        if severity != "medium":
            return False

        if outcome != OUTCOME_ASK_USER:
            return False

        key = f"{reason}:{goal_id or 'none'}"
        if ask_history is None:
            return True
        if key in ask_history:
            return False
        ask_history[key] = 1.0
        return True
