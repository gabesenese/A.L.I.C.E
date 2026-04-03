"""Goal-aware verification for unified action execution."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass
class GoalVerificationReport:
    accepted: bool
    goal_satisfied: bool
    partial_success: bool
    target_match_score: float
    ambiguity_flags: List[str]
    issues: List[str]
    recommended_next_action: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GoalActionVerifier:
    """Checks whether an action result satisfied the requested goal and target."""

    def verify(
        self, request: Any, tool_result: Dict[str, Any]
    ) -> GoalVerificationReport:
        issues: List[str] = []
        ambiguity_flags: List[str] = []

        success = bool((tool_result or {}).get("success", False))
        status = str((tool_result or {}).get("status", "")).strip().lower()
        data = (
            (tool_result or {}).get("data")
            if isinstance((tool_result or {}).get("data"), dict)
            else {}
        )

        target_score = self._target_match_score(request, data)
        if target_score < 0.5:
            ambiguity_flags.append("target_mismatch")
            issues.append("target_match_low")

        required_target_actions = {"read", "delete", "update", "append"}
        action = str(getattr(request, "action", "") or "").strip().lower()
        if action in required_target_actions and not self._has_target(request, data):
            ambiguity_flags.append("target_ambiguous")
            issues.append("missing_target")

        partial_success = status == "partial" or (not success and bool(data))
        goal_satisfied = bool(success and target_score >= 0.5 and not ambiguity_flags)
        accepted = bool(
            (
                status
                in {
                    "success",
                    "partial",
                    "failed",
                    "verification_failed",
                    "not_handled",
                }
            )
            and target_score >= 0.3
        )

        if goal_satisfied:
            next_action = "continue"
        elif partial_success:
            next_action = "clarify_then_continue"
        elif (tool_result or {}).get("retryable", False):
            next_action = "retry"
        else:
            next_action = "escalate"

        return GoalVerificationReport(
            accepted=accepted,
            goal_satisfied=goal_satisfied,
            partial_success=partial_success,
            target_match_score=target_score,
            ambiguity_flags=ambiguity_flags,
            issues=issues,
            recommended_next_action=next_action,
        )

    def _target_match_score(self, request: Any, data: Dict[str, Any]) -> float:
        target_spec = getattr(request, "target_spec", {}) or {}
        if not isinstance(target_spec, dict) or not target_spec:
            return 0.7

        candidates = {
            str(v).strip().lower()
            for v in data.values()
            if isinstance(v, (str, int, float)) and str(v).strip()
        }
        if not candidates:
            return 0.4

        desired = {
            str(v).strip().lower()
            for v in target_spec.values()
            if isinstance(v, (str, int, float)) and str(v).strip()
        }
        if not desired:
            return 0.7

        matches = 0
        for needle in desired:
            if any(needle in got or got in needle for got in candidates):
                matches += 1

        return matches / max(1, len(desired))

    def _has_target(self, request: Any, data: Dict[str, Any]) -> bool:
        target_spec = getattr(request, "target_spec", {}) or {}
        if isinstance(target_spec, dict) and target_spec:
            return True

        params = getattr(request, "params", {}) or {}
        if any(
            params.get(k)
            for k in ("target", "path", "filename", "note_id", "title", "note_title")
        ):
            return True

        if any(
            data.get(k)
            for k in ("target", "path", "filename", "note_id", "title", "note_title")
        ):
            return True

        return False


_goal_action_verifier: GoalActionVerifier | None = None


def get_goal_action_verifier() -> GoalActionVerifier:
    global _goal_action_verifier
    if _goal_action_verifier is None:
        _goal_action_verifier = GoalActionVerifier()
    return _goal_action_verifier
