"""Goal-aware verification for unified action execution."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

from ai.core.entity_registry import get_entity_registry


@dataclass
class GoalVerificationReport:
    accepted: bool
    goal_satisfied: bool
    partial_success: bool
    tool_succeeded: bool
    target_matched: bool
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

        tool_succeeded = bool(success or status in {"success", "partial"})
        target_score = self._target_match_score(request, data)
        target_matched = bool(target_score >= 0.6)
        if target_score < 0.5:
            ambiguity_flags.append("target_mismatch")
            issues.append("target_match_low")

        required_target_actions = {"read", "delete", "update", "append", "open"}
        action = str(getattr(request, "action", "") or "").strip().lower()
        if action in required_target_actions and not self._has_target(request, data):
            ambiguity_flags.append("target_ambiguous")
            issues.append("missing_target")

        domain_ok, domain_issues = self._domain_specific_checks(
            request=request,
            action=action,
            data=data,
            tool_result=tool_result,
        )
        issues.extend(domain_issues)
        if not domain_ok:
            ambiguity_flags.append("domain_postcondition_unmet")

        partial_success = bool(
            status == "partial"
            or (tool_succeeded and (not target_matched or not domain_ok))
            or (not success and bool(data))
        )
        goal_satisfied = bool(
            success and target_matched and domain_ok and not ambiguity_flags
        )
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
        elif ambiguity_flags:
            next_action = "clarify_then_continue"
        elif partial_success:
            next_action = "verify_target_then_continue"
        elif (tool_result or {}).get("retryable", False):
            next_action = "retry"
        else:
            next_action = "escalate"

        return GoalVerificationReport(
            accepted=accepted,
            goal_satisfied=goal_satisfied,
            partial_success=partial_success,
            tool_succeeded=tool_succeeded,
            target_matched=target_matched,
            target_match_score=target_score,
            ambiguity_flags=ambiguity_flags,
            issues=issues,
            recommended_next_action=next_action,
        )

    def _domain_specific_checks(
        self,
        *,
        request: Any,
        action: str,
        data: Dict[str, Any],
        tool_result: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        plugin = str(getattr(request, "plugin", "") or "").strip().lower()
        issues: List[str] = []
        tool_succeeded = bool(
            bool((tool_result or {}).get("success", False))
            or str((tool_result or {}).get("status", "")).strip().lower()
            in {"success", "partial"}
        )
        if not tool_succeeded:
            return True, []

        if plugin == "notes" and action in {"create", "read", "update", "delete"}:
            if action == "create" and not any(
                data.get(k) for k in ("note_id", "id", "title", "note_title")
            ):
                issues.append("notes_create_missing_identifier")
            if action == "read" and not any(
                data.get(k)
                for k in (
                    "content",
                    "body",
                    "text",
                    "note",
                    "note_id",
                    "id",
                    "title",
                    "note_title",
                    "target",
                )
            ):
                issues.append("notes_read_missing_content")

        if plugin == "email" and action in {"send", "draft", "read"}:
            if action == "send" and not any(
                data.get(k) for k in ("message_id", "id", "sent", "status")
            ):
                issues.append("email_send_unconfirmed")
            if action == "read" and not any(
                data.get(k) for k in ("subject", "messages", "body", "content")
            ):
                issues.append("email_read_missing_payload")

        if plugin == "calendar" and action in {"create", "update", "read", "list"}:
            if action in {"create", "update"} and not any(
                data.get(k) for k in ("event_id", "id", "event", "title")
            ):
                issues.append("calendar_write_missing_event_reference")
            if action in {"read", "list"} and not any(
                data.get(k) for k in ("events", "event", "items")
            ):
                issues.append("calendar_read_missing_events")

        if plugin in {"file_operations", "files", "file"} and action in {
            "read",
            "write",
            "delete",
            "update",
            "append",
        }:
            if not any(data.get(k) for k in ("path", "file", "filename", "target")):
                issues.append("file_operation_missing_path_echo")

        if tool_succeeded and not issues:
            return True, []
        return len(issues) == 0, issues

    def _target_match_score(self, request: Any, data: Dict[str, Any]) -> float:
        target_spec = getattr(request, "target_spec", {}) or {}
        if not isinstance(target_spec, dict) or not target_spec:
            return 0.7

        registry = None
        try:
            registry = get_entity_registry()
        except Exception:
            registry = None

        candidates = {
            str(v).strip().lower()
            for v in data.values()
            if isinstance(v, (str, int, float)) and str(v).strip()
        }
        if registry and candidates:
            expanded_candidates = set(candidates)
            for cand in list(candidates):
                resolved = str(registry.resolve_label(cand) or "").strip().lower()
                if resolved:
                    expanded_candidates.add(resolved)
            candidates = expanded_candidates
        if not candidates:
            return 0.4

        desired = {
            str(v).strip().lower()
            for v in target_spec.values()
            if isinstance(v, (str, int, float)) and str(v).strip()
        }
        if registry and desired:
            expanded_desired = set(desired)
            for want in list(desired):
                resolved = str(registry.resolve_label(want) or "").strip().lower()
                if resolved:
                    expanded_desired.add(resolved)
            desired = expanded_desired
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
