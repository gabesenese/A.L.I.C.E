"""Post-tool verification for plugin execution results."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List

_NEGATIVE_MARKERS = (
    "failed",
    "error",
    "could not",
    "couldn't",
    "unable",
    "not available",
)

_INTENT_EXPECTED_DATA_KEYS = {
    "weather": ("temperature", "forecast", "condition"),
    "notes": ("notes", "count", "note_title", "content"),
    "email": ("emails", "message_id", "subject", "from"),
    "calendar": ("events", "event_id", "date"),
    "file_operations": ("path", "files", "content"),
    "system": ("command", "stdout", "stderr"),
}

_ALLOWED_STATUS = {"ok", "error", "partial", "skipped"}


@dataclass
class ToolVerificationResult:
    accepted: bool
    confidence: float
    tool_succeeded: bool
    goal_satisfied: bool
    issues: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ToolResultVerifier:
    def verify(
        self, *, intent: str, user_input: str, plugin_result: Dict[str, Any]
    ) -> ToolVerificationResult:
        issues: List[str] = []

        if not isinstance(plugin_result, dict):
            return ToolVerificationResult(
                accepted=False,
                confidence=0.0,
                tool_succeeded=False,
                goal_satisfied=False,
                issues=["plugin_result is not a dict"],
            )

        plugin_name = str(plugin_result.get("plugin") or "")
        action = str(plugin_result.get("action") or "")
        response = str(plugin_result.get("response") or "")
        data = plugin_result.get("data")
        success = bool(plugin_result.get("success", False))
        status = str(plugin_result.get("status") or "").strip().lower()
        diagnostics = plugin_result.get("diagnostics")
        confidence_field = plugin_result.get("confidence")

        if not plugin_name:
            issues.append("missing plugin name")
        if not action:
            issues.append("missing plugin action")
        if not response and not data:
            issues.append("empty plugin output")
        if (
            response
            and any(m in response.lower() for m in _NEGATIVE_MARKERS)
            and success
        ):
            issues.append("success flag contradicts response text")
        if status and status not in _ALLOWED_STATUS:
            issues.append("invalid status field")
        if status == "error" and success:
            issues.append("success flag contradicts status=error")
        if status == "ok" and not success:
            issues.append("status=ok contradicts success flag")

        if confidence_field is not None:
            try:
                c_val = float(confidence_field)
                if c_val < 0.0 or c_val > 1.0:
                    issues.append("confidence field out of range")
            except Exception:
                issues.append("confidence field is not numeric")

        if diagnostics is not None and not isinstance(diagnostics, dict):
            issues.append("diagnostics field must be a dict when provided")
        intent_prefix = (intent or "").split(":", 1)[0]

        if (
            success
            and intent_prefix == "weather"
            and "weather" not in plugin_name.lower()
        ):
            issues.append("unexpected plugin for weather intent")

        if success and isinstance(data, dict):
            expected = _INTENT_EXPECTED_DATA_KEYS.get(intent_prefix, ())
            if expected and not any(k in data for k in expected):
                issues.append("tool output missing expected data fields for intent")

            if intent_prefix == "notes" and "count" in data and "notes" in data:
                notes = data.get("notes")
                count = data.get("count")
                if (
                    isinstance(notes, list)
                    and isinstance(count, int)
                    and count < len(notes)
                ):
                    issues.append("notes count is inconsistent with listed notes")

        user_lower = (user_input or "").lower()
        if (
            success
            and any(w in user_lower for w in ("count", "how many", "list", "show"))
            and not data
        ):
            issues.append(
                "goal likely unsatisfied: user requested structured output but tool data is empty"
            )

        if success and intent_prefix in {"system", "file_operations"} and not data:
            issues.append("high-impact action lacks structured execution data")

        critical_issue = any(
            phrase in issue
            for issue in issues
            for phrase in (
                "contradicts",
                "missing expected data fields",
                "goal likely unsatisfied",
                "invalid status field",
                "out of range",
                "lacks structured execution data",
            )
        )

        confidence = 1.0
        confidence -= 0.25 * len(issues)
        if not success:
            confidence -= 0.25
        confidence = max(0.0, min(1.0, confidence))

        if critical_issue:
            confidence = min(confidence, 0.35)

        goal_satisfied = bool(success and confidence >= 0.7)
        accepted = bool(confidence >= 0.6 and goal_satisfied)

        return ToolVerificationResult(
            accepted=accepted,
            confidence=confidence,
            tool_succeeded=success,
            goal_satisfied=goal_satisfied,
            issues=issues,
        )


_tool_result_verifier: ToolResultVerifier | None = None


def get_tool_result_verifier() -> ToolResultVerifier:
    global _tool_result_verifier
    if _tool_result_verifier is None:
        _tool_result_verifier = ToolResultVerifier()
    return _tool_result_verifier
