"""Lightweight tool result verification for Foundation 3 execution safety."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List


@dataclass
class VerificationResult:
    accepted: bool
    confidence: float
    issues: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ToolResultVerifier:
    """Small schema and consistency verifier for normalized tool results."""

    def verify(self, result: Dict[str, Any]) -> VerificationResult:
        issues: List[str] = []

        if not isinstance(result, dict):
            return VerificationResult(False, 0.0, ["result_not_dict"])

        required = [
            "success",
            "status",
            "data",
            "message",
            "confidence",
            "retryable",
            "side_effects",
        ]
        for key in required:
            if key not in result:
                issues.append(f"missing_{key}")

        if not isinstance(result.get("success"), bool):
            issues.append("success_not_bool")

        status = str(result.get("status", "")).strip().lower()
        if status not in {
            "success",
            "failed",
            "partial",
            "not_handled",
            "verification_failed",
        }:
            issues.append("invalid_status")

        conf = result.get("confidence")
        try:
            conf_val = float(conf)
            if conf_val < 0.0 or conf_val > 1.0:
                issues.append("confidence_out_of_range")
        except Exception:
            issues.append("confidence_not_numeric")
            conf_val = 0.0

        if not isinstance(result.get("data"), dict):
            issues.append("data_not_dict")

        if not isinstance(result.get("retryable"), bool):
            issues.append("retryable_not_bool")

        if not isinstance(result.get("side_effects"), list):
            issues.append("side_effects_not_list")

        if result.get("success") and status in {
            "failed",
            "verification_failed",
            "not_handled",
        }:
            issues.append("success_status_conflict")

        if (not result.get("success")) and status == "success":
            issues.append("failure_status_conflict")

        accepted = len(issues) == 0
        confidence = conf_val if accepted else min(conf_val, 0.4)

        return VerificationResult(
            accepted=accepted, confidence=confidence, issues=issues
        )


_tool_result_verifier: ToolResultVerifier | None = None


def get_tool_result_verifier() -> ToolResultVerifier:
    global _tool_result_verifier
    if _tool_result_verifier is None:
        _tool_result_verifier = ToolResultVerifier()
    return _tool_result_verifier
