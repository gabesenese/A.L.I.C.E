"""Verification helpers for planner/task execution outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List


@dataclass
class ExecutionVerification:
    accepted: bool
    confidence: float
    issues: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExecutionVerifier:
    """Checks that planned execution produced a usable answer payload."""

    def verify_task_result(
        self,
        *,
        intent: str,
        result: Any,
        all_results: Dict[str, Any] | Dict[int, Any] | None,
    ) -> ExecutionVerification:
        issues: List[str] = []
        normalized_intent = str(intent or "").lower().strip()
        result_text = str(result or "").strip()
        steps = all_results if isinstance(all_results, dict) else {}

        if not result_text:
            issues.append("empty_result")

        if normalized_intent in {"study_topic", "learning:study_topic"}:
            non_empty_steps = 0
            for value in steps.values():
                if str(value or "").strip():
                    non_empty_steps += 1
            if non_empty_steps < 2:
                issues.append("insufficient_study_steps")

        if normalized_intent.startswith("conversation:") and len(result_text) < 12:
            issues.append("conversation_result_too_short")

        confidence = 1.0 - (0.25 * len(issues))
        confidence = max(0.0, min(1.0, confidence))
        accepted = len(issues) == 0

        return ExecutionVerification(
            accepted=accepted,
            confidence=confidence,
            issues=issues,
        )


_execution_verifier: ExecutionVerifier | None = None


def get_execution_verifier() -> ExecutionVerifier:
    global _execution_verifier
    if _execution_verifier is None:
        _execution_verifier = ExecutionVerifier()
    return _execution_verifier
