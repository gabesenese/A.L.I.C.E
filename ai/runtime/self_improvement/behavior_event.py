from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4


FAILURE_KINDS = {
    "routing",
    "memory",
    "greeting_tone",
    "continuity_claim",
    "local_execution",
    "tool_execution",
    "response_grounding",
    "passive_response",
    "runtime_error",
    "test_failure",
    "project_state",
    "unknown",
}

SOURCES = {
    "user_correction",
    "routing_failure_log",
    "verification_failure",
    "local_execution_error",
    "test_failure",
    "manual_audit",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BehaviorEvent:
    event_id: str
    timestamp: str
    source: str
    user_input: str
    alice_response: str
    route: str
    intent: str
    trace_id: str
    failure_kind: str
    symptom: str
    expected_behavior: str
    actual_behavior: str
    severity: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    related_files: List[str] = field(default_factory=list)
    session_id: str = ""
    user_id: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_behavior_event(
    *,
    source: str,
    user_input: str = "",
    alice_response: str = "",
    route: str = "",
    intent: str = "",
    trace_id: str = "",
    failure_kind: str = "unknown",
    symptom: str = "",
    expected_behavior: str = "",
    actual_behavior: str = "",
    severity: str = "medium",
    evidence: Dict[str, Any] | None = None,
    related_files: List[str] | None = None,
    session_id: str = "",
    user_id: str = "default",
) -> BehaviorEvent:
    normalized_source = source if source in SOURCES else "manual_audit"
    kind = failure_kind if failure_kind in FAILURE_KINDS else "unknown"
    return BehaviorEvent(
        event_id=str(uuid4()),
        timestamp=_now_iso(),
        source=normalized_source,
        user_input=str(user_input or ""),
        alice_response=str(alice_response or ""),
        route=str(route or ""),
        intent=str(intent or ""),
        trace_id=str(trace_id or ""),
        failure_kind=kind,
        symptom=str(symptom or ""),
        expected_behavior=str(expected_behavior or ""),
        actual_behavior=str(actual_behavior or ""),
        severity=str(severity or "medium"),
        evidence=dict(evidence or {}),
        related_files=list(related_files or []),
        session_id=str(session_id or ""),
        user_id=str(user_id or "default"),
    )
