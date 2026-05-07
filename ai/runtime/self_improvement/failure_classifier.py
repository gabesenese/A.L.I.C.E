from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import List

from ai.runtime.self_improvement.behavior_event import BehaviorEvent


@dataclass
class FailureClassification:
    failure_kind: str
    confidence: float
    signals: List[str] = field(default_factory=list)
    likely_layer: str = ""
    likely_files: List[str] = field(default_factory=list)
    severity: str = "medium"
    reason: str = ""

    def to_dict(self):
        return asdict(self)


def classify_failure(event: BehaviorEvent) -> FailureClassification:
    text = " ".join(
        [
            event.symptom,
            event.actual_behavior,
            event.expected_behavior,
            str(event.evidence or {}),
        ]
    ).lower()
    signals: List[str] = []
    likely_files: List[str] = []
    kind = "unknown"
    layer = "runtime"
    confidence = 0.45
    reason = "insufficient specific signals"

    def set_class(
        k: str, files: List[str], rsn: str, conf: float, lyr: str, sig: str
    ) -> None:
        nonlocal kind, likely_files, reason, confidence, layer
        kind = k
        likely_files = files
        reason = rsn
        confidence = conf
        layer = lyr
        signals.append(sig)

    if any(s in text for s in ("wrong route", "misroute", "clarify", "intent", "file_operations")):
        set_class(
            "routing",
            [
                "ai/core/routing/route_arbiter.py",
                "ai/core/routing/evidence_contracts.py",
                "ai/runtime/boundaries/boundary_factory.py",
            ],
            "routing signals detected",
            0.86,
            "routing",
            "routing_keywords",
        )
    elif any(s in text for s in ("remember", "old memory", "fake memory")):
        set_class(
            "memory",
            [
                "ai/runtime/continuity_claim_guard.py",
                "ai/memory/project_memory.py",
                "ai/memory/personal_memory.py",
                "ai/memory/memory_extractor.py",
            ],
            "memory consistency signals detected",
            0.82,
            "memory",
            "memory_keywords",
        )
    elif any(s in text for s in ("machine learning last time", "unsupported claim", "last time", "we were discussing")):
        set_class(
            "continuity_claim",
            [
                "ai/runtime/continuity_claim_guard.py",
                "ai/runtime/greeting_surface_policy.py",
            ],
            "unsupported continuity claim signals detected",
            0.9,
            "response_guard",
            "continuity_keywords",
        )
    elif any(s in text for s in ("greeting", "dry", "assistant-like", "forced", "bland")):
        set_class(
            "greeting_tone",
            [
                "ai/runtime/greeting_surface_policy.py",
                "tests/golden/test_greeting_memory_grounding.py",
            ],
            "greeting tone overcorrection signals detected",
            0.84,
            "surface_policy",
            "greeting_keywords",
        )
    elif any(s in text for s in ("target not found", "could not find file", "analyze file", "list files")):
        set_class(
            "local_execution",
            [
                "ai/runtime/local_actions/file_resolver.py",
                "ai/runtime/local_actions/local_action_executor.py",
                "ai/runtime/local_actions/code_analyzer.py",
            ],
            "local execution/file resolution signals detected",
            0.83,
            "local_execution",
            "local_execution_keywords",
        )
    elif any(s in text for s in ("fake", "not grounded", "claimed it inspected")):
        set_class(
            "response_grounding",
            [
                "ai/runtime/continuity_claim_guard.py",
                "ai/runtime/response_momentum_policy.py",
                "ai/runtime/verification_adapter.py",
            ],
            "response grounding signals detected",
            0.79,
            "response",
            "grounding_keywords",
        )
    elif any(s in text for s in ("too passive", "let me know", "assistant")):
        set_class(
            "passive_response",
            [
                "ai/runtime/response_momentum_policy.py",
                "ai/runtime/next_step_policy.py",
            ],
            "passive response style signals detected",
            0.78,
            "response",
            "passive_keywords",
        )
    elif any(s in text for s in ("tool_failed", "tool execution", "plugin failed")):
        set_class(
            "tool_execution",
            [
                "ai/runtime/tool_execution_adapter.py",
                "ai/runtime/contract_pipeline.py",
            ],
            "tool execution failure signals detected",
            0.81,
            "tools",
            "tool_execution_keywords",
        )
    elif any(s in text for s in ("test failed", "pytest", "assertionerror")):
        set_class(
            "test_failure",
            list(event.related_files or []),
            "test failure signals detected",
            0.8,
            "tests",
            "test_failure_keywords",
        )
    elif any(s in text for s in ("project state", "objective not set", "what are we fixing")):
        set_class(
            "project_state",
            [
                "ai/memory/project_memory.py",
                "ai/runtime/operator_state.py",
                "ai/runtime/next_step_policy.py",
            ],
            "project state continuity signals detected",
            0.78,
            "state",
            "project_state_keywords",
        )
    elif any(s in text for s in ("traceback", "exception", "attributeerror", "runtimeerror")):
        tb = str(event.evidence.get("traceback") or "")
        tb_files = re.findall(r"([a-zA-Z0-9_/\\.-]+\.py)", tb)
        set_class(
            "runtime_error",
            list(tb_files or event.related_files or []),
            "runtime traceback/error signals detected",
            0.88,
            "runtime",
            "runtime_error_keywords",
        )

    return FailureClassification(
        failure_kind=kind,
        confidence=confidence,
        signals=signals,
        likely_layer=layer,
        likely_files=likely_files,
        severity=event.severity or "medium",
        reason=reason,
    )
