from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from ai.runtime.self_improvement.improvement_hypothesis import ImprovementHypothesis


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PatchPlan:
    plan_id: str
    hypothesis_id: str
    summary: str
    target_files: List[str] = field(default_factory=list)
    proposed_changes: List[str] = field(default_factory=list)
    tests_to_add: List[str] = field(default_factory=list)
    tests_to_run: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    safety_notes: List[str] = field(default_factory=list)
    requires_approval: bool = True
    can_auto_apply: bool = False
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _existing(paths: List[str]) -> List[str]:
    return [p for p in paths if Path(p).exists()]


def build_patch_plan(
    hypothesis: ImprovementHypothesis,
    repo_files: List[str] | None = None,
) -> PatchPlan:
    targets = list(hypothesis.likely_files or [])
    if repo_files:
        file_set = set(repo_files)
        targets = [p for p in targets if p in file_set]
    targets = _existing(targets) or targets

    kind = str(hypothesis.failure_kind or "unknown")
    proposed_changes = [
        "Implement minimal, test-driven updates in target layer.",
        "Preserve existing safety guards and avoid broad behavior rewrites.",
    ]
    tests_to_add = []
    tests_to_run = ["pytest"]
    acceptance = [
        "Fix is grounded in recorded evidence.",
        "No unsupported continuity or fake capability claims.",
        "No silent source modification without explicit approval gate.",
    ]
    safety = [
        "Source-code edits require explicit approval before application.",
        "Do not auto-merge or run destructive operations.",
    ]

    if kind == "greeting_tone":
        tests_to_add = ["tests/golden/test_greeting_memory_grounding.py"]
        tests_to_run = [
            "pytest tests/golden/test_greeting_memory_grounding.py",
            "pytest tests/integration/test_contract_pipeline.py",
            "pytest",
        ]
        proposed_changes.append(
            "Adjust greeting surface selection to presence-first companion style while preserving continuity suppression."
        )
    elif kind == "routing":
        tests_to_add = ["tests/golden/test_route_arbiter_scoring.py"]
        tests_to_run = [
            "pytest tests/golden/test_routing_evidence_contracts.py",
            "pytest tests/golden/test_route_arbiter_scoring.py",
            "pytest tests/integration/test_contract_pipeline.py",
            "pytest",
        ]
        proposed_changes.append(
            "Refine route/evidence scoring and clarification fallback conditions."
        )
    elif kind in {"memory", "continuity_claim"}:
        tests_to_add = ["tests/golden/test_companion_memory_flows.py"]
        tests_to_run = [
            "pytest tests/golden/test_companion_memory_flows.py",
            "pytest tests/golden/test_greeting_memory_grounding.py",
            "pytest tests/integration/test_project_memory.py",
            "pytest",
        ]
    elif kind == "local_execution":
        tests_to_add = ["tests/golden/test_operator_code_flows.py"]
        tests_to_run = [
            "pytest tests/golden/test_operator_code_flows.py",
            "pytest tests/integration/test_contract_pipeline.py",
            "pytest",
        ]
    elif kind == "runtime_error":
        tests_to_add = ["tests/integration/test_runtime_modes.py"]
        tests_to_run = [
            "pytest tests/integration/test_runtime_modes.py",
            "pytest",
        ]

    return PatchPlan(
        plan_id=str(uuid4()),
        hypothesis_id=str(hypothesis.hypothesis_id),
        summary=str(hypothesis.hypothesis or "Proposed runtime fix"),
        target_files=targets,
        proposed_changes=proposed_changes,
        tests_to_add=tests_to_add,
        tests_to_run=tests_to_run,
        acceptance_criteria=acceptance,
        safety_notes=safety,
        requires_approval=True,
        can_auto_apply=False,
    )

