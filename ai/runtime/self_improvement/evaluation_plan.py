from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

from ai.runtime.self_improvement.patch_plan import PatchPlan


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EvaluationPlan:
    evaluation_id: str
    patch_plan_id: str
    commands: List[str] = field(default_factory=list)
    golden_tests: List[str] = field(default_factory=list)
    manual_checks: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    regression_checks: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_evaluation_plan(patch_plan: PatchPlan) -> EvaluationPlan:
    commands = list(patch_plan.tests_to_run or [])
    if "pytest" not in commands:
        commands.append("pytest")
    golden = [cmd for cmd in commands if "tests/golden/" in cmd]
    manual_checks = [
        "Validate behavior against audit acceptance criteria.",
        "Confirm no unsupported continuity or fake-capability claims.",
    ]
    success = [
        "Targeted tests pass.",
        "No regression in contract pipeline and routing evidence behavior.",
    ]
    regression = [
        "Run full pytest suite.",
        "Review changed runtime metadata for safety flags.",
    ]
    return EvaluationPlan(
        evaluation_id=str(uuid4()),
        patch_plan_id=str(patch_plan.plan_id),
        commands=commands,
        golden_tests=golden,
        manual_checks=manual_checks,
        success_criteria=success,
        regression_checks=regression,
    )

