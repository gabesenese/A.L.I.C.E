from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict

from ai.runtime.self_improvement.patch_plan import PatchPlan


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ApprovalDecision:
    approved: bool
    approval_type: str = "none"
    approved_by: str = ""
    scope: str = ""
    notes: str = ""
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def requires_approval(patch_plan: PatchPlan) -> bool:
    return bool(patch_plan.requires_approval)


def assert_approval_for_source_change(decision: ApprovalDecision | None) -> None:
    if decision is None:
        raise PermissionError("approval_required_for_source_changes")
    if not bool(decision.approved):
        raise PermissionError("source_change_not_approved")
    if str(decision.approval_type or "").strip().lower() not in {
        "source_change",
        "test_change",
        "git_operation",
        "merge",
        "manual",
    }:
        raise PermissionError("invalid_approval_type")


def assert_approval_for_git_operation(decision: ApprovalDecision | None) -> None:
    if decision is None:
        raise PermissionError("approval_required_for_git_operation")
    if not bool(decision.approved):
        raise PermissionError("git_operation_not_approved")
    approval_type = str(decision.approval_type or "").strip().lower()
    if approval_type not in {"git_operation", "manual"}:
        raise PermissionError("invalid_git_approval_type")


def assert_approval_for_merge(decision: ApprovalDecision | None) -> None:
    if decision is None:
        raise PermissionError("approval_required_for_merge")
    if not bool(decision.approved):
        raise PermissionError("merge_not_approved")
    approval_type = str(decision.approval_type or "").strip().lower()
    if approval_type not in {"merge", "manual"}:
        raise PermissionError("invalid_merge_approval_type")
