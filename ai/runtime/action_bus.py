from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict
from uuid import uuid4

from ai.runtime.approval_ledger import approval_matches


@dataclass
class ActionRequest:
    action_id: str
    name: str
    target: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    risk_level: str = "safe_read"
    requires_approval: bool = False
    approved: bool = False
    approval_id: str = ""
    source: str = ""
    expected_result: str = ""


@dataclass
class ActionResult:
    action_id: str
    name: str
    target: str = ""
    success: bool = False
    data: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    verified: bool = False
    risk_level: str = "safe_read"
    requires_approval: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ActionBus:
    def __init__(self, approval_lookup=None, approval_consume=None) -> None:
        self._executors: Dict[str, Callable[[ActionRequest], ActionResult]] = {}
        self._approval_lookup = approval_lookup
        self._approval_consume = approval_consume

    def register(self, name: str, executor: Callable[[ActionRequest], ActionResult]) -> None:
        self._executors[str(name or "").strip()] = executor

    @staticmethod
    def _action_requires_approval(action_request: ActionRequest) -> bool:
        risk = str(action_request.risk_level or "safe_read").strip().lower()
        if bool(action_request.requires_approval):
            return True
        return risk in {"safe_write", "destructive", "external"}

    def can_execute(self, action_request: ActionRequest) -> bool:
        action_name = str(action_request.name or "").strip()
        if action_name not in self._executors:
            return False
        if self._action_requires_approval(action_request):
            return bool(action_request.approved)
        return True

    def _blocked_result(self, request: ActionRequest, error: str) -> ActionResult:
        approval_required = self._action_requires_approval(request)
        return ActionResult(
            action_id=request.action_id,
            name=str(request.name or "").strip(),
            target=request.target,
            success=False,
            error=error,
            evidence={
                "source": "action_bus",
                "approval_required": bool(approval_required),
                "approved": False,
                "approval_id": str(request.approval_id or ""),
                "approval_error": error,
                "risk_level": str(request.risk_level or "safe_read"),
            },
            verified=False,
            risk_level=str(request.risk_level or "safe_read"),
            requires_approval=bool(approval_required),
        )

    def execute(self, action_request: ActionRequest) -> ActionResult:
        action_name = str(action_request.name or "").strip()
        approval_required = self._action_requires_approval(action_request)
        if action_name not in self._executors:
            return ActionResult(
                action_id=action_request.action_id,
                name=action_name,
                target=action_request.target,
                success=False,
                error="unknown_action",
                evidence={"source": "action_bus"},
                verified=False,
                risk_level=str(action_request.risk_level or "safe_read"),
                requires_approval=approval_required,
            )

        if approval_required:
            if not str(action_request.approval_id or "").strip() or not bool(action_request.approved):
                return self._blocked_result(action_request, "approval_required")
            if not callable(self._approval_lookup):
                return self._blocked_result(action_request, "approval_not_found")
            record = self._approval_lookup(str(action_request.approval_id or ""))
            if not record:
                return self._blocked_result(action_request, "approval_not_found")
            record_approved = bool(getattr(record, "approved", False) if not isinstance(record, dict) else record.get("approved"))
            record_consumed = bool(getattr(record, "consumed", False) if not isinstance(record, dict) else record.get("consumed"))
            if (not record_approved) or record_consumed:
                return self._blocked_result(action_request, "approval_not_valid")
            if not approval_matches(record, action_request):
                return self._blocked_result(action_request, "approval_mismatch")

        if not self.can_execute(action_request):
            return self._blocked_result(action_request, "approval_required")

        out = self._executors[action_name](action_request)
        approval_consumed = False
        if approval_required and callable(self._approval_consume):
            approval_consumed = bool(self._approval_consume(str(action_request.approval_id or "")))
        out.evidence = dict(out.evidence or {})
        out.evidence.setdefault("action", action_name)
        out.evidence.setdefault("source", "action_bus")
        out.evidence.setdefault("approval_required", bool(approval_required))
        out.evidence.setdefault("approved", bool(action_request.approved))
        out.evidence.setdefault("approval_id", str(action_request.approval_id or ""))
        out.evidence.setdefault(
            "approval_consumed", bool(approval_consumed if approval_required else True)
        )
        out.evidence.setdefault("risk_level", str(action_request.risk_level or "safe_read"))
        out.risk_level = str(out.risk_level or action_request.risk_level or "safe_read")
        out.requires_approval = bool(out.requires_approval or approval_required)
        out.verified = bool(out.success and out.evidence)
        return out


def action_result_from_local_execution(
    *,
    action_name: str,
    local_execution: dict,
    target: str = "",
) -> ActionResult:
    payload = dict(local_execution or {})
    success = bool(payload.get("success"))
    inspected = str(payload.get("inspected_file") or "").strip()
    analysis = dict(payload.get("analysis") or {})
    evidence: Dict[str, Any] = {"source": "local_execution"}
    if inspected:
        evidence["inspected_file"] = inspected
    if analysis:
        evidence["analysis"] = analysis
    if isinstance(payload.get("files"), list):
        files = list(payload.get("files") or [])
        evidence["files"] = files
        evidence["count"] = len(files)
    if payload.get("count") is not None and "count" not in evidence:
        evidence["count"] = payload.get("count")
    if payload.get("active_objective") is not None:
        evidence["active_objective"] = payload.get("active_objective")
    if payload.get("current_focus") is not None:
        evidence["current_focus"] = payload.get("current_focus")
    return ActionResult(
        action_id=f"action_{uuid4().hex[:12]}",
        name=str(action_name or "").strip(),
        target=str(target or inspected or "").strip(),
        success=success,
        data={"local_execution": payload},
        error=str(payload.get("error") or ""),
        evidence=evidence,
        verified=bool(success and (inspected or analysis or evidence.get("files"))),
        risk_level="safe_read",
        requires_approval=False,
    )
