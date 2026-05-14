from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict
from uuid import uuid4


@dataclass
class ActionRequest:
    action_id: str
    name: str
    target: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    risk_level: str = "safe_read"
    requires_approval: bool = False
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
    def __init__(self) -> None:
        self._executors: Dict[str, Callable[[ActionRequest], ActionResult]] = {}

    def register(self, name: str, executor: Callable[[ActionRequest], ActionResult]) -> None:
        self._executors[str(name or "").strip()] = executor

    def can_execute(self, action_request: ActionRequest) -> bool:
        if action_request.requires_approval:
            return False
        if action_request.risk_level in {"destructive", "external"}:
            return False
        return action_request.name in self._executors

    def execute(self, action_request: ActionRequest) -> ActionResult:
        action_name = str(action_request.name or "").strip()
        if action_name not in self._executors:
            return ActionResult(
                action_id=action_request.action_id,
                name=action_name,
                target=action_request.target,
                success=False,
                error="unknown_action",
                evidence={"source": "action_bus"},
                verified=False,
                risk_level=action_request.risk_level,
                requires_approval=action_request.requires_approval,
            )
        if not self.can_execute(action_request):
            return ActionResult(
                action_id=action_request.action_id,
                name=action_name,
                target=action_request.target,
                success=False,
                error="approval_required",
                evidence={"source": "action_bus", "can_execute": False},
                verified=False,
                risk_level=action_request.risk_level,
                requires_approval=action_request.requires_approval,
            )
        out = self._executors[action_name](action_request)
        out.evidence = dict(out.evidence or {})
        out.evidence.setdefault("action", action_name)
        out.evidence.setdefault("source", "action_bus")
        out.risk_level = str(out.risk_level or action_request.risk_level or "safe_read")
        out.requires_approval = bool(out.requires_approval or action_request.requires_approval)
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
