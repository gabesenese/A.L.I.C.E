from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict


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
        if not self.can_execute(action_request):
            return ActionResult(
                action_id=action_request.action_id,
                name=action_request.name,
                target=action_request.target,
                success=False,
                error="action_not_allowed_or_not_registered",
                evidence={"can_execute": False},
                verified=False,
            )
        out = self._executors[action_request.name](action_request)
        out.evidence = dict(out.evidence or {})
        out.evidence.setdefault("action", action_request.name)
        out.verified = bool(out.success and out.evidence)
        return out
