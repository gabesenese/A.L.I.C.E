from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List


@dataclass
class Objective:
    text: str = ""
    mode: str = "general"


@dataclass
class PlanStep:
    name: str
    status: str = "pending"


@dataclass
class Observation:
    name: str
    detail: str = ""


@dataclass
class AgentLoopResult:
    active: bool = False
    objective: Dict[str, Any] = field(default_factory=dict)
    plan_steps: List[Dict[str, Any]] = field(default_factory=list)
    executed_steps: List[str] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    verification: Dict[str, Any] = field(default_factory=dict)
    next_step: str = ""
    blocked_reason: str = ""
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active": bool(self.active),
            "objective": dict(self.objective or {}),
            "plan_steps": list(self.plan_steps or []),
            "executed_steps": list(self.executed_steps or []),
            "observations": list(self.observations or []),
            "verification": dict(self.verification or {}),
            "next_step": self.next_step,
            "blocked_reason": self.blocked_reason,
            "updated_at": self.updated_at,
        }


def build_agent_loop_state(
    *,
    user_input: str,
    route: str,
    intent: str,
    local_execution: Dict[str, Any] | None = None,
    active_objective: str = "",
) -> Dict[str, Any]:
    objective = active_objective or "Improve Alice into an agentic companion/operator"
    plan = [
        {"name": "set_objective", "status": "completed"},
        {"name": "execute_safe_step", "status": "completed" if route == "local" else "pending"},
        {"name": "observe_result", "status": "completed" if local_execution else "pending"},
        {"name": "verify_result", "status": "completed" if local_execution else "pending"},
    ]
    obs = []
    if local_execution:
        obs.append(
            {
                "name": "local_execution",
                "detail": f"action={local_execution.get('action','')} success={local_execution.get('success', False)}",
            }
        )
    result = AgentLoopResult(
        active=route in {"local", "llm"},
        objective={"text": objective, "mode": "code_inspection" if str(intent).startswith("code:") else "general"},
        plan_steps=plan,
        executed_steps=["execute_safe_step"] if route == "local" else [],
        observations=obs,
        verification={"accepted": bool(local_execution.get("success")) if local_execution else True},
        next_step="Continue with the next safe inspection step.",
        blocked_reason="" if route in {"local", "llm"} else "unsupported_route",
    )
    return result.to_dict()
