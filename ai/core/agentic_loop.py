"""Core agentic control loop for perception → reasoning → goals → decision → execution → learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Optional


@dataclass
class AgenticCycleReport:
    perceived: Dict[str, Any] = field(default_factory=dict)
    reasoning: Dict[str, Any] = field(default_factory=dict)
    goals: Dict[str, Any] = field(default_factory=dict)
    decision: Dict[str, Any] = field(default_factory=dict)
    execution: Dict[str, Any] = field(default_factory=dict)
    learning: Dict[str, Any] = field(default_factory=dict)
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class AgenticLoop:
    """
    Lightweight loop coordinator.

    This does not replace existing orchestration layers; it standardizes
    a deterministic cycle with explicit stage outputs for monitoring.
    """

    def __init__(
        self,
        *,
        perceive_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        reason_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        goal_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        decide_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        execute_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        learn_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self.perceive_fn = perceive_fn or (lambda payload: dict(payload))
        self.reason_fn = reason_fn or (lambda state: {"summary": "no_reasoner"})
        self.goal_fn = goal_fn or (lambda state: {"goal": "maintain_stability"})
        self.decide_fn = decide_fn or (lambda state: {"action": "noop"})
        self.execute_fn = execute_fn or (lambda decision: {"status": "skipped"})
        self.learn_fn = learn_fn or (lambda feedback: {"adapted": False})
        self._memory: Dict[str, Any] = {"last_goal": "", "last_action": "", "cycles": 0}

    def run_cycle(self, payload: Dict[str, Any]) -> AgenticCycleReport:
        report = AgenticCycleReport()
        report.perceived = dict(self.perceive_fn(payload) or {})

        reason_input = {"perceived": report.perceived, "memory": dict(self._memory)}
        report.reasoning = dict(self.reason_fn(reason_input) or {})

        goal_input = {
            "perceived": report.perceived,
            "reasoning": report.reasoning,
            "memory": dict(self._memory),
        }
        report.goals = dict(self.goal_fn(goal_input) or {})

        decision_input = {
            "goals": report.goals,
            "reasoning": report.reasoning,
            "perceived": report.perceived,
            "memory": dict(self._memory),
        }
        report.decision = dict(self.decide_fn(decision_input) or {})

        report.execution = dict(self.execute_fn(report.decision) or {})

        learning_input = {
            "decision": report.decision,
            "execution": report.execution,
            "goals": report.goals,
        }
        report.learning = dict(self.learn_fn(learning_input) or {})

        self._memory["last_goal"] = str(report.goals.get("goal") or "")
        self._memory["last_action"] = str(report.decision.get("action") or "")
        self._memory["cycles"] = int(self._memory.get("cycles", 0)) + 1
        self._memory["last_updated"] = datetime.utcnow().isoformat()

        return report

    def snapshot(self) -> Dict[str, Any]:
        return dict(self._memory)
