"""
Reasoning planner for decomposition-first execution.

This module provides a lightweight planning brain that turns a high-level
request into executable steps with explicit decisions and verification checks.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Dict, List, Optional

from ai.planning.task_planner import (
    ExecutionPlan,
    PlanStep,
    StepStatus,
    TaskPlanner,
)


@dataclass
class TaskRepresentation:
    """Canonical task object used by the planner."""

    task_id: str
    user_request: str
    objective: str
    constraints: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class ExecutionDecision:
    step_id: int
    action: str
    reason: str


@dataclass
class ReasoningTrace:
    thought: str
    action: str
    result: str
    decision: str
    timestamp: float = field(default_factory=time.time)


class ReasoningPlanner:
    """
    Planner brain with explicit decomposition, step decisions, and verification.
    """

    def create_task_representation(
        self,
        user_request: str,
        *,
        constraints: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> TaskRepresentation:
        objective = self._extract_objective(user_request)
        return TaskRepresentation(
            task_id=f"task-{uuid.uuid4().hex[:10]}",
            user_request=user_request,
            objective=objective,
            constraints=list(constraints or []),
            context=dict(context or {}),
        )

    def __init__(self) -> None:
        # Reuse the canonical planner model to avoid divergent plan schemas.
        self._task_planner = TaskPlanner()
        self._traces_by_plan_id: Dict[str, List[ReasoningTrace]] = {}

    def generate_steps(self, task: TaskRepresentation) -> List[PlanStep]:
        """
        Generate executable steps for a task.

        The output is intentionally consistent to support deterministic execution
        and verification.
        """
        step_specs: List[Dict[str, str]] = [
            {
                "title": "Clarify target outcome",
                "action": "analyze_request",
                "expected": "Clear outcome definition",
            },
            {
                "title": "Collect required inputs",
                "action": "collect_inputs",
                "expected": "All required context/tools identified",
            },
            {
                "title": "Execute core operation",
                "action": "execute_operation",
                "expected": "Primary objective completed",
            },
            {
                "title": "Verify result quality",
                "action": "verify_result",
                "expected": "Output meets objective and constraints",
            },
            {
                "title": "Stress-test edge cases",
                "action": "validate_edge_cases",
                "expected": "Edge cases are identified and handled",
            },
            {
                "title": "Finalize and summarize",
                "action": "finalize_output",
                "expected": "Final output is concise and production-ready",
            },
        ]

        complexity = self._estimate_complexity(task)
        target_steps = min(6, max(3, ceil(2 + (complexity * 4))))
        step_specs = step_specs[:target_steps]

        steps: List[PlanStep] = []
        for idx, spec in enumerate(step_specs):
            step_id = idx + 1
            depends_on = [idx] if idx > 0 else []
            steps.append(
                PlanStep(
                    step_id=step_id,
                    action=spec["action"],
                    params={"title": spec["title"], "expected_output": spec["expected"]},
                    depends_on=depends_on,
                )
            )
        return steps

    def create_plan(self, task: TaskRepresentation) -> ExecutionPlan:
        # Use the canonical TaskPlanner for intent-centric plans; fallback to
        # deterministic generic decomposition when no intent is supplied.
        intent = str(task.context.get("intent") or "question")
        entities = dict(task.context.get("entities") or {"query": task.user_request})
        context = dict(task.context)
        plan = self._task_planner.create_plan(intent=intent, entities=entities, context=context)
        if len(plan.steps) < 3:
            plan.steps = self.generate_steps(task)

        critical_path = self.estimate_critical_path(plan)

        self._traces_by_plan_id[plan.plan_id] = [
            ReasoningTrace(
                thought="Decompose request into deterministic sequence",
                action="generate_steps",
                result=f"Generated {len(plan.steps)} steps; critical_path={critical_path}",
                decision="proceed_with_plan",
            )
        ]
        return plan

    def decide_next_step(self, plan: ExecutionPlan) -> Optional[ExecutionDecision]:
        completed_ids = {s.step_id for s in plan.steps if s.status == StepStatus.COMPLETED}
        step = plan.get_next_step(completed_ids)
        if step is None:
            return None
        decision = ExecutionDecision(
            step_id=step.step_id,
            action=step.action,
            reason="Dependencies satisfied and step is pending",
        )
        self._trace(plan.plan_id).append(
            ReasoningTrace(
                thought=f"Evaluate next step {step.step_id}",
                action=step.action,
                result="selected",
                decision=decision.reason,
            )
        )
        return decision

    def verify_step_result(self, step: PlanStep, result: Any) -> bool:
        """Weighted verification: structural validity + expectation coverage."""
        if result is None:
            return False
        if isinstance(result, str) and not result.strip():
            return False
        if isinstance(result, dict) and result.get("error"):
            return False

        structural_score = 1.0
        if isinstance(result, dict) and not result:
            structural_score = 0.2

        expected = str((step.params or {}).get("expected_output") or "")
        if not expected:
            return structural_score >= 0.5

        expected_tokens = self._tokenize(expected)
        result_tokens = self._tokenize(str(result))
        if not expected_tokens:
            return structural_score >= 0.5

        overlap = len(expected_tokens.intersection(result_tokens)) / float(len(expected_tokens))
        quality_score = (0.6 * structural_score) + (0.4 * overlap)
        return quality_score >= 0.45

    def apply_step_result(self, plan: ExecutionPlan, step_id: str, result: Any) -> bool:
        step_id_text = str(step_id).strip()
        if step_id_text.upper().startswith("S"):
            step_id_text = step_id_text[1:]
        try:
            normalized_step_id = int(step_id_text)
        except ValueError:
            return False

        step = next((s for s in plan.steps if s.step_id == normalized_step_id), None)
        if step is None:
            return False

        step.status = StepStatus.RUNNING
        ok = self.verify_step_result(step, result)
        if ok:
            step.status = StepStatus.COMPLETED
            step.result = result
            self._trace(plan.plan_id).append(
                ReasoningTrace(
                    thought=f"Validate result for {step.step_id}",
                    action="verify_step_result",
                    result="pass",
                    decision="mark_completed",
                )
            )
            return True

        step.status = StepStatus.FAILED
        step.error = "Verification failed"
        step.result = result
        self._trace(plan.plan_id).append(
            ReasoningTrace(
                thought=f"Validate result for {step.step_id}",
                action="verify_step_result",
                result="fail",
                decision="mark_failed",
            )
        )
        return False

    def debug_trace_view(self, plan: ExecutionPlan) -> str:
        goal = getattr(plan, "goal", "")
        lines = [f"Plan {plan.plan_id}: {goal}"]
        lines.append(f"CriticalPath={self.estimate_critical_path(plan)}")
        for trace in self._trace(plan.plan_id):
            lines.append(
                f"Thought={trace.thought} | Action={trace.action} | Result={trace.result} | Decision={trace.decision}"
            )
        return "\n".join(lines)

    def estimate_critical_path(self, plan: ExecutionPlan) -> int:
        """Longest dependency-chain length (DAG critical path) via DP."""
        depth_cache: Dict[int, int] = {}
        by_id = {int(step.step_id): step for step in plan.steps}

        def depth(step_id: int) -> int:
            if step_id in depth_cache:
                return depth_cache[step_id]
            step = by_id.get(step_id)
            if step is None:
                return 0
            deps = [int(d) for d in (step.depends_on or [])]
            if not deps:
                depth_cache[step_id] = 1
                return 1
            val = 1 + max(depth(d) for d in deps)
            depth_cache[step_id] = val
            return val

        return max((depth(int(s.step_id)) for s in plan.steps), default=0)

    def _trace(self, plan_id: str) -> List[ReasoningTrace]:
        if plan_id not in self._traces_by_plan_id:
            self._traces_by_plan_id[plan_id] = []
        return self._traces_by_plan_id[plan_id]

    def _extract_objective(self, user_request: str) -> str:
        text = (user_request or "").strip()
        if not text:
            return "Handle user request"
        # Remove filler and keep a compact objective phrase.
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"^(please|can you|could you|i need you to)\s+", "", text, flags=re.I)
        return text[0].upper() + text[1:]

    def _estimate_complexity(self, task: TaskRepresentation) -> float:
        text = f"{task.user_request} {' '.join(task.constraints)}".lower()
        words = len(text.split())
        conjunctions = len(re.findall(r"\b(and|then|after|before|while|also)\b", text))
        conditional = len(re.findall(r"\b(if|unless|except|when)\b", text))
        action_verbs = len(re.findall(r"\b(build|create|design|refactor|analyze|verify|test|deploy)\b", text))

        score = 0.0
        score += min(0.35, words / 40.0)
        score += min(0.20, conjunctions * 0.07)
        score += min(0.20, conditional * 0.08)
        score += min(0.15, action_verbs * 0.05)
        score += min(0.10, len(task.constraints) * 0.04)
        return max(0.0, min(1.0, score))

    def _tokenize(self, text: str) -> set[str]:
        return {tok for tok in re.findall(r"[a-zA-Z0-9_]+", (text or "").lower()) if len(tok) > 2}
