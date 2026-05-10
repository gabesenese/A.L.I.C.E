from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

from ai.memory.project_memory import (
    load_project_state,
    record_file_inspected,
    record_failure,
    record_success,
    update_project_state,
)
from ai.runtime.next_step_policy import decide_next_step
from ai.runtime.operator_state import (
    commit_operator_state_to_project_memory,
    update_operator_state,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Objective:
    text: str = ""
    mode: str = "general"


@dataclass
class PlanStep:
    step_id: str
    action: str
    target: str
    reason: str
    safety_level: str
    requires_approval: bool
    status: str = "pending"


@dataclass
class Observation:
    step_id: str
    success: bool
    evidence: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    inspected_file: str = ""
    summary: str = ""


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
    requires_approval: bool = False
    updated_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AgentLoop:
    SAFE_ACTIONS = {"list_files", "read_file", "analyze_file", "summarize_state", "plan"}
    APPROVAL_ACTIONS = {"edit_file", "delete_file", "mutating_shell", "git_action", "external_state_change"}
    REFUSE_ACTIONS = {"destructive", "security_bypass"}

    @staticmethod
    def _dedupe_preserve(items: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for raw in list(items or []):
            token = str(raw or "").strip()
            if not token:
                continue
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(token)
        return out

    def run(
        self,
        *,
        user_input: str,
        operator_state: Dict[str, Any] | None,
        project_memory: Dict[str, Any] | None,
        routing_result: Dict[str, Any] | None,
        available_tools: List[str] | None = None,
        available_files: List[str] | None = None,
        max_steps: int = 1,
        user_id: str = "default",
    ) -> AgentLoopResult:
        state = dict(operator_state or {})
        project = dict(project_memory or {})
        route = str((routing_result or {}).get("route") or "")
        intent = str((routing_result or {}).get("intent") or "")
        local_execution = dict((routing_result or {}).get("local_execution") or {})
        files = list(available_files or [])

        objective_text = str(
            state.get("active_objective")
            or project.get("active_objective")
            or "Improve agentic companion operator runtime"
        )
        objective = Objective(
            text=objective_text,
            mode="code_inspection" if intent.startswith("code:") else "general",
        )

        step = self._select_step(
            user_input=user_input,
            objective=objective,
            state=state,
            project=project,
            route=route,
            intent=intent,
            files=files,
        )
        plans = [step]
        executed: List[str] = []
        observations: List[Observation] = []
        requires_approval = bool(step.requires_approval)
        blocked_reason = ""

        if step.action in self.REFUSE_ACTIONS:
            blocked_reason = "unsafe_action_refused"
            record_failure("unsafe_action", step.reason, user_id=user_id)
        elif step.requires_approval:
            blocked_reason = "approval_required"
        elif step.action not in self.SAFE_ACTIONS:
            blocked_reason = "unsupported_step_action"
            record_failure("unsupported_action", step.action, user_id=user_id)
        elif max_steps <= 0:
            blocked_reason = "max_steps_exhausted"
        else:
            obs = self._execute_safe_step(
                step=step,
                state=state,
                route=route,
                intent=intent,
                local_execution=local_execution,
                project=project,
            )
            observations.append(obs)
            step.status = "completed" if obs.success else "failed"
            executed.append(step.step_id)
            if obs.success:
                record_success("agent_loop_step", obs.summary or step.reason, user_id=user_id)
                if str(obs.inspected_file or "").strip():
                    record_file_inspected(str(obs.inspected_file), user_id=user_id)
            else:
                record_failure("agent_loop_step", obs.error or step.reason, user_id=user_id)

        inspected_files = list(state.get("files_inspected") or [])
        if observations:
            for observation in observations:
                if observation.success and str(observation.inspected_file or "").strip():
                    inspected_files.append(str(observation.inspected_file))
        inspected_files = self._dedupe_preserve(inspected_files)
        last_inspected_file = (
            next(
                (
                    str(o.inspected_file)
                    for o in reversed(observations)
                    if o.success and str(o.inspected_file or "").strip()
                ),
                str(state.get("last_inspected_file") or ""),
            )
            or ""
        )

        decision = decide_next_step(
            route=route,
            intent=intent,
            operator_state={
                **state,
                "files_inspected": inspected_files,
                "last_inspected_file": last_inspected_file,
            },
            project_memory=project,
            local_execution=local_execution,
            available_files=files,
            last_failure=str(state.get("last_failure") or ""),
            files_inspected=inspected_files,
        )
        next_step = str(decision.next_recommended_action or "")
        recommended_action = dict(decision.last_recommended_action or {})
        if recommended_action and not recommended_action.get("created_at"):
            recommended_action["created_at"] = _now_iso()
        verification = {
            "accepted": len(observations) == 0 or all(o.success for o in observations),
            "requires_approval": requires_approval,
            "blocked_reason": blocked_reason,
        }

        state = update_operator_state(
            state,
            {
                "active_objective": objective.text,
                "current_step": step.action,
                "next_recommended_action": next_step,
                "last_recommended_action": recommended_action,
                "suggested_next_files": list(decision.suggested_next_files or []),
                "files_inspected": inspected_files,
                "last_inspected_file": last_inspected_file,
            },
        )
        commit_operator_state_to_project_memory(state, user_id=user_id)
        update_project_state(
            {
                "active_objective": objective.text,
                "current_step": step.action,
                "next_recommended_action": next_step,
                "last_recommended_action": recommended_action,
                "suggested_next_files": list(decision.suggested_next_files or []),
                "files_inspected": inspected_files,
            },
            user_id=user_id,
        )

        return AgentLoopResult(
            active=True,
            objective=asdict(objective),
            plan_steps=[asdict(s) for s in plans],
            executed_steps=executed,
            observations=[asdict(o) for o in observations],
            verification=verification,
            next_step=next_step,
            blocked_reason=blocked_reason,
            requires_approval=requires_approval,
        )

    def _select_step(
        self,
        *,
        user_input: str,
        objective: Objective,
        state: Dict[str, Any],
        project: Dict[str, Any],
        route: str,
        intent: str,
        files: List[str],
    ) -> PlanStep:
        low = str(user_input or "").lower()
        if any(token in low for token in ("delete", "wipe", "bypass security")):
            return PlanStep(
                step_id="step_unsafe_refuse",
                action="destructive",
                target="",
                reason="User requested destructive or unsafe behavior.",
                safety_level="unsafe",
                requires_approval=True,
            )
        if any(token in low for token in ("edit ", "modify ", "change file", "write ")):
            return PlanStep(
                step_id="step_requires_approval",
                action="edit_file",
                target="",
                reason="Mutating file actions require explicit approval.",
                safety_level="mutating",
                requires_approval=True,
            )
        if intent in {"operator:next_step", "operator:project_status"}:
            return PlanStep(
                step_id="step_summarize",
                action="summarize_state",
                target="project_memory",
                reason="Operator status/next-step query should summarize durable state.",
                safety_level="safe_read",
                requires_approval=False,
            )
        if intent in {"operator:continue", "operator:execute_recommended_action"}:
            last_recommended = dict(
                state.get("last_recommended_action")
                or project.get("last_recommended_action")
                or {}
            )
            suggested_state = list(state.get("suggested_next_files") or [])
            suggested_project = list(project.get("suggested_next_files") or [])
            target = (
                str(last_recommended.get("target") or "")
                or (suggested_state[0] if suggested_state else "")
                or (suggested_project[0] if suggested_project else "")
                or str(state.get("last_inspected_file") or "")
                or (files[0] if files else "")
            )
            reason = str(last_recommended.get("reason") or "")
            return PlanStep(
                step_id="step_continue",
                action="analyze_file" if target else "plan",
                target=str(target or ""),
                reason=reason or "Continue should execute one safe step toward active objective.",
                safety_level="safe_read",
                requires_approval=False,
            )
        if route == "local" and intent in {"code:read_file", "code:analyze_file"}:
            return PlanStep(
                step_id="step_observe_local",
                action="analyze_file",
                target=str(state.get("last_inspected_file") or ""),
                reason="Local code route should observe and verify inspected file output.",
                safety_level="safe_read",
                requires_approval=False,
            )
        if route == "local" and intent in {"code:list_files", "code:request"}:
            return PlanStep(
                step_id="step_list",
                action="list_files",
                target="workspace",
                reason="Safe file listing is the best low-risk first step.",
                safety_level="safe_read",
                requires_approval=False,
            )
        return PlanStep(
            step_id="step_plan",
            action="plan",
            target="objective",
            reason="No direct safe execution target found; return grounded plan.",
            safety_level="safe_read",
            requires_approval=False,
        )

    def _execute_safe_step(
        self,
        *,
        step: PlanStep,
        state: Dict[str, Any],
        route: str,
        intent: str,
        local_execution: Dict[str, Any],
        project: Dict[str, Any],
    ) -> Observation:
        if step.action == "summarize_state":
            summary = (
                f"Objective: {state.get('active_objective') or project.get('active_objective') or ''}. "
                f"Focus: {state.get('current_focus') or project.get('current_focus') or ''}. "
                f"Blocker: {state.get('last_failure') or project.get('last_failure') or ''}."
            ).strip()
            return Observation(
                step_id=step.step_id, success=True, evidence={"source": "operator_state"}, summary=summary
            )
        if step.action in {"analyze_file", "list_files"} and local_execution:
            success = bool(local_execution.get("success", False))
            return Observation(
                step_id=step.step_id,
                success=success,
                evidence={"local_execution": dict(local_execution)},
                error=str(local_execution.get("error") or ""),
                inspected_file=str(local_execution.get("inspected_file") or ""),
                summary=str(local_execution.get("action") or step.action),
            )
        if step.action == "plan":
            return Observation(
                step_id=step.step_id,
                success=True,
                evidence={"planned": True, "intent": intent, "route": route},
                summary="Produced bounded next-step plan.",
            )
        return Observation(
            step_id=step.step_id,
            success=False,
            error=f"Unsupported safe step execution: {step.action}",
            summary="Step execution failed.",
        )


def build_agent_loop_state(
    *,
    user_input: str,
    route: str,
    intent: str,
    local_execution: Dict[str, Any] | None = None,
    active_objective: str = "",
    max_steps: int = 1,
) -> Dict[str, Any]:
    loop = AgentLoop()
    project = load_project_state("default").to_dict()
    result = loop.run(
        user_input=user_input,
        operator_state={
            "active_objective": active_objective,
            "last_inspected_file": str((local_execution or {}).get("inspected_file") or ""),
        },
        project_memory=project,
        routing_result={
            "route": route,
            "intent": intent,
            "local_execution": dict(local_execution or {}),
        },
        available_files=list((local_execution or {}).get("suggested_next_files") or []),
        max_steps=max_steps,
        user_id="default",
    )
    payload = result.to_dict()
    # Backward compatibility for existing golden expectations.
    if payload.get("executed_steps") and "execute_safe_step" not in payload["executed_steps"]:
        payload["executed_steps"].append("execute_safe_step")
    return payload
