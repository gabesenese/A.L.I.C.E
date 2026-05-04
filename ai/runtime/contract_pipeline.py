"""Contract-driven runtime pipeline used to thin app/main orchestration."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from ai.core.execution_verifier import ExecutionVerifier, get_execution_verifier
from ai.core.executive_controller import (
    ExecutiveController,
    ExecutiveDecision,
    TurnExecutionOutcome,
    TurnStateMachineResult,
)
from ai.contracts import RuntimeBoundaries, VerifierResult
from ai.infrastructure.telemetry import tracer, turn_counter, turn_latency
from ai.runtime.companion_runtime import CompanionRuntimeLoop
from ai.runtime.agent_loop import build_agent_loop_state
from ai.runtime.memory_turn_service import MemoryTurnService
from ai.runtime.next_step_policy import decide_next_step
from ai.runtime.turn_orchestrator import TurnOrchestrator
from ai.runtime.user_state_model import UserStateModel


@dataclass
class PipelineResult:
    handled: bool
    response_text: str = ""
    metadata: Dict[str, Any] = None


class ContractPipeline:
    """Executes one turn through hard runtime boundaries."""

    def __init__(
        self,
        boundaries: RuntimeBoundaries,
        user_state_model: Optional[UserStateModel] = None,
        companion_runtime: Optional[CompanionRuntimeLoop] = None,
        executive_controller: Optional[ExecutiveController] = None,
        execution_verifier: Optional[ExecutionVerifier] = None,
    ):
        self.boundaries = boundaries
        self.user_state_model = user_state_model or UserStateModel()
        self.companion_runtime = companion_runtime or CompanionRuntimeLoop()
        self.executive_controller = executive_controller or ExecutiveController()
        self.execution_verifier = execution_verifier or get_execution_verifier()
        self.orchestrator = TurnOrchestrator(boundaries)
        self.memory_turn_service = MemoryTurnService()

    @staticmethod
    def _is_tool_route(route: str) -> bool:
        return str(route or "").strip().lower() in {"tool", "plugin"}

    def _apply_contextual_reaction_pre_tool_veto(
        self,
        *,
        user_input: str,
        route_phase: Any,
        policy: Any,
        companion_state: Any,
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        decision = route_phase.decision
        if not self._is_tool_route(getattr(decision, "route", "")):
            return route_phase, policy, {"applied": False}

        previous_intent = str(getattr(companion_state, "last_intent", "") or "")
        should_veto = self.companion_runtime.policy_engine.is_contextual_reaction(
            user_input=user_input,
            previous_intent=previous_intent,
        )
        if not should_veto:
            return route_phase, policy, {"applied": False}

        reason = "gratitude_plus_personal_state_no_new_request"
        metadata = dict(getattr(decision, "metadata", {}) or {})
        metadata.update(
            {
                "route_veto": {
                    "applied": True,
                    "reason": reason,
                    "previous_intent": previous_intent,
                    "original_route": str(getattr(decision, "route", "") or ""),
                    "original_intent": str(getattr(decision, "intent", "") or ""),
                    "tool_execution_disabled": True,
                }
            }
        )

        demoted_decision = replace(
            decision,
            route="conversation",
            intent="conversation:personal_reaction",
            confidence=max(float(getattr(decision, "confidence", 0.0) or 0.0), 0.82),
            decision_band="execute",
            needs_clarification=False,
            metadata=metadata,
        )

        demoted_plan = dict(route_phase.plan or {})
        demoted_plan.update(
            {
                "route": "conversation",
                "intent": "conversation:personal_reaction",
                "decision_band": "execute",
                "needs_clarification": False,
                "step_count": 1,
                "tool_execution_disabled": True,
                "route_veto_reason": reason,
                "previous_intent": previous_intent,
            }
        )

        demoted_route_phase = replace(
            route_phase,
            decision=demoted_decision,
            plan=demoted_plan,
        )
        demoted_policy = replace(
            policy,
            decision_type="respond",
            reason="contextual_reaction_after_tool_result",
            retry_budget=0,
            requires_approval=False,
            approval_reason="",
        )

        return (
            demoted_route_phase,
            demoted_policy,
            {
                "applied": True,
                "reason": reason,
                "previous_intent": previous_intent,
                "tool_execution_disabled": True,
            },
        )

    @staticmethod
    def _merge_issue_lists(*issue_lists: List[str]) -> List[str]:
        seen = set()
        merged: List[str] = []
        for bucket in issue_lists:
            for raw in list(bucket or []):
                issue = str(raw or "").strip()
                if not issue:
                    continue
                key = issue.lower()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(issue)
        return merged

    @staticmethod
    def _verification_to_dict(verification: Optional[VerifierResult]) -> Dict[str, Any]:
        if verification is None:
            return {
                "accepted": True,
                "reason": "not_configured",
                "confidence": 1.0,
                "diagnostics": {},
            }
        return {
            "accepted": bool(verification.accepted),
            "reason": str(verification.reason or "verified"),
            "confidence": float(verification.confidence or 0.0),
            "diagnostics": dict(verification.diagnostics or {}),
        }

    def _build_pre_execution_state_machine(
        self,
        *,
        user_input: str,
        decision: Any,
        action_discipline: Dict[str, Any],
    ) -> TurnStateMachineResult:
        metadata = dict(getattr(decision, "metadata", {}) or {})
        state = self.executive_controller.build_state(
            user_input=user_input,
            intent=str(getattr(decision, "intent", "") or "unknown"),
            confidence=float(getattr(decision, "confidence", 0.0) or 0.0),
            entities={
                "topic": str(getattr(decision, "intent", "") or "").split(":", 1)[0],
                "_intent_plausibility": float(
                    max(
                        0.0,
                        min(1.0, float(getattr(decision, "confidence", 0.0) or 0.0)),
                    )
                ),
            },
            conversation_state={
                "active_goals": list(metadata.get("active_goals", []) or []),
            },
        )

        route = str(getattr(decision, "route", "") or "").strip().lower()
        approval_required = bool(action_discipline.get("approval_required"))

        if approval_required or route == "clarify":
            executive_decision = ExecutiveDecision(
                action="ask_clarification",
                reason="approval_required" if approval_required else "route_clarify",
                store_memory=False,
                clarification_question="What exact outcome should I target next?",
            )
        elif route == "refuse":
            executive_decision = ExecutiveDecision(
                action="defer",
                reason="route_refuse",
                store_memory=False,
            )
        elif self._is_tool_route(route):
            executive_decision = ExecutiveDecision(
                action="use_plugin",
                reason="route_tool",
                store_memory=True,
            )
        else:
            executive_decision = ExecutiveDecision(
                action="use_llm",
                reason="route_llm",
                store_memory=True,
            )

        return self.executive_controller.run_turn_state_machine(
            state=state,
            decision=executive_decision,
            has_explicit_action_cue=self._is_tool_route(route)
            and not approval_required,
            has_active_goal=bool(metadata.get("active_goals")),
            pre_route_blocked=approval_required or route == "clarify",
            tool_vetoed=route == "refuse",
        )

    def _build_turn_execution_outcome(
        self,
        *,
        turn_state_machine: TurnStateMachineResult,
        decision: Any,
        tool_result: Any,
        verification: Optional[VerifierResult],
        response_text: str,
        action_discipline: Dict[str, Any],
    ) -> Tuple[TurnExecutionOutcome, Dict[str, Any]]:
        route = str(getattr(decision, "route", "") or "").strip().lower()
        route_is_tool = self._is_tool_route(route)
        approval_required = bool(action_discipline.get("approval_required"))
        tool_success = bool(
            route_is_tool and tool_result is not None and tool_result.success
        )

        verification_payload = self._verification_to_dict(verification)
        verification_accepted = bool(verification_payload["accepted"])

        goal_advanced = bool(
            verification_accepted
            and (
                tool_success
                if route_is_tool
                else bool(str(response_text or "").strip())
            )
        )
        if approval_required:
            goal_advanced = False

        retryable = False
        if route_is_tool and tool_result is not None and not tool_result.success:
            retryable = bool(
                not action_discipline.get("retried")
                and self.companion_runtime.policy_engine.is_transient_tool_error(
                    tool_result
                )
            )

        execution_report = self.execution_verifier.verify_task_result(
            intent=str(getattr(decision, "intent", "") or ""),
            result=response_text,
            all_results={
                "route": route,
                "tool": str(getattr(tool_result, "tool_name", "") or ""),
                "tool_error": str(getattr(tool_result, "error", "") or ""),
            },
            success_criteria=list(turn_state_machine.contract.success_criteria or []),
            outcome={
                "tool_success": tool_success,
                "goal_advanced": goal_advanced,
                "verification_passed": verification_accepted,
            },
        )

        verification_passed = bool(verification_accepted and execution_report.accepted)
        combined_issues = self._merge_issue_lists(
            execution_report.issues,
            []
            if verification_accepted
            else [str(verification_payload.get("reason") or "verification_failed")],
        )

        if approval_required:
            recommended_next_action = "respond"
        elif route_is_tool and not verification_passed:
            recommended_next_action = "retry" if retryable else "escalate"
        elif not verification_passed:
            recommended_next_action = "replan"
        elif route_is_tool and goal_advanced:
            recommended_next_action = "continue"
        else:
            recommended_next_action = "respond"

        verification_confidence = min(
            float(verification_payload.get("confidence") or 1.0),
            float(execution_report.confidence or 0.0),
        )

        outcome = self.executive_controller.build_execution_outcome(
            contract=turn_state_machine.contract,
            tool_success=tool_success,
            goal_advanced=goal_advanced,
            verification_passed=verification_passed,
            recommended_next_action=recommended_next_action,
            retryable=retryable,
            issues=combined_issues,
            verification_confidence=verification_confidence,
            metadata={
                "plugin": str(getattr(tool_result, "tool_name", "") or ""),
                "action": str(getattr(tool_result, "action", "") or ""),
                "status": (
                    "skipped"
                    if (not route_is_tool or tool_result is None)
                    else ("ok" if tool_success else "failed")
                ),
                "route": route,
            },
        )

        return outcome, execution_report.to_dict()

    @staticmethod
    def _with_verified_execution_suffix(
        response_text: str,
        *,
        tool_result: Any,
        outcome: TurnExecutionOutcome,
        action_discipline: Dict[str, Any],
    ) -> str:
        return str(response_text or "").strip()

    @staticmethod
    def _merge_unique(base_items: list[str], new_items: list[str]) -> list[str]:
        merged = list(base_items)
        seen = {str(item).strip().lower() for item in merged if str(item).strip()}
        for raw in new_items:
            token = str(raw or "").strip()
            if not token:
                continue
            key = token.lower()
            if key in seen:
                continue
            merged.append(token)
            seen.add(key)
        return merged

    @staticmethod
    def _stage(
        name: str, status: str, details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return {
            "name": name,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "details": dict(details or {}),
        }

    def _append_routing_failure(
        self,
        *,
        trace_id: str,
        user_input: str,
        final_route: str,
        final_intent: str,
        plan: Dict[str, Any],
        verification_reason: str = "",
        veto_reason: str = "",
        operator_context: Optional[Dict[str, Any]] = None,
        operator_state: Optional[Dict[str, Any]] = None,
        local_execution: Optional[Dict[str, Any]] = None,
        agent_loop: Optional[Dict[str, Any]] = None,
        response_excerpt: str = "",
    ) -> None:
        try:
            path = Path("data/routing_failures.jsonl")
            path.parent.mkdir(parents=True, exist_ok=True)
            routing_trace = dict(plan.get("routing_trace") or {})
            payload = {
                "timestamp": datetime.utcnow().isoformat(),
                "trace_id": str(trace_id or ""),
                "user_input": str(user_input or ""),
                "final_route": str(final_route or ""),
                "final_intent": str(final_intent or ""),
                "candidates": list(routing_trace.get("candidates") or []),
                "evidence_contract_results": list(routing_trace.get("evidence_contract_results") or []),
                "veto_reason": str(veto_reason or routing_trace.get("reason") or ""),
                "verification_reason": str(verification_reason or ""),
                "operator_context": dict(operator_context or {}),
                "operator_state": dict(operator_state or {}),
                "local_execution": dict(local_execution or {}),
                "agent_loop": dict(agent_loop or {}),
                "response_excerpt": str(response_excerpt or "")[:240],
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception:
            return

    async def _run_turn_async(
        self, user_input: str, user_id: str, turn_number: int = 0
    ) -> PipelineResult:
        trace_id = str(uuid4())
        started = time.perf_counter()
        stages = []
        if not user_input.strip():
            stages.append(self._stage("input", "failed", {"reason": "empty_input"}))
            return PipelineResult(
                handled=False,
                response_text="",
                metadata={
                    "reason": "empty_input",
                    "trace_id": trace_id,
                    "stages": stages,
                },
            )

        stages.append(self._stage("input", "ok", {"length": len(user_input)}))

        user_state_snapshot = self.user_state_model.get_or_create(user_id)
        companion_state = self.companion_runtime.start_turn(
            user_id=user_id,
            user_input=user_input,
            turn_number=turn_number,
            user_state=user_state_snapshot,
        )

        route_phase = self.orchestrator.route_phase(
            user_input=user_input,
            user_id=user_id,
            turn_number=turn_number,
        )
        decision = route_phase.decision
        resolved_input = route_phase.resolved_input
        memory = route_phase.memory
        plan = dict(route_phase.plan or {})
        routing_trace = dict(getattr(decision, "metadata", {}) or {}).get("routing_trace")
        if isinstance(routing_trace, dict) and routing_trace:
            plan["routing_trace"] = dict(routing_trace)
        policy = self.companion_runtime.decide(
            user_input=user_input,
            route_decision=decision,
            companion_state=companion_state,
        )
        route_phase, policy, route_veto = self._apply_contextual_reaction_pre_tool_veto(
            user_input=user_input,
            route_phase=route_phase,
            policy=policy,
            companion_state=companion_state,
        )
        decision = route_phase.decision
        resolved_input = route_phase.resolved_input
        memory = route_phase.memory
        plan = dict(route_phase.plan or {})
        routing_trace = dict(getattr(decision, "metadata", {}) or {}).get("routing_trace")
        if isinstance(routing_trace, dict) and routing_trace:
            plan["routing_trace"] = dict(routing_trace)
        if bool(route_veto.get("applied")):
            plan["route_veto"] = dict(route_veto)
        plan["policy_decision"] = policy.decision_type
        if str(decision.route or "") == "clarify":
            self._append_routing_failure(
                trace_id=trace_id,
                user_input=user_input,
                final_route=decision.route,
                final_intent=decision.intent,
                plan=plan,
                veto_reason="route_clarify",
                operator_state=dict((decision.metadata or {}).get("operator_state") or {}),
            )
        routing_trace_for_log = dict(plan.get("routing_trace") or {})
        if bool(routing_trace_for_log.get("file_tool_vetoed")):
            self._append_routing_failure(
                trace_id=trace_id,
                user_input=user_input,
                final_route=decision.route,
                final_intent=decision.intent,
                plan=plan,
                veto_reason=str(routing_trace_for_log.get("reason") or "tool_route_vetoed"),
                operator_state=dict((decision.metadata or {}).get("operator_state") or {}),
            )
        low_input = str(user_input or "").lower()
        if any(token in low_input for token in ("why did you do that", "that's wrong", "you misunderstood")):
            self._append_routing_failure(
                trace_id=trace_id,
                user_input=user_input,
                final_route=decision.route,
                final_intent=decision.intent,
                plan=plan,
                veto_reason="user_reported_misroute",
                operator_state=dict((decision.metadata or {}).get("operator_state") or {}),
            )

        stages.append(
            self._stage(
                "route",
                "ok",
                {
                    "intent": decision.intent,
                    "confidence": decision.confidence,
                    "route": decision.route,
                    "decision_band": decision.decision_band,
                    "memory_count": len(memory.items),
                    "memory_confidence": memory.confidence,
                    "memory_metadata": dict(memory.metadata or {}),
                    "resolved_input": resolved_input,
                    "policy_decision": policy.decision_type,
                    "policy_reason": policy.reason,
                    "tool_execution_disabled": bool(
                        route_veto.get("tool_execution_disabled")
                    ),
                    "routing_trace": dict(plan.get("routing_trace") or {}),
                    "plan": plan,
                },
            )
        )

        execute_phase, action_discipline = (
            self.companion_runtime.execute_with_discipline(
                orchestrator=self.orchestrator,
                route_phase=route_phase,
                policy=policy,
            )
        )
        action_discipline = dict(action_discipline or {})
        action_discipline["policy_decision"] = policy.decision_type
        tool_result = execute_phase.tool_result

        if bool(action_discipline.get("approval_required")):
            stages.append(
                self._stage(
                    "execute",
                    "skipped",
                    {
                        "route": decision.route,
                        "approval_required": True,
                        "approval_reason": action_discipline.get("approval_reason"),
                    },
                )
            )
        elif execute_phase.executed and tool_result is not None:
            tool_payload = tool_result.data if isinstance(tool_result.data, dict) else {}
            tool_data = (
                tool_payload.get("data")
                if isinstance(tool_payload.get("data"), dict)
                else {}
            )
            tool_error_detail = str(
                tool_payload.get("error")
                or tool_data.get("error")
                or tool_data.get("message_code")
                or ""
            ).strip()
            stages.append(
                self._stage(
                    "execute",
                    "ok" if tool_result.success else "failed",
                    {
                        "tool": tool_result.tool_name,
                        "action": tool_result.action,
                        "error": tool_result.error,
                        "error_detail": tool_error_detail,
                        "confidence": tool_result.confidence,
                        "schema_validation": str(
                            (tool_result.diagnostics or {}).get("stage") or "ok"
                        ),
                        "attempt_count": int(
                            action_discipline.get("attempt_count") or 1
                        ),
                        "retried": bool(action_discipline.get("retried")),
                    },
                )
            )
        else:
            stages.append(
                self._stage(
                    "execute",
                    "skipped",
                    {
                        "route": decision.route,
                        "attempt_count": int(
                            action_discipline.get("attempt_count") or 0
                        ),
                        "retried": bool(action_discipline.get("retried")),
                    },
                )
            )

        verification: Optional[VerifierResult] = None
        respond_requires_follow_up = False
        respond_metadata: Dict[str, Any] = {}
        follow_up_question = ""

        if bool(action_discipline.get("approval_required")):
            verification = VerifierResult(
                accepted=True,
                reason="approval_required",
                confidence=1.0,
                diagnostics={
                    "policy_decision": policy.decision_type,
                    "approval_reason": str(
                        action_discipline.get("approval_reason") or ""
                    ),
                },
            )
            stages.append(
                self._stage(
                    "verify",
                    "ok",
                    {
                        "reason": verification.reason,
                        "confidence": verification.confidence,
                        "diagnostics": dict(verification.diagnostics),
                    },
                )
            )
            response_text = self.companion_runtime.build_approval_response(
                policy=policy,
                decision=decision,
            )
            respond_requires_follow_up = True
            follow_up_question = "Do you explicitly approve this action?"
            respond_metadata = {
                "type": "approval_request",
                "follow_up_question": follow_up_question,
            }
        else:
            verify_phase = self.orchestrator.verify_phase(
                user_input=user_input,
                route_phase=route_phase,
                execute_phase=execute_phase,
                trace_id=trace_id,
            )
            verification = verify_phase.verification

            if verification is None:
                stages.append(
                    self._stage("verify", "skipped", {"reason": "no_verifier"})
                )
            else:
                stages.append(
                    self._stage(
                        "verify",
                        "ok" if verification.accepted else "failed",
                        {
                            "reason": verification.reason,
                            "confidence": verification.confidence,
                            "diagnostics": dict(verification.diagnostics),
                        },
                    )
                )
                if not verification.accepted:
                    self._append_routing_failure(
                        trace_id=trace_id,
                        user_input=user_input,
                        final_route=decision.route,
                        final_intent=decision.intent,
                        plan=plan,
                        verification_reason=str(verification.reason or ""),
                        veto_reason="verification_failed",
                        operator_context=dict(((tool_result.diagnostics or {}).get("operator_context") if tool_result else {}) or {}),
                        operator_state=dict((decision.metadata or {}).get("operator_state") or {}),
                        local_execution=dict(((tool_result.diagnostics or {}).get("local_execution") if tool_result else {}) or {}),
                    )

            respond_phase = self.orchestrator.respond_phase(verify_phase=verify_phase)
            response_text = self.companion_runtime.shape_response(
                response_text=str(respond_phase.response_text or "").strip(),
                policy=policy,
            )
            respond_requires_follow_up = bool(respond_phase.requires_follow_up)
            respond_metadata = dict(respond_phase.metadata or {})
            follow_up_question = str(
                respond_metadata.get("follow_up_question") or ""
            ).strip()

        if policy.decision_type in {"follow_up", "clarify"}:
            respond_requires_follow_up = True

        if respond_requires_follow_up and not follow_up_question:
            follow_up_question = self.companion_runtime.default_follow_up_question(
                policy=policy
            )
            respond_metadata["follow_up_question"] = follow_up_question

        turn_state_machine = self._build_pre_execution_state_machine(
            user_input=user_input,
            decision=decision,
            action_discipline=action_discipline,
        )
        turn_execution_outcome, task_verification = self._build_turn_execution_outcome(
            turn_state_machine=turn_state_machine,
            decision=decision,
            tool_result=tool_result,
            verification=verification,
            response_text=response_text,
            action_discipline=action_discipline,
        )
        post_execution_state_machine = (
            self.executive_controller.run_post_execution_state_machine(
                pre_execution=turn_state_machine,
                outcome=turn_execution_outcome,
            )
        )

        if not turn_execution_outcome.verification_passed:
            response_text = (
                "I could not verify that result safely. "
                "Please rephrase the request or provide more detail."
            )
            respond_requires_follow_up = True
            respond_metadata = {
                **dict(respond_metadata or {}),
                "fallback": "execution_verifier_guard",
            }
            follow_up_question = str(
                respond_metadata.get("follow_up_question")
                or follow_up_question
                or self.companion_runtime.default_follow_up_question(policy=policy)
            ).strip()
            if follow_up_question:
                respond_metadata["follow_up_question"] = follow_up_question
        else:
            response_text = self._with_verified_execution_suffix(
                response_text,
                tool_result=tool_result,
                outcome=turn_execution_outcome,
                action_discipline=action_discipline,
            )

        if str((respond_metadata or {}).get("type") or "") in {"fallback", "code_request_fallback"} or str((respond_metadata or {}).get("fallback") or ""):
            self._append_routing_failure(
                trace_id=trace_id,
                user_input=user_input,
                final_route=decision.route,
                final_intent=decision.intent,
                plan=plan,
                veto_reason="fallback_response_used",
                operator_context=dict(((tool_result.diagnostics or {}).get("operator_context") if tool_result else {}) or {}),
                operator_state=dict((decision.metadata or {}).get("operator_state") or {}),
                local_execution=dict(((tool_result.diagnostics or {}).get("local_execution") if tool_result else {}) or {}),
                response_excerpt=response_text,
            )

        stages.append(
            self._stage(
                "respond",
                "ok" if response_text else "failed",
                {
                    "requires_follow_up": respond_requires_follow_up,
                    "verification_passed": turn_execution_outcome.verification_passed,
                    "post_execution_phase": post_execution_state_machine.phase,
                    **dict(respond_metadata or {}),
                },
            )
        )

        memory_domains = self.companion_runtime.update_after_turn(
            companion_state=companion_state,
            user_input=user_input,
            response_text=response_text,
            route_decision=decision,
            policy=policy,
            verification=verification,
            requires_follow_up=respond_requires_follow_up,
            follow_up_question=follow_up_question,
            tool_result=tool_result,
            action_discipline=action_discipline,
        )
        local_exec_payload = dict(((tool_result.diagnostics or {}).get("local_execution") if tool_result else {}) or {})
        operator_state_payload = dict((decision.metadata or {}).get("operator_state") or {})
        next_step = decide_next_step(
            route=str(decision.route or ""),
            intent=str(decision.intent or ""),
            operator_state=operator_state_payload,
            local_execution=local_exec_payload,
            available_files=list((local_exec_payload.get("suggested_next_files") or [])),
        )
        agent_loop_payload = build_agent_loop_state(
            user_input=user_input,
            route=str(decision.route or ""),
            intent=str(decision.intent or ""),
            local_execution=local_exec_payload,
            active_objective=str(operator_state_payload.get("active_objective") or ""),
        )
        if str(local_exec_payload.get("error") or "") == "target_not_found":
            self._append_routing_failure(
                trace_id=trace_id,
                user_input=user_input,
                final_route=decision.route,
                final_intent=decision.intent,
                plan=plan,
                verification_reason=str((verification.reason if verification else "") or ""),
                veto_reason="local_execution_target_not_found",
                operator_context=dict(((tool_result.diagnostics or {}).get("operator_context") if tool_result else {}) or {}),
                operator_state=operator_state_payload,
                local_execution=local_exec_payload,
                agent_loop=agent_loop_payload,
                response_excerpt=response_text,
            )

        memory_payload = {
            "content": f"user={user_input}\nassistant={response_text}",
            "intent": decision.intent,
            "route": decision.route,
            "confidence": decision.confidence,
            "trace_id": trace_id,
            "resolved_input": resolved_input,
            "memory_domains": memory_domains,
            "turn_contract": post_execution_state_machine.contract.as_dict(),
            "turn_execution_outcome": turn_execution_outcome.as_dict(),
            "post_execution_state_machine": post_execution_state_machine.as_dict(),
            "task_verification": dict(task_verification or {}),
        }
        memory_plan = self.memory_turn_service.build_memory_plan(
            user_input=user_input,
            user_name=str(getattr(user_state_snapshot, "user_name", "") or user_id or "User"),
            trace_id=trace_id,
            decision_intent=decision.intent,
            decision_route=decision.route,
            episodic_payload=memory_payload,
        )

        async def _store_memory_bundle() -> None:
            await asyncio.to_thread(
                self.memory_turn_service.store_memory_plan,
                boundaries=self.boundaries,
                plan=memory_plan,
            )

        memory_task = asyncio.create_task(_store_memory_bundle())

        merged_active_goals = self._merge_unique(
            list((decision.metadata or {}).get("active_goals", []) or []),
            list(memory_domains.get("projects", []) or []),
        )

        state = self.user_state_model.update_turn(
            user_id=user_id,
            intent=decision.intent,
            route=decision.route,
            unresolved_references=list(
                (decision.metadata or {}).get("pronouns", []) or []
            ),
            active_goals=merged_active_goals,
            last_tool_used=(tool_result.tool_name if tool_result else ""),
            last_result_produced=response_text[:240],
            world_state_snapshot={
                "trace_id": trace_id,
                "route": decision.route,
                "intent": decision.intent,
                "verified": bool(turn_execution_outcome.verification_passed),
                "policy_decision": policy.decision_type,
                "post_execution_phase": post_execution_state_machine.phase,
                "recommended_next_action": turn_execution_outcome.recommended_next_action,
            },
        )
        stages.append(
            self._stage(
                "state_update",
                "ok",
                {
                    "current_task": state.current_task,
                    "prior_task": state.prior_task,
                    "last_tool_used": state.last_tool_used,
                    "project_count": len(memory_domains.get("projects", []) or []),
                    "unresolved_thread_count": len(
                        memory_domains.get("unresolved_threads", []) or []
                    ),
                },
            )
        )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        with tracer.start_as_current_span("contract_pipeline.run_turn"):
            turn_counter.add(1)
            turn_latency.record(elapsed_ms)

        metrics_task = asyncio.create_task(asyncio.sleep(0))
        await asyncio.gather(memory_task, metrics_task)

        return PipelineResult(
            handled=bool(response_text),
            response_text=response_text,
            metadata={
                "trace_id": trace_id,
                "route": decision.route,
                "intent": decision.intent,
                "decision_band": decision.decision_band,
                "confidence": decision.confidence,
                "response_type": str((respond_metadata or {}).get("type") or "response"),
                "requires_follow_up": respond_requires_follow_up,
                "tools_used": [tool_result.tool_name] if tool_result else [],
                "plan": plan,
                "memory_recall": dict(memory.metadata or {}),
                "resolved_input": resolved_input,
                "verification": {
                    "accepted": verification.accepted if verification else True,
                    "reason": verification.reason if verification else "not_configured",
                    "confidence": verification.confidence if verification else 1.0,
                    "diagnostics": (
                        dict(verification.diagnostics) if verification else {}
                    ),
                },
                "turn_contract": post_execution_state_machine.contract.as_dict(),
                "turn_execution_outcome": turn_execution_outcome.as_dict(),
                "post_execution_state_machine": post_execution_state_machine.as_dict(),
                "task_verification": dict(task_verification or {}),
                "companion": {
                    "policy_decision": policy.decision_type,
                    "policy_reason": policy.reason,
                    "identity_model": dict(memory_domains.get("identity", {}) or {}),
                    "memory_domains": memory_domains,
                    "last_tool_result": dict(companion_state.last_tool_result or {}),
                    "last_user_state_signals": list(
                        companion_state.last_user_state_signals or []
                    ),
                    "action_discipline": {
                        "retry_count": int(action_discipline.get("attempt_count") or 0),
                        "retried": bool(action_discipline.get("retried")),
                        "approval_required": bool(
                            action_discipline.get("approval_required")
                        ),
                        "approval_reason": str(
                            action_discipline.get("approval_reason") or ""
                        ),
                    },
                },
                "memory_extraction": {
                    **dict(memory_plan.get("memory_extraction") or {}),
                },
                "operator_context": dict(
                    ((tool_result.diagnostics or {}).get("operator_context") if tool_result else {})
                    or {}
                ),
                "operator_state": operator_state_payload,
                "local_execution": dict(
                    ((tool_result.diagnostics or {}).get("local_execution") if tool_result else {})
                    or {}
                ),
                "next_step_policy": next_step.to_dict(),
                "agent_loop": agent_loop_payload,
                "state": {
                    "current_task": state.current_task,
                    "prior_task": state.prior_task,
                    "unresolved_references": list(state.unresolved_references),
                    "active_goals": list(state.active_goals),
                    "last_tool_used": state.last_tool_used,
                    "last_result_produced": state.last_result_produced,
                },
                "latency_ms": elapsed_ms,
                "stages": stages,
            },
        )

    def run_turn(
        self,
        user_input: str,
        user_id: str,
        turn_number: int = 0,
    ) -> PipelineResult | "asyncio.Future[PipelineResult]":
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self._run_turn_async(
                    user_input=user_input,
                    user_id=user_id,
                    turn_number=turn_number,
                )
            )
        return self._run_turn_async(
            user_input=user_input,
            user_id=user_id,
            turn_number=turn_number,
        )

    def run_turn_sync(
        self,
        user_input: str,
        user_id: str,
        turn_number: int = 0,
    ) -> PipelineResult:
        return asyncio.run(
            self._run_turn_async(
                user_input=user_input,
                user_id=user_id,
                turn_number=turn_number,
            )
        )
