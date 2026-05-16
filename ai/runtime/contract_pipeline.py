"""Contract-driven runtime pipeline used to thin app/main orchestration."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, replace
from datetime import datetime
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
from ai.runtime.pipeline.metadata_builder import PipelineMetadataBuilder
from ai.runtime.pipeline.routing_failure_logger import RoutingFailureLogger
from ai.runtime.greeting_surface_policy import render_grounded_greeting, validate_chat_greeting
from ai.runtime.operator_response_surface import (
    normalize_response_paragraphs,
    strip_meta_response_artifacts,
)
from ai.runtime.response_momentum_policy import (
    apply_response_momentum,
    strip_passive_followup_sentences,
)
from ai.runtime.claim_verifier import verify_response_claims
from ai.runtime.action_bus import action_result_from_local_execution
from ai.runtime.turn_orchestrator import TurnOrchestrator
from ai.runtime.user_state_model import UserStateModel
from ai.runtime.turn_mode_policy import classify_turn_mode
from ai.memory.project_memory import load_project_state, update_project_state
from ai.runtime.self_improvement.improvement_loop import ImprovementLoop
from ai.core.constraint_preference_extractor import ConstraintPreferenceExtractor
from ai.runtime.companion_state import (
    load_companion_state,
    sync_from_operator_project_state,
    update_companion_state,
)


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
        self.routing_failure_logger = RoutingFailureLogger()
        self._greeting_session_state_by_user: Dict[str, Dict[str, Any]] = {}
        self.constraint_extractor = ConstraintPreferenceExtractor()

    @staticmethod
    def _normalize_design_constraints(raw_constraints: List[str]) -> List[str]:
        mapping = {
            "no_external_apis": "no external APIs",
            "local_first": "local-first",
            "unique_architecture": "unique architecture",
            "frameworks_as_reference_only": "frameworks as references only",
            "avoid_bundled_framework_feel": "avoid bundled-framework feel",
            "maximum_value_minimal_effort": "maximum value, minimal effort",
            "advanced_engineering_lean_implementation": "advanced engineering, lean implementation",
        }
        out: List[str] = []
        for token in list(raw_constraints or []):
            label = mapping.get(str(token or "").strip())
            if label and label not in out:
                out.append(label)
        return out

    def _maybe_record_behavior_event(
        self,
        *,
        user_id: str,
        source: str,
        user_input: str,
        alice_response: str = "",
        route: str = "",
        intent: str = "",
        trace_id: str = "",
        failure_kind: str = "unknown",
        symptom: str = "",
        expected_behavior: str = "",
        actual_behavior: str = "",
        severity: str = "medium",
        evidence: Optional[Dict[str, Any]] = None,
        related_files: Optional[List[str]] = None,
    ) -> None:
        try:
            loop = ImprovementLoop(user_id=user_id)
            event = loop.observe_event(
                source=source,
                user_input=user_input,
                alice_response=alice_response,
                route=route,
                intent=intent,
                trace_id=trace_id,
                failure_kind=failure_kind,
                symptom=symptom,
                expected_behavior=expected_behavior,
                actual_behavior=actual_behavior,
                severity=severity,
                evidence=dict(evidence or {}),
                related_files=list(related_files or []),
                user_id=user_id,
            )
            loop.maybe_auto_audit(event)
        except Exception:
            return

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

    @staticmethod
    def _resolve_response_surface(
        *,
        decision_intent: str,
        decision_route: str,
        response_type: str,
    ) -> str:
        intent = str(decision_intent or "").strip().lower()
        route = str(decision_route or "").strip().lower()
        rtype = str(response_type or "").strip().lower()
        if intent.endswith("greeting") or intent == "greeting" or rtype == "greeting_grounded":
            return "greeting"
        if "educational" in intent or "educational" in rtype or "research_explain" in intent:
            return "educational"
        if "approval" in rtype:
            return "approval"
        if rtype in {"local_execution_error", "weather_tool_fallback", "fallback"}:
            return "error"
        if intent.startswith("operator:") or intent.startswith("code:") or route == "local":
            return "operator"
        if route in {"tool", "plugin"} or rtype in {"tool_response", "local_code_request"}:
            return "tool_result"
        if intent.startswith("conversation:") or route in {"llm", "conversation"}:
            return "casual"
        return "error"

    @staticmethod
    def _greeting_task_intake_tokens() -> Tuple[str, ...]:
        return (
            "what should we work on",
            "what would you like to work on",
            "what are we working on",
            "what would you like to start",
            "what do you want to work on",
            "how can i help",
            "how may i assist",
            "what can i do",
            "what do you need",
            "ready when you are",
            "let me know",
        )

    @staticmethod
    def _greeting_soft_continuity_tokens() -> Tuple[str, ...]:
        return (
            "see you again",
            "great to see you again",
            "good to see you again",
            "nice to see you again",
            "back again",
            "here again",
        )

    @classmethod
    def _contains_task_intake_greeting(cls, text: str) -> bool:
        low = str(text or "").lower()
        return any(token in low for token in cls._greeting_task_intake_tokens())

    @classmethod
    def _contains_soft_continuity_greeting(cls, text: str) -> bool:
        low = str(text or "").lower()
        return any(token in low for token in cls._greeting_soft_continuity_tokens())

    @staticmethod
    def _is_pure_greeting_input(user_input: str) -> bool:
        normalized = " ".join(str(user_input or "").lower().replace(",", " ").split())
        return normalized in {
            "hi",
            "hi alice",
            "hey",
            "hey alice",
            "hello",
            "hello alice",
            "yo",
            "yo alice",
        }

    @staticmethod
    def is_clear_concept_breakdown_request(
        user_input: str, active_concept_thread: Dict[str, Any] | None = None
    ) -> bool:
        low = str(user_input or "").lower().strip()
        if not low:
            return False
        phrases = (
            "break this down",
            "break it down",
            "with today's technology",
            "with todays technology",
            "explain the layers",
            "go deeper",
            "what would that look like",
            "how would that work",
            "what are the parts",
            "what would we need",
            "break",
            "layers",
        )
        if any(phrase in low for phrase in phrases):
            return True
        has_thread = bool(
            str((active_concept_thread or {}).get("topic") or "").strip()
        )
        if not has_thread:
            return False
        short_followup = len([w for w in low.split() if w]) <= 7
        if short_followup and any(
            token in low for token in ("this", "that", "it", "exactly", "deeper")
        ):
            return True
        return False

    @staticmethod
    def _concept_breakdown_skeleton() -> str:
        return (
            "Break it into layers:\n\n"
            "1. Model brain\n"
            "2. Memory\n"
            "3. Tools\n"
            "4. Background event loop\n"
            "5. Relevance filter\n"
            "6. Planner\n"
            "7. Approval layer\n"
            "8. Notification/UI layer"
        )

    @staticmethod
    def _contains_generic_clarification_fallback(text: str) -> bool:
        low = str(text or "").lower()
        blocked = (
            "what exact result do you want",
            "can you clarify",
            "what would you like me to focus on",
            "tell me more about what you mean",
        )
        return any(token in low for token in blocked)

    def _retry_concept_refinement_breakdown(
        self,
        *,
        user_input: str,
        active_concept_thread: Dict[str, Any] | None = None,
    ) -> str:
        llm_generate_fn = getattr(self.boundaries.response, "llm_generate_fn", None)
        if not llm_generate_fn:
            return ""
        concept = dict(active_concept_thread or {})
        prompt = (
            "You are Alice.\n"
            "The user is asking to break down the active concept with today's technology.\n"
            "Answer directly.\n"
            "Do not ask what result they want.\n"
            "Do not mention files or codebase unless the user asks to implement it.\n"
            "Keep it conceptual and practical.\n"
            "Use a layered breakdown.\n"
            "Return only the answer text.\n\n"
            f"Active concept thread: {concept or {'topic': 'proactive AI companion'}}\n"
            f"User message:\n{str(user_input or '').strip()}\n"
        )
        try:
            try:
                out = str(llm_generate_fn(prompt=prompt) or "").strip()
            except TypeError:
                out = str(llm_generate_fn(prompt) or "").strip()
        except Exception:
            return ""
        out = strip_meta_response_artifacts(out)
        out = normalize_response_paragraphs(out)
        if self._contains_generic_clarification_fallback(out):
            return ""
        return str(out or "").strip()

    @classmethod
    def _repair_greeting_task_intake(
        cls,
        *,
        user_input: str,
        current_text: str,
        llm_generate_fn: Any,
    ) -> Tuple[str, Dict[str, Any]]:
        current = str(current_text or "").strip()
        has_task_intake = cls._contains_task_intake_greeting(current)
        has_soft_continuity = cls._contains_soft_continuity_greeting(current)
        if not has_task_intake and not has_soft_continuity:
            return current, {
                "applied": False,
                "retry_attempted": False,
                "accepted": bool(current),
                "reasons": [],
            }
        reasons: list[str] = []
        if has_task_intake:
            reasons.extend(["task_intake_greeting", "assistant_service_language"])
        if has_soft_continuity:
            reasons.append("unsupported_soft_continuity_claim")
        if not llm_generate_fn:
            return "", {
                "applied": True,
                "retry_attempted": False,
                "accepted": False,
                "reasons": reasons,
            }
        reasons_line = ", ".join(dict.fromkeys(reasons))
        extra_constraints = ""
        if has_soft_continuity:
            extra_constraints = (
                "Do not say 'again', 'back again', or imply prior conversation continuity.\n"
            )
        prompt = (
            "You are Alice speaking to Gabriel.\n\n"
            "The previous greeting was rejected for:\n"
            f"{reasons_line}\n\n"
            "Write a natural greeting only.\n"
            "Use only the current user message.\n"
            "Do not ask what to work on.\n"
            "Do not ask how you can help.\n"
            "Do not use task-intake language.\n"
            f"{extra_constraints}"
            "Keep it 1-3 short sentences.\n"
            "Return only the greeting.\n\n"
            f"User message:\n{str(user_input or '').strip()}\n"
        )
        try:
            try:
                candidate = str(llm_generate_fn(prompt=prompt) or "").strip()
            except TypeError:
                candidate = str(llm_generate_fn(prompt) or "").strip()
        except Exception:
            candidate = ""
        if (
            not candidate
            or cls._contains_task_intake_greeting(candidate)
            or cls._contains_soft_continuity_greeting(candidate)
        ):
            return "", {
                "applied": True,
                "retry_attempted": True,
                "accepted": False,
                "reasons": reasons,
            }
        validation = validate_chat_greeting(candidate, pure_greeting=True)
        if not validation.valid:
            return "", {
                "applied": True,
                "retry_attempted": True,
                "accepted": False,
                "reasons": list(validation.reasons or reasons),
            }
        return candidate, {
            "applied": True,
            "retry_attempted": True,
            "accepted": True,
            "reasons": reasons,
        }

    @classmethod
    def _build_response_generation_metadata(
        cls,
        *,
        decision_intent: str,
        decision_route: str,
        response_type: str,
        generated_by: str,
        respond_metadata: Dict[str, Any],
        response_generation_details: Dict[str, Any],
        llm_generate_fn: Any,
        llm_model_name: str,
        claim_verifier_applied: bool,
    ) -> Dict[str, Any]:
        surface = cls._resolve_response_surface(
            decision_intent=decision_intent,
            decision_route=decision_route,
            response_type=response_type,
        )
        generated = str(generated_by or "").strip().lower()
        operator_ack = dict(response_generation_details.get("operator_ack") or {})
        model_used = False
        if surface == "greeting":
            model_used = generated in {"llm", "llm_retry"}
        elif str(response_type or "").strip().lower() in {
            "llm_response",
            "educational_explain",
            "greeting_grounded",
        }:
            model_used = True
        elif surface == "operator":
            model_used = bool(operator_ack.get("model_used"))
        elif surface == "casual":
            model_used = str(response_type or "").strip().lower() == "llm_response"

        validation_applied = bool(
            surface == "greeting"
            or bool(operator_ack.get("validation_applied"))
            or bool((respond_metadata or {}).get("validation_passed") is not None)
        )
        fallback_used = bool(
            str(response_type or "").strip().lower() in {"fallback"}
            or bool((respond_metadata or {}).get("fallback"))
        )
        if surface in {"greeting", "casual", "educational", "operator"}:
            fallback_used = False
        model_name = str(llm_model_name or "").strip()
        if not model_name and llm_generate_fn:
            model_name = "alice-ollama"
        if not model_name:
            model_name = "none"
        return {
            "model_used": bool(model_used),
            "model_name": model_name,
            "surface": surface,
            "validation_applied": bool(validation_applied),
            "claim_verifier_applied": bool(claim_verifier_applied),
            "fallback_used": bool(fallback_used),
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
        routing_trace = dict(plan.get("routing_trace") or {})
        payload = self.routing_failure_logger.build_payload(
            trace_id=str(trace_id or ""),
            user_input=str(user_input or ""),
            final_route=str(final_route or ""),
            final_intent=str(final_intent or ""),
            candidates=list(routing_trace.get("candidates") or []),
            evidence_contract_results=list(
                routing_trace.get("evidence_contract_results") or []
            ),
            veto_reason=str(veto_reason or routing_trace.get("reason") or ""),
            verification_reason=str(verification_reason or ""),
            operator_context=dict(operator_context or {}),
            operator_state=dict(operator_state or {}),
            local_execution=dict(local_execution or {}),
            agent_loop=dict(agent_loop or {}),
            response_excerpt=str(response_excerpt or "")[:240],
        )
        self.routing_failure_logger.append(payload)

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
        extracted_constraints = self.constraint_extractor.extract(user_input).get(
            "constraints", []
        )
        design_constraints = self._normalize_design_constraints(
            list(extracted_constraints or [])
        )
        if design_constraints:
            try:
                update_project_state(
                    {"design_constraints": design_constraints},
                    user_id=str(user_id or "default"),
                )
            except Exception:
                pass

        user_state_snapshot = self.user_state_model.get_or_create(user_id)
        companion_state = self.companion_runtime.start_turn(
            user_id=user_id,
            user_input=user_input,
            turn_number=turn_number,
            user_state=user_state_snapshot,
        )
        low_input = str(user_input or "").lower()
        if any(
            phrase in low_input
            for phrase in (
                "that's wrong",
                "you misunderstood",
                "this feels forced",
                "too dry",
                "still assistant-like",
                "not accurate",
            )
        ):
            self._maybe_record_behavior_event(
                user_id=user_id,
                source="user_correction",
                user_input=user_input,
                trace_id=trace_id,
                failure_kind="unknown",
                symptom="User correction signal received.",
                expected_behavior="Grounded response aligned with user correction.",
                actual_behavior="User reported incorrect or forced behavior.",
                severity="medium",
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
        perception_frame = dict((decision.metadata or {}).get("perception_frame") or {})
        project_state_payload = load_project_state(str(user_id or "default")).to_dict()
        operator_state_for_companion = dict((decision.metadata or {}).get("operator_state") or {})
        companion_profile_state = sync_from_operator_project_state(
            operator_state_for_companion,
            project_state_payload,
            user_id=str(user_id or "default"),
        )
        if perception_frame:
            update_companion_state(
                {
                    "last_user_request": str(user_input or ""),
                    "last_actual_request": str(perception_frame.get("actual_request") or ""),
                    "active_topic": str(perception_frame.get("topic") or companion_profile_state.active_topic),
                    "energy_signal": str(perception_frame.get("user_energy_signal") or companion_profile_state.energy_signal),
                    "mood_signal": str(perception_frame.get("user_mood_signal") or companion_profile_state.mood_signal),
                    "time_context": str(perception_frame.get("time_reference") or companion_profile_state.time_context),
                    "user_context": {
                        "social_context": str(perception_frame.get("social_context") or ""),
                    },
                },
                user_id=str(user_id or "default"),
            )
            companion_profile_state = load_companion_state(str(user_id or "default"))
        routing_trace = dict(getattr(decision, "metadata", {}) or {}).get(
            "routing_trace"
        )
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
        routing_trace = dict(getattr(decision, "metadata", {}) or {}).get(
            "routing_trace"
        )
        if isinstance(routing_trace, dict) and routing_trace:
            plan["routing_trace"] = dict(routing_trace)
        if bool(route_veto.get("applied")):
            plan["route_veto"] = dict(route_veto)
        plan["policy_decision"] = policy.decision_type
        if str(decision.route or "") == "clarify":
            self._maybe_record_behavior_event(
                user_id=user_id,
                source="routing_failure_log",
                user_input=user_input,
                route=str(decision.route or ""),
                intent=str(decision.intent or ""),
                trace_id=trace_id,
                failure_kind="routing",
                symptom="route clarify unexpectedly",
                expected_behavior="Route should execute or respond when evidence is sufficient.",
                actual_behavior="Route selected clarify.",
                severity="medium",
                evidence={"routing_trace": dict(plan.get("routing_trace") or {})},
                related_files=[
                    "ai/core/routing/route_arbiter.py",
                    "ai/core/routing/evidence_contracts.py",
                ],
            )
            self._append_routing_failure(
                trace_id=trace_id,
                user_input=user_input,
                final_route=decision.route,
                final_intent=decision.intent,
                plan=plan,
                veto_reason="route_clarify",
                operator_state=dict(
                    (decision.metadata or {}).get("operator_state") or {}
                ),
            )
        routing_trace_for_log = dict(plan.get("routing_trace") or {})
        if bool(routing_trace_for_log.get("file_tool_vetoed")):
            self._append_routing_failure(
                trace_id=trace_id,
                user_input=user_input,
                final_route=decision.route,
                final_intent=decision.intent,
                plan=plan,
                veto_reason=str(
                    routing_trace_for_log.get("reason") or "tool_route_vetoed"
                ),
                operator_state=dict(
                    (decision.metadata or {}).get("operator_state") or {}
                ),
            )
        low_input = str(user_input or "").lower()
        if any(
            token in low_input
            for token in ("why did you do that", "that's wrong", "you misunderstood")
        ):
            self._append_routing_failure(
                trace_id=trace_id,
                user_input=user_input,
                final_route=decision.route,
                final_intent=decision.intent,
                plan=plan,
                veto_reason="user_reported_misroute",
                operator_state=dict(
                    (decision.metadata or {}).get("operator_state") or {}
                ),
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
            tool_payload = (
                tool_result.data if isinstance(tool_result.data, dict) else {}
            )
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
        unshaped_response_text = ""
        proposed_response_text = ""

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
            proposed_response_text = str(
                getattr(verify_phase.proposed_response, "text", "") or ""
            ).strip()

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
                        operator_context=dict(
                            (
                                (tool_result.diagnostics or {}).get("operator_context")
                                if tool_result
                                else {}
                            )
                            or {}
                        ),
                        operator_state=dict(
                            (decision.metadata or {}).get("operator_state") or {}
                        ),
                        local_execution=dict(
                            (
                                (tool_result.diagnostics or {}).get("local_execution")
                                if tool_result
                                else {}
                            )
                            or {}
                        ),
                    )

            respond_phase = self.orchestrator.respond_phase(verify_phase=verify_phase)
            unshaped_response_text = str(respond_phase.response_text or "").strip()
            response_text = self.companion_runtime.shape_response(
                response_text=unshaped_response_text,
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
            is_greeting_turn = str(getattr(decision, "intent", "") or "").endswith(
                "greeting"
            ) or str(getattr(decision, "intent", "") or "") == "greeting"
            active_concept_thread = dict(
                (decision.metadata or {}).get("active_concept_thread") or {}
            )
            verification_reason = str(verification.reason or "") if verification else ""
            verification_diag = dict(verification.diagnostics or {}) if verification else {}
            verification_reasons = [
                str(item or "").strip()
                for item in list(verification_diag.get("reasons") or [])
                if str(item or "").strip()
            ]
            turn_mode_for_verification = classify_turn_mode(
                user_input=user_input,
                intent=str(getattr(decision, "intent", "") or ""),
                route=str(getattr(decision, "route", "") or ""),
                operator_state=dict((decision.metadata or {}).get("operator_state") or {}),
                project_memory=load_project_state(str(user_id or "default")).to_dict(),
            )
            can_salvage_followup_claims = (
                verification_reason == "unsupported_claims"
                and turn_mode_for_verification in {"educational_explain", "clarification", "casual_companion"}
            )
            clear_concept_breakdown_request = self.is_clear_concept_breakdown_request(
                user_input=user_input,
                active_concept_thread=active_concept_thread,
            )
            is_concept_refinement_turn = (
                str(getattr(decision, "intent", "") or "").strip().lower()
                == "conversation:concept_refinement"
                or turn_mode_for_verification == "concept_refinement"
                or (
                    bool(active_concept_thread)
                    and clear_concept_breakdown_request
                )
                or (
                    clear_concept_breakdown_request
                    and str(getattr(decision, "route", "") or "").strip().lower()
                    == "llm"
                    and str(getattr(decision, "intent", "") or "")
                    .strip()
                    .lower()
                    .startswith("conversation:")
                )
            )
            clear_concept_breakdown = bool(
                is_concept_refinement_turn and clear_concept_breakdown_request
            )
            concept_refinement_repair_applied = False
            concept_refinement_skeleton_used = False
            if is_greeting_turn or (
                verification
                and str(verification.reason or "") == "unsupported_continuity_claim"
            ):
                op_state = dict((decision.metadata or {}).get("operator_state") or {})
                user_name = str(getattr(user_state_snapshot, "user_name", "") or "")
                greeting_session = dict(
                    self._greeting_session_state_by_user.get(str(user_id), {}) or {}
                )
                greeting = render_grounded_greeting(
                    user_name=user_name,
                    operator_state=op_state,
                    session_state=greeting_session,
                    user_input=user_input,
                    llm_generate=getattr(
                        self.boundaries.response, "llm_generate_fn", None
                    ),
                )
                self._greeting_session_state_by_user[str(user_id)] = dict(
                    greeting.session_state
                )
                response_text = str(greeting.text or "").strip()
            elif can_salvage_followup_claims:
                mode = (
                    "educational_explain"
                    if turn_mode_for_verification == "educational_explain"
                    else ("clarification" if turn_mode_for_verification == "clarification" else "companion_chat")
                )
                cleaned = strip_passive_followup_sentences(
                    str(proposed_response_text or unshaped_response_text or response_text),
                    mode=mode,
                )
                response_text = str(cleaned or "").strip()
                respond_requires_follow_up = bool(not response_text)
            elif clear_concept_breakdown:
                retried_concept = self._retry_concept_refinement_breakdown(
                    user_input=user_input,
                    active_concept_thread=active_concept_thread,
                )
                if retried_concept:
                    response_text = str(retried_concept or "").strip()
                    concept_refinement_repair_applied = True
                else:
                    response_text = self._concept_breakdown_skeleton()
                    concept_refinement_skeleton_used = True
                respond_requires_follow_up = False
            else:
                response_text = (
                    "I could not verify that result safely. "
                    "Please rephrase the request or provide more detail."
                )
            self._maybe_record_behavior_event(
                user_id=user_id,
                source="verification_failure",
                user_input=user_input,
                alice_response=response_text,
                route=str(decision.route or ""),
                intent=str(decision.intent or ""),
                trace_id=trace_id,
                failure_kind=(
                    "continuity_claim"
                    if verification
                    and str(verification.reason or "") == "unsupported_continuity_claim"
                    else "response_grounding"
                ),
                symptom=str(verification.reason or "verification_failed")
                if verification
                else "verification_failed",
                expected_behavior="Verified grounded response.",
                actual_behavior="Verifier rejected the proposed response.",
                severity="medium",
                evidence={
                    "verification_reason": str(verification.reason or "")
                    if verification
                    else "",
                    "verification_diagnostics": dict(verification.diagnostics or {})
                    if verification
                    else {},
                },
                related_files=[
                    "ai/runtime/continuity_claim_guard.py",
                    "ai/runtime/contract_pipeline.py",
                ],
            )
            if not (
                (can_salvage_followup_claims and response_text)
                or concept_refinement_repair_applied
                or concept_refinement_skeleton_used
            ):
                respond_requires_follow_up = True
            respond_metadata = {
                **dict(respond_metadata or {}),
                "fallback": "execution_verifier_guard",
            }
            if can_salvage_followup_claims and response_text:
                respond_metadata["fallback"] = ""
                respond_metadata["verification_rewrite"] = "removed_unsupported_followup_claims"
            if concept_refinement_repair_applied:
                respond_metadata["fallback"] = ""
                respond_metadata["verification_rewrite"] = "concept_refinement_retry"
            if concept_refinement_skeleton_used:
                respond_metadata["fallback"] = ""
                respond_metadata["verification_rewrite"] = "concept_refinement_skeleton"
            if is_greeting_turn or (
                verification
                and str(verification.reason or "") == "unsupported_continuity_claim"
            ):
                respond_metadata.update(
                    {
                        "greeting_memory_policy": "active_state_only",
                        "broad_memory_suppressed": True,
                        "active_objective_used": bool(
                            greeting.active_objective_used
                        ),
                        "greeting_style": str(greeting.greeting_style),
                        "suppressed_project_menu": bool(
                            greeting.suppressed_project_menu
                        ),
                        "repeated_greeting": bool(greeting.repeated_greeting),
                        "generated_by": str(greeting.generated_by),
                        "warmth_level": str(greeting.warmth_level),
                        "companion_tone": bool(greeting.companion_tone),
                        "assistant_like_prompt_suppressed": bool(
                            greeting.assistant_like_prompt_suppressed
                        ),
                        "validation_passed": bool(greeting.validation_passed),
                        "validation_reasons": list(greeting.validation_reasons),
                        "continuity_guard_applied": bool(greeting.continuity_guard_applied),
                        "continuity_claims": dict(greeting.continuity_claims or {}),
                        "llm_candidate_rejected": bool(greeting.llm_candidate_rejected),
                    }
                )
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

        if str((respond_metadata or {}).get("type") or "") in {
            "fallback",
        } or str((respond_metadata or {}).get("fallback") or ""):
            self._append_routing_failure(
                trace_id=trace_id,
                user_input=user_input,
                final_route=decision.route,
                final_intent=decision.intent,
                plan=plan,
                veto_reason="fallback_response_used",
                operator_context=dict(
                    (
                        (tool_result.diagnostics or {}).get("operator_context")
                        if tool_result
                        else {}
                    )
                    or {}
                ),
                operator_state=dict(
                    (decision.metadata or {}).get("operator_state") or {}
                ),
                local_execution=dict(
                    (
                        (tool_result.diagnostics or {}).get("local_execution")
                        if tool_result
                        else {}
                    )
                    or {}
                ),
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
        local_exec_payload = dict(
            (
                (tool_result.diagnostics or {}).get("local_execution")
                if tool_result
                else {}
            )
            or {}
        )
        operator_state_payload = dict(
            (decision.metadata or {}).get("operator_state") or {}
        )
        inspected_file = str(local_exec_payload.get("inspected_file") or "").strip()
        if inspected_file:
            existing_inspected = list(operator_state_payload.get("files_inspected") or [])
            if inspected_file not in existing_inspected:
                existing_inspected.append(inspected_file)
            operator_state_payload["files_inspected"] = existing_inspected
            operator_state_payload["last_inspected_file"] = inspected_file
            last_rec = dict(operator_state_payload.get("last_recommended_action") or {})
            if str(last_rec.get("target") or "").strip() == inspected_file:
                operator_state_payload["last_recommended_action"] = {}
                operator_state_payload["next_recommended_action"] = ""
        next_step = decide_next_step(
            route=str(decision.route or ""),
            intent=str(decision.intent or ""),
            operator_state=operator_state_payload,
            local_execution=local_exec_payload,
            available_files=list(
                (local_exec_payload.get("suggested_next_files") or [])
            ),
            memory_recall=dict(memory.metadata or {}),
            routing_trace=dict(plan.get("routing_trace") or {}),
            last_failure=str(local_exec_payload.get("error") or ""),
        )
        stored_recommended_action = dict(next_step.last_recommended_action or {})
        if stored_recommended_action and not stored_recommended_action.get("created_at"):
            stored_recommended_action["created_at"] = datetime.utcnow().isoformat()
        if stored_recommended_action or str(next_step.next_recommended_action or "").strip():
            try:
                update_project_state(
                    {
                        "next_recommended_action": str(next_step.next_recommended_action or ""),
                        "last_recommended_action": stored_recommended_action,
                        "suggested_next_files": list(next_step.suggested_next_files or []),
                    },
                    user_id=str(user_id or "default"),
                )
            except Exception:
                pass
        if stored_recommended_action:
            operator_state_payload["last_recommended_action"] = stored_recommended_action
        if next_step.suggested_next_files:
            operator_state_payload["suggested_next_files"] = list(next_step.suggested_next_files or [])
        if str(next_step.next_recommended_action or "").strip():
            operator_state_payload["next_recommended_action"] = str(next_step.next_recommended_action or "")
        response_generation_details: Dict[str, Any] = {}
        response_text = apply_response_momentum(
            user_input=user_input,
            response_text=response_text,
            intent=str(decision.intent or ""),
            route=str(decision.route or ""),
            operator_state=operator_state_payload,
            project_memory=load_project_state(str(user_id or "default")).to_dict(),
            local_execution=local_exec_payload,
            next_step=str(next_step.next_recommended_action or ""),
            llm_generate=getattr(self.boundaries.response, "llm_generate_fn", None),
            perception_frame=perception_frame,
            companion_state=companion_profile_state.to_dict(),
            response_generation_metadata=response_generation_details,
        )
        concept_thread_for_fallback = dict(
            (decision.metadata or {}).get("active_concept_thread") or {}
        )
        concept_refinement_turn = (
            str(decision.intent or "").strip().lower() == "conversation:concept_refinement"
            or bool(concept_thread_for_fallback)
            or (
                self.is_clear_concept_breakdown_request(
                    user_input=user_input,
                    active_concept_thread=concept_thread_for_fallback,
                )
                and str(decision.route or "").strip().lower() == "llm"
                and str(decision.intent or "").strip().lower().startswith("conversation:")
            )
        )
        clear_breakdown_request = bool(
            concept_refinement_turn
            and self.is_clear_concept_breakdown_request(
                user_input=user_input,
                active_concept_thread=concept_thread_for_fallback,
            )
        )
        if clear_breakdown_request and self._contains_generic_clarification_fallback(
            response_text
        ):
            retried_breakdown = self._retry_concept_refinement_breakdown(
                user_input=user_input,
                active_concept_thread=concept_thread_for_fallback,
            )
            response_text = str(
                retried_breakdown or self._concept_breakdown_skeleton()
            ).strip()
            respond_metadata = {
                **dict(respond_metadata or {}),
                "verification_rewrite": "concept_refinement_fallback_repair",
            }
        tool_data_payload = dict((tool_result.data or {}) if tool_result else {})
        standardized_action_result = dict(tool_data_payload.get("action_result") or {})
        if not standardized_action_result and local_exec_payload:
            standardized_action_result = action_result_from_local_execution(
                action_name=str(local_exec_payload.get("action") or decision.intent or "inspect_file"),
                local_execution=local_exec_payload,
                target=str(local_exec_payload.get("inspected_file") or ""),
            ).to_dict()
        deletion_payload = dict(
            tool_data_payload.get("deletion_result")
            or tool_data_payload.get("memory_delete")
            or {}
        )
        memory_claim_payload = dict(
            tool_data_payload.get("memory_result")
            or (
                tool_data_payload
                if str(decision.intent or "").startswith("memory:")
                else {}
            )
            or {}
        )
        background_events = list(
            tool_data_payload.get("background_events")
            or (companion_profile_state.to_dict().get("background_events") or [])
            or []
        )
        if policy.decision_type == "follow_up" and not background_events:
            background_events = [{"source": "policy_follow_up"}]
        claim_verification = verify_response_claims(
            response_text,
            user_input=user_input,
            route=str(decision.route or ""),
            intent=str(decision.intent or ""),
            local_execution=local_exec_payload,
            action_result=standardized_action_result or tool_data_payload,
            memory_result=memory_claim_payload,
            deletion_result=deletion_payload,
            operator_state=operator_state_payload,
            project_memory=load_project_state(str(user_id or "default")).to_dict(),
            background_events=background_events,
        )
        response_text = normalize_response_paragraphs(
            str(claim_verification.verified_text or "").strip()
        )
        turn_mode = classify_turn_mode(
            user_input=user_input,
            intent=str(decision.intent or ""),
            route=str(decision.route or ""),
            operator_state=operator_state_payload,
            project_memory=load_project_state(str(user_id or "default")).to_dict(),
        )
        if (
            str(decision.intent or "").strip().lower() == "greeting"
            or turn_mode == "greeting"
            or self._is_pure_greeting_input(user_input)
        ):
            repaired_text, greeting_guard = self._repair_greeting_task_intake(
                user_input=user_input,
                current_text=response_text,
                llm_generate_fn=getattr(self.boundaries.response, "llm_generate_fn", None),
            )
            if bool(greeting_guard.get("applied")):
                respond_metadata = {
                    **dict(respond_metadata or {}),
                    "greeting_task_intake_guard": dict(greeting_guard or {}),
                    "validation_passed": bool(greeting_guard.get("accepted")),
                    "validation_reasons": list(greeting_guard.get("reasons") or []),
                    "generated_by": (
                        "llm_retry"
                        if bool(greeting_guard.get("accepted"))
                        and bool(greeting_guard.get("retry_attempted"))
                        else str((respond_metadata or {}).get("generated_by") or "none")
                    ),
                }
            response_text = str(repaired_text or "").strip()
        agent_loop_payload = build_agent_loop_state(
            user_input=user_input,
            route=str(decision.route or ""),
            intent=str(decision.intent or ""),
            local_execution=local_exec_payload,
            active_objective=str(operator_state_payload.get("active_objective") or ""),
        )
        if str(local_exec_payload.get("error") or "") == "target_not_found":
            self._maybe_record_behavior_event(
                user_id=user_id,
                source="local_execution_error",
                user_input=user_input,
                alice_response=response_text,
                route=str(decision.route or ""),
                intent=str(decision.intent or ""),
                trace_id=trace_id,
                failure_kind="local_execution",
                symptom="local execution target_not_found",
                expected_behavior="Resolve target file or offer safe list fallback.",
                actual_behavior="Requested file target could not be resolved.",
                severity="medium",
                evidence={"local_execution": dict(local_exec_payload or {})},
                related_files=[
                    "ai/runtime/local_actions/file_resolver.py",
                    "ai/runtime/local_actions/local_action_executor.py",
                ],
            )
            self._append_routing_failure(
                trace_id=trace_id,
                user_input=user_input,
                final_route=decision.route,
                final_intent=decision.intent,
                plan=plan,
                verification_reason=str(
                    (verification.reason if verification else "") or ""
                ),
                veto_reason="local_execution_target_not_found",
                operator_context=dict(
                    (
                        (tool_result.diagnostics or {}).get("operator_context")
                        if tool_result
                        else {}
                    )
                    or {}
                ),
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
            user_name=str(
                getattr(user_state_snapshot, "user_name", "") or user_id or "User"
            ),
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

        metadata_payload = PipelineMetadataBuilder.build(
            {
                "trace_id": trace_id,
                "route": decision.route,
                "intent": decision.intent,
                "decision_band": decision.decision_band,
                "confidence": decision.confidence,
            },
            response_type=str((respond_metadata or {}).get("type") or "response"),
            requires_follow_up=respond_requires_follow_up,
            tools_used=[tool_result.tool_name] if tool_result else [],
            plan=plan,
            memory_recall=dict(memory.metadata or {}),
            resolved_input=resolved_input,
            verification={
                "accepted": verification.accepted if verification else True,
                "reason": verification.reason if verification else "not_configured",
                "confidence": verification.confidence if verification else 1.0,
                "diagnostics": (dict(verification.diagnostics) if verification else {}),
            },
        )
        metadata_payload.update(
            {
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
                    (
                        (tool_result.diagnostics or {}).get("operator_context")
                        if tool_result
                        else {}
                    )
                    or {}
                ),
                "operator_state": operator_state_payload,
                "local_execution": dict(
                    (
                        (tool_result.diagnostics or {}).get("local_execution")
                        if tool_result
                        else {}
                    )
                    or {}
                ),
                "action_result": standardized_action_result,
                "next_step_policy": next_step.to_dict(),
                "perception_frame": perception_frame,
                "companion_state": companion_profile_state.to_dict(),
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
                "claim_verifier_applied": True,
                "claim_verifier_valid": bool(claim_verification.valid),
                "unsupported_claims": list(claim_verification.unsupported_claims),
                "claim_verifier_reasons": list(claim_verification.reasons),
                "claim_verifier_evidence_used": dict(claim_verification.evidence_used),
            }
        )
        if "continuity_claims" in (respond_metadata or {}):
            metadata_payload["continuity_claims"] = dict(
                respond_metadata.get("continuity_claims") or {}
            )
        greeting_policy = {
            "greeting_memory_policy": str(
                (respond_metadata or {}).get("greeting_memory_policy") or ""
            ),
            "broad_memory_suppressed": bool(
                (respond_metadata or {}).get("broad_memory_suppressed")
            ),
            "active_objective_used": bool(
                (respond_metadata or {}).get("active_objective_used")
            ),
            "greeting_style": str((respond_metadata or {}).get("greeting_style") or ""),
            "suppressed_project_menu": bool(
                (respond_metadata or {}).get("suppressed_project_menu")
            ),
            "repeated_greeting": bool((respond_metadata or {}).get("repeated_greeting")),
            "generated_by": str((respond_metadata or {}).get("generated_by") or ""),
            "warmth_level": str((respond_metadata or {}).get("warmth_level") or ""),
            "companion_tone": bool((respond_metadata or {}).get("companion_tone")),
            "assistant_like_prompt_suppressed": bool(
                (respond_metadata or {}).get("assistant_like_prompt_suppressed")
            ),
            "validation_passed": bool((respond_metadata or {}).get("validation_passed", True)),
            "validation_reasons": list((respond_metadata or {}).get("validation_reasons") or []),
        }
        if any(greeting_policy.values()):
            metadata_payload["greeting_metadata"] = greeting_policy
        metadata_payload["response_generation"] = self._build_response_generation_metadata(
            decision_intent=str(decision.intent or ""),
            decision_route=str(decision.route or ""),
            response_type=str((respond_metadata or {}).get("type") or "response"),
            generated_by=str((respond_metadata or {}).get("generated_by") or ""),
            respond_metadata=dict(respond_metadata or {}),
            response_generation_details=dict(response_generation_details or {}),
            llm_generate_fn=getattr(self.boundaries.response, "llm_generate_fn", None),
            llm_model_name=str(getattr(self.boundaries.response, "llm_model_name", "") or ""),
            claim_verifier_applied=True,
        )
        if isinstance((respond_metadata or {}).get("context_frame"), dict):
            metadata_payload["context_frame"] = dict(
                (respond_metadata or {}).get("context_frame") or {}
            )
        if str((respond_metadata or {}).get("context_block") or "").strip():
            metadata_payload["context_block"] = str(
                (respond_metadata or {}).get("context_block") or ""
            )
        return PipelineResult(
            handled=bool(response_text),
            response_text=response_text,
            metadata=metadata_payload,
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
