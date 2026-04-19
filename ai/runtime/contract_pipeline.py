"""Contract-driven runtime pipeline used to thin app/main orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

from ai.contracts import RuntimeBoundaries, VerifierResult
from ai.runtime.companion_runtime import CompanionRuntimeLoop
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
    ):
        self.boundaries = boundaries
        self.user_state_model = user_state_model or UserStateModel()
        self.companion_runtime = companion_runtime or CompanionRuntimeLoop()
        self.orchestrator = TurnOrchestrator(boundaries)

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

    def run_turn(
        self, user_input: str, user_id: str, turn_number: int = 0
    ) -> PipelineResult:
        trace_id = str(uuid4())
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
        policy = self.companion_runtime.decide(
            user_input=user_input,
            route_decision=decision,
            companion_state=companion_state,
        )
        plan["policy_decision"] = policy.decision_type

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
                    "resolved_input": resolved_input,
                    "policy_decision": policy.decision_type,
                    "policy_reason": policy.reason,
                    "plan": plan,
                },
            )
        )

        execute_phase, action_discipline = self.companion_runtime.execute_with_discipline(
            orchestrator=self.orchestrator,
            route_phase=route_phase,
            policy=policy,
        )
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
            stages.append(
                self._stage(
                    "execute",
                    "ok" if tool_result.success else "failed",
                    {
                        "tool": tool_result.tool_name,
                        "action": tool_result.action,
                        "error": tool_result.error,
                        "confidence": tool_result.confidence,
                        "schema_validation": str(
                            (tool_result.diagnostics or {}).get("stage") or "ok"
                        ),
                        "attempt_count": int(action_discipline.get("attempt_count") or 1),
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
                        "attempt_count": int(action_discipline.get("attempt_count") or 0),
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
                    "approval_reason": str(action_discipline.get("approval_reason") or ""),
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
                stages.append(self._stage("verify", "skipped", {"reason": "no_verifier"}))
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

            respond_phase = self.orchestrator.respond_phase(verify_phase=verify_phase)
            response_text = self.companion_runtime.shape_response(
                response_text=str(respond_phase.response_text or "").strip(),
                policy=policy,
            )
            respond_requires_follow_up = bool(respond_phase.requires_follow_up)
            respond_metadata = dict(respond_phase.metadata or {})
            follow_up_question = str(respond_metadata.get("follow_up_question") or "").strip()

        if policy.decision_type in {"follow_up", "clarify"}:
            respond_requires_follow_up = True

        if respond_requires_follow_up and not follow_up_question:
            follow_up_question = self.companion_runtime.default_follow_up_question(
                policy=policy
            )
            respond_metadata["follow_up_question"] = follow_up_question

        stages.append(
            self._stage(
                "respond",
                "ok" if response_text else "failed",
                {
                    "requires_follow_up": respond_requires_follow_up,
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
            action_discipline=action_discipline,
        )

        self.boundaries.memory.store(
            {
                "content": f"user={user_input}\nassistant={response_text}",
                "intent": decision.intent,
                "route": decision.route,
                "confidence": decision.confidence,
                "trace_id": trace_id,
                "resolved_input": resolved_input,
                "memory_domains": memory_domains,
            }
        )

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
                "verified": bool(verification.accepted if verification else True),
                "policy_decision": policy.decision_type,
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

        return PipelineResult(
            handled=bool(response_text),
            response_text=response_text,
            metadata={
                "trace_id": trace_id,
                "route": decision.route,
                "intent": decision.intent,
                "decision_band": decision.decision_band,
                "confidence": decision.confidence,
                "requires_follow_up": respond_requires_follow_up,
                "plan": plan,
                "resolved_input": resolved_input,
                "verification": {
                    "accepted": verification.accepted if verification else True,
                    "reason": verification.reason if verification else "not_configured",
                    "confidence": verification.confidence if verification else 1.0,
                    "diagnostics": (
                        dict(verification.diagnostics) if verification else {}
                    ),
                },
                "companion": {
                    "policy_decision": policy.decision_type,
                    "policy_reason": policy.reason,
                    "identity_model": dict(memory_domains.get("identity", {}) or {}),
                    "memory_domains": memory_domains,
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
                "state": {
                    "current_task": state.current_task,
                    "prior_task": state.prior_task,
                    "unresolved_references": list(state.unresolved_references),
                    "active_goals": list(state.active_goals),
                    "last_tool_used": state.last_tool_used,
                    "last_result_produced": state.last_result_produced,
                },
                "stages": stages,
            },
        )
