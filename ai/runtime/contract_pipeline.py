"""Contract-driven runtime pipeline used to thin app/main orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

from ai.contracts import RuntimeBoundaries
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
    ):
        self.boundaries = boundaries
        self.user_state_model = user_state_model or UserStateModel()
        self.orchestrator = TurnOrchestrator(boundaries)

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

        route_phase = self.orchestrator.route_phase(
            user_input=user_input,
            user_id=user_id,
            turn_number=turn_number,
        )
        decision = route_phase.decision
        resolved_input = route_phase.resolved_input
        memory = route_phase.memory
        plan = dict(route_phase.plan or {})

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
                    "plan": plan,
                },
            )
        )

        execute_phase = self.orchestrator.execute_phase(route_phase=route_phase)
        tool_result = execute_phase.tool_result
        if execute_phase.executed and tool_result is not None:
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
                    },
                )
            )
        else:
            stages.append(self._stage("execute", "skipped", {"route": decision.route}))

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
        response_text = str(respond_phase.response_text or "").strip()
        stages.append(
            self._stage(
                "respond",
                "ok" if response_text else "failed",
                {
                    "requires_follow_up": respond_phase.requires_follow_up,
                    **dict(respond_phase.metadata or {}),
                },
            )
        )

        self.boundaries.memory.store(
            {
                "content": f"user={user_input}\nassistant={response_text}",
                "intent": decision.intent,
                "route": decision.route,
                "confidence": decision.confidence,
                "trace_id": trace_id,
                "resolved_input": resolved_input,
            }
        )

        state = self.user_state_model.update_turn(
            user_id=user_id,
            intent=decision.intent,
            route=decision.route,
            unresolved_references=list(
                (decision.metadata or {}).get("pronouns", []) or []
            ),
            active_goals=list((decision.metadata or {}).get("active_goals", []) or []),
            last_tool_used=(tool_result.tool_name if tool_result else ""),
            last_result_produced=response_text[:240],
            world_state_snapshot={
                "trace_id": trace_id,
                "route": decision.route,
                "intent": decision.intent,
                "verified": bool(verification.accepted if verification else True),
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
                "requires_follow_up": respond_phase.requires_follow_up,
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
