"""Contract-driven runtime pipeline used to thin app/main orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

from ai.contracts import (
    MemoryRequest,
    ResponseRequest,
    RouterRequest,
    RuntimeBoundaries,
    ToolInvocation,
    VerifierRequest,
)
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

        decision = self.boundaries.routing.route(
            RouterRequest(user_input=user_input, turn_number=turn_number)
        )
        stages.append(
            self._stage(
                "interpretation",
                "ok",
                {
                    "intent": decision.intent,
                    "confidence": decision.confidence,
                    "route": decision.route,
                    "decision_band": decision.decision_band,
                },
            )
        )

        resolved_input = str(
            (decision.metadata or {}).get("resolved_input") or user_input
        )

        memory = self.boundaries.memory.recall(
            MemoryRequest(query=resolved_input, user_id=user_id, max_items=8)
        )
        stages.append(
            self._stage(
                "context_resolution",
                "ok",
                {
                    "memory_count": len(memory.items),
                    "memory_confidence": memory.confidence,
                },
            )
        )

        plan = {
            "route": decision.route,
            "intent": decision.intent,
            "decision_band": decision.decision_band,
            "needs_clarification": decision.needs_clarification,
            "step_count": 1 if decision.route in {"llm", "clarify"} else 2,
        }
        stages.append(self._stage("planning", "ok", {"plan": plan}))

        tool_result = None
        if decision.route in {"tool", "plugin"}:
            tool_name = (
                decision.intent.split(":", 1)[0]
                if ":" in decision.intent
                else decision.intent
            )
            tool_result = self.boundaries.tools.execute(
                ToolInvocation(
                    tool_name=tool_name,
                    action=decision.intent,
                    params={
                        "intent": decision.intent,
                        "query": resolved_input,
                        "entities": {},
                        "context": {"memory_count": len(memory.items)},
                    },
                )
            )
            stages.append(
                self._stage(
                    "tool_execution",
                    "ok" if tool_result.success else "failed",
                    {
                        "tool": tool_result.tool_name,
                        "action": tool_result.action,
                        "error": tool_result.error,
                        "confidence": tool_result.confidence,
                        "schema_validation": str(
                            tool_result.diagnostics.get("stage") or "ok"
                        ),
                    },
                )
            )
        else:
            stages.append(
                self._stage("tool_execution", "skipped", {"route": decision.route})
            )

        response = self.boundaries.response.generate(
            ResponseRequest(
                user_input=user_input,
                decision=decision,
                memory=memory,
                tool_result=tool_result,
                metadata={"resolved_input": resolved_input},
            )
        )
        stages.append(
            self._stage(
                "response_generation",
                "ok" if response.text.strip() else "failed",
                {
                    "confidence": response.confidence,
                    "requires_follow_up": response.requires_follow_up,
                },
            )
        )

        verification = None
        if self.boundaries.verifier is not None:
            verification = self.boundaries.verifier.verify(
                VerifierRequest(
                    user_input=user_input,
                    decision=decision,
                    memory=memory,
                    proposed_response=response,
                    tool_result=tool_result,
                    metadata={"trace_id": trace_id},
                )
            )
        if verification is None:
            stages.append(
                self._stage("verification", "skipped", {"reason": "no_verifier"})
            )
        else:
            stages.append(
                self._stage(
                    "verification",
                    "ok" if verification.accepted else "failed",
                    {
                        "reason": verification.reason,
                        "confidence": verification.confidence,
                        "diagnostics": dict(verification.diagnostics),
                    },
                )
            )
            if not verification.accepted:
                response_text = (
                    "I could not verify that result safely. "
                    "Please rephrase the request or provide more detail."
                )
                stages.append(
                    self._stage("response", "ok", {"fallback": "verification_guard"})
                )
                return PipelineResult(
                    handled=True,
                    response_text=response_text,
                    metadata={
                        "trace_id": trace_id,
                        "route": decision.route,
                        "intent": decision.intent,
                        "decision_band": decision.decision_band,
                        "confidence": decision.confidence,
                        "plan": plan,
                        "verification": {
                            "accepted": verification.accepted,
                            "reason": verification.reason,
                            "confidence": verification.confidence,
                            "diagnostics": dict(verification.diagnostics),
                        },
                        "stages": stages,
                    },
                )

        self.boundaries.memory.store(
            {
                "content": f"user={user_input}\nassistant={response.text}",
                "intent": decision.intent,
                "route": decision.route,
                "confidence": decision.confidence,
                "trace_id": trace_id,
                "resolved_input": resolved_input,
            }
        )
        stages.append(self._stage("response", "ok", {"stored": True}))

        state = self.user_state_model.update_turn(
            user_id=user_id,
            intent=decision.intent,
            route=decision.route,
            unresolved_references=list(
                (decision.metadata or {}).get("pronouns", []) or []
            ),
            active_goals=list((decision.metadata or {}).get("active_goals", []) or []),
            last_tool_used=(tool_result.tool_name if tool_result else ""),
            last_result_produced=response.text[:240],
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
            handled=bool(response.text.strip()),
            response_text=response.text,
            metadata={
                "trace_id": trace_id,
                "route": decision.route,
                "intent": decision.intent,
                "decision_band": decision.decision_band,
                "confidence": decision.confidence,
                "requires_follow_up": response.requires_follow_up,
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
