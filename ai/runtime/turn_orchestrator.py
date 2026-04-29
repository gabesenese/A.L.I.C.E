"""Explicit turn phases: route -> execute -> verify -> respond."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ai.contracts import (
    MemoryRequest,
    ResponseRequest,
    RouterRequest,
    RuntimeBoundaries,
    ToolInvocation,
    ToolResult,
    VerifierRequest,
    VerifierResult,
    ResponseOutput,
    RouterDecision,
    MemoryResult,
)


@dataclass(frozen=True)
class RoutePhaseResult:
    decision: RouterDecision
    resolved_input: str
    memory: MemoryResult
    plan: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutePhaseResult:
    tool_result: Optional[ToolResult] = None
    executed: bool = False


@dataclass(frozen=True)
class VerifyPhaseResult:
    proposed_response: ResponseOutput
    verification: Optional[VerifierResult]


@dataclass(frozen=True)
class RespondPhaseResult:
    response_text: str
    requires_follow_up: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class TurnOrchestrator:
    """Contract-driven turn orchestrator with explicit runtime phases."""

    def __init__(self, boundaries: RuntimeBoundaries) -> None:
        self.boundaries = boundaries

    def route_phase(
        self,
        *,
        user_input: str,
        user_id: str,
        turn_number: int,
    ) -> RoutePhaseResult:
        decision = self.boundaries.routing.route(
            RouterRequest(user_input=user_input, turn_number=turn_number)
        )

        resolved_input = str(
            (decision.metadata or {}).get("resolved_input") or user_input
        )
        memory = self.boundaries.memory.recall(
            MemoryRequest(query=resolved_input, user_id=user_id, max_items=8)
        )

        plan = {
            "route": decision.route,
            "intent": decision.intent,
            "decision_band": decision.decision_band,
            "needs_clarification": decision.needs_clarification,
            "step_count": 1 if decision.route in {"llm", "clarify", "refuse"} else 2,
        }

        return RoutePhaseResult(
            decision=decision,
            resolved_input=resolved_input,
            memory=memory,
            plan=plan,
        )

    def execute_phase(
        self,
        *,
        route_phase: RoutePhaseResult,
    ) -> ExecutePhaseResult:
        decision = route_phase.decision
        if decision.route not in {"tool", "plugin"}:
            return ExecutePhaseResult(tool_result=None, executed=False)

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
                    "query": route_phase.resolved_input,
                    "entities": {},
                    "context": {"memory_count": len(route_phase.memory.items)},
                },
            )
        )
        return ExecutePhaseResult(tool_result=tool_result, executed=True)

    def verify_phase(
        self,
        *,
        user_input: str,
        route_phase: RoutePhaseResult,
        execute_phase: ExecutePhaseResult,
        trace_id: str,
    ) -> VerifyPhaseResult:
        proposed = self.boundaries.response.generate(
            ResponseRequest(
                user_input=user_input,
                decision=route_phase.decision,
                memory=route_phase.memory,
                tool_result=execute_phase.tool_result,
                metadata={"resolved_input": route_phase.resolved_input},
            )
        )

        verification = None
        if self.boundaries.verifier is not None:
            verification = self.boundaries.verifier.verify(
                VerifierRequest(
                    user_input=user_input,
                    decision=route_phase.decision,
                    memory=route_phase.memory,
                    proposed_response=proposed,
                    tool_result=execute_phase.tool_result,
                    metadata={"trace_id": trace_id},
                )
            )

        return VerifyPhaseResult(proposed_response=proposed, verification=verification)

    def respond_phase(
        self,
        *,
        verify_phase: VerifyPhaseResult,
    ) -> RespondPhaseResult:
        verification = verify_phase.verification
        proposed = verify_phase.proposed_response

        if verification is not None and not verification.accepted:
            response_text = (
                "I could not verify that result safely. "
                "Please rephrase the request or provide more detail."
            )
            return RespondPhaseResult(
                response_text=response_text,
                requires_follow_up=True,
                metadata={"fallback": "verification_guard"},
            )

        return RespondPhaseResult(
            response_text=str(proposed.text or "").strip(),
            requires_follow_up=bool(proposed.requires_follow_up),
            metadata={
                "type": str((proposed.metadata or {}).get("type") or "response"),
                "follow_up_question": str(proposed.follow_up_question or ""),
            },
        )
