"""Contract-driven runtime pipeline used to thin app/main orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ai.contracts import (
    MemoryRequest,
    ResponseRequest,
    RouterRequest,
    RuntimeBoundaries,
    ToolInvocation,
)


@dataclass
class PipelineResult:
    handled: bool
    response_text: str = ""
    metadata: Dict[str, Any] = None


class ContractPipeline:
    """Executes one turn through hard runtime boundaries."""

    def __init__(self, boundaries: RuntimeBoundaries):
        self.boundaries = boundaries

    def run_turn(self, user_input: str, user_id: str, turn_number: int = 0) -> PipelineResult:
        if not user_input.strip():
            return PipelineResult(handled=False, response_text="", metadata={"reason": "empty_input"})

        decision = self.boundaries.routing.route(
            RouterRequest(user_input=user_input, turn_number=turn_number)
        )

        memory = self.boundaries.memory.recall(
            MemoryRequest(query=user_input, user_id=user_id, max_items=8)
        )

        tool_result = None
        if decision.route in {"tool", "plugin"}:
            tool_name = decision.intent.split(":", 1)[0] if ":" in decision.intent else decision.intent
            tool_result = self.boundaries.tools.execute(
                ToolInvocation(
                    tool_name=tool_name,
                    action=decision.intent,
                    params={
                        "intent": decision.intent,
                        "query": user_input,
                        "entities": {},
                        "context": {"memory_count": len(memory.items)},
                    },
                )
            )

        response = self.boundaries.response.generate(
            ResponseRequest(
                user_input=user_input,
                decision=decision,
                memory=memory,
                tool_result=tool_result,
            )
        )

        self.boundaries.memory.store(
            {
                "content": f"user={user_input}\nassistant={response.text}",
                "intent": decision.intent,
                "route": decision.route,
                "confidence": decision.confidence,
            }
        )

        return PipelineResult(
            handled=bool(response.text.strip()),
            response_text=response.text,
            metadata={
                "route": decision.route,
                "intent": decision.intent,
                "confidence": decision.confidence,
                "requires_follow_up": response.requires_follow_up,
            },
        )
