"""Response authority contract for final publication decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ai.runtime.fallback_policy import RuntimeFallbackPolicy, FallbackDecision


@dataclass(frozen=True)
class ResponseAuthorityOutcome:
    action: str
    text: str
    reason: str


class ResponseAuthorityContract:
    """Enforce final response authority for LLM turns.

    Contract:
    - If LLM is accepted, publish that response.
    - If LLM is rejected, try refinement first.
    - If refinement fails, use deterministic fallback.
    """

    def __init__(self, fallback_policy: Optional[RuntimeFallbackPolicy] = None) -> None:
        self._fallback_policy = fallback_policy or RuntimeFallbackPolicy()

    def start_turn(self, turn_key: str = "") -> None:
        self._fallback_policy.start_turn(turn_key)

    def resolve_llm_turn(
        self,
        *,
        accepted: bool,
        response: str,
        refine_fn: Callable[[str], Optional[str]],
        deterministic_fn: Callable[[], Optional[str]],
    ) -> ResponseAuthorityOutcome:
        decision: FallbackDecision = self._fallback_policy.refine_then_deterministic(
            response=response,
            accepted=accepted,
            refine_fn=refine_fn,
            deterministic_fn=deterministic_fn,
        )
        return ResponseAuthorityOutcome(
            action=decision.action,
            text=decision.text,
            reason=decision.reason,
        )
