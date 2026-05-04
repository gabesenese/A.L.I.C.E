from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class NextStepDecision:
    next_recommended_action: str = ""
    suggested_next_files: List[str] = None
    should_offer_action: bool = False
    should_continue_without_question: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "next_recommended_action": self.next_recommended_action,
            "suggested_next_files": list(self.suggested_next_files or []),
            "should_offer_action": bool(self.should_offer_action),
            "should_continue_without_question": bool(self.should_continue_without_question),
        }


def decide_next_step(
    *,
    route: str,
    intent: str,
    operator_state: Dict[str, Any] | None,
    local_execution: Dict[str, Any] | None,
    available_files: List[str] | None,
) -> NextStepDecision:
    files = list(available_files or [])
    state = dict(operator_state or {})
    local = dict(local_execution or {})
    preferred = [
        "ai/runtime/turn_orchestrator.py",
        "ai/runtime/contract_pipeline.py",
        "ai/runtime/alice_contract_factory.py",
        "ai/runtime/companion_runtime.py",
        "ai/core/route_coordinator.py",
        "ai/memory/memory_extractor.py",
        "ai/memory/personal_memory.py",
    ]
    suggested = [p for p in preferred if p in files][:5]

    if str(local.get("error") or "") == "target_not_found":
        close = list(local.get("close_matches") or [])
        if close:
            return NextStepDecision(
                next_recommended_action="Choose one of the close-matched files for inspection.",
                suggested_next_files=close[:5],
                should_offer_action=True,
                should_continue_without_question=False,
            )
        return NextStepDecision(
            next_recommended_action="List workspace files, then pick one file for inspection.",
            suggested_next_files=suggested,
            should_offer_action=True,
            should_continue_without_question=False,
        )

    if route == "local" and intent in {"code:request", "code:list_files"}:
        return NextStepDecision(
            next_recommended_action=(
                "Inspect ai/runtime/turn_orchestrator.py next because it controls route -> execute -> verify -> respond."
                if "ai/runtime/turn_orchestrator.py" in files
                else "Pick one file target and inspect it next."
            ),
            suggested_next_files=suggested or files[:5],
            should_offer_action=True,
            should_continue_without_question=True,
        )

    if route == "local" and intent == "code:analyze_file":
        return NextStepDecision(
            next_recommended_action="Inspect a directly related runtime/routing file next.",
            suggested_next_files=suggested or files[:5],
            should_offer_action=True,
            should_continue_without_question=True,
        )

    return NextStepDecision(
        next_recommended_action=str(state.get("next_recommended_action") or ""),
        suggested_next_files=list(state.get("suggested_next_files") or []),
        should_offer_action=False,
        should_continue_without_question=False,
    )
