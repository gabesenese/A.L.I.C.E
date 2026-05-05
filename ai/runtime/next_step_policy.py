from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class NextStepDecision:
    next_recommended_action: str = ""
    suggested_next_files: List[str] = None
    should_offer_action: bool = False
    should_continue_without_question: bool = False
    should_continue: bool = False
    should_ask_user: bool = False
    blocker_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "next_recommended_action": self.next_recommended_action,
            "suggested_next_files": list(self.suggested_next_files or []),
            "should_offer_action": bool(self.should_offer_action),
            "should_continue_without_question": bool(self.should_continue_without_question),
            "should_continue": bool(self.should_continue),
            "should_ask_user": bool(self.should_ask_user),
            "blocker_reason": self.blocker_reason,
        }


def decide_next_step(
    *,
    route: str,
    intent: str,
    operator_state: Dict[str, Any] | None,
    local_execution: Dict[str, Any] | None,
    available_files: List[str] | None,
    memory_recall: Dict[str, Any] | None = None,
    routing_trace: Dict[str, Any] | None = None,
    last_failure: str = "",
) -> NextStepDecision:
    files = list(available_files or [])
    state = dict(operator_state or {})
    local = dict(local_execution or {})
    preferred = [
        "ai/runtime/turn_orchestrator.py",
        "ai/runtime/contract_pipeline.py",
        "ai/runtime/alice_contract_factory.py",
        "ai/runtime/companion_runtime.py",
        "ai/core/routing/route_arbiter.py",
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
                should_continue=False,
                should_ask_user=True,
                blocker_reason="target_not_found_with_close_matches",
            )
        return NextStepDecision(
            next_recommended_action="List workspace files, then pick one file for inspection.",
            suggested_next_files=suggested,
            should_offer_action=True,
            should_continue_without_question=False,
            should_continue=False,
            should_ask_user=True,
            blocker_reason="target_not_found_no_matches",
        )

    active_objective = str(state.get("active_objective") or "")
    files_inspected = list(state.get("files_inspected") or [])
    if active_objective and not files_inspected:
        return NextStepDecision(
            next_recommended_action="Next best move: inspect ai/runtime/turn_orchestrator.py because it controls route -> execute -> verify -> respond.",
            suggested_next_files=suggested or files[:5],
            should_offer_action=True,
            should_continue_without_question=True,
            should_continue=True,
            should_ask_user=False,
        )

    failure_signal = str(last_failure or state.get("last_failure") or "").lower()
    if "routing" in failure_signal:
        routing_files = [f for f in files if any(k in f for k in ("route_arbiter.py", "evidence_contracts.py", "routing_adapter.py"))][:5]
        return NextStepDecision(
            next_recommended_action="Next best move: inspect routing arbitration and evidence contracts.",
            suggested_next_files=routing_files or suggested or files[:5],
            should_offer_action=True,
            should_continue=True,
            should_ask_user=False,
        )
    if "memory" in failure_signal:
        memory_files = [f for f in files if any(k in f for k in ("memory_extractor.py", "personal_memory.py", "memory_turn_service.py"))][:5]
        return NextStepDecision(
            next_recommended_action="Next best move: inspect memory extraction and recall wiring.",
            suggested_next_files=memory_files or suggested or files[:5],
            should_offer_action=True,
            should_continue=True,
            should_ask_user=False,
        )
    if "local" in failure_signal or "target_not_found" in failure_signal:
        local_files = [f for f in files if any(k in f for k in ("file_resolver.py", "local_action_executor.py"))][:5]
        return NextStepDecision(
            next_recommended_action="Next best move: inspect local file resolution and execution dispatch.",
            suggested_next_files=local_files or suggested or files[:5],
            should_offer_action=True,
            should_continue=True,
            should_ask_user=False,
        )

    if route == "local" and intent in {"code:request", "code:list_files"}:
        return NextStepDecision(
            next_recommended_action=(
                "Next best move: inspect ai/runtime/turn_orchestrator.py because it controls route -> execute -> verify -> respond."
                if "ai/runtime/turn_orchestrator.py" in files
                else "Pick one file target and inspect it next."
            ),
            suggested_next_files=suggested or files[:5],
            should_offer_action=True,
            should_continue_without_question=True,
            should_continue=True,
            should_ask_user=False,
        )

    if route == "local" and intent == "code:analyze_file":
        return NextStepDecision(
            next_recommended_action="Inspect a directly related runtime/routing file next.",
            suggested_next_files=suggested or files[:5],
            should_offer_action=True,
            should_continue_without_question=True,
            should_continue=True,
            should_ask_user=False,
        )

    return NextStepDecision(
        next_recommended_action=str(state.get("next_recommended_action") or ""),
        suggested_next_files=list(state.get("suggested_next_files") or []),
        should_offer_action=False,
        should_continue_without_question=False,
        should_continue=False,
        should_ask_user=not bool(state.get("next_recommended_action")),
        blocker_reason="no_grounded_next_step" if not state.get("next_recommended_action") else "",
    )
