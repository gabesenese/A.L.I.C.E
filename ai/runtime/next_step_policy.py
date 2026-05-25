from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class CandidateNextAction:
    action: str
    target: str = ""
    reason: str = ""
    confidence: float = 0.0
    priority: float = 0.0
    source: str = "policy"
    safety_level: str = "safe_read"


@dataclass
class NextStepDecision:
    next_recommended_action: str = ""
    last_recommended_action: Dict[str, Any] = field(default_factory=dict)
    candidate_actions: List[Dict[str, Any]] = field(default_factory=list)
    suggested_next_files: List[str] = field(default_factory=list)
    should_continue: bool = False
    should_ask_user: bool = False
    blocker_reason: str = ""
    confidence: float = 0.0

    # Backward compatibility fields.
    should_offer_action: bool = False
    should_continue_without_question: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _file_exists(rel_path: str) -> bool:
    try:
        return Path(rel_path).exists()
    except Exception:
        return False


def _candidate(
    *,
    action: str,
    target: str,
    reason: str,
    confidence: float,
    priority: float,
    source: str,
    safety_level: str = "safe_read",
) -> CandidateNextAction:
    return CandidateNextAction(
        action=action,
        target=target,
        reason=reason,
        confidence=max(0.0, min(1.0, confidence)),
        priority=priority,
        source=source,
        safety_level=safety_level,
    )


def _suggestion_from_failure(
    *, failure: str, available_files: List[str], files_inspected: List[str]
) -> List[CandidateNextAction]:
    low = str(failure or "").lower()
    candidates: List[CandidateNextAction] = []

    def add_file(
        action: str, path: str, reason: str, source: str, confidence: float
    ) -> None:
        if path in files_inspected and "blocker" not in reason.lower():
            return
        if path in available_files or _file_exists(path):
            candidates.append(
                _candidate(
                    action=action,
                    target=path,
                    reason=reason,
                    confidence=confidence,
                    priority=confidence,
                    source=source,
                )
            )

    if "greeting" in low:
        add_file(
            "inspect_file",
            "ai/runtime/greeting_surface_policy.py",
            "Latest blocker mentions greeting behavior.",
            "failure_signal",
            0.85,
        )
        add_file(
            "inspect_file",
            "ai/runtime/continuity_claim_guard.py",
            "Greeting safety depends on continuity guard behavior.",
            "failure_signal",
            0.8,
        )
    if any(
        token in low for token in ("routing", "misroute", "clarification", "intent")
    ):
        for path in (
            "ai/core/routing/route_arbiter.py",
            "ai/core/routing/evidence_contracts.py",
            "ai/runtime/routing_adapter.py",
        ):
            add_file(
                "inspect_file",
                path,
                "Latest blocker mentions routing/intent behavior.",
                "failure_signal",
                0.84,
            )
    if any(token in low for token in ("memory", "recall")):
        for path in (
            "ai/memory/memory_extractor.py",
            "ai/memory/personal_memory.py",
            "ai/memory/project_memory.py",
            "ai/runtime/memory_turn_service.py",
        ):
            add_file(
                "inspect_file",
                path,
                "Latest blocker mentions memory/recall behavior.",
                "failure_signal",
                0.83,
            )
    if any(token in low for token in ("target_not_found", "local file", "resolver")):
        for path in (
            "ai/runtime/local_actions/file_resolver.py",
            "ai/runtime/local_actions/local_action_executor.py",
        ):
            add_file(
                "inspect_file",
                path,
                "Latest blocker mentions local file resolution.",
                "failure_signal",
                0.82,
            )
    if "agent loop" in low or "skeleton" in low or "inactive" in low:
        add_file(
            "inspect_file",
            "ai/runtime/agent_loop.py",
            "Agent loop behavior is still incomplete.",
            "failure_signal",
            0.88,
        )
    if any(token in low for token in ("startup", "shutdown", "lifecycle")):
        for path in (
            "app/main.py",
            "app/runtime_modes.py",
            "ai/runtime/lifecycle_manager.py",
        ):
            add_file(
                "inspect_file",
                path,
                "Failure signal points to runtime lifecycle startup/shutdown path.",
                "failure_signal",
                0.81,
            )
    if any(token in low for token in ("passive", "generic", "momentum")):
        for path in (
            "ai/runtime/response_momentum_policy.py",
            "ai/runtime/response_adapter.py",
        ):
            add_file(
                "inspect_file",
                path,
                "Response momentum signal suggests passive surface behavior.",
                "failure_signal",
                0.8,
            )
    return candidates


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
    project_memory: Dict[str, Any] | None = None,
    recent_files_changed: List[str] | None = None,
    files_inspected: List[str] | None = None,
) -> NextStepDecision:
    state = dict(operator_state or {})
    local = dict(local_execution or {})
    dict(memory_recall or {})
    proj = dict(project_memory or {})
    files = list(available_files or [])
    inspected = list(files_inspected or state.get("files_inspected") or [])
    changed = list(recent_files_changed or proj.get("files_changed") or [])
    failure = str(
        last_failure
        or local.get("error")
        or state.get("last_failure")
        or proj.get("last_failure")
        or ""
    )

    candidates: List[CandidateNextAction] = []
    candidates.extend(
        _suggestion_from_failure(
            failure=failure, available_files=files, files_inspected=inspected
        )
    )

    active_objective = str(
        state.get("active_objective") or proj.get("active_objective") or ""
    )
    if active_objective and not candidates:
        inspected_set = {str(p or "").lower() for p in inspected}
        if "ai/runtime/agent_loop.py" not in inspected_set:
            candidates.append(
                _candidate(
                    action="inspect_file",
                    target="ai/runtime/agent_loop.py",
                    reason="Active objective exists; agent loop should drive next safe step.",
                    confidence=0.72,
                    priority=0.72,
                    source="active_objective",
                )
            )
        else:
            followups = (
                (
                    "ai/runtime/next_step_policy.py",
                    "It decides what Alice should do after each safe step.",
                    0.76,
                ),
                (
                    "ai/runtime/operator_state.py",
                    "It stores active objective, current focus, inspected files, and recommendations.",
                    0.74,
                ),
                (
                    "ai/runtime/response_momentum_policy.py",
                    "It shapes whether Alice advances with momentum or drifts into passive responses.",
                    0.72,
                ),
                (
                    "ai/runtime/contract_pipeline.py",
                    "It coordinates route -> execute -> verify -> respond handoff.",
                    0.7,
                ),
                (
                    "ai/memory/project_memory.py",
                    "It persists durable objective and recommendation state across turns.",
                    0.68,
                ),
            )
            for path, reason_text, conf in followups:
                if path.lower() in inspected_set:
                    continue
                if path in files or _file_exists(path):
                    candidates.append(
                        _candidate(
                            action="inspect_file",
                            target=path,
                            reason=reason_text,
                            confidence=conf,
                            priority=conf,
                            source="active_objective_followup",
                        )
                    )
            if not candidates:
                candidates.append(
                    _candidate(
                        action="summarize_findings",
                        target="inspected_files",
                        reason="All primary runtime files have been inspected; synthesize findings before continuing.",
                        confidence=0.69,
                        priority=0.69,
                        source="active_objective_followup",
                    )
                )

    if str(local.get("error") or "") == "target_not_found":
        close = list(local.get("close_matches") or [])
        if close:
            return NextStepDecision(
                next_recommended_action="Choose one close-matched file and inspect it next.",
                candidate_actions=[
                    asdict(
                        _candidate(
                            action="inspect_file",
                            target=path,
                            reason="Local resolver returned close match after missing target.",
                            confidence=0.76,
                            priority=0.76,
                            source="local_execution",
                        )
                    )
                    for path in close[:5]
                ],
                suggested_next_files=close[:5],
                should_continue=False,
                should_ask_user=True,
                blocker_reason="target_not_found_with_close_matches",
                confidence=0.76,
                should_offer_action=True,
                should_continue_without_question=False,
            )

    # Prefer existing unchanged candidates, then deprioritize already-changed files.
    candidates.sort(
        key=lambda c: (
            -float(c.priority),
            c.target in changed,
            c.target in inspected,
        )
    )
    top = candidates[0] if candidates else None
    suggested_files = [c.target for c in candidates if c.target][:5]

    if top:
        top_payload = asdict(top)
        return NextStepDecision(
            next_recommended_action=(
                f"{top.action.replace('_', ' ')} {top.target} because {top.reason}"
            ).strip(),
            last_recommended_action={
                **top_payload,
                "requires_approval": False,
                "created_at": "",
            },
            candidate_actions=[asdict(c) for c in candidates[:8]],
            suggested_next_files=suggested_files,
            should_continue=True,
            should_ask_user=False,
            blocker_reason="",
            confidence=float(top.confidence),
            should_offer_action=True,
            should_continue_without_question=True,
        )

    remembered_next = str(
        state.get("next_recommended_action")
        or proj.get("next_recommended_action")
        or ""
    )
    return NextStepDecision(
        next_recommended_action=remembered_next,
        last_recommended_action=dict(
            state.get("last_recommended_action")
            or proj.get("last_recommended_action")
            or {}
        ),
        candidate_actions=[],
        suggested_next_files=list(state.get("suggested_next_files") or []),
        should_continue=bool(remembered_next),
        should_ask_user=not bool(remembered_next),
        blocker_reason="no_grounded_next_step" if not remembered_next else "",
        confidence=0.45 if remembered_next else 0.0,
        should_offer_action=bool(remembered_next),
        should_continue_without_question=bool(remembered_next),
    )
