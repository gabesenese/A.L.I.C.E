"""Central companion runtime loop for per-turn state and policy decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Tuple

from ai.contracts import RouterDecision, ToolResult, VerifierResult
from ai.runtime.turn_orchestrator import (
    ExecutePhaseResult,
    RoutePhaseResult,
    TurnOrchestrator,
)


def _dedupe_keep_order(values: List[str], limit: int) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for raw in values:
        token = str(raw or "").strip()
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(token)
    return ordered[:limit]


@dataclass
class IdentityModel:
    user_id: str
    relationship_mode: str = "collaborative"
    trust_band: str = "medium"
    continuity_score: float = 0.5


@dataclass
class CompanionMemoryDomains:
    identity: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    projects: List[str] = field(default_factory=list)
    causal_lessons: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_threads: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "identity": dict(self.identity),
            "preferences": dict(self.preferences),
            "projects": list(self.projects),
            "causal_lessons": [dict(item) for item in self.causal_lessons],
            "unresolved_threads": list(self.unresolved_threads),
        }


@dataclass
class CompanionState:
    identity_model: IdentityModel
    memory_domains: CompanionMemoryDomains = field(
        default_factory=CompanionMemoryDomains
    )
    turn_number: int = 0
    last_user_input: str = ""
    last_intent: str = ""
    last_route: str = ""
    last_response_excerpt: str = ""
    last_tool_result: Dict[str, Any] = field(default_factory=dict)
    last_user_state_signals: List[str] = field(default_factory=list)


@dataclass
class PolicyDecision:
    decision_type: str
    reason: str
    retry_budget: int = 0
    requires_approval: bool = False
    approval_reason: str = ""


class CompanionPolicyEngine:
    """Policy engine deciding whether to respond, act, follow-up, or clarify."""

    _approval_terms = (
        "force push",
        "push --force",
        "overwrite",
        "wipe",
        "destroy",
        "drop table",
        "sudo",
        "rm -rf",
    )
    _transient_error_terms = (
        "timeout",
        "timed out",
        "temporarily unavailable",
        "temporary failure",
        "connection",
        "rate limit",
        "429",
        "retry",
    )
    _contextual_reaction_gratitude_terms = (
        "thanks",
        "thank you",
        "appreciate it",
        "good to know",
        "letting me know",
    )
    _contextual_reaction_state_terms = (
        "cold",
        "sick",
        "flu",
        "fever",
        "headache",
        "under the weather",
        "not feeling well",
        "tired",
        "exhausted",
        "bipolar weather",
        "weather has been",
        "got a cold",
    )
    _contextual_reaction_request_terms = (
        "can you",
        "could you",
        "please",
        "what's",
        "what is",
        "how",
        "when",
        "where",
        "show",
        "tell me",
        "check",
        "forecast",
        "temperature",
        "temp",
        "rain",
        "snow",
        "humidity",
        "wind",
        "chance",
        "should i",
    )

    def decide(
        self,
        *,
        user_input: str,
        route_decision: RouterDecision,
        companion_state: CompanionState,
    ) -> PolicyDecision:
        route = str(route_decision.route or "").lower()
        band = str(route_decision.decision_band or "").lower()

        if (
            route == "clarify"
            or route_decision.needs_clarification
            or band == "clarify"
        ):
            return PolicyDecision(
                decision_type="clarify",
                reason="clarification_required",
                retry_budget=0,
            )

        if route in {"tool", "plugin", "local"}:
            if self.is_contextual_reaction(
                user_input=user_input,
                previous_intent=companion_state.last_intent,
            ):
                return PolicyDecision(
                    decision_type="respond",
                    reason="contextual_reaction_after_tool_result",
                    retry_budget=0,
                )

            requires_approval, approval_reason = self.requires_approval(
                user_input=user_input,
                intent=route_decision.intent,
            )
            return PolicyDecision(
                decision_type="act",
                reason="tool_or_local_route",
                retry_budget=1,
                requires_approval=requires_approval,
                approval_reason=approval_reason,
            )

        if (
            str(route_decision.intent or "").startswith("conversation:")
            and companion_state.memory_domains.unresolved_threads
        ):
            return PolicyDecision(
                decision_type="follow_up",
                reason="unresolved_threads_present",
                retry_budget=0,
            )

        return PolicyDecision(
            decision_type="respond",
            reason="default_response",
            retry_budget=0,
        )

    def requires_approval(self, *, user_input: str, intent: str) -> Tuple[bool, str]:
        text = f"{str(user_input or '').lower()} {str(intent or '').lower()}"
        for marker in self._approval_terms:
            if marker in text:
                return True, f"approval_marker:{marker}"
        return False, ""

    def is_transient_tool_error(self, tool_result: ToolResult) -> bool:
        if tool_result.success:
            return False
        error_text = f"{tool_result.error} {(tool_result.diagnostics or {}).get('error', '')}".lower()
        return any(marker in error_text for marker in self._transient_error_terms)

    def is_contextual_reaction(self, *, user_input: str, previous_intent: str) -> bool:
        prior_intent = str(previous_intent or "").strip().lower()
        if not prior_intent.startswith("weather:"):
            return False

        text = str(user_input or "").strip().lower()
        if not text:
            return False

        has_personal_state = any(
            marker in text for marker in self._contextual_reaction_state_terms
        )
        has_direct_request = "?" in text or any(
            marker in text for marker in self._contextual_reaction_request_terms
        )

        return has_personal_state and not has_direct_request


class CompanionRuntimeLoop:
    """Central runtime loop that keeps companion state coherent every turn."""

    _project_pattern = re.compile(
        r"\b(project|feature|milestone|roadmap|repo|repository|build|test suite|automation)\b",
        re.IGNORECASE,
    )
    _user_state_signal_patterns = (
        ("cold", re.compile(r"\b(cold|chilly|freezing)\b", re.IGNORECASE)),
        (
            "sick",
            re.compile(
                r"\b(sick|ill|under the weather|not feeling well|flu|fever)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "tired",
            re.compile(r"\b(tired|exhausted|drained|burned out)\b", re.IGNORECASE),
        ),
        (
            "stressed",
            re.compile(r"\b(stressed|anxious|overwhelmed|panic)\b", re.IGNORECASE),
        ),
    )

    def __init__(self, policy_engine: CompanionPolicyEngine | None = None) -> None:
        self.policy_engine = policy_engine or CompanionPolicyEngine()
        self._states: Dict[str, CompanionState] = {}

    def start_turn(
        self,
        *,
        user_id: str,
        user_input: str,
        turn_number: int,
        user_state: Any,
    ) -> CompanionState:
        key = str(user_id or "default").strip() or "default"
        state = self._states.get(key)
        if state is None:
            state = CompanionState(identity_model=IdentityModel(user_id=key))
            self._states[key] = state

        state.turn_number = int(turn_number or 0)
        state.last_user_input = str(user_input or "").strip()[:320]
        state.memory_domains.identity = self._build_identity_snapshot(
            state=state,
            user_state=user_state,
        )
        state.memory_domains.preferences = dict(
            getattr(user_state, "preferences", {}) or {}
        )

        project_hints = self._extract_project_hints(user_input)
        if project_hints:
            state.memory_domains.projects = _dedupe_keep_order(
                list(state.memory_domains.projects) + project_hints,
                limit=12,
            )

        return state

    def decide(
        self,
        *,
        user_input: str,
        route_decision: RouterDecision,
        companion_state: CompanionState,
    ) -> PolicyDecision:
        return self.policy_engine.decide(
            user_input=user_input,
            route_decision=route_decision,
            companion_state=companion_state,
        )

    def execute_with_discipline(
        self,
        *,
        orchestrator: TurnOrchestrator,
        route_phase: RoutePhaseResult,
        policy: PolicyDecision,
    ) -> Tuple[ExecutePhaseResult, Dict[str, Any]]:
        if policy.requires_approval:
            return (
                ExecutePhaseResult(tool_result=None, executed=False),
                {
                    "attempt_count": 0,
                    "retried": False,
                    "approval_required": True,
                    "approval_reason": policy.approval_reason,
                },
            )

        if policy.decision_type != "act":
            return (
                ExecutePhaseResult(tool_result=None, executed=False),
                {
                    "attempt_count": 0,
                    "retried": False,
                    "approval_required": False,
                },
            )

        max_attempts = max(1, 1 + int(policy.retry_budget or 0))
        last_phase = ExecutePhaseResult(tool_result=None, executed=False)

        for attempt in range(1, max_attempts + 1):
            phase = orchestrator.execute_phase(route_phase=route_phase)
            last_phase = phase

            if not phase.executed or phase.tool_result is None:
                return phase, {
                    "attempt_count": attempt if phase.executed else 0,
                    "retried": attempt > 1,
                    "approval_required": False,
                }

            normalized = self._normalize_tool_result(
                tool_result=phase.tool_result,
                attempt=attempt,
                max_attempts=max_attempts,
            )
            last_phase = ExecutePhaseResult(
                tool_result=normalized, executed=phase.executed
            )

            if normalized.success:
                return last_phase, {
                    "attempt_count": attempt,
                    "retried": attempt > 1,
                    "approval_required": False,
                }

            if not self.policy_engine.is_transient_tool_error(normalized):
                return last_phase, {
                    "attempt_count": attempt,
                    "retried": attempt > 1,
                    "approval_required": False,
                }

        return last_phase, {
            "attempt_count": max_attempts,
            "retried": max_attempts > 1,
            "approval_required": False,
        }

    def build_approval_response(
        self, *, policy: PolicyDecision, decision: RouterDecision
    ) -> str:
        intent = str(decision.intent or "action")
        reason = str(policy.approval_reason or "safety_check")
        return (
            "I can run that action, but I need explicit approval first. "
            f"Reply with 'approve {intent}' to continue. "
            f"(reason: {reason})"
        )

    def shape_response(
        self,
        *,
        response_text: str,
        policy: PolicyDecision,
    ) -> str:
        text = str(response_text or "").strip()
        if not text:
            return ""

        if policy.decision_type == "follow_up":
            follow_up_suffix = (
                "If you want, I can keep tracking this thread and follow up next turn."
            )
            if follow_up_suffix.lower() not in text.lower():
                return f"{text} {follow_up_suffix}"

        return text

    def default_follow_up_question(self, *, policy: PolicyDecision) -> str:
        if policy.decision_type == "clarify":
            return "What exact outcome should I target next?"
        if policy.decision_type == "follow_up":
            return "Want me to keep this thread active and check in next turn?"
        if policy.decision_type == "act":
            return "Do you want me to run another action for this?"
        return "What should we tackle next?"

    def update_after_turn(
        self,
        *,
        companion_state: CompanionState,
        user_input: str,
        response_text: str,
        route_decision: RouterDecision,
        policy: PolicyDecision,
        verification: VerifierResult | None,
        requires_follow_up: bool,
        follow_up_question: str,
        tool_result: ToolResult | None,
        action_discipline: Dict[str, Any],
    ) -> Dict[str, Any]:
        companion_state.last_intent = str(route_decision.intent or "")
        companion_state.last_route = str(route_decision.route or "")
        companion_state.last_response_excerpt = str(response_text or "").strip()[:280]
        companion_state.last_user_state_signals = self._extract_user_state_signals(
            user_input
        )
        if tool_result is not None:
            companion_state.last_tool_result = self._summarize_tool_result(tool_result)

        if verification and verification.accepted:
            companion_state.identity_model.continuity_score = min(
                1.0,
                companion_state.identity_model.continuity_score + 0.03,
            )
        elif verification and not verification.accepted:
            companion_state.identity_model.continuity_score = max(
                0.0,
                companion_state.identity_model.continuity_score - 0.06,
            )

        project_hints = self._extract_project_hints(user_input)
        active_goals = list(
            (route_decision.metadata or {}).get("active_goals", []) or []
        )
        companion_state.memory_domains.projects = _dedupe_keep_order(
            list(companion_state.memory_domains.projects)
            + active_goals
            + project_hints,
            limit=12,
        )

        if verification and not verification.accepted:
            companion_state.memory_domains.causal_lessons.append(
                {
                    "reason": verification.reason,
                    "intent": str(route_decision.intent or ""),
                    "route": str(route_decision.route or ""),
                }
            )

        if bool(action_discipline.get("retried")):
            companion_state.memory_domains.causal_lessons.append(
                {
                    "reason": "tool_retry_attempted",
                    "intent": str(route_decision.intent or ""),
                    "attempt_count": int(action_discipline.get("attempt_count") or 0),
                }
            )

        unresolved = list(companion_state.memory_domains.unresolved_threads)
        if requires_follow_up:
            unresolved.append(
                str(follow_up_question or user_input or "follow_up").strip()
            )
        elif policy.decision_type == "follow_up" and unresolved:
            unresolved = unresolved[1:]

        companion_state.memory_domains.unresolved_threads = _dedupe_keep_order(
            unresolved,
            limit=8,
        )

        companion_state.memory_domains.causal_lessons = (
            companion_state.memory_domains.causal_lessons[-16:]
        )
        companion_state.memory_domains.identity = self._build_identity_snapshot(
            state=companion_state,
            user_state=None,
        )

        return companion_state.memory_domains.as_dict()

    def _normalize_tool_result(
        self,
        *,
        tool_result: ToolResult,
        attempt: int,
        max_attempts: int,
    ) -> ToolResult:
        data = dict(tool_result.data or {})
        diagnostics = dict(tool_result.diagnostics or {})
        diagnostics.update(
            {
                "schema_version": "tool_result.v1",
                "attempt": attempt,
                "max_attempts": max_attempts,
            }
        )
        data.setdefault("response", str(data.get("response") or ""))
        data.setdefault("success", bool(tool_result.success))
        data.setdefault("plugin", str(data.get("plugin") or tool_result.tool_name))

        return ToolResult(
            success=bool(tool_result.success),
            tool_name=str(tool_result.tool_name or ""),
            action=str(tool_result.action or ""),
            data=data,
            error=str(tool_result.error or ""),
            confidence=float(tool_result.confidence or 0.0),
            diagnostics=diagnostics,
        )

    def _extract_user_state_signals(self, user_input: str) -> List[str]:
        text = str(user_input or "").strip()
        if not text:
            return []

        detected: List[str] = []
        for label, pattern in self._user_state_signal_patterns:
            if pattern.search(text):
                detected.append(label)

        return _dedupe_keep_order(detected, limit=8)

    def _summarize_tool_result(self, tool_result: ToolResult) -> Dict[str, Any]:
        data = dict(tool_result.data or {})
        return {
            "tool_name": str(tool_result.tool_name or ""),
            "action": str(tool_result.action or ""),
            "success": bool(tool_result.success),
            "confidence": float(tool_result.confidence or 0.0),
            "error": str(tool_result.error or ""),
            "response_excerpt": str(data.get("response") or "").strip()[:180],
        }

    def _extract_project_hints(self, user_input: str) -> List[str]:
        text = str(user_input or "").strip()
        if not text:
            return []

        lowered = text.lower()
        if not self._project_pattern.search(lowered):
            return []

        hints: List[str] = []
        fragments = re.split(r"[,;]|\band\b", text)
        for fragment in fragments:
            cleaned = str(fragment or "").strip()
            if not cleaned:
                continue
            if self._project_pattern.search(cleaned):
                hints.append(cleaned[:120])

        return _dedupe_keep_order(hints, limit=6)

    def _build_identity_snapshot(
        self,
        *,
        state: CompanionState,
        user_state: Any,
    ) -> Dict[str, Any]:
        snapshot = {
            "user_id": state.identity_model.user_id,
            "relationship_mode": state.identity_model.relationship_mode,
            "trust_band": state.identity_model.trust_band,
            "continuity_score": round(float(state.identity_model.continuity_score), 3),
        }

        if user_state is not None:
            updated_at = str(getattr(user_state, "updated_at", "") or "").strip()
            if updated_at:
                snapshot["last_user_state_update"] = updated_at

        return snapshot
