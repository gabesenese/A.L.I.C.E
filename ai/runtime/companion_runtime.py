"""Central companion runtime loop for per-turn state and policy decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ai.contracts import RouterDecision, ToolResult, VerifierResult
from brain.personality import PersonalityLayer
from memory.world_model import WorldModel, get_world_model
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
        # Risk classifier gate: high-risk intents require approval
        try:
            from ai.infrastructure.policy import get_risk_classifier

            rc = get_risk_classifier()
            risk = rc.classify(
                intent=str(intent or ""), user_input=str(user_input or "")
            )
            if risk == "high" and rc.requires_confirmation(risk):
                return (
                    True,
                    f"risk_classifier:high_risk:{str(intent or '').split(':')[0]}",
                )
        except Exception:
            pass
        # Reversibility gate: irreversible actions need approval even at medium risk
        try:
            from ai.core.reversibility_scorer import get_reversibility_scorer

            rs = get_reversibility_scorer()
            rev = rs.score(intent=str(intent or ""), user_input=str(user_input or ""))
            if rev < 0.20:
                label = rs.reversibility_label(rev)
                return True, f"reversibility:{label}:{str(intent or '').split(':')[0]}"
        except Exception:
            pass
        return False, ""

    def is_transient_tool_error(self, tool_result: ToolResult) -> bool:
        if tool_result.success:
            return False
        error_text = f"{tool_result.error} {(tool_result.diagnostics or {}).get('error', '')}".lower()
        return any(marker in error_text for marker in self._transient_error_terms)

    # Tool-domain intents that can trigger contextual reactions on follow-up
    _tool_intent_prefixes = (
        "weather:",
        "notes:",
        "email:",
        "calendar:",
        "system:",
        "music:",
        "reminder:",
        "search:",
    )

    def is_contextual_reaction(self, *, user_input: str, previous_intent: str) -> bool:
        prior_intent = str(previous_intent or "").strip().lower()
        if not any(prior_intent.startswith(p) for p in self._tool_intent_prefixes):
            return False

        text = str(user_input or "").strip().lower()
        if not text:
            return False

        # Gratitude / acknowledgement after any tool result
        has_gratitude = any(
            marker in text for marker in self._contextual_reaction_gratitude_terms
        )
        if has_gratitude:
            return True

        # Personal state reaction (currently only meaningful after weather)
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
        # Require first-person context so weather descriptions don't trigger these
        (
            "cold",
            re.compile(
                r"\b(?:i'?m|i\s+feel(?:ing)?|feeling)\s+(?:\w+\s+)?(cold|chilly|freezing)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "sick",
            re.compile(
                r"\b(?:i'?m|i\s+feel(?:ing)?|feeling)\s+(?:\w+\s+)?(?:sick|ill|not\s+well)\b"
                r"|\b(?:under the weather|not feeling well|flu|fever)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "tired",
            re.compile(
                r"\b(?:i'?m|i\s+feel(?:ing)?|feeling)\s+(?:\w+\s+)?(tired|exhausted|drained|burned out)\b"
                r"|\b(burned out|drained)\b",
                re.IGNORECASE,
            ),
        ),
        (
            "stressed",
            re.compile(
                r"\b(?:i'?m|i\s+feel(?:ing)?|feeling)\s+(?:\w+\s+)?(stressed|anxious|overwhelmed)\b"
                r"|\b(panic(?:king)?)\b",
                re.IGNORECASE,
            ),
        ),
    )

    def __init__(
        self,
        policy_engine: CompanionPolicyEngine | None = None,
        world_model: WorldModel | None = None,
    ) -> None:
        self.policy_engine = policy_engine or CompanionPolicyEngine()
        self.world_model = world_model or get_world_model()
        self.personality_layer = PersonalityLayer(world_model=self.world_model)
        self._states: Dict[str, CompanionState] = {}

        # Start background memory maintenance (non-blocking daemon thread)
        try:
            from ai.memory.maintenance_scheduler import start_maintenance_scheduler

            start_maintenance_scheduler()
        except Exception:
            pass

    def start_turn(
        self,
        *,
        user_id: str,
        user_input: str,
        turn_number: int,
        user_state: Any,
    ) -> CompanionState:
        key = str(user_id or "default").strip() or "default"
        is_new_session = key not in self._states
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

        # On the first turn of a new session inject world model continuity so
        # the LLM knows what was in progress without Gabriel having to recap.
        if is_new_session or int(turn_number or 0) <= 1:
            try:
                wm = self.world_model.snapshot()
                continuity: Dict[str, Any] = {}
                current_goals = [
                    str(g.get("goal") or g.get("text") or "")
                    for g in list(wm.get("user", {}).get("current_goals") or [])[:4]
                    if g.get("goal") or g.get("text")
                ]
                open_tasks = [
                    str(t.get("text") or "")
                    for t in list(wm.get("environment", {}).get("open_tasks") or [])[:3]
                    if t.get("text")
                ]
                active_threads = [
                    str(t.get("text") or "")
                    for t in list(
                        wm.get("alice_state", {}).get("active_threads") or []
                    )[:3]
                    if t.get("text")
                ]
                if current_goals:
                    continuity["active_goals"] = current_goals
                if open_tasks:
                    continuity["open_tasks"] = open_tasks
                if active_threads:
                    continuity["active_threads"] = active_threads
                if continuity:
                    state.memory_domains.identity["session_continuity"] = continuity
            except Exception:
                pass

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

            # On failure, consult cross-plugin fallback chain
            if not self.policy_engine.is_transient_tool_error(normalized):
                try:
                    from ai.core.cross_plugin_fallback import (
                        get_cross_plugin_fallback_chain,
                    )

                    intent = str(route_phase.decision.intent or "")
                    error_type = str(
                        (normalized.data or {}).get("error") or normalized.error or ""
                    )
                    chain = get_cross_plugin_fallback_chain().get_chain(
                        intent, error_type
                    )
                    if chain:
                        diag = dict(normalized.diagnostics or {})
                        diag["fallback_chain"] = [s.plugin for s in chain]
                        diag["fallback_available"] = True
                        from ai.contracts import ToolResult

                        annotated = ToolResult(
                            success=normalized.success,
                            tool_name=normalized.tool_name,
                            action=normalized.action,
                            data=dict(normalized.data or {}),
                            error=normalized.error,
                            confidence=normalized.confidence,
                            diagnostics=diag,
                        )
                        last_phase = ExecutePhaseResult(
                            tool_result=annotated, executed=phase.executed
                        )
                except Exception:
                    pass
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
        action_label = (
            intent.split(":")[-1].replace("_", " ") if ":" in intent else intent
        )

        # Build a dry-run preview for high-risk actions
        dry_run_preview = self._dry_run_preview(decision=decision)
        preview_block = (
            f"\n\nDry-run preview: {dry_run_preview}" if dry_run_preview else ""
        )

        return (
            f"I can {action_label}, but this action is flagged as high-risk and needs your explicit approval first.{preview_block}\n\n"
            f"Reply with 'approve {intent}' to proceed, or rephrase if you want something different. "
            f"(risk reason: {reason})"
        )

    @staticmethod
    def _dry_run_preview(*, decision: RouterDecision) -> str:
        """Generate a one-line dry-run description of what the action would do."""
        intent = str(decision.intent or "")
        meta = dict(decision.metadata or {})
        resolved = str(meta.get("resolved_input") or "").strip()

        intent_lower = intent.lower()
        target = str(meta.get("target_file") or "").strip() or (
            resolved[:60] if resolved else ""
        )

        if "delete" in intent_lower or "remove" in intent_lower:
            return f"Would permanently delete: {target or 'the specified target'}"
        if "send" in intent_lower or "email" in intent_lower:
            return f"Would send a message to: {target or 'the specified recipient'}"
        if "push" in intent_lower or "deploy" in intent_lower:
            return f"Would push/deploy: {target or 'the current changes'}"
        if "overwrite" in intent_lower or "write" in intent_lower:
            return f"Would overwrite: {target or 'the specified file'}"
        if target:
            return f"Would execute '{intent_lower}' on: {target}"
        return f"Would execute: {intent}"

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
        if companion_state.last_user_state_signals:
            self._persist_emotional_signals(companion_state.last_user_state_signals)

        # Layer 1 — capture explicit style corrections
        self._capture_explicit_preference(
            user_input, companion_state.last_response_excerpt
        )

        # Layer 2 — update behavioral profile
        self._update_behavioral_profile(
            user_input, response_text, str(route_decision.intent or "")
        )

        # Layer 3 — collect turn pair for future fine-tuning
        quality = (
            "verified" if (verification and verification.accepted) else "unverified"
        )
        self._collect_turn_pair(user_input, response_text, quality=quality)
        self._log_turn_evaluation(
            user_input=user_input,
            response_text=response_text,
            intent=str(route_decision.intent or "unknown"),
            verified=bool(verification and verification.accepted),
        )

        if tool_result is not None:
            companion_state.last_tool_result = self._summarize_tool_result(tool_result)
            # Advance any active goals whose next_action matches what was just executed
            if tool_result.success:
                try:
                    from ai.goals.goal_engine import get_goal_engine

                    get_goal_engine().ingest_completed_intent(
                        intent=str(route_decision.intent or ""),
                        user_input=user_input,
                    )
                except Exception:
                    pass

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
        tool_success = bool(tool_result.success) if tool_result is not None else None
        # Record topic mentions for cross-session reinforcement
        try:
            self.world_model.record_topic_mentions(user_input)
        except Exception:
            pass

        world_snapshot = self.world_model.update_from_turn(
            user_input=user_input,
            response_text=response_text,
            route=str(route_decision.route or ""),
            intent=str(route_decision.intent or ""),
            requires_follow_up=bool(requires_follow_up),
            tool_success=tool_success,
        )
        personality = self.personality_layer.update_after_turn(
            user_input=user_input,
            response_text=response_text,
            conversation_topics=list(companion_state.memory_domains.projects)
            + [str(route_decision.intent or "")],
        )
        world_snapshot = self.world_model.snapshot()
        companion_state.memory_domains.identity["world_model_updated_at"] = str(
            world_snapshot.get("updated_at") or ""
        )
        companion_state.memory_domains.preferences["personality"] = dict(personality)

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

    @staticmethod
    def _persist_emotional_signals(signals: List[str]) -> None:
        try:
            from ai.identity.user_identity import load_identity, save_identity

            identity = load_identity()
            noted_at = datetime.now(timezone.utc).isoformat()
            for sig in signals:
                identity.emotional_history.append({"signal": sig, "noted_at": noted_at})
            identity.emotional_history = identity.emotional_history[-10:]
            save_identity(identity)
        except Exception:
            pass

    # Layer 1 — explicit preference capture
    _PREFERENCE_SIGNALS: List[tuple] = [
        (
            re.compile(
                r"\b(be more concise|too long|keep it short|shorter|less verbose|brief(er)?|stop repeating)\b",
                re.I,
            ),
            "response_length",
            "brief",
        ),
        (
            re.compile(
                r"\b(more detail|go deeper|elaborate|explain more|in depth)\b", re.I
            ),
            "response_length",
            "detailed",
        ),
        (
            re.compile(r"\b(too formal|be casual|more casual|relax a bit)\b", re.I),
            "tone",
            "casual",
        ),
        (
            re.compile(r"\b(more formal|be professional|professional tone)\b", re.I),
            "tone",
            "formal",
        ),
        (
            re.compile(
                r"\b(no (bullet|list|bullets)|stop (listing|using bullets))\b", re.I
            ),
            "format",
            "prose",
        ),
        (
            re.compile(r"\b(use (bullets|lists)|bullet points? please)\b", re.I),
            "format",
            "bullets",
        ),
        (
            re.compile(r"\b(no (emojis?|icons)|stop using emojis?)\b", re.I),
            "emojis",
            "never",
        ),
    ]
    _CONFIRMATION_RE = re.compile(
        r"\b(exactly (right|what I needed?)|perfect(,| that)?|spot on|keep (doing|that)|yes exactly|that'?s? (it|correct|perfect))\b",
        re.I,
    )

    @staticmethod
    def _capture_explicit_preference(user_input: str, last_response: str) -> None:
        text = str(user_input or "").strip()
        if not text:
            return
        prefs: Dict[str, str] = {}
        for pattern, key, value in CompanionRuntimeLoop._PREFERENCE_SIGNALS:
            if pattern.search(text):
                prefs[key] = value
        if not prefs:
            return
        try:
            from ai.identity.user_identity import load_identity, save_identity

            identity = load_identity()
            identity.learned_preferences.update(prefs)
            save_identity(identity)
        except Exception:
            pass

    # Layer 2 — behavioral pattern update
    @staticmethod
    def _update_behavioral_profile(
        user_input: str, response_text: str, intent: str
    ) -> None:
        try:
            from ai.learning.user_profile_engine import get_profile_engine

            get_profile_engine().record_interaction(
                user_input=user_input,
                alice_response=response_text,
                intent=intent,
            )
        except Exception:
            pass
        try:
            from ai.personality.personality_evolution import get_evolution_engine

            get_evolution_engine().learn_from_interaction(
                user_id="default",
                user_input=user_input,
                alice_response=response_text,
            )
        except Exception:
            pass

    # Layer 3 — turn pair collection for fine-tuning and learning evaluation
    _TRAINING_PATH = Path("data/autolearn/training_pairs.jsonl")
    _EVAL_PATH = Path("data/evaluations/evaluations.jsonl")

    @staticmethod
    def _collect_turn_pair(
        user_input: str, response_text: str, *, quality: str
    ) -> None:
        try:
            CompanionRuntimeLoop._TRAINING_PATH.parent.mkdir(
                parents=True, exist_ok=True
            )
            record = json.dumps(
                {
                    "input": str(user_input or "").strip()[:800],
                    "response": str(response_text or "").strip()[:1200],
                    "quality": quality,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
            )
            with open(CompanionRuntimeLoop._TRAINING_PATH, "a", encoding="utf-8") as f:
                f.write(record + "\n")
        except Exception:
            pass

    @staticmethod
    def _log_turn_evaluation(
        user_input: str,
        response_text: str,
        intent: str,
        verified: bool,
    ) -> None:
        import uuid

        try:
            CompanionRuntimeLoop._EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
            score = 85 if verified else 45
            record = json.dumps(
                {
                    "interaction_id": str(uuid.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "user_input": str(user_input or "").strip()[:800],
                    "alice_response": str(response_text or "").strip()[:1200],
                    "expected_data": {},
                    "overall_score": score,
                    "accuracy_score": score,
                    "completeness_score": score,
                    "naturalness_score": score,
                    "conciseness_score": score,
                    "what_worked": "verified_by_pipeline" if verified else "",
                    "what_needs_improvement": "" if verified else "verification_failed",
                    "suggested_improvement": None
                    if verified
                    else "Improve response quality or tool execution.",
                    "action_type": str(intent or "unknown"),
                    "alice_confidence": 0.85 if verified else 0.4,
                },
                ensure_ascii=False,
            )
            with open(CompanionRuntimeLoop._EVAL_PATH, "a", encoding="utf-8") as f:
                f.write(record + "\n")
        except Exception:
            pass

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
