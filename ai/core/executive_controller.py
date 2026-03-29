"""
Executive control layer for high-level action decisions.

This module keeps a structured internal reasoning state and outputs a compact
decision signal for routing behavior. It is intentionally not chain-of-thought.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List


@dataclass
class ReasoningState:
    user_intent: str
    source_text: str
    topic: str
    confidence: float
    intent_plausibility: float
    conversation_goal: str
    user_goal: str
    depth_level: int
    planner_hint: str = ""
    planner_depth: int = 1
    route_bias: str = "balanced"
    tool_budget: int = 1
    plan: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "user_intent": self.user_intent,
            "source_text": self.source_text,
            "topic": self.topic,
            "confidence": self.confidence,
            "intent_plausibility": self.intent_plausibility,
            "conversation_goal": self.conversation_goal,
            "user_goal": self.user_goal,
            "depth_level": self.depth_level,
            "planner_hint": self.planner_hint,
            "planner_depth": self.planner_depth,
            "route_bias": self.route_bias,
            "tool_budget": self.tool_budget,
            "plan": list(self.plan),
        }


@dataclass
class ExecutiveDecision:
    action: (
        str  # use_plugin | use_llm | ask_clarification | ignore | answer_direct | defer
    )
    reason: str
    store_memory: bool = True
    clarification_question: str = ""


class ExecutiveController:
    """Produces high-level decisions from compact state and runtime hints."""

    TOOL_DOMAINS = (
        "notes:",
        "email:",
        "calendar:",
        "file_operations:",
        "memory:",
        "reminder:",
        "system:",
        "weather:",
        "time:",
    )
    SIMPLE_SCAFFOLD_INTENTS = {
        "conversation:help",
        "conversation:ack",
        "conversation:acknowledgment",
        "thanks",
        "greeting",
        "conversation:clarification_needed",
    }
    HELP_OPENER_RE = re.compile(
        r"\b(i\s+need\s+help|can\s+you\s+help|help\s+me|help\s+with\s+this|help\s+with\s+my\s+project)\b",
        re.IGNORECASE,
    )
    SEMANTIC_STOPWORDS = {
        "the", "a", "an", "and", "or", "to", "of", "for", "with", "in", "on", "at", "by",
        "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these", "those",
        "i", "you", "we", "they", "he", "she", "them", "my", "your", "our", "their", "me",
        "can", "could", "would", "should", "want", "need", "help", "today", "world", "real",
        "learn", "about", "explain", "tell", "please",
    }

    def __init__(self) -> None:
        # Adaptive routing matrix (updated by reflection loop).
        self._routing_weights: Dict[str, float] = {
            "tools": 1.0,
            "memory": 1.0,
            "rag": 1.0,
            "llm": 1.0,
            "clarify": 1.0,
            "search": 1.0,
            "reject": 1.0,
            "defer": 1.0,
        }

    def build_state(
        self,
        user_input: str,
        intent: str,
        confidence: float,
        entities: Dict[str, Any],
        conversation_state: Dict[str, Any],
    ) -> ReasoningState:
        entities = entities or {}
        conversation_state = conversation_state or {}

        topic = str(
            entities.get("topic") or conversation_state.get("conversation_topic") or ""
        ).strip()

        conversation_goal = str(
            conversation_state.get("conversation_goal") or ""
        ).strip()
        user_goal = str(
            conversation_state.get("user_goal") or entities.get("goal") or ""
        ).strip()
        depth_level = int(conversation_state.get("depth_level") or 0)
        intent_plausibility = max(
            0.0,
            min(1.0, float(entities.get("_intent_plausibility", 1.0) or 1.0)),
        )
        planner_hint = str(conversation_state.get("planner_hint") or "").strip().lower()
        planner_depth = int(conversation_state.get("planner_depth") or 1)
        route_bias = (
            str(conversation_state.get("route_bias") or "balanced").strip().lower()
        )
        tool_budget = int(conversation_state.get("tool_budget") or 1)

        plan = self._derive_plan(
            intent=intent, topic=topic, depth_level=depth_level, user_input=user_input
        )

        return ReasoningState(
            user_intent=intent or "unknown",
            source_text=str(user_input or ""),
            topic=topic,
            confidence=max(0.0, min(1.0, float(confidence or 0.0))),
            intent_plausibility=intent_plausibility,
            conversation_goal=conversation_goal,
            user_goal=user_goal,
            depth_level=depth_level,
            planner_hint=planner_hint,
            planner_depth=max(1, min(4, planner_depth)),
            route_bias=route_bias or "balanced",
            tool_budget=max(0, min(3, tool_budget)),
            plan=plan,
        )

    def should_preempt_for_plausibility(
        self,
        state: ReasoningState,
        *,
        has_explicit_action_cue: bool,
        intent_candidates: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Pre-routing plausibility gate to stop low-confidence tool trajectories early."""
        candidates = intent_candidates or []
        conf = max(0.0, min(1.0, float(state.confidence)))
        plausibility = max(0.0, min(1.0, float(state.intent_plausibility)))
        intent = (state.user_intent or "").lower().strip()

        if intent.startswith("conversation:"):
            return {"block": False, "reason": "conversation_intent"}

        if plausibility < 0.38:
            return {
                "block": True,
                "reason": "pre_route_low_plausibility",
                "question": "I may be misclassifying your intent. Do you want a concrete tool action or a normal explanation?",
            }

        if (not has_explicit_action_cue) and conf < 0.50 and plausibility < 0.60:
            return {
                "block": True,
                "reason": "pre_route_uncertain_without_action_cue",
                "question": "Before I route this, what exact outcome do you want me to perform?",
            }

        if len(candidates) > 1:
            top = float(candidates[0].get("score", 0.0))
            second = float(candidates[1].get("score", 0.0))
            if (
                (top - second) < 0.06
                and plausibility < 0.68
                and not has_explicit_action_cue
            ):
                return {
                    "block": True,
                    "reason": "pre_route_ambiguous_candidates",
                    "question": "I see multiple likely intents. Should I execute a tool or stay in discussion mode?",
                }

        return {"block": False, "reason": "allowed"}

    def decide(
        self,
        state: ReasoningState,
        *,
        is_pure_conversation: bool,
        has_explicit_action_cue: bool,
        has_active_goal: bool,
        force_plugins_for_notes: bool,
    ) -> ExecutiveDecision:
        if len(state.user_intent.strip()) == 0:
            return ExecutiveDecision(
                action="ignore",
                reason="empty_intent",
                store_memory=False,
            )

        if self._should_force_native_scaffold(
            state=state,
            has_explicit_action_cue=has_explicit_action_cue,
            has_active_goal=has_active_goal,
        ):
            return ExecutiveDecision(
                action="answer_direct",
                reason="native_conversation_scaffold",
                store_memory=True,
            )

        normalized_intent = (state.user_intent or "").lower().strip()
        if (
            normalized_intent == "conversation:clarification_needed"
            and state.confidence >= 0.45
        ):
            return ExecutiveDecision(
                action="use_llm",
                reason="clarification_answer_requested",
                store_memory=True,
            )

        scores = self.score_decisions(
            state,
            is_pure_conversation=is_pure_conversation,
            has_explicit_action_cue=has_explicit_action_cue,
            has_active_goal=has_active_goal,
            force_plugins_for_notes=force_plugins_for_notes,
        )
        uncertainty = self.uncertainty_behavior(state, scores)
        if uncertainty in ("clarify", "defer", "reject"):
            if uncertainty == "clarify":
                return ExecutiveDecision(
                    action="ask_clarification",
                    reason="uncertainty_clarify",
                    store_memory=False,
                    clarification_question="I can help, but I am not fully certain. Did you want action X or explanation Y?",
                )
            if uncertainty == "defer":
                return ExecutiveDecision(
                    action="defer",
                    reason="uncertainty_defer",
                    store_memory=False,
                )
            return ExecutiveDecision(
                action="ignore",
                reason="uncertainty_reject",
                store_memory=False,
            )

        winner = max(scores.items(), key=lambda kv: kv[1])[0]

        if winner == "clarify":
            return ExecutiveDecision(
                action="ask_clarification",
                reason="score_clarify",
                store_memory=False,
                clarification_question="I want to help accurately. Do you mean tool action, explanation, or search?",
            )
        if winner == "reject":
            return ExecutiveDecision(
                action="ignore",
                reason="score_reject",
                store_memory=False,
            )
        if winner in ("tools", "search"):
            return ExecutiveDecision(
                action="use_plugin", reason=f"score_{winner}", store_memory=True
            )
        if winner == "memory":
            return ExecutiveDecision(
                action="use_llm", reason="score_memory", store_memory=True
            )

        return ExecutiveDecision(
            action="use_llm", reason="score_llm", store_memory=True
        )

    def _is_help_opener(self, state: ReasoningState) -> bool:
        text = " ".join(
            str(x)
            for x in (state.source_text, state.topic, state.user_goal, state.user_intent)
            if x
        )
        return bool(self.HELP_OPENER_RE.search(text))

    def _should_force_native_scaffold(
        self,
        *,
        state: ReasoningState,
        has_explicit_action_cue: bool,
        has_active_goal: bool,
    ) -> bool:
        normalized_intent = (state.user_intent or "").lower().strip()
        if normalized_intent in self.SIMPLE_SCAFFOLD_INTENTS:
            return not has_explicit_action_cue
        if normalized_intent == "conversation:general" and self._is_help_opener(state):
            return (not has_explicit_action_cue) and (not has_active_goal)
        return False

    def should_veto_tool_execution(
        self,
        *,
        user_input: str,
        intent: str,
        confidence: float,
        intent_plausibility: float,
        intent_candidates: List[Dict[str, Any]] | None,
    ) -> Dict[str, Any]:
        """Final guard before plugin execution to prevent high-cost misroutes."""
        normalized_intent = (intent or "").lower().strip()
        conf = max(0.0, min(1.0, float(confidence or 0.0)))
        plausibility = max(0.0, min(1.0, float(intent_plausibility or 0.0)))
        candidates = intent_candidates or []
        text_lower = (user_input or "").lower()

        if normalized_intent.startswith("conversation:"):
            return {"veto": False, "reason": "conversation_intent"}

        conversational_cues = (
            "brainstorm",
            "idea",
            "explore",
            "how might",
            "could we",
            "strategy",
        )
        if any(
            cue in text_lower for cue in conversational_cues
        ) and not normalized_intent.startswith("conversation:"):
            return {
                "veto": True,
                "reason": "conversational_input_not_actionable",
                "question": "This sounds like discussion mode. Do you want brainstorming help or a concrete tool action?",
            }

        if plausibility < 0.46:
            return {
                "veto": True,
                "reason": "low_intent_plausibility",
                "question": "I am not confident this intent is correct. Can you clarify the action you want?",
            }

        if conf < 0.42 and plausibility < 0.62:
            return {
                "veto": True,
                "reason": "low_confidence_and_plausibility",
                "question": "I need one more detail before triggering a tool. What exact outcome do you want?",
            }

        if candidates and len(candidates) > 1:
            top = float(candidates[0].get("score", 0.0))
            second = float(candidates[1].get("score", 0.0))
            if (top - second) < 0.08 and conf < 0.70:
                return {
                    "veto": True,
                    "reason": "ambiguous_top_intents",
                    "question": "I see multiple possible intents. Should I execute a tool command or keep this conversational?",
                }

        return {"veto": False, "reason": "allowed"}

    def should_use_planner(
        self, state: ReasoningState, scores: Dict[str, float]
    ) -> bool:
        """Executive authority for when planning is required before response."""
        intent = (state.user_intent or "").lower()
        if intent.startswith("learning:") or intent.startswith("question:"):
            return True
        if state.depth_level >= 2 and state.topic:
            return True
        if scores.get("tools", 0.0) >= 0.75 and state.user_goal:
            return True
        return False

    def uncertainty_behavior(
        self, state: ReasoningState, scores: Dict[str, float]
    ) -> str:
        """Return proceed | clarify | defer | reject based on confidence and score ambiguity."""
        conf = max(0.0, min(1.0, float(state.confidence)))
        plausibility = max(0.0, min(1.0, float(state.intent_plausibility)))
        normalized_intent = (state.user_intent or "").lower().strip()
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[0][1] if ranked else 0.0
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top - second

        if normalized_intent == "conversation:clarification_needed" and conf >= 0.45:
            return "proceed"

        if plausibility < 0.32:
            return "clarify"
        if conf < 0.20 and not state.topic:
            return "reject"
        if conf < 0.35:
            return "defer"
        if plausibility < 0.55 and conf < 0.65:
            return "clarify"
        if conf < 0.60 and margin < 0.12:
            return "clarify"
        return "proceed"

    def format_candidate_uncertainty(
        self,
        intent_candidates: List[Dict[str, Any]] | None,
        *,
        limit: int = 3,
    ) -> str:
        """Return a compact top-k candidate summary for user-facing clarification."""
        candidates = intent_candidates or []
        if not candidates:
            return ""
        top = sorted(
            candidates,
            key=lambda c: float(c.get("score", c.get("confidence", 0.0)) or 0.0),
            reverse=True,
        )[: max(1, int(limit or 3))]
        parts: List[str] = []
        for cand in top:
            label = str(cand.get("intent") or "unknown")
            score = float(cand.get("score", cand.get("confidence", 0.0)) or 0.0)
            parts.append(f"{label} ({score * 100:.0f}%)")
        return "I am not fully certain. Top possibilities: " + ", ".join(parts) + "."

    def score_decisions(
        self,
        state: ReasoningState,
        *,
        is_pure_conversation: bool,
        has_explicit_action_cue: bool,
        has_active_goal: bool,
        force_plugins_for_notes: bool,
    ) -> Dict[str, float]:
        """Return a probabilistic-like score table for routing decisions."""
        text_intent = (state.user_intent or "").lower()
        scores = {
            "tools": 0.0,
            "memory": 0.0,
            "rag": 0.0,
            "llm": 0.0,
            "clarify": 0.0,
            "search": 0.0,
            "reject": 0.0,
            "defer": 0.0,
        }

        conf = max(0.0, min(1.0, state.confidence))
        normalized_intent = (state.user_intent or "").lower().strip()
        scores["llm"] = 0.40 + (0.40 * conf)
        scores["memory"] = 0.25 + (0.25 * (1.0 if state.topic else 0.0))
        scores["rag"] = 0.20 + (0.30 * (1.0 if state.user_goal else 0.0))
        scores["clarify"] = 0.15 + (0.55 * (1.0 - conf))

        if normalized_intent == "conversation:clarification_needed":
            scores["llm"] += 0.30
            scores["clarify"] -= 0.30

        if has_explicit_action_cue or has_active_goal:
            scores["tools"] += 0.55
        if any(text_intent.startswith(domain) for domain in self.TOOL_DOMAINS):
            scores["tools"] += 0.35
        if force_plugins_for_notes:
            scores["tools"] += 0.60
        if is_pure_conversation:
            scores["llm"] += 0.20

        if self._is_help_opener(state):
            # Generic help-openers should steer toward native brief guidance,
            # not deep LLM reasoning.
            scores["llm"] -= 0.45
            scores["memory"] -= 0.18
            scores["rag"] -= 0.15
            scores["clarify"] += 0.20

        if normalized_intent in self.SIMPLE_SCAFFOLD_INTENTS:
            scores["llm"] -= 0.30
            scores["clarify"] += 0.15
            scores["defer"] -= 0.10
        if conf < 0.35 and not state.topic:
            scores["clarify"] += 0.35
        if conf < 0.20 and not has_explicit_action_cue and not state.user_goal:
            scores["reject"] += 0.55
        if conf < 0.35:
            scores["defer"] += 0.40
        if "search" in text_intent or "research" in text_intent:
            scores["search"] += 0.60

        # Intent plausibility and runtime planner/cognition influences.
        plausibility = max(0.0, min(1.0, float(state.intent_plausibility)))
        if plausibility < 0.55:
            scores["clarify"] += 0.20
            scores["defer"] += 0.10
            scores["tools"] -= 0.12
            scores["search"] -= 0.08
        if plausibility < 0.40:
            scores["clarify"] += 0.25
            scores["tools"] -= 0.20
            scores["search"] -= 0.15

        if state.route_bias == "clarify_first":
            scores["clarify"] += 0.20
            scores["tools"] -= 0.10
        elif state.route_bias == "tool_first":
            scores["tools"] += 0.12
        elif state.route_bias == "goal_first":
            scores["llm"] += 0.08
            scores["memory"] += 0.06

        if state.tool_budget <= 0:
            scores["tools"] -= 0.30
            scores["search"] -= 0.20
        elif state.tool_budget >= 2 and (has_explicit_action_cue or has_active_goal):
            scores["tools"] += 0.12

        if state.planner_depth >= 3:
            scores["llm"] += 0.08
            scores["clarify"] += 0.04

        if state.planner_hint == "increase_structure_depth":
            scores["llm"] += 0.05
            scores["memory"] += 0.03

        for route in list(scores.keys()):
            scores[route] *= float(self._routing_weights.get(route, 1.0))

        # Normalize to 0..1 to keep all downstream thresholds stable.
        for k, v in list(scores.items()):
            scores[k] = max(0.0, min(1.0, float(v)))
        return scores

    def apply_reflection(self, reflection: Dict[str, Any]) -> None:
        """Adjust routing matrix based on reflection feedback."""
        if not isinstance(reflection, dict):
            return
        adjustments = reflection.get("routing_adjustments", {}) or {}
        for route, delta in adjustments.items():
            if route not in self._routing_weights:
                continue
            current = float(self._routing_weights.get(route, 1.0))
            self._routing_weights[route] = max(0.5, min(1.5, current + float(delta)))

    def get_routing_weights(self) -> Dict[str, float]:
        return dict(self._routing_weights)

    def derive_runtime_controls(
        self,
        state: ReasoningState,
        scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """Derive runtime controls that affect route preference, depth, and tool usage."""
        ranked = sorted((scores or {}).items(), key=lambda kv: kv[1], reverse=True)
        top_route = ranked[0][0] if ranked else "llm"
        top_score = float(ranked[0][1]) if ranked else 0.0

        thinking_depth = max(1, min(4, int(state.planner_depth or 1)))
        if state.depth_level >= 3:
            thinking_depth = max(thinking_depth, 3)
        if state.route_bias == "goal_first":
            thinking_depth = min(4, thinking_depth + 1)

        allow_tools = bool(state.tool_budget > 0 and state.intent_plausibility >= 0.45)
        if state.route_bias == "clarify_first" and state.intent_plausibility < 0.65:
            allow_tools = False

        routing_preference = "balanced"
        if top_route in ("tools", "search") and top_score >= 0.62 and allow_tools:
            routing_preference = "tool_first"
        elif top_route in ("clarify", "defer", "reject"):
            routing_preference = "clarify_first"
        elif top_route in ("llm", "memory"):
            routing_preference = "llm_first"

        max_tool_hops = (
            0 if not allow_tools else max(1, min(3, int(state.tool_budget or 1)))
        )
        return {
            "routing_preference": routing_preference,
            "thinking_depth": thinking_depth,
            "allow_tools": allow_tools,
            "max_tool_hops": max_tool_hops,
        }

    def save_weights(self, path: str) -> None:
        """Persist routing weights to disk as JSON."""
        import json
        import os

        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._routing_weights, f, indent=2)
        except Exception:
            pass

    def load_weights(self, path: str, *, decay: float = 0.05) -> None:
        """
        Load routing weights from disk.
        Applies a small decay toward 1.0 so stale biases fade after a restart.
        """
        import json

        try:
            with open(path, "r", encoding="utf-8") as f:
                stored = json.load(f)
            if not isinstance(stored, dict):
                return
            for key, value in stored.items():
                if key in self._routing_weights and isinstance(value, (int, float)):
                    loaded = float(value)
                    # Pull toward 1.0 (neutral) by the decay factor each restart
                    decayed = loaded + (1.0 - loaded) * decay
                    self._routing_weights[key] = max(0.5, min(1.5, decayed))
        except (FileNotFoundError, ValueError):
            pass
        except Exception:
            pass

    def evaluate_response(
        self,
        *,
        user_input: str,
        intent: str,
        response: str,
        route: str,
        context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Response acceptance gate before content is sent to the user."""
        context = context or {}
        resp = (response or "").strip()
        if not resp:
            return {
                "accepted": False,
                "score": 0.0,
                "reason": "empty_response",
                "fallback_action": "clarify",
            }

        semantic_guard = self._semantic_fidelity_guard(
            user_input=user_input,
            intent=intent,
            response=resp,
        )
        if semantic_guard is not None:
            return semantic_guard

        user_tokens = set(self._tokens(user_input))
        resp_tokens = set(self._tokens(resp))
        overlap = 0.0
        if user_tokens:
            overlap = len(user_tokens.intersection(resp_tokens)) / max(
                len(user_tokens), 1
            )

        uncertain_markers = (
            "i'm not sure",
            "i am not sure",
            "maybe",
            "possibly",
            "i don't know",
            "not certain",
        )
        uncertain_penalty = (
            0.35 if any(m in resp.lower() for m in uncertain_markers) else 0.0
        )

        generic_markers = (
            "it depends",
            "in general",
            "there are many factors",
            "cannot be determined",
        )
        generic_penalty = (
            0.20 if any(m in resp.lower() for m in generic_markers) else 0.0
        )

        score = max(
            0.0, min(1.0, 0.55 + (0.55 * overlap) - uncertain_penalty - generic_penalty)
        )
        plan_adherence = self._response_plan_adherence(resp, context)
        goal_alignment = float((context or {}).get("goal_alignment", 1.0) or 1.0)
        goal_alignment = max(0.0, min(1.0, goal_alignment))
        plan = (context or {}).get("response_plan", {}) or {}
        required_sections = (
            plan.get("required_sections", []) if isinstance(plan, dict) else []
        )
        format_hint = (
            str(plan.get("format_hint", "")).lower() if isinstance(plan, dict) else ""
        )
        _needs_steps = (
            ("steps" in required_sections)
            or ("numbered" in format_hint)
            or (
                str(plan.get("response_type", "")).lower()
                in ("instruction", "troubleshooting")
            )
        )
        _has_steps = any(f"{i}." in resp.lower() for i in range(1, 6)) or (
            "step" in resp.lower()
        )
        if _needs_steps and not _has_steps:
            return {
                "accepted": False,
                "score": 0.0,
                "reason": "plan_violation_missing_steps",
                "fallback_action": "safe_reply",
            }

        score = max(
            0.0,
            min(
                1.0,
                (0.65 * score) + (0.20 * plan_adherence) + (0.15 * goal_alignment),
            ),
        )
        threshold = 0.52 if route == "llm" else 0.48

        if score >= threshold:
            return {
                "accepted": True,
                "score": score,
                "reason": (
                    "accepted"
                    if plan_adherence >= 0.60
                    else "accepted_low_plan_adherence"
                ),
                "fallback_action": "",
            }

        fallback_action = (
            "clarify" if (overlap < 0.2 or plan_adherence < 0.45) else "safe_reply"
        )
        return {
            "accepted": False,
            "score": score,
            "reason": (
                "low_alignment" if goal_alignment >= 0.35 else "goal_misalignment"
            ),
            "fallback_action": fallback_action,
        }

    def _semantic_anchor_tokens(self, text: str) -> List[str]:
        tokens = []
        for token in self._tokens(text):
            low = token.lower().strip()
            if len(low) < 4:
                continue
            if low in self.SEMANTIC_STOPWORDS:
                continue
            tokens.append(low)
        # Preserve stable order with uniqueness.
        seen = set()
        ordered: List[str] = []
        for tok in tokens:
            if tok in seen:
                continue
            seen.add(tok)
            ordered.append(tok)
        return ordered

    def _semantic_fidelity_guard(
        self,
        *,
        user_input: str,
        intent: str,
        response: str,
    ) -> Dict[str, Any] | None:
        """Reject responses that lose core user meaning or inject obvious nonsense."""
        low_resp = (response or "").lower()
        if "person 'an ai'" in low_resp or "general_assistance" in low_resp:
            return {
                "accepted": False,
                "score": 0.0,
                "reason": "semantic_noise_in_response",
                "fallback_action": "clarify",
            }

        normalized_intent = (intent or "").lower().strip()
        if not (
            normalized_intent.startswith("conversation:")
            or normalized_intent.startswith("learning:")
            or normalized_intent in {"greeting", "thanks"}
        ):
            return None

        user_anchors = self._semantic_anchor_tokens(user_input)
        if len(user_anchors) < 2:
            return None

        response_tokens = set(self._tokens(response))
        overlap = [tok for tok in user_anchors if tok in response_tokens]
        if not overlap:
            return {
                "accepted": False,
                "score": 0.0,
                "reason": "semantic_core_missing",
                "fallback_action": "clarify",
            }

        # Guard against programming-language drift on conceptual assistant architecture questions.
        programming_drift_terms = {"polymorphism", "interface", "inheritance", "encapsulation", "oop"}
        if programming_drift_terms.intersection(response_tokens) and not programming_drift_terms.intersection(set(user_anchors)):
            return {
                "accepted": False,
                "score": 0.0,
                "reason": "semantic_drift_programming_domain",
                "fallback_action": "clarify",
            }

        return None

    def _response_plan_adherence(self, response: str, context: Dict[str, Any]) -> float:
        """Return 0..1 estimate of how well the response follows the active response plan."""
        plan = (context or {}).get("response_plan", {}) or {}
        if not isinstance(plan, dict) or not plan:
            return 1.0

        resp_low = (response or "").lower()
        score = 0.0
        checks = 0.0

        resp_type = str(plan.get("response_type", "")).strip().lower()
        format_hint = str(plan.get("format_hint", "")).strip().lower()
        required_sections = plan.get("required_sections", []) or []

        # Required sections are soft checks via lexical cues.
        section_markers = {
            "answer": ("answer", "in short", "summary"),
            "explanation": ("because", "means", "works by"),
            "example": ("for example", "example", "e.g."),
            "steps": ("1.", "2.", "step", "first", "next", "then"),
            "expected_result": ("expected", "result", "you should see"),
            "root_cause": ("root cause", "caused by", "reason"),
            "fix": ("fix", "solution", "change", "update"),
            "plan": ("plan", "phase", "milestone"),
            "risks": ("risk", "trade-off", "constraint"),
            "check_understanding": (
                "does that make sense",
                "want me to",
                "would you like",
            ),
        }

        for sec in required_sections:
            checks += 1.0
            markers = section_markers.get(str(sec), ())
            if markers and any(m in resp_low for m in markers):
                score += 1.0

        # Format adherence check.
        checks += 1.0
        if "numbered" in format_hint:
            if any(f"{i}." in resp_low for i in range(1, 6)):
                score += 1.0
        elif "structured" in format_hint:
            if "\n" in response and (":" in response or "- " in response):
                score += 1.0
        else:
            score += 1.0

        # Type-level sanity check.
        checks += 1.0
        if resp_type in ("instruction", "troubleshooting"):
            if any(f"{i}." in resp_low for i in range(1, 6)) or "step" in resp_low:
                score += 1.0
        elif resp_type == "explanation":
            if "because" in resp_low or "example" in resp_low:
                score += 1.0
        elif resp_type == "debugging":
            if any(m in resp_low for m in ("error", "fix", "cause", "traceback")):
                score += 1.0
        else:
            score += 1.0

        if checks <= 0.0:
            return 1.0
        return max(0.0, min(1.0, score / checks))

    def decide_learning(
        self,
        *,
        relevance: float,
        confidence: float,
        novelty: float,
        risk: float,
    ) -> str:
        """Executive authority over learning writes.

        Returns one of: store | reject | queue_review | temporary
        """
        relevance = max(0.0, min(1.0, float(relevance)))
        confidence = max(0.0, min(1.0, float(confidence)))
        novelty = max(0.0, min(1.0, float(novelty)))
        risk = max(0.0, min(1.0, float(risk)))

        utility = (0.45 * relevance) + (0.35 * confidence) + (0.20 * novelty)
        if risk >= 0.70:
            return "reject"
        if utility >= 0.72 and risk <= 0.35:
            return "store"
        if utility >= 0.55 and risk <= 0.55:
            return "temporary"
        if utility >= 0.45:
            return "queue_review"
        return "reject"

    def format_reasoning_state(self, state: ReasoningState) -> str:
        """Render compact, non-CoT internal state for system context."""
        lines = [
            "Internal reasoning state (system-only):",
            f"- user_intent: {state.user_intent}",
            f"- topic: {state.topic or 'unknown'}",
            f"- confidence: {state.confidence:.2f}",
            f"- conversation_goal: {state.conversation_goal or 'general_assistance'}",
            f"- user_goal: {state.user_goal or 'none'}",
            f"- depth_level: {state.depth_level}",
        ]
        if state.plan:
            lines.append(f"- plan: {' | '.join(state.plan)}")
        return "\n".join(lines)

    def _derive_plan(
        self, intent: str, topic: str, depth_level: int, user_input: str
    ) -> List[str]:
        lowered_intent = (intent or "").lower()
        lowered_input = (user_input or "").lower()

        if lowered_intent.startswith("learning:") or lowered_intent.startswith(
            "question:"
        ):
            if (
                depth_level >= 3
                or "example" in lowered_input
                or "code" in lowered_input
            ):
                return [
                    "explain succinctly",
                    "give concrete example",
                    "offer deeper explanation",
                ]
            return [
                "explain simply",
                "give example",
                "offer deeper explanation",
            ]

        if topic:
            return ["answer current question", "keep topic continuity"]

        return ["answer directly", "ask clarification if ambiguity remains"]

    def _tokens(self, text: str) -> List[str]:
        import re

        return re.findall(r"[a-z0-9']+", (text or "").lower())


_executive_controller: ExecutiveController | None = None


def get_executive_controller() -> ExecutiveController:
    global _executive_controller
    if _executive_controller is None:
        _executive_controller = ExecutiveController()
    return _executive_controller
