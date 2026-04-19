"""Foundational NLP orchestration layers for agent-grade behavior."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

_FILLER_RE = re.compile(r"\b(?:um+|uh+|er+|ah+|like)\b", re.IGNORECASE)
_HEDGE_RE = re.compile(r"\b(?:kind of|sort of|i guess|maybe)\b", re.IGNORECASE)


@dataclass
class TurnBudget:
    started_at: float = field(default_factory=time.perf_counter)
    budget_ms: float = 120.0

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000.0

    def remaining_ms(self) -> float:
        return max(0.0, self.budget_ms - self.elapsed_ms())


@dataclass
class PlanState:
    goal: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)
    domain: str = ""
    turns_seen: int = 0


class FoundationLayers:
    """Small, composable layers for robust assistant routing."""

    def __init__(self, budget_ms: float = 120.0):
        self._budget_ms = float(budget_ms)
        self._plan = PlanState()
        self._metrics = {"turns": 0, "clarifications": 0, "safety_blocks": 0}
        self._clarification_conf_threshold = 0.58
        self._clarification_margin_threshold = 0.35
        self._deep_stage_skip_threshold = 0.95

        try:
            from ai.optimization.runtime_thresholds import get_thresholds

            thresholds = get_thresholds()
            self._clarification_conf_threshold = float(
                thresholds.get(
                    "foundation_clarification_confidence",
                    self._clarification_conf_threshold,
                )
            )
            self._clarification_margin_threshold = float(
                thresholds.get(
                    "foundation_clarification_margin",
                    self._clarification_margin_threshold,
                )
            )
            self._deep_stage_skip_threshold = float(
                thresholds.get(
                    "foundation_deep_stage_skip_threshold",
                    self._deep_stage_skip_threshold,
                )
            )
        except Exception:
            # Keep constructor dependency-light; fallback defaults are already set.
            pass

    # 1) ASR-aware normalization (text-first variant; safe for non-voice input too).
    def normalize_input(self, text: str) -> str:
        normalized = str(text or "")
        normalized = _FILLER_RE.sub(" ", normalized)
        normalized = _HEDGE_RE.sub(" ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    # 2) Multi-turn plan memory.
    def update_plan_memory(
        self, *, intent: str, parsed_command: Dict[str, Any], text: str
    ) -> Dict[str, Any]:
        self._plan.turns_seen += 1
        if (
            parsed_command.get("object_type")
            and parsed_command.get("object_type") != "unknown"
        ):
            self._plan.domain = str(parsed_command.get("object_type"))
        if parsed_command.get("title_hint"):
            self._plan.goal = str(parsed_command.get("title_hint"))
        if "by " in text.lower():
            self._plan.constraints["deadline_hint"] = text
        if intent:
            self._plan.constraints["last_intent"] = intent
        return {
            "goal": self._plan.goal,
            "domain": self._plan.domain,
            "constraints": dict(self._plan.constraints),
            "turns_seen": self._plan.turns_seen,
        }

    # 3) Grounded world-state integration (extensible, no hard dependency).
    def apply_grounding(
        self,
        parsed_command: Dict[str, Any],
        world_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = dict(world_state or {})
        hints = {}
        if "location" in state:
            hints["location"] = state["location"]
        if "timezone" in state:
            hints["timezone"] = state["timezone"]
        if "is_stale" in state:
            hints["is_stale"] = bool(state.get("is_stale"))
            hints["requires_refresh"] = bool(state.get("is_stale"))
        if "age_seconds" in state:
            hints["age_seconds"] = float(state.get("age_seconds") or 0.0)
        if hints:
            parsed_command.setdefault("modifiers", {})["grounding_hints"] = hints
        return hints

    # 4) Proactive clarification policy.
    def clarification_policy(
        self,
        *,
        plugin_scores: Dict[str, float],
        confidence: float,
        profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ordered = sorted(plugin_scores.items(), key=lambda x: x[1], reverse=True)
        top = ordered[0][1] if ordered else 0.0
        second = ordered[1][1] if len(ordered) > 1 else 0.0
        margin = top - second
        needs = bool(
            confidence < self._clarification_conf_threshold
            or margin < self._clarification_margin_threshold
        )
        if needs:
            self._metrics["clarifications"] += 1
        prompt = ""
        if needs and ordered:
            top_name = ordered[0][0]
            second_name = ordered[1][0] if len(ordered) > 1 else "conversation"
            brevity = str((profile or {}).get("response_brevity") or "balanced").lower()
            if brevity == "concise":
                prompt = f"{top_name} or {second_name}?"
            else:
                prompt = (
                    f"I can handle this as {top_name} or {second_name}. "
                    f"Which one do you want?"
                )
        return {
            "needs_clarification": needs,
            "top_margin": margin,
            "top_score": top,
            "prompt": prompt,
            "thresholds": {
                "confidence": self._clarification_conf_threshold,
                "margin": self._clarification_margin_threshold,
            },
        }

    def response_style_hint(
        self, profile: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        prefs = dict(profile or {})
        return {
            "brevity": str(prefs.get("response_brevity") or "balanced"),
            "confirmation": str(prefs.get("confirmation_style") or "safety_first"),
            "risk_tolerance": str(prefs.get("risk_tolerance") or "medium"),
        }

    def tune_from_outcome(
        self,
        *,
        false_clarification: bool = False,
        wrong_tool_execution: bool = False,
    ) -> Dict[str, float]:
        """Adaptive threshold calibration from runtime outcomes (P0)."""
        if false_clarification:
            self._clarification_conf_threshold = max(
                0.45, self._clarification_conf_threshold - 0.01
            )
            self._clarification_margin_threshold = max(
                0.20, self._clarification_margin_threshold - 0.01
            )
        if wrong_tool_execution:
            self._clarification_conf_threshold = min(
                0.75, self._clarification_conf_threshold + 0.02
            )
            self._clarification_margin_threshold = min(
                0.50, self._clarification_margin_threshold + 0.015
            )
        return {
            "confidence": round(self._clarification_conf_threshold, 4),
            "margin": round(self._clarification_margin_threshold, 4),
        }

    # 5) Evaluation harness (online turn metrics).
    def record_turn(
        self, *, confidence: float, clarification: bool, safety_blocked: bool
    ) -> Dict[str, Any]:
        self._metrics["turns"] += 1
        if clarification:
            self._metrics["clarifications"] += 1
        if safety_blocked:
            self._metrics["safety_blocks"] += 1
        return {
            "turns": self._metrics["turns"],
            "clarifications": self._metrics["clarifications"],
            "safety_blocks": self._metrics["safety_blocks"],
            "last_confidence": float(confidence),
        }

    # 6) Latency budget + staged inference.
    def new_budget(self) -> TurnBudget:
        return TurnBudget(budget_ms=self._budget_ms)

    def should_run_deep_stage(
        self, budget: TurnBudget, *, shallow_confidence: float
    ) -> bool:
        if shallow_confidence >= self._deep_stage_skip_threshold:
            return False
        return budget.remaining_ms() > 20.0

    # 7) Safety / authorization gate.
    def authorization_policy(self, *, intent: str, text: str) -> Dict[str, Any]:
        low = (text or "").lower()
        high_risk_intents = {
            "system:shutdown",
            "system:reboot",
            "email:delete",
            "notes:delete",
        }
        high_risk_language = {"delete all", "wipe", "format", "shutdown", "reboot"}
        blocked = intent in high_risk_intents or any(
            cue in low for cue in high_risk_language
        )
        return {
            "allowed": not blocked,
            "requires_confirmation": blocked,
            "reason": "high_risk_action" if blocked else "allowed",
        }
