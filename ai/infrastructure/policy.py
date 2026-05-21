"""
Policy for A.L.I.C.E
Decides execute vs ask (clarify/confirm) given resolver output, optional Goal JSON, and confidence.
"""

import logging
import re
from typing import Optional
from dataclasses import dataclass

from ai.planning.goal_from_llm import GoalJSON

logger = logging.getLogger(__name__)


class RiskClassifier:
    """Classifies an action/intent as low, medium, or high risk.

    Risk level drives approval requirements and dry-run enforcement.
    """

    _HIGH_RISK = re.compile(
        r"\b(delete|remove|drop|wipe|destroy|purge|rm\b|rm -rf|force.push|"
        r"overwrite|truncate|format|kill|terminate|sudo|chmod|chown|"
        r"send.email|send.message|post|publish|deploy.to.prod|push.to.main)\b",
        re.IGNORECASE,
    )
    _MEDIUM_RISK = re.compile(
        r"\b(edit|modify|update|patch|write|save|create|install|run|execute|"
        r"move|rename|copy|migrate|restart|reload|enable|disable|push|merge)\b",
        re.IGNORECASE,
    )
    _LOW_RISK = re.compile(
        r"\b(read|get|list|show|open|view|check|inspect|search|find|query|"
        r"fetch|describe|summarize|explain|analyze|compare|preview)\b",
        re.IGNORECASE,
    )

    def classify(self, *, intent: str, user_input: str = "") -> str:
        """Return 'high', 'medium', or 'low'."""
        text = f"{intent} {user_input}".lower()
        if self._HIGH_RISK.search(text):
            return "high"
        if self._MEDIUM_RISK.search(text):
            return "medium"
        if self._LOW_RISK.search(text):
            return "low"
        return "medium"  # default conservative

    def requires_confirmation(self, risk_level: str) -> bool:
        return risk_level == "high"

    def requires_dry_run_offer(self, risk_level: str) -> bool:
        return risk_level in ("high", "medium")


_risk_classifier: RiskClassifier | None = None


def get_risk_classifier() -> RiskClassifier:
    global _risk_classifier
    if _risk_classifier is None:
        _risk_classifier = RiskClassifier()
    return _risk_classifier


@dataclass
class PolicyDecision:
    """Result of policy: execute or ask."""

    execute: bool
    clarification_question: Optional[str] = None


def get_policy_decision(
    intent_confidence: float,
    has_goal: bool,
    goal_json: Optional[GoalJSON] = None,
    plugin_available: bool = False,
    tool_path_threshold: Optional[float] = None,
    ask_threshold: Optional[float] = None,
    intent: str = "",
) -> PolicyDecision:
    """
    Decide whether to execute (tool/plugin path) or ask for clarification.

    Confidence bands:
      >= high_band  → execute directly
      medium..high  → execute (record feedback opportunity)
      low..medium   → clarify, but also attempt execute if goal_json says so
      < low_band    → always clarify

    Args:
        intent_confidence: Router confidence (0–1).
        has_goal: Whether resolver has an active goal.
        goal_json: Optional Goal from LLM (when router was not confident).
        plugin_available: Whether a plugin matched.
        tool_path_threshold: High-confidence execute band lower bound.
        ask_threshold: Low-confidence always-ask upper bound.
        intent: Intent string for per-intent threshold lookup.

    Returns:
        PolicyDecision(execute=True) or PolicyDecision(execute=False, clarification_question="...").
    """
    if tool_path_threshold is None or ask_threshold is None:
        try:
            from ai.optimization.runtime_thresholds import get_thresholds
            thresholds = get_thresholds()
        except Exception:
            thresholds = {}
        if tool_path_threshold is None:
            tool_path_threshold = float(thresholds.get("tool_path_confidence", 0.7))
        if ask_threshold is None:
            ask_threshold = float(thresholds.get("ask_threshold", 0.5))

    # Per-intent threshold overrides: high-stakes actions need higher confidence
    intent_prefix = str(intent or "").split(":")[0].lower()
    _high_stakes = {"local", "operator", "code"}
    if intent_prefix in _high_stakes:
        tool_path_threshold = max(tool_path_threshold, 0.80)
        ask_threshold = max(ask_threshold, 0.55)

    # LLM explicitly said "ask" and gave a question
    if goal_json and goal_json.action == "ask" and goal_json.clarification_question:
        logger.info("[Policy] Goal JSON says ask -> returning clarification")
        return PolicyDecision(
            execute=False, clarification_question=goal_json.clarification_question
        )

    # Band 4: very low confidence — always clarify
    if intent_confidence < ask_threshold:
        if goal_json and goal_json.action == "execute":
            logger.info("[Policy] Low confidence but Goal JSON says execute -> executing")
            return PolicyDecision(execute=True)
        logger.info(f"[Policy] Low confidence ({intent_confidence:.2f}) -> ask")
        return PolicyDecision(
            execute=False,
            clarification_question=(
                goal_json.clarification_question if goal_json else None
            ),
        )

    # Band 2-3: medium confidence — execute but note feedback opportunity
    if intent_confidence < tool_path_threshold:
        if has_goal or plugin_available:
            logger.info(f"[Policy] Medium confidence ({intent_confidence:.2f}) -> execute with feedback flag")
            return PolicyDecision(execute=True)
        # No goal or plugin — clarify
        return PolicyDecision(
            execute=False,
            clarification_question=(
                goal_json.clarification_question if goal_json else None
            ),
        )

    # Band 1: high confidence — execute directly
    return PolicyDecision(execute=True)
