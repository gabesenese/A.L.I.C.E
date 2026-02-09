"""
Policy for A.L.I.C.E
Decides execute vs ask (clarify/confirm) given resolver output, optional Goal JSON, and confidence.
"""

import logging
from typing import Optional
from dataclasses import dataclass

from ai.planning.goal_from_llm import GoalJSON

logger = logging.getLogger(__name__)

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
        tool_path_threshold: float = 0.7,
        ask_threshold: float = 0.5,
) -> PolicyDecision:
    """
    Decide whether to execute (tool/plugin path) or ask for clarification.

    Args:
        intent_confidence: Router confidence (0–1).
        has_goal: Whether resolver has an active goal.
        goal_json: Optional Goal from LLM (when router was not confident).
        plugin_available: Whether a plugin matched (caller can pass False before trying plugins).
        tool_path_threshold: Above this → prefer execute when we have goal or plugin.
        ask_threshold: Below this → ask unless goal_json says execute and we're sure.

    Returns:
        PolicyDecision(execute=True) or PolicyDecision(execute=False, clarification_question="...").
    """
    # LLM explicitly said "ask" and gave a question
    if goal_json and goal_json.action == "ask" and goal_json.clarification_question:
        logger.info("[Policy] Goal JSON says ask → returning clarification")
        return PolicyDecision(execute=False, clarification_question=goal_json.clarification_question)
    

    # Very low confidence -> ask (unless goal_json gave us a clear execute)
    if intent_confidence < ask_threshold:
        if goal_json and goal_json.action == "execute":
            logger.info("[Policy] Low confidence but Goal JSON says execute -> executing")
            return PolicyDecision(execute=True)
        logger.info(f"[Policy] Low confidence ({intent_confidence:.2f}) -> ask")
        return PolicyDecision(
            execute=False,
            clarification_question=goal_json.clarification_question if goal_json else None,
        )
    
    # Confident enough for tool path
    if intent_confidence >= tool_path_threshold and (has_goal or plugin_available):
        return PolicyDecision(execute=True)
    
    # Default: execute (let plugin/LLM path handle it)
    return PolicyDecision(execute=True)
