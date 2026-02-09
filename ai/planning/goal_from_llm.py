"""
Goal-from-LLM for A.L.I.C.E
When the router is not confident, the LLM produces a structured Goal JSON
(target intent, target item, execute vs ask) so the resolver and policy can act on it.
"""

import json
import re
import logging
from typing import Optional, Any, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GoalJSON:
    """Structured goal produced by the LLM when intent is unclear."""
    target_intent: str
    target_item: Optional[str] = None  # e.g. "the grocery list", "email 2"
    action: str = "execute"  # "execute" | "ask"
    clarification_question: Optional[str] = None


def _parse_goal_response(raw: str) -> Optional[GoalJSON]:
    """Extract GoalJSON from LLM response (JSON block or line)."""
    raw = (raw or "").strip()
    if not raw:
        return None
    # Try to find a JSON object
    match = re.search(r"\{[^{}]*\"target_intent\"[^{}]*\}", raw)
    if not match:
        match = re.search(r"\{[^{}]+\}", raw)
    if match:
        try:
            obj = json.loads(match.group(0))
            target_intent = obj.get("target_intent") or obj.get("intent") or "conversation:general"
            if not isinstance(target_intent, str):
                target_intent = "conversation:general"
            action = obj.get("action", "execute")
            if action not in ("execute", "ask"):
                action = "execute"
            return GoalJSON(
                target_intent=target_intent,
                target_item=obj.get("target_item"),
                action=action,
                clarification_question=obj.get("clarification_question"),
            )
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def get_goal_from_llm(
    user_input: str,
    llm_chat: Callable[[str, bool], str],
    current_goal_description: Optional[str] = None,
    detected_intent: Optional[str] = None,
) -> Optional[GoalJSON]:
    """
    Ask the LLM to produce a structured Goal JSON for unclear user input.

    Args:
        user_input: What the user said.
        llm_chat: Callable (prompt: str, use_history: bool) -> str (e.g. self.llm.chat).
        current_goal_description: Current active goal description if any.
        detected_intent: Intent from router (may be low confidence).

    Returns:
        GoalJSON or None if LLM failed or returned invalid JSON.
    """
    context = []
    if current_goal_description:
        context.append(f"Current goal: {current_goal_description[:150]}")
    if detected_intent:
        context.append(f"Detected intent (uncertain): {detected_intent}")
    context_str = "\n".join(context) if context else "No current goal."

    prompt = f"""You are a goal classifier. The user said something ambiguous. Output ONLY a single JSON object, no other text.

User input: "{user_input[:300]}"
{context_str}

Output a JSON object with exactly these keys:
- "target_intent": one of email, notes, calendar, music, weather, time, conversation:general, conversation:question, conversation:ack
- "target_item": optional string (e.g. "the grocery list", "latest email") if they refer to something specific
- "action": "execute" or "ask" — use "ask" only if you truly cannot infer what to do and need one short clarification question
- "clarification_question": optional string, only if action is "ask"

Example: {{"target_intent": "notes", "target_item": "shopping list", "action": "execute"}}
Example: {{"target_intent": "conversation:general", "action": "ask", "clarification_question": "Do you want me to search the web or just chat about it?"}}

JSON:"""

    try:
        response = llm_chat(prompt, use_history=False)
        return _parse_goal_response(response)
    except Exception as e:
        logger.warning(f"[GoalFromLLM] LLM call failed: {e}")
        return None