"""
Global Goal Resolver for A.L.I.C.E
Tracks user goals across turns, resolves goal references ("cancel that", "actually do X"),
and aligns intent with the current goal so A.L.I.C.E follows through correctly.
"""

import re
import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .world_state import WorldState, ActiveGoal, get_world_state

logger = logging.getLogger(__name__)


@dataclass
class GoalResolution:
    """Result of resolving the current user goal."""
    goal: Optional[ActiveGoal]
    intent: str
    entities: Dict[str, Any]
    cancelled: bool
    revised: bool
    message: Optional[str] = None


class GlobalGoalResolver:
    """
    Tracks and resolves user goals. Handles:
    - New goal: user states a goal -> push and use it
    - Cancel: "cancel that", "never mind" -> pop/clear goal
    - Revise: "actually do X instead" -> replace goal
    - Reference: "do it", "go ahead" -> reuse current goal
    """

    CANCEL_PATTERNS = [
        r"\b(?:cancel|abort|never\s+mind|forget\s+it|don\'?t\s+do\s+that)\b",
        r"\b(?:stop|undo)\s+that\b",
        r"\bno\s*,?\s*(?:forget\s+it|cancel)\b",
    ]
    REVISE_PATTERNS = [
        r"\bactually\s+(.+)$",
        r"\binstead\s+(.+)$",
        r"\b(?:no\s*,?\s*)?(?:do|make|create|delete|remove)\s+(.+)$",
    ]
    REFERENCE_PATTERNS = [
        r"\b(?:do\s+it|go\s+ahead|proceed|yes)\s*\.?\s*$",
        r"\b(?:that\'?s\s+what\s+i\s+meant|correct|right)\s*\.?\s*$",
    ]

    def __init__(self, world_state: Optional[WorldState] = None):
        self.world_state = world_state or get_world_state()

    def resolve(
        self,
        user_input: str,
        intent: str,
        entities: Dict[str, Any],
        resolved_input: Optional[str] = None,
    ) -> GoalResolution:
        """
        Resolve current goal from user input and intent.
        Returns goal (or None), possibly updated intent/entities, and flags.
        """
        cancelled = False
        revised = False
        msg: Optional[str] = None
        inp = (resolved_input or user_input).strip().lower()

        # 1) Cancel: "cancel that", "never mind"
        for pat in self.CANCEL_PATTERNS:
            if re.search(pat, inp):
                self.world_state.set_goal(None)
                cancelled = True
                msg = "Understood. Cancelled."
                return GoalResolution(
                    goal=None,
                    intent="conversation:ack",
                    entities={},
                    cancelled=True,
                    revised=False,
                    message=msg,
                )

        # 2) Revise: "actually delete the shopping list" -> new goal, replace current
        for pat in self.REVISE_PATTERNS:
            m = re.search(pat, inp, re.IGNORECASE | re.DOTALL)
            if not m:
                continue
            rest = m.group(1).strip() if m.lastindex else ""
            if len(rest) < 3:
                continue
            goal_id = f"goal_{uuid.uuid4().hex[:8]}"
            new_goal = ActiveGoal(
                goal_id=goal_id,
                description=rest,
                intent=intent,
                entities=dict(entities),
            )
            self.world_state.set_goal(new_goal)
            revised = True
            return GoalResolution(
                goal=new_goal,
                intent=intent,
                entities=entities,
                cancelled=False,
                revised=True,
                message=None,
            )

        # 3) Reference: "do it", "go ahead" -> reuse current goal
        for pat in self.REFERENCE_PATTERNS:
            if re.search(pat, inp):
                cur = self.world_state.active_goal
                if cur:
                    return GoalResolution(
                        goal=cur,
                        intent=cur.intent,
                        entities=cur.entities,
                        cancelled=False,
                        revised=False,
                        message=None,
                    )
                return GoalResolution(
                    goal=None,
                    intent=intent,
                    entities=entities,
                    cancelled=False,
                    revised=False,
                    message="I'm not sure what to do—could you say it again?",
                )

        # 4) Check if current input relates to existing goal
        current_goal = self.world_state.active_goal
        if current_goal:
            # Check if this input is related to the current goal
            goal_keywords = set(current_goal.description.lower().split())
            input_keywords = set(inp.split())
            # Check entity overlap
            goal_entities = set(str(v).lower() for v in current_goal.entities.values() if v)
            input_entities = set(str(v).lower() for v in entities.values() if v)
            
            # If intent matches or entities overlap, continue with current goal
            if (intent == current_goal.intent or 
                len(goal_keywords & input_keywords) >= 2 or
                len(goal_entities & input_entities) > 0 or
                any(word in inp for word in current_goal.description.lower().split()[:5])):
                # Update goal with latest input but keep same goal
                current_goal.entities = {**current_goal.entities, **entities}
                # Update description if it adds new info
                if len(user_input) > len(current_goal.description):
                    current_goal.description = user_input[:200]
                return GoalResolution(
                    goal=current_goal,
                    intent=intent if intent != "conversation:general" else current_goal.intent,
                    entities={**current_goal.entities, **entities},
                    cancelled=False,
                    revised=False,
                    message=None,
                )
        
        # 5) New goal: create only if this is clearly a new task
        # Don't create goal for simple acknowledgments or follow-ups
        if intent in ["conversation:ack", "conversation:general"] and len(inp.split()) < 5:
            # Short follow-up, likely related to current goal
            if current_goal:
                return GoalResolution(
                    goal=current_goal,
                    intent=current_goal.intent,
                    entities=current_goal.entities,
                    cancelled=False,
                    revised=False,
                    message=None,
                )
        
        # Create new goal for substantial new requests
        goal_id = f"goal_{uuid.uuid4().hex[:8]}"
        goal = ActiveGoal(
            goal_id=goal_id,
            description=user_input[:200],
            intent=intent,
            entities=dict(entities),
        )
        self.world_state.set_goal(goal)
        return GoalResolution(
            goal=goal,
            intent=intent,
            entities=entities,
            cancelled=False,
            revised=False,
            message=None,
        )

    def get_current_goal(self) -> Optional[ActiveGoal]:
        return self.world_state.active_goal

    def mark_goal_completed(self) -> None:
        self.world_state.set_goal(None)


_goal_resolver: Optional[GlobalGoalResolver] = None


def get_goal_resolver(world_state: Optional[WorldState] = None) -> GlobalGoalResolver:
    global _goal_resolver
    if _goal_resolver is None:
        _goal_resolver = GlobalGoalResolver(world_state=world_state)
    return _goal_resolver
