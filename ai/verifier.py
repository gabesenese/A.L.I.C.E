"""
Verifier for A.L.I.C.E
Validates that actions succeeded and user goals were fulfilled.
Uses world state and plugin results so A.L.I.C.E can correct or clarify when needed.
"""

import re
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .world_state import WorldState, get_world_state

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of verifying an action or goal."""
    verified: bool
    action_succeeded: bool
    goal_fulfilled: bool
    message: Optional[str] = None
    suggested_follow_up: Optional[str] = None


class Verifier:
    """
    Verifies actions and goal fulfillment. Checks:
    - Action success: plugin reported success and response looks consistent
    - Goal fulfillment: response matches what the user asked for
    """

    FAILURE_INDICATORS = [
        r"\b(?:fail|failed|error|couldn\'?t|could\s+not)\b",
        r"\b(?:not\s+found|doesn\'?t\s+exist)\b",
        r"\b(?:sorry|unable|invalid)\b",
    ]
    SUCCESS_INDICATORS = [
        r"\b(?:done|completed|archived|deleted|created|added|sent)\b",
        r"\b(?:ok|success|✓|✅)\b",
        r"^[\s\S]*\b(?:successfully|done)\b[\s\S]*$",
    ]

    def __init__(self, world_state: Optional[WorldState] = None):
        self.world_state = world_state or get_world_state()

    def verify(
        self,
        plugin_result: Dict[str, Any],
        goal_intent: Optional[str] = None,
        goal_description: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify plugin result against reported success and optional goal.
        Returns verification outcome and optional message/follow-up.
        """
        success = plugin_result.get("success", False)
        response = (plugin_result.get("response") or plugin_result.get("message") or "").strip()

        action_succeeded = self._check_action_success(success, response)
        goal_fulfilled = True
        if goal_intent or goal_description:
            goal_fulfilled = self._check_goal_fulfilled(
                success, response, goal_intent, goal_description
            )

        verified = action_succeeded and goal_fulfilled
        msg: Optional[str] = None
        follow_up: Optional[str] = None

        if not action_succeeded and not success:
            msg = "That didn't complete as expected."
            follow_up = "Would you like me to try again or do something else?"
        elif not goal_fulfilled and success:
            msg = "I did that, but it may not be what you had in mind."
            follow_up = "Say what you’d like changed and I’ll adjust."

        return VerificationResult(
            verified=verified,
            action_succeeded=action_succeeded,
            goal_fulfilled=goal_fulfilled,
            message=msg,
            suggested_follow_up=follow_up,
        )

    def _check_action_success(self, reported: bool, response: str) -> bool:
        """Infer from response text whether the action really succeeded."""
        if not reported:
            for pat in self.FAILURE_INDICATORS:
                if re.search(pat, response, re.IGNORECASE):
                    return False
            return False
        for pat in self.FAILURE_INDICATORS:
            if re.search(pat, response, re.IGNORECASE):
                return False
        return True

    def _check_goal_fulfilled(
        self,
        success: bool,
        response: str,
        goal_intent: Optional[str],
        goal_description: Optional[str],
    ) -> bool:
        """Heuristic: goal about delete/remove -> look for archived/deleted/done in response."""
        if not success:
            return False
        lower = response.lower()
        if goal_intent and ("delete" in goal_intent or "remove" in goal_intent):
            return bool(re.search(r"\b(?:archived|deleted|removed|done)\b", lower))
        if goal_intent and ("create" in goal_intent or "add" in goal_intent):
            return bool(re.search(r"\b(?:created|added|done|saved)\b", lower))
        return True


_verifier: Optional[Verifier] = None


def get_verifier(world_state: Optional[WorldState] = None) -> Verifier:
    global _verifier
    if _verifier is None:
        _verifier = Verifier(world_state=world_state)
    return _verifier
