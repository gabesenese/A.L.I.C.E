"""
Unified Policy Manager for A.L.I.C.E
Single point for all policy and decision making
Consolidates:
- LLMPolicy (rate limiting, approvals)
- ExecutionPolicy (execute vs ask decisions)
- ConfidencePolicy (threshold-based routing)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LLMCallType(Enum):
    """Types of LLM calls for rate limiting"""
    CHAT = "chat"
    KNOWLEDGE = "knowledge"
    PARSE = "parse"
    PHRASE = "phrase"
    AUDIT = "audit"


@dataclass
class PolicyDecision:
    """Unified policy decision result"""
    allowed: bool
    reason: str
    should_ask_user: bool = False
    alternative_action: Optional[str] = None
    confidence_threshold: Optional[float] = None


class PolicyManager:
    """Single point for all policy and decision making"""

    def __init__(self) -> None:
        # Import dependencies
        try:
            from ai.core.llm_policy import LLMPolicy
            self.llm_policy = LLMPolicy.get_instance()
        except Exception as e:
            logger.warning(f"LLM policy not available: {e}")
            self.llm_policy = None

        try:
            from ai.optimization.runtime_thresholds import get_thresholds
            self.thresholds = get_thresholds()
        except Exception as e:
            logger.error(f"Failed to load thresholds: {e}")
            self.thresholds = {
                'tool_path_confidence': 0.7,
                'goal_path_confidence': 0.6,
                'ask_threshold': 0.5
            }

        logger.info("[PolicyManager] Initialized unified policy system")

    def can_call_llm(
        self,
        call_type: LLMCallType,
        user_input: str,
        confidence: Optional[float] = None
    ) -> Tuple[bool, str]:
        """
        Check if LLM call is allowed considering ALL policies:
        1. Rate limiting (LLMBudgetPolicy)
        2. User approval requirements (LLMPolicy)
        3. Confidence thresholds (ConfidencePolicy)

        Args:
            call_type: Type of LLM call
            user_input: User's input
            confidence: Optional confidence score

        Returns:
            Tuple of (allowed, reason)
        """
        # Check rate limits
        if self.llm_policy and not self.llm_policy.can_make_call(call_type.value):
            return False, f"Rate limit exceeded for {call_type.value} calls"

        # Check user approval needs
        if self.llm_policy:
            needs_approval = self.llm_policy.needs_user_approval(call_type.value)
            if needs_approval and not self.llm_policy.has_user_approval(call_type.value):
                return False, f"User approval required for {call_type.value} calls"

        # Check confidence thresholds if provided
        if confidence is not None:
            min_confidence = self.thresholds.get('tool_path_confidence', 0.7)
            if confidence < min_confidence:
                return False, f"Confidence {confidence:.2f} below threshold {min_confidence}"

        return True, "All policies satisfied"

    def should_execute(
        self,
        intent_confidence: float,
        has_goal: bool,
        goal_json: Optional[Dict[str, Any]] = None
    ) -> PolicyDecision:
        """
        Decide whether to execute or ask for clarification
        Integrates: infrastructure/policy.py logic + runtime thresholds

        Args:
            intent_confidence: Confidence in intent classification
            has_goal: Whether a goal was detected
            goal_json: Optional goal structure

        Returns:
            PolicyDecision with execution recommendation
        """
        tool_threshold = self.thresholds.get('tool_path_confidence', 0.7)
        ask_threshold = self.thresholds.get('ask_threshold', 0.5)
        goal_threshold = self.thresholds.get('goal_path_confidence', 0.6)

        # High confidence - execute directly
        if intent_confidence >= tool_threshold:
            return PolicyDecision(
                allowed=True,
                reason=f"High confidence ({intent_confidence:.2f} >= {tool_threshold})",
                should_ask_user=False,
                confidence_threshold=tool_threshold
            )

        # Goal detected with sufficient confidence
        if has_goal and goal_json and intent_confidence >= goal_threshold:
            return PolicyDecision(
                allowed=True,
                reason=f"Goal detected with sufficient confidence ({intent_confidence:.2f} >= {goal_threshold})",
                should_ask_user=False,
                confidence_threshold=goal_threshold
            )

        # Medium confidence - ask for clarification
        if intent_confidence >= ask_threshold:
            return PolicyDecision(
                allowed=False,
                reason=f"Medium confidence ({intent_confidence:.2f}), requesting clarification",
                should_ask_user=True,
                alternative_action="ask_for_clarification",
                confidence_threshold=ask_threshold
            )

        # Low confidence - fallback to conversation
        return PolicyDecision(
            allowed=False,
            reason=f"Low confidence ({intent_confidence:.2f} < {ask_threshold})",
            should_ask_user=True,
            alternative_action="general_conversation",
            confidence_threshold=ask_threshold
        )

    def load_dynamic_thresholds(self, threshold_file: Optional[str] = None) -> bool:
        """
        Reload thresholds from file

        Args:
            threshold_file: Optional path to threshold file

        Returns:
            True if reloaded successfully
        """
        try:
            # Clear cache and reload
            import ai.optimization.runtime_thresholds as rt
            if hasattr(rt, '_cached'):
                rt._cached = {}

            from ai.optimization.runtime_thresholds import get_thresholds
            self.thresholds = get_thresholds()

            logger.info(f"[PolicyManager] Reloaded thresholds: {self.thresholds}")
            return True
        except Exception as e:
            logger.error(f"Failed to reload thresholds: {e}")
            return False

    def get_threshold(self, key: str, default: float = 0.7) -> float:
        """
        Get specific threshold value

        Args:
            key: Threshold key
            default: Default value if not found

        Returns:
            Threshold value
        """
        return self.thresholds.get(key, default)

    def update_threshold(self, key: str, value: float) -> bool:
        """
        Update a specific threshold

        Args:
            key: Threshold key
            value: New value

        Returns:
            True if updated successfully
        """
        try:
            from ai.optimization.runtime_thresholds import update_thresholds
            update_thresholds({key: value})

            # Reload thresholds
            self.load_dynamic_thresholds()
            return True
        except Exception as e:
            logger.error(f"Failed to update threshold '{key}': {e}")
            return False

    def get_all_thresholds(self) -> Dict[str, float]:
        """Get all current thresholds"""
        return self.thresholds.copy()

    def get_policy_summary(self) -> Dict[str, Any]:
        """
        Get summary of current policy state

        Returns:
            Policy summary dictionary
        """
        summary = {
            'thresholds': self.get_all_thresholds(),
            'llm_policy_available': self.llm_policy is not None
        }

        if self.llm_policy:
            try:
                summary['llm_rate_limits'] = {
                    call_type.value: self.llm_policy.can_make_call(call_type.value)
                    for call_type in LLMCallType
                }
            except Exception as e:
                logger.error(f"Failed to get LLM rate limits: {e}")

        return summary


# Singleton instance
_policy_manager: Optional[PolicyManager] = None


def get_policy_manager() -> PolicyManager:
    """Get or create the PolicyManager singleton"""
    global _policy_manager
    if _policy_manager is None:
        _policy_manager = PolicyManager()
    return _policy_manager
