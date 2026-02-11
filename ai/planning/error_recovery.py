"""
Error Recovery & Replanning
============================
Handles errors during autonomous execution and replans when needed.
Makes Alice resilient to failures and able to adapt to changing circumstances.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Strategies for error recovery"""
    RETRY = "retry"  # Try the same step again
    SKIP = "skip"  # Skip this step and continue
    REPLAN = "replan"  # Replan remaining steps
    ESCALATE = "escalate"  # Ask user for help
    ABORT = "abort"  # Give up on this goal


@dataclass
class ErrorRecord:
    """Record of an error that occurred"""
    step_id: str
    error_type: str
    error_message: str
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    recovery_attempted: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None
    recovery_successful: bool = False


class ErrorRecoverySystem:
    """
    Intelligent error recovery and replanning system.

    Capabilities:
    - Classifies errors by type and severity
    - Selects appropriate recovery strategy
    - Retries with exponential backoff
    - Replans when steps become obsolete
    - Escalates to user when stuck
    - Learns from recovery successes/failures
    """

    def __init__(
        self,
        max_retries: int = 3,
        max_replans: int = 2,
        escalation_threshold: int = 5
    ):
        self.max_retries = max_retries
        self.max_replans = max_replans
        self.escalation_threshold = escalation_threshold

        # Error tracking
        self.error_history: List[ErrorRecord] = []
        self.retry_counts: Dict[str, int] = {}
        self.replan_counts: Dict[str, int] = {}

        # Recovery success tracking
        self.successful_recoveries: Dict[str, RecoveryStrategy] = {}

    def handle_error(
        self,
        goal_id: str,
        step_id: str,
        error: Exception,
        context: Dict[str, Any]
    ) -> Tuple[RecoveryStrategy, Optional[str]]:
        """
        Handle an error and determine recovery strategy.

        Args:
            goal_id: ID of the goal that failed
            step_id: ID of the step that failed
            error: The exception that occurred
            context: Execution context

        Returns:
            (Recovery strategy, optional message for user)
        """
        # Classify error
        error_type = self._classify_error(error)

        # Record error
        error_record = ErrorRecord(
            step_id=step_id,
            error_type=error_type,
            error_message=str(error)
        )
        self.error_history.append(error_record)

        # Determine recovery strategy
        strategy = self._select_recovery_strategy(
            goal_id,
            step_id,
            error_type,
            context
        )

        error_record.recovery_strategy = strategy
        error_record.recovery_attempted = True

        logger.info(f"Error recovery strategy: {strategy.value} for {error_type}")

        # Generate message if escalating
        message = None
        if strategy == RecoveryStrategy.ESCALATE:
            message = self._generate_escalation_message(goal_id, step_id, error)

        return strategy, message

    def _classify_error(self, error: Exception) -> str:
        """Classify error by type"""
        error_str = str(error).lower()
        error_type_name = type(error).__name__

        # Network/connection errors
        if 'connection' in error_str or 'timeout' in error_str or 'network' in error_str:
            return 'network_error'

        # File system errors
        if 'filenotfound' in error_type_name.lower() or 'no such file' in error_str:
            return 'file_not_found'

        if 'permission' in error_str:
            return 'permission_denied'

        # Syntax/code errors
        if 'syntax' in error_str or 'syntaxerror' in error_type_name.lower():
            return 'syntax_error'

        # Import errors
        if 'import' in error_str or 'module' in error_str:
            return 'import_error'

        # Resource errors
        if 'memory' in error_str or 'disk' in error_str:
            return 'resource_error'

        # Default
        return 'unknown_error'

    def _select_recovery_strategy(
        self,
        goal_id: str,
        step_id: str,
        error_type: str,
        context: Dict[str, Any]
    ) -> RecoveryStrategy:
        """Select appropriate recovery strategy"""

        # Check retry count
        retry_key = f"{goal_id}:{step_id}"
        retry_count = self.retry_counts.get(retry_key, 0)

        # Check replan count
        replan_count = self.replan_counts.get(goal_id, 0)

        # Check total error count
        total_errors = len([e for e in self.error_history if e.step_id == step_id])

        # Strategy selection logic
        if total_errors >= self.escalation_threshold:
            return RecoveryStrategy.ESCALATE

        # Transient errors - retry
        if error_type in ['network_error', 'timeout', 'resource_error']:
            if retry_count < self.max_retries:
                self.retry_counts[retry_key] = retry_count + 1
                return RecoveryStrategy.RETRY
            else:
                # Too many retries - replan
                if replan_count < self.max_replans:
                    self.replan_counts[goal_id] = replan_count + 1
                    return RecoveryStrategy.REPLAN
                else:
                    return RecoveryStrategy.ESCALATE

        # File not found - might need to adjust path or skip
        if error_type == 'file_not_found':
            if retry_count < 1:
                self.retry_counts[retry_key] = retry_count + 1
                return RecoveryStrategy.RETRY
            else:
                return RecoveryStrategy.SKIP

        # Code errors - likely need replanning or user help
        if error_type in ['syntax_error', 'import_error']:
            if replan_count < self.max_replans:
                self.replan_counts[goal_id] = replan_count + 1
                return RecoveryStrategy.REPLAN
            else:
                return RecoveryStrategy.ESCALATE

        # Permission denied - escalate immediately
        if error_type == 'permission_denied':
            return RecoveryStrategy.ESCALATE

        # Unknown errors - try once, then escalate
        if retry_count == 0:
            self.retry_counts[retry_key] = 1
            return RecoveryStrategy.RETRY
        else:
            return RecoveryStrategy.ESCALATE

    def _generate_escalation_message(
        self,
        goal_id: str,
        step_id: str,
        error: Exception
    ) -> str:
        """Generate message for user escalation"""
        recent_errors = [
            e for e in self.error_history[-5:]
            if e.step_id == step_id
        ]

        message = f"""Autonomous execution needs assistance with goal {goal_id}.

Error: {type(error).__name__}: {str(error)}

I've tried {len(recent_errors)} times but haven't been able to resolve this.
Would you like me to:
1. Continue trying with a different approach
2. Skip this step and proceed
3. Abort this goal

What would you prefer?"""

        return message

    def should_replan(
        self,
        goal_id: str,
        current_plan: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> bool:
        """
        Determine if replanning is needed.

        Args:
            goal_id: Goal ID
            current_plan: Current list of steps
            context: Execution context

        Returns:
            True if replanning is recommended
        """
        # Check failure rate
        failed_steps = sum(1 for step in current_plan if step.get('status') == 'failed')
        total_steps = len(current_plan)

        if total_steps > 0 and failed_steps / total_steps > 0.3:
            return True  # More than 30% failed

        # Check if we're stuck
        completed_recently = context.get('steps_completed_last_hour', 0)
        if completed_recently == 0 and len(current_plan) > 3:
            return True  # Not making progress

        # Check replan count
        replan_count = self.replan_counts.get(goal_id, 0)
        if replan_count >= self.max_replans:
            return False  # Already replanned too many times

        return False

    def reset_for_goal(self, goal_id: str):
        """Reset error tracking for a goal"""
        # Clear retry counts for this goal
        keys_to_remove = [k for k in self.retry_counts.keys() if k.startswith(f"{goal_id}:")]
        for key in keys_to_remove:
            del self.retry_counts[key]

        # Clear replan count
        if goal_id in self.replan_counts:
            del self.replan_counts[goal_id]

        logger.info(f"Reset error tracking for goal: {goal_id}")

    def get_stats(self) -> Dict[str, Any]:
        """Get recovery statistics"""
        total_errors = len(self.error_history)
        recovery_attempted = len([e for e in self.error_history if e.recovery_attempted])
        recoveries_successful = len([e for e in self.error_history if e.recovery_successful])

        success_rate = recoveries_successful / recovery_attempted if recovery_attempted > 0 else 0.0

        return {
            'total_errors': total_errors,
            'recovery_attempts': recovery_attempted,
            'successful_recoveries': recoveries_successful,
            'success_rate': success_rate,
            'active_retries': len(self.retry_counts),
            'active_replans': len(self.replan_counts)
        }


# Global singleton
_recovery_system = None


def get_recovery_system(
    max_retries: int = 3,
    max_replans: int = 2
) -> ErrorRecoverySystem:
    """Get or create global error recovery system"""
    global _recovery_system
    if _recovery_system is None:
        _recovery_system = ErrorRecoverySystem(
            max_retries=max_retries,
            max_replans=max_replans
        )
    return _recovery_system


def get_error_recovery(
    max_retries: int = 3,
    max_replans: int = 2
) -> ErrorRecoverySystem:
    """Alias for get_recovery_system - for consistency with other modules"""
    return get_recovery_system(max_retries=max_retries, max_replans=max_replans)
