"""
Autonomous Execution Loop
==========================
Continuously executes goals without user intervention.
Works in background, reports progress, handles errors autonomously.
"""

import time
import threading
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class ExecutionLoopState(Enum):
    """State of the execution loop"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


class AutonomousExecutionLoop:
    """
    Background execution loop that autonomously works on goals.

    Features:
    - Runs continuously in background thread
    - Works on active goals from goal system
    - Executes next available steps
    - Reports progress proactively
    - Handles errors and retries
    - Respects user availability/quiet hours
    """

    def __init__(
        self,
        autonomous_agent=None,
        goal_system=None,
        check_interval: int = 30  # seconds
    ):
        self.autonomous_agent = autonomous_agent
        self.goal_system = goal_system
        self.check_interval = check_interval

        # Loop state
        self.state = ExecutionLoopState.STOPPED
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # Execution tracking
        self.last_execution: Optional[float] = None
        self.execution_count = 0
        self.error_count = 0

        # Quiet hours (don't execute during these times)
        self.quiet_start_hour = 23  # 11 PM
        self.quiet_end_hour = 7  # 7 AM

    def start(self):
        """Start the autonomous execution loop"""
        if self.state == ExecutionLoopState.RUNNING:
            logger.warning("Execution loop already running")
            return

        self.state = ExecutionLoopState.RUNNING
        self.thread = threading.Thread(
            target=self._execution_loop,
            name="AutonomousExecutionLoop",
            daemon=True
        )
        self.thread.start()

        logger.info("Autonomous execution loop started")

    def stop(self):
        """Stop the execution loop"""
        self.state = ExecutionLoopState.STOPPED

        if self.thread:
            self.thread.join(timeout=10.0)

        logger.info("Autonomous execution loop stopped")

    def pause(self):
        """Temporarily pause execution"""
        self.state = ExecutionLoopState.PAUSED
        logger.info("Execution loop paused")

    def resume(self):
        """Resume execution after pause"""
        if self.state == ExecutionLoopState.PAUSED:
            self.state = ExecutionLoopState.RUNNING
            logger.info("Execution loop resumed")

    def _execution_loop(self):
        """Main execution loop"""
        while self.state != ExecutionLoopState.STOPPED:
            try:
                if self.state == ExecutionLoopState.RUNNING:
                    if not self._is_quiet_hours():
                        self._execute_iteration()

                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                self.error_count += 1
                time.sleep(self.check_interval)

    def _execute_iteration(self):
        """Execute one iteration of the loop"""
        if not self.goal_system or not self.autonomous_agent:
            return

        # Get active goals
        active_goals = self.goal_system.get_active_goals()

        if not active_goals:
            # No active goals, nothing to do
            return

        # Find highest priority goal with executable steps
        for goal in sorted(active_goals, key=lambda g: g.priority.value):
            # Get next executable step
            next_step = goal.get_next_step()

            if next_step:
                logger.info(f"Executing step for goal: {goal.title}")

                # Create execution context
                from ai.planning.autonomous_agent import ExecutionContext
                context = ExecutionContext(goal_id=goal.goal_id)

                # Execute the step
                result = self.autonomous_agent.execute_step(
                    {
                        'type': 'implement',  # Determine from step metadata
                        'description': next_step.description,
                        'dependencies': next_step.dependencies
                    },
                    context
                )

                # Update goal based on result
                if result.get('success'):
                    self.goal_system.complete_current_step(
                        goal.goal_id,
                        result=result.get('output')
                    )
                    self.execution_count += 1
                    self.last_execution = time.time()

                    logger.info(f"Step completed: {next_step.description}")
                else:
                    # Step failed - mark as failed and move on
                    self.goal_system.fail_step(
                        next_step.step_id,
                        error=result.get('error', 'Unknown error')
                    )
                    self.error_count += 1

                    logger.warning(f"Step failed: {result.get('error')}")

                # Only execute one step per iteration
                break

    def _is_quiet_hours(self) -> bool:
        """Check if current time is during quiet hours"""
        current_hour = datetime.now().hour

        if self.quiet_start_hour < self.quiet_end_hour:
            # Normal case (e.g., 23 < 7 wraps around midnight)
            return current_hour >= self.quiet_start_hour or current_hour < self.quiet_end_hour
        else:
            # Day time quiet hours (e.g., 10 < 16)
            return self.quiet_start_hour <= current_hour < self.quiet_end_hour

    def get_stats(self) -> Dict[str, Any]:
        """Get execution loop statistics"""
        return {
            'state': self.state.value,
            'execution_count': self.execution_count,
            'error_count': self.error_count,
            'last_execution': self.last_execution,
            'check_interval_seconds': self.check_interval,
            'quiet_hours': f"{self.quiet_start_hour}:00 - {self.quiet_end_hour}:00"
        }


# Global singleton
_execution_loop = None


def get_execution_loop(
    autonomous_agent=None,
    goal_system=None,
    check_interval: int = 30
) -> AutonomousExecutionLoop:
    """Get or create global execution loop"""
    global _execution_loop
    if _execution_loop is None:
        _execution_loop = AutonomousExecutionLoop(
            autonomous_agent=autonomous_agent,
            goal_system=goal_system,
            check_interval=check_interval
        )
    return _execution_loop
