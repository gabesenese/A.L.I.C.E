"""
Planning Facade for A.L.I.C.E
Goals, tasks, and autonomous execution
"""

from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

try:
    from ai.planning.goal_tracker import GoalTracker
except ImportError:
    GoalTracker = None

try:
    from ai.planning.task_executor import TaskExecutor
except ImportError:
    TaskExecutor = None


class PlanningFacade:
    """Facade for goal planning and task execution"""

    def __init__(self, safe_mode: bool = True) -> None:
        # Goal tracker
        try:
            self.goal_manager = GoalTracker() if GoalTracker else None
        except Exception as e:
            logger.warning(f"Goal tracker not available: {e}")
            self.goal_manager = None

        # Task executor
        try:
            self.executor = TaskExecutor(safe_mode=safe_mode) if TaskExecutor else None
        except Exception as e:
            logger.warning(f"Task executor not available: {e}")
            self.executor = None

        # Goal decomposer (not yet implemented)
        self.decomposer = None

        logger.info("[PlanningFacade] Initialized planning systems")

    def detect_goal(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Detect if user input contains a goal

        Args:
            user_input: User's message

        Returns:
            Goal structure or None
        """
        if not self.goal_manager:
            return None

        try:
            return self.goal_manager.detect_goal(user_input)
        except Exception as e:
            logger.error(f"Failed to detect goal: {e}")
            return None

    def create_goal(
        self,
        description: str,
        user_input: str
    ) -> Optional[str]:
        """
        Create and store a new goal

        Args:
            description: Goal description
            user_input: Original user input

        Returns:
            Goal ID or None
        """
        if not self.goal_manager:
            return None

        try:
            return self.goal_manager.create_goal(description, user_input)
        except Exception as e:
            logger.error(f"Failed to create goal: {e}")
            return None

    def decompose_goal(
        self,
        goal_description: str
    ) -> List[Dict[str, Any]]:
        """
        Break down goal into actionable steps

        Args:
            goal_description: Description of the goal

        Returns:
            List of steps
        """
        if not self.decomposer:
            return []

        try:
            return self.decomposer.decompose(goal_description)
        except Exception as e:
            logger.error(f"Failed to decompose goal: {e}")
            return []

    def execute_task(
        self,
        task_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a task with given parameters

        Args:
            task_type: Type of task to execute
            parameters: Task parameters

        Returns:
            Execution result
        """
        if not self.executor:
            return {
                "success": False,
                "error": "Task executor not available"
            }

        try:
            # Route to appropriate executor method
            if task_type == "open_application":
                app_path = parameters.get('app_path', '')
                result = self.executor.open_application(app_path)
            elif task_type == "run_command":
                command = parameters.get('command', '')
                result = self.executor.run_command(command)
            else:
                return {
                    "success": False,
                    "error": f"Unknown task type: {task_type}"
                }

            return {
                "success": result.success,
                "output": result.output,
                "error": result.error
            }
        except Exception as e:
            logger.error(f"Failed to execute task: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_active_goals(self) -> List[Dict[str, Any]]:
        """
        Get all active goals

        Returns:
            List of active goals
        """
        if not self.goal_manager:
            return []

        try:
            return self.goal_manager.get_active_goals()
        except Exception as e:
            logger.error(f"Failed to get active goals: {e}")
            return []


# Singleton instance
_planning_facade: Optional[PlanningFacade] = None


def get_planning_facade(safe_mode: bool = True) -> PlanningFacade:
    """Get or create the PlanningFacade singleton"""
    global _planning_facade
    if _planning_facade is None:
        _planning_facade = PlanningFacade(safe_mode=safe_mode)
    return _planning_facade
