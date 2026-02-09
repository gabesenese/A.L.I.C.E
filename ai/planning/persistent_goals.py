"""
Persistent Goal System
=======================
Long-running goal management with multi-session persistence,
progress tracking, and autonomous execution.

Enables Alice to work on complex goals over days/weeks like Jarvis.
"""

import json
import time
import uuid
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class GoalStatus(Enum):
    """Goal execution status"""
    CREATED = "created"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GoalPriority(Enum):
    """Goal priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class GoalStep:
    """A single step in achieving a goal"""
    step_id: str
    description: str
    status: str = "pending"
    dependencies: List[str] = field(default_factory=list)
    result: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None


@dataclass
class Goal:
    """A long-running goal"""
    goal_id: str
    title: str
    description: str
    status: GoalStatus = GoalStatus.CREATED
    priority: GoalPriority = GoalPriority.NORMAL

    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    deadline: Optional[float] = None

    steps: List[GoalStep] = field(default_factory=list)
    current_step: Optional[str] = None

    progress: float = 0.0
    success_criteria: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)

    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_step(self, description: str, dependencies: List[str] = None) -> GoalStep:
        """Add a step to this goal"""
        step = GoalStep(
            step_id=str(uuid.uuid4()),
            description=description,
            dependencies=dependencies or []
        )
        self.steps.append(step)
        return step

    def get_step(self, step_id: str) -> Optional[GoalStep]:
        """Get a step by ID"""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def complete_step(self, step_id: str, result: str = None):
        """Mark a step as completed"""
        step = self.get_step(step_id)
        if step:
            step.status = "completed"
            step.completed_at = time.time()
            step.result = result
            self.updated_at = time.time()
            self._update_progress()

    def fail_step(self, step_id: str, error: str):
        """Mark a step as failed"""
        step = self.get_step(step_id)
        if step:
            step.status = "failed"
            step.error = error
            self.updated_at = time.time()

    def get_next_step(self) -> Optional[GoalStep]:
        """Get the next executable step"""
        completed_steps = {s.step_id for s in self.steps if s.status == "completed"}

        for step in self.steps:
            if step.status == "pending":
                # Check if all dependencies are met
                if all(dep in completed_steps for dep in step.dependencies):
                    return step

        return None

    def _update_progress(self):
        """Update goal progress based on completed steps"""
        if not self.steps:
            self.progress = 0.0
            return

        completed = sum(1 for s in self.steps if s.status == "completed")
        self.progress = completed / len(self.steps)


class PersistentGoalSystem:
    """
    Manages long-running goals that persist across sessions.

    Features:
    - Multi-session goal persistence
    - Progress tracking and visualization
    - Dependency management between steps
    - Automatic resumption on startup
    - Blocker detection and handling
    - Success criteria validation
    """

    def __init__(self, storage_path: str = "data/goals"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Active goals in memory
        self.goals: Dict[str, Goal] = {}

        # Load existing goals
        self._load_goals()

    def _load_goals(self):
        """Load all goals from storage"""
        goals_file = self.storage_path / "active_goals.json"

        if goals_file.exists():
            try:
                with open(goals_file, 'r') as f:
                    data = json.load(f)

                for goal_data in data.get('goals', []):
                    goal = self._deserialize_goal(goal_data)
                    self.goals[goal.goal_id] = goal

                logger.info(f"Loaded {len(self.goals)} goals from storage")

            except Exception as e:
                logger.error(f"Error loading goals: {e}")

    def _save_goals(self):
        """Save all goals to storage"""
        goals_file = self.storage_path / "active_goals.json"

        try:
            data = {
                'goals': [self._serialize_goal(goal) for goal in self.goals.values()],
                'last_updated': time.time()
            }

            with open(goals_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving goals: {e}")

    def _serialize_goal(self, goal: Goal) -> Dict[str, Any]:
        """Convert goal to JSON-serializable dict"""
        return {
            'goal_id': goal.goal_id,
            'title': goal.title,
            'description': goal.description,
            'status': goal.status.value,
            'priority': goal.priority.value,
            'created_at': goal.created_at,
            'updated_at': goal.updated_at,
            'started_at': goal.started_at,
            'completed_at': goal.completed_at,
            'deadline': goal.deadline,
            'steps': [asdict(step) for step in goal.steps],
            'current_step': goal.current_step,
            'progress': goal.progress,
            'success_criteria': goal.success_criteria,
            'blockers': goal.blockers,
            'context': goal.context,
            'metadata': goal.metadata
        }

    def _deserialize_goal(self, data: Dict[str, Any]) -> Goal:
        """Convert dict to Goal object"""
        goal = Goal(
            goal_id=data['goal_id'],
            title=data['title'],
            description=data['description'],
            status=GoalStatus(data['status']),
            priority=GoalPriority(data['priority']),
            created_at=data['created_at'],
            updated_at=data['updated_at'],
            started_at=data.get('started_at'),
            completed_at=data.get('completed_at'),
            deadline=data.get('deadline'),
            current_step=data.get('current_step'),
            progress=data.get('progress', 0.0),
            success_criteria=data.get('success_criteria', []),
            blockers=data.get('blockers', []),
            context=data.get('context', {}),
            metadata=data.get('metadata', {})
        )

        # Deserialize steps
        for step_data in data.get('steps', []):
            step = GoalStep(**step_data)
            goal.steps.append(step)

        return goal

    def create_goal(
        self,
        title: str,
        description: str,
        priority: GoalPriority = GoalPriority.NORMAL,
        deadline: Optional[datetime] = None,
        success_criteria: List[str] = None
    ) -> Goal:
        """
        Create a new goal.

        Args:
            title: Short goal title
            description: Detailed description
            priority: Goal priority
            deadline: Optional deadline
            success_criteria: List of success criteria

        Returns:
            Created Goal object
        """
        goal = Goal(
            goal_id=str(uuid.uuid4()),
            title=title,
            description=description,
            priority=priority,
            deadline=deadline.timestamp() if deadline else None,
            success_criteria=success_criteria or []
        )

        self.goals[goal.goal_id] = goal
        self._save_goals()

        logger.info(f"Created goal: {title} ({goal.goal_id})")
        return goal

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID"""
        return self.goals.get(goal_id)

    def get_active_goals(self) -> List[Goal]:
        """Get all active (not completed/cancelled) goals"""
        return [
            goal for goal in self.goals.values()
            if goal.status not in [GoalStatus.COMPLETED, GoalStatus.CANCELLED, GoalStatus.FAILED]
        ]

    def get_goals_by_priority(self, priority: GoalPriority) -> List[Goal]:
        """Get goals by priority level"""
        return [
            goal for goal in self.goals.values()
            if goal.priority == priority
        ]

    def update_goal_status(self, goal_id: str, status: GoalStatus):
        """Update goal status"""
        goal = self.get_goal(goal_id)
        if goal:
            goal.status = status
            goal.updated_at = time.time()

            if status == GoalStatus.IN_PROGRESS and not goal.started_at:
                goal.started_at = time.time()
            elif status == GoalStatus.COMPLETED:
                goal.completed_at = time.time()
                goal.progress = 1.0

            self._save_goals()

    def add_blocker(self, goal_id: str, blocker: str):
        """Add a blocker to a goal"""
        goal = self.get_goal(goal_id)
        if goal:
            goal.blockers.append(blocker)
            if goal.status == GoalStatus.IN_PROGRESS:
                goal.status = GoalStatus.BLOCKED
            goal.updated_at = time.time()
            self._save_goals()

    def remove_blocker(self, goal_id: str, blocker: str):
        """Remove a blocker from a goal"""
        goal = self.get_goal(goal_id)
        if goal and blocker in goal.blockers:
            goal.blockers.remove(blocker)

            # Resume if no more blockers
            if not goal.blockers and goal.status == GoalStatus.BLOCKED:
                goal.status = GoalStatus.IN_PROGRESS

            goal.updated_at = time.time()
            self._save_goals()

    def plan_goal(self, goal_id: str, steps: List[str]) -> bool:
        """
        Add execution steps to a goal.

        Args:
            goal_id: Goal ID
            steps: List of step descriptions

        Returns:
            True if successful
        """
        goal = self.get_goal(goal_id)
        if not goal:
            return False

        goal.status = GoalStatus.PLANNING

        for i, step_desc in enumerate(steps):
            # Create dependencies based on sequential order
            dependencies = [goal.steps[i-1].step_id] if i > 0 else []
            goal.add_step(step_desc, dependencies)

        goal.status = GoalStatus.IN_PROGRESS
        goal.updated_at = time.time()
        self._save_goals()

        logger.info(f"Planned {len(steps)} steps for goal: {goal.title}")
        return True

    def execute_next_step(self, goal_id: str) -> Optional[GoalStep]:
        """
        Get the next executable step for a goal.

        Args:
            goal_id: Goal ID

        Returns:
            Next step to execute, or None
        """
        goal = self.get_goal(goal_id)
        if not goal:
            return None

        next_step = goal.get_next_step()
        if next_step:
            next_step.status = "in_progress"
            next_step.started_at = time.time()
            goal.current_step = next_step.step_id
            goal.updated_at = time.time()
            self._save_goals()

        return next_step

    def complete_current_step(self, goal_id: str, result: str = None):
        """Mark current step as completed"""
        goal = self.get_goal(goal_id)
        if not goal or not goal.current_step:
            return

        goal.complete_step(goal.current_step, result)

        # Check if goal is complete
        if all(s.status == "completed" for s in goal.steps):
            self.update_goal_status(goal_id, GoalStatus.COMPLETED)

        self._save_goals()

    def get_goal_summary(self, goal_id: str) -> Dict[str, Any]:
        """Get a summary of goal progress"""
        goal = self.get_goal(goal_id)
        if not goal:
            return {}

        completed_steps = sum(1 for s in goal.steps if s.status == "completed")
        total_steps = len(goal.steps)

        time_elapsed = None
        if goal.started_at:
            time_elapsed = time.time() - goal.started_at

        time_remaining = None
        if goal.deadline:
            time_remaining = goal.deadline - time.time()

        return {
            'goal_id': goal.goal_id,
            'title': goal.title,
            'status': goal.status.value,
            'priority': goal.priority.value,
            'progress': goal.progress,
            'completed_steps': completed_steps,
            'total_steps': total_steps,
            'blockers': goal.blockers,
            'time_elapsed_hours': time_elapsed / 3600 if time_elapsed else None,
            'time_remaining_hours': time_remaining / 3600 if time_remaining else None,
            'is_overdue': goal.deadline and time.time() > goal.deadline
        }

    def get_all_goals_summary(self) -> Dict[str, Any]:
        """Get summary of all goals"""
        active = self.get_active_goals()

        return {
            'total_goals': len(self.goals),
            'active_goals': len(active),
            'completed_goals': len([g for g in self.goals.values() if g.status == GoalStatus.COMPLETED]),
            'blocked_goals': len([g for g in self.goals.values() if g.status == GoalStatus.BLOCKED]),
            'critical_goals': len(self.get_goals_by_priority(GoalPriority.CRITICAL)),
            'goals': [self.get_goal_summary(g.goal_id) for g in active]
        }

    def resume_on_startup(self) -> List[Goal]:
        """
        Get goals to resume on startup.

        Returns:
            List of goals that were in progress
        """
        in_progress = [
            goal for goal in self.goals.values()
            if goal.status in [GoalStatus.IN_PROGRESS, GoalStatus.PAUSED]
        ]

        if in_progress:
            logger.info(f"Resuming {len(in_progress)} goals from previous session")

        return in_progress


# Global singleton
_goal_system = None


def get_goal_system(storage_path: str = "data/goals") -> PersistentGoalSystem:
    """Get or create global goal system"""
    global _goal_system
    if _goal_system is None:
        _goal_system = PersistentGoalSystem(storage_path=storage_path)
    return _goal_system
