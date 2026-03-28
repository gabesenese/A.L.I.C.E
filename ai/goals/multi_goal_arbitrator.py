"""
Multi-Goal Arbitrator - Tier 4: Mission-Aligned Execution

Tracks 3-5 active missions with priority/deadline.
Prevents goal confusion and ensures focused execution.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class GoalPriority(Enum):
    """Priority levels for goals."""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    DEFER = 1


@dataclass
class ActiveGoal:
    """Represents a goal being tracked."""
    
    goal_id: str
    description: str
    priority: GoalPriority
    created_at: str
    deadline: Optional[str] = None  # ISO format if set
    status: str = "active"  # active, in_progress, paused, completed, abandoned
    progress_percent: float = 0.0
    steps_completed: int = 0
    steps_total: int = 1
    related_intents: List[str] = None
    assigned_resources: List[str] = None
    blocking_issues: List[str] = None
    notes: str = ""
    last_touched: str = None
    
    def __post_init__(self):
        if self.related_intents is None:
            self.related_intents = []
        if self.assigned_resources is None:
            self.assigned_resources = []
        if self.blocking_issues is None:
            self.blocking_issues = []
        if not self.last_touched:
            self.last_touched = self.created_at
    
    def is_urgent(self) -> bool:
        """Check if goal is urgent."""
        if self.priority in (GoalPriority.CRITICAL, GoalPriority.HIGH):
            return True
        
        if self.deadline:
            deadline = datetime.fromisoformat(self.deadline)
            return datetime.now() > deadline - timedelta(hours=1)
        
        return False
    
    def is_overdue(self) -> bool:
        """Check if goal is overdue."""
        if not self.deadline:
            return False
        
        deadline = datetime.fromisoformat(self.deadline)
        return datetime.now() > deadline and self.status != "completed"
    
    def progress(self) -> float:
        """Calculate progress as percentage."""
        if self.steps_total > 0:
            return (self.steps_completed / self.steps_total) * 100.0
        return self.progress_percent


class MultiGoalArbitrator:
    """Manages multiple active goals with priority arbitration."""
    
    def __init__(self, max_active_goals: int = 5):
        """
        Args:
            max_active_goals: Maximum concurrent goals to track
        """
        self.max_active_goals = max_active_goals
        self.goals: Dict[str, ActiveGoal] = {}
        self.goal_counter = 0
        self.execution_history: List[Dict[str, Any]] = []
        self.current_focus: Optional[str] = None
    
    def add_goal(
        self,
        description: str,
        priority: GoalPriority = GoalPriority.MEDIUM,
        deadline: Optional[str] = None,
        steps_total: int = 1,
        related_intents: List[str] = None,
    ) -> ActiveGoal:
        """
        Add a new goal to track.
        
        Args:
            description: What the user is trying to achieve
            priority: How important this is
            deadline: Optional deadline (ISO format)
            steps_total: How many steps to complete
            related_intents: Intents typically used for this goal
        
        Returns:
            The created goal
        """
        # Check capacity
        active_count = sum(1 for g in self.goals.values() if g.status == "active")
        if active_count >= self.max_active_goals:
            logger.warning(f"[MultiGoal] At capacity ({self.max_active_goals} goals). Pausing lowest priority.")
            self._pause_lowest_priority_goal()
        
        self.goal_counter += 1
        goal_id = f"goal_{self.goal_counter}"
        
        goal = ActiveGoal(
            goal_id=goal_id,
            description=description,
            priority=priority,
            created_at=datetime.now().isoformat(),
            deadline=deadline,
            steps_total=steps_total,
            related_intents=related_intents or [],
        )
        
        self.goals[goal_id] = goal
        logger.info(f"[MultiGoal] Added goal '{description}' (priority={priority.name}, deadline={deadline})")
        
        return goal
    
    def focus_on_goal(self, goal_id: str) -> ActiveGoal:
        """Explicitly focus on a specific goal."""
        if goal_id not in self.goals:
            logger.warning(f"[MultiGoal] Goal not found: {goal_id}")
            return None
        
        goal = self.goals[goal_id]
        self.current_focus = goal_id
        goal.status = "in_progress"
        goal.last_touched = datetime.now().isoformat()
        
        logger.info(f"[MultiGoal] Focus switched to: {goal.description}")
        
        return goal
    
    def auto_select_next_focus(self) -> ActiveGoal:
        """Automatically select the next goal to focus on."""
        active_goals = [g for g in self.goals.values() if g.status == "active"]
        
        if not active_goals:
            logger.info("[MultiGoal] No active goals")
            return None
        
        # Sort by: urgency > overdue > priority > oldest
        def sort_key(g):
            overdue_penalty = -1000 if g.is_overdue() else 0
            urgent_penalty = -500 if g.is_urgent() else 0
            priority_val = -g.priority.value * 100
            age = (datetime.now() - datetime.fromisoformat(g.created_at)).total_seconds()
            
            return (overdue_penalty + urgent_penalty + priority_val, age)
        
        next_goal = sorted(active_goals, key=sort_key)[0]
        return self.focus_on_goal(next_goal.goal_id)
    
    def update_progress(
        self,
        goal_id: str,
        steps_completed: int = None,
        progress_percent: float = None,
    ) -> None:
        """Update progress on a goal."""
        if goal_id not in self.goals:
            logger.warning(f"[MultiGoal] Goal not found: {goal_id}")
            return
        
        goal = self.goals[goal_id]
        
        if steps_completed is not None:
            goal.steps_completed = min(steps_completed, goal.steps_total)
            goal.progress_percent = goal.progress()
        
        if progress_percent is not None:
            goal.progress_percent = min(progress_percent, 100.0)
        
        goal.last_touched = datetime.now().isoformat()
        
        logger.info(f"[MultiGoal] Progress on '{goal.description}': {goal.progress_percent:.0f}%")
    
    def complete_goal(self, goal_id: str, notes: str = "") -> None:
        """Mark a goal as completed."""
        if goal_id not in self.goals:
            return
        
        goal = self.goals[goal_id]
        goal.status = "completed"
        goal.progress_percent = 100.0
        goal.notes = notes
        goal.last_touched = datetime.now().isoformat()
        
        logger.info(f"[MultiGoal] Goal completed: {goal.description}")
        
        # Auto-focus on next goal
        self.auto_select_next_focus()
    
    def pause_goal(self, goal_id: str, reason: str = "") -> None:
        """Pause a goal (but keep tracking it)."""
        if goal_id not in self.goals:
            return
        
        goal = self.goals[goal_id]
        goal.status = "paused"
        goal.notes = reason
        
        logger.info(f"[MultiGoal] Goal paused: {goal.description}")
    
    def resume_goal(self, goal_id: str) -> None:
        """Resume a paused goal."""
        if goal_id not in self.goals:
            return
        
        goal = self.goals[goal_id]
        goal.status = "active"
        
        logger.info(f"[MultiGoal] Goal resumed: {goal.description}")
    
    def abandon_goal(self, goal_id: str, reason: str = "") -> None:
        """Abandon a goal."""
        if goal_id not in self.goals:
            return
        
        goal = self.goals[goal_id]
        goal.status = "abandoned"
        goal.notes = reason
        
        logger.info(f"[MultiGoal] Goal abandoned: {goal.description}")
    
    def _pause_lowest_priority_goal(self) -> None:
        """When at capacity, pause the lowest priority active goal."""
        active_goals = [g for g in self.goals.values() if g.status == "active" and not g.is_overdue()]
        
        if not active_goals:
            return
        
        lowest = min(active_goals, key=lambda g: (g.priority.value, datetime.fromisoformat(g.created_at)))
        self.pause_goal(lowest.goal_id, "Paused to make room for higher priority goal")
    
    def resolve_goal_conflict(self, intent: str) -> Optional[ActiveGoal]:
        """
        When user input doesn't clearly map to a goal, try to resolve which one they mean.
        
        Returns:
            The most likely goal, or None if ambiguous
        """
        # Find goals that could be relevant
        candidates = []
        
        for goal in self.goals.values():
            if goal.status != "active":
                continue
            
            # Check if intent matches related_intents
            if any(rel in intent for rel in goal.related_intents):
                candidates.append((goal, 0.9))  # Strong match
            
            # Check if description keywords appear in intent
            description_words = set(goal.description.lower().split())
            intent_words = set(intent.lower().split())
            overlap = len(description_words & intent_words)
            if overlap >= 2:
                candidates.append((goal, 0.7))  # Weak match
        
        if not candidates:
            return None
        
        # Return highest scored
        return sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]
    
    def get_active_goals(self) -> List[ActiveGoal]:
        """Get all active goals."""
        return [g for g in self.goals.values() if g.status == "active"]
    
    def get_goal_status(self, goal_id: str) -> Dict[str, Any]:
        """Get detailed status of a goal."""
        if goal_id not in self.goals:
            return {}
        
        goal = self.goals[goal_id]
        
        return {
            "goal_id": goal.goal_id,
            "description": goal.description,
            "status": goal.status,
            "priority": goal.priority.name,
            "progress": goal.progress(),
            "is_urgent": goal.is_urgent(),
            "is_overdue": goal.is_overdue(),
            "deadline": goal.deadline,
            "steps": f"{goal.steps_completed}/{goal.steps_total}",
            "blocking_issues": goal.blocking_issues,
            "notes": goal.notes,
        }
    
    def get_goals_summary(self) -> str:
        """Get human-readable summary of all goals."""
        active = self.get_active_goals()
        
        if not active:
            return "No active goals."
        
        lines = ["Active Goals:"]
        
        for goal in sorted(active, key=lambda g: g.priority.value, reverse=True):
            urgent = " ⚠ URGENT" if goal.is_urgent() else ""
            overdue = " ⛔ OVERDUE" if goal.is_overdue() else ""
            progress_bar = "█" * int(goal.progress() / 10) + "░" * (10 - int(goal.progress() / 10))
            
            lines.append(
                f"  • [{progress_bar}] {goal.description} "
                f"({goal.priority.name}){urgent}{overdue}"
            )
        
        return "\n".join(lines)
    
    def record_action_for_goal(self, goal_id: str, action: str, result: str) -> None:
        """Record an action taken toward a goal."""
        if goal_id not in self.goals:
            return
        
        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "goal_id": goal_id,
            "action": action,
            "result": result,
        })
