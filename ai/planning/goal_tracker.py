"""
Goal Tracker: Manages active goals, subgoals, and learns which tool sequences succeed/fail.

Implements dynamic goal management in the reasoning engine to track multi-step tasks
and learn patterns about what sequences of tool calls typically succeed.
"""

import json
import os
from typing import Dict, List, Any, Optional, Set
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, asdict, field
import logging

logger = logging.getLogger(__name__)


class GoalStatus(Enum):
    """Status of a goal"""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    ABANDONED = "abandoned"


class ToolCallStatus(Enum):
    """Status of a tool call within a goal"""
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ToolCall:
    """Record of a single tool call"""
    tool_name: str
    action: str  # e.g., "email_read", "file_write"
    arguments: Dict[str, Any] = field(default_factory=dict)
    status: ToolCallStatus = ToolCallStatus.PENDING
    timestamp: str = None
    duration_ms: float = 0
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self):
        return {
            **asdict(self),
            'status': self.status.value if isinstance(self.status, ToolCallStatus) else self.status
        }


@dataclass
class SubgoalRecord:
    """Record of a subgoal within a larger goal"""
    id: str
    name: str
    parent_goal_id: Optional[str] = None
    status: GoalStatus = GoalStatus.ACTIVE
    tool_calls: List[ToolCall] = field(default_factory=list)
    created_at: str = None
    completed_at: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def add_tool_call(self, tool_call: ToolCall):
        """Add a tool call to this subgoal"""
        self.tool_calls.append(tool_call)
    
    def mark_completed(self):
        """Mark subgoal as completed"""
        self.status = GoalStatus.COMPLETED
        self.completed_at = datetime.now().isoformat()
    
    def mark_failed(self):
        """Mark subgoal as failed"""
        self.status = GoalStatus.FAILED
        self.completed_at = datetime.now().isoformat()


@dataclass
class GoalRecord:
    """Record of an active goal"""
    id: str
    description: str
    intent: str
    status: GoalStatus = GoalStatus.ACTIVE
    subgoals: List[SubgoalRecord] = field(default_factory=list)
    tool_call_sequence: List[ToolCall] = field(default_factory=list)
    created_at: str = None
    completed_at: Optional[str] = None
    success: bool = False
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
    
    def add_tool_call(self, tool_call: ToolCall):
        """Record a tool call for this goal"""
        self.tool_call_sequence.append(tool_call)
    
    def add_subgoal(self, subgoal: SubgoalRecord):
        """Add a subgoal"""
        self.subgoals.append(subgoal)
    
    def mark_completed(self):
        """Mark goal as completed"""
        self.status = GoalStatus.COMPLETED
        self.completed_at = datetime.now().isoformat()
        self.success = True
    
    def mark_failed(self):
        """Mark goal as failed"""
        self.status = GoalStatus.FAILED
        self.completed_at = datetime.now().isoformat()
        self.success = False
    
    def get_tool_sequence(self) -> List[str]:
        """Get sequence of tool calls as (tool, action) tuples"""
        return [(tc.tool_name, tc.action) for tc in self.tool_call_sequence]
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'description': self.description,
            'intent': self.intent,
            'status': self.status.value if isinstance(self.status, GoalStatus) else self.status,
            'tool_sequence': self.get_tool_sequence(),
            'success': self.success,
            'subgoal_count': len(self.subgoals),
            'tool_call_count': len(self.tool_call_sequence),
            'created_at': self.created_at,
            'completed_at': self.completed_at
        }


class GoalTracker:
    """
    Tracks active goals, subgoals, and tool call sequences.
    Learns which sequences typically succeed or fail.
    """
    
    def __init__(self, history_file: str = "data/reasoning/goal_history.jsonl"):
        """Initialize goal tracker"""
        self.history_file = history_file
        self.active_goals: Dict[str, GoalRecord] = {}
        self.goal_counter = 0
        self.tool_sequence_stats: Dict[str, Dict[str, Any]] = {}  # Sequence → success rate
        
        os.makedirs(os.path.dirname(history_file) or '.', exist_ok=True)
        self._load_history()
    
    def create_goal(self, description: str, intent: str) -> str:
        """
        Create a new goal.
        
        Args:
            description: Goal description
            intent: Detected intent for this goal
            
        Returns:
            Goal ID
        """
        self.goal_counter += 1
        goal_id = f"goal_{self.goal_counter}_{datetime.now().timestamp()}"
        
        goal = GoalRecord(
            id=goal_id,
            description=description,
            intent=intent
        )
        
        self.active_goals[goal_id] = goal
        logger.info(f"[GoalTracker] Created goal {goal_id}: {description}")
        
        return goal_id
    
    def add_subgoal(self, goal_id: str, name: str) -> Optional[str]:
        """
        Add a subgoal to an active goal.
        
        Args:
            goal_id: Parent goal ID
            name: Subgoal name
            
        Returns:
            Subgoal ID or None if goal not found
        """
        if goal_id not in self.active_goals:
            logger.warning(f"Goal {goal_id} not found")
            return None
        
        goal = self.active_goals[goal_id]
        subgoal_id = f"subgoal_{len(goal.subgoals)}_{goal_id}"
        
        subgoal = SubgoalRecord(
            id=subgoal_id,
            name=name,
            parent_goal_id=goal_id
        )
        
        goal.add_subgoal(subgoal)
        logger.info(f"[GoalTracker] Added subgoal to {goal_id}: {name}")
        
        return subgoal_id
    
    def record_tool_call(
        self,
        goal_id: str,
        tool_name: str,
        action: str,
        arguments: Dict[str, Any] = None,
        status: ToolCallStatus = ToolCallStatus.SUCCESS,
        duration_ms: float = 0,
        error: Optional[str] = None
    ):
        """
        Record a tool call within a goal.
        
        Args:
            goal_id: Goal ID
            tool_name: Tool name
            action: Action performed
            arguments: Tool arguments
            status: Result status
            duration_ms: Execution time in milliseconds
            error: Error message if failed
        """
        if goal_id not in self.active_goals:
            logger.warning(f"Goal {goal_id} not found")
            return
        
        goal = self.active_goals[goal_id]
        tool_call = ToolCall(
            tool_name=tool_name,
            action=action,
            arguments=arguments or {},
            status=status,
            duration_ms=duration_ms,
            error=error
        )
        
        goal.add_tool_call(tool_call)
        
        status_str = "" if status == ToolCallStatus.SUCCESS else ""
        logger.info(f"[GoalTracker] Tool call {status_str}: {tool_name}.{action} (in goal {goal_id})")
    
    def complete_goal(self, goal_id: str, success: bool):
        """
        Mark a goal as complete.
        
        Args:
            goal_id: Goal ID
            success: Whether goal succeeded
        """
        if goal_id not in self.active_goals:
            logger.warning(f"Goal {goal_id} not found")
            return
        
        goal = self.active_goals[goal_id]
        
        if success:
            goal.mark_completed()
            logger.info(f"[GoalTracker] Goal {goal_id} COMPLETED")
        else:
            goal.mark_failed()
            logger.info(f"[GoalTracker] Goal {goal_id} FAILED")
        
        # Learn from this goal
        self._learn_from_goal(goal)
        
        # Save goal
        self._save_goal(goal)
        
        # Remove from active goals
        del self.active_goals[goal_id]
    
    def _learn_from_goal(self, goal: GoalRecord):
        """
        Learn from a completed goal.
        Track which tool sequences succeed/fail.
        """
        sequence = tuple(goal.get_tool_sequence())
        
        if sequence not in self.tool_sequence_stats:
            self.tool_sequence_stats[sequence] = {
                'count': 0,
                'successes': 0,
                'failures': 0,
                'success_rate': 0.0
            }
        
        stats = self.tool_sequence_stats[sequence]
        stats['count'] += 1
        
        if goal.success:
            stats['successes'] += 1
        else:
            stats['failures'] += 1
        
        stats['success_rate'] = stats['successes'] / stats['count']
        
        logger.info(f"[GoalTracker] Learned sequence {sequence}: "
                   f"success_rate={stats['success_rate']:.1%} ({stats['successes']}/{stats['count']})")
    
    def _save_goal(self, goal: GoalRecord):
        """Save completed goal to history file"""
        with open(self.history_file, 'a') as f:
            f.write(json.dumps(goal.to_dict()) + '\n')
    
    def _load_history(self):
        """Load goal history from file"""
        if not os.path.exists(self.history_file):
            return
        
        with open(self.history_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        sequence = tuple(record['tool_sequence'])
                        if 'success_rate' in record or record.get('success'):
                            self.tool_sequence_stats[sequence] = {
                                'count': 1,
                                'successes': 1 if record.get('success') else 0,
                                'failures': 0 if record.get('success') else 1,
                                'success_rate': float(record.get('success_rate', 1.0 if record.get('success') else 0.0))
                            }
                    except Exception as e:
                        logger.warning(f"Failed to load history record: {e}")
    
    def get_sequence_stats(self, sequence: tuple) -> Optional[Dict[str, Any]]:
        """
        Get success statistics for a tool sequence.
        
        Args:
            sequence: Tuple of (tool, action) pairs
            
        Returns:
            Statistics dict or None if sequence not seen
        """
        return self.tool_sequence_stats.get(sequence)
    
    def get_top_sequences(self, k: int = 10) -> List[tuple]:
        """
        Get top K most successful tool sequences.
        
        Args:
            k: Number of top sequences to return
            
        Returns:
            List of top sequences by success rate
        """
        sorted_sequences = sorted(
            self.tool_sequence_stats.items(),
            key=lambda x: (x[1]['success_rate'], x[1]['count']),
            reverse=True
        )
        return [seq for seq, _ in sorted_sequences[:k]]
    
    def recommend_sequence(self, intent: str) -> Optional[List[tuple]]:
        """
        Recommend best tool sequences for an intent.
        
        Args:
            intent: Intent for the goal
            
        Returns:
            List of recommended sequences or None
        """
        # Find goals with this intent that succeeded
        successful_sequences = [
            seq for seq, stats in self.tool_sequence_stats.items()
            if stats['success_rate'] > 0.7  # High success rate
        ]
        
        if successful_sequences:
            logger.info(f"[GoalTracker] Recommending {len(successful_sequences)} sequences for {intent}")
            return successful_sequences[:3]  # Top 3 recommendations
        
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about tracked goals and sequences"""
        total_sequences = len(self.tool_sequence_stats)
        successful_sequences = sum(1 for s in self.tool_sequence_stats.values() if s['success_rate'] > 0.5)
        avg_success_rate = (
            sum(s['success_rate'] for s in self.tool_sequence_stats.values()) / total_sequences
            if total_sequences > 0 else 0
        )
        
        return {
            'active_goals': len(self.active_goals),
            'total_sequences_learned': total_sequences,
            'successful_sequences': successful_sequences,
            'average_success_rate': avg_success_rate,
            'top_sequences': [
                {
                    'sequence': seq,
                    'success_rate': stats['success_rate'],
                    'count': stats['count']
                }
                for seq, stats in sorted(
                    self.tool_sequence_stats.items(),
                    key=lambda x: (x[1]['success_rate'], x[1]['count']),
                    reverse=True
                )[:5]
            ]
        }
