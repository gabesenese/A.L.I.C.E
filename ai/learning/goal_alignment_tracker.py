"""
Goal Alignment Tracker - Tier 1: User Goal Alignment Signal

Tracks whether ALICE's responses actually satisfied the user.
Collects implicit and explicit feedback to improve routing/quality metrics.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class FeedbackSignal(Enum):
    """Types of user feedback signals."""
    EXPLICIT_YES = "explicit_yes"  # "That helped"
    EXPLICIT_NO = "explicit_no"  # "That didn't work"
    IMPLICIT_REPEAT = "implicit_repeat"  # User asks same thing again = implicit no
    IMPLICIT_FOLLOWUP = "implicit_followup"  # User asks related thing = implicit yes
    IMPLICIT_ABANDON = "implicit_abandon"  # User abandons line of questioning = implicit no
    NEUTRAL = "neutral"  # No clear signal


@dataclass
class AlignmentEntry:
    """Record of a single interaction's goal alignment."""
    
    turn_number: int
    timestamp: str
    user_input: str
    alice_response: str
    intent: str
    goal: str  # What the user is trying to accomplish
    response_type: str
    # Feedback signals
    feedback_signal: FeedbackSignal = FeedbackSignal.NEUTRAL
    feedback_confidence: float = 0.5  # How confident we are in the signal
    was_helpful: bool = False
    user_satisfaction: Optional[float] = None  # 0.0-1.0 if known
    # Diagnostics
    routing_path: str = ""  # Which path Alice took
    tool_used: Optional[str] = None
    success: bool = True
    # Context for learning
    context_tags: List[str] = field(default_factory=list)


class GoalAlignmentTracker:
    """Tracks success of responses in achieving user goals."""
    
    def __init__(self):
        self.entries: List[AlignmentEntry] = []
        self.turn_count = 0
        self.last_user_input = ""
        self.goal_history: Dict[str, int] = {}  # goal -> count
        self.alignment_stats = {
            "total_interactions": 0,
            "helpful_count": 0,
            "feedback_received": 0,
            "implicit_signal_count": 0,
            "explicit_signal_count": 0,
        }
    
    def record_interaction(
        self,
        user_input: str,
        alice_response: str,
        intent: str,
        goal: str,
        response_type: str,
        routing_path: str = "",
        tool_used: Optional[str] = None,
        success: bool = True,
        context_tags: List[str] = None,
    ) -> AlignmentEntry:
        """
        Record an interaction for later feedback analysis.
        """
        self.turn_count += 1
        context_tags = context_tags or []
        
        entry = AlignmentEntry(
            turn_number=self.turn_count,
            timestamp=datetime.now().isoformat(),
            user_input=user_input,
            alice_response=alice_response,
            intent=intent,
            goal=goal,
            response_type=response_type,
            routing_path=routing_path,
            tool_used=tool_used,
            success=success,
            context_tags=context_tags,
        )
        
        self.entries.append(entry)
        self.goal_history[goal] = self.goal_history.get(goal, 0) + 1
        self.alignment_stats["total_interactions"] += 1
        self.last_user_input = user_input
        
        logger.info(f"[Alignment] Recorded turn {self.turn_count}: goal='{goal}' via {routing_path}")
        
        return entry
    
    def record_explicit_feedback(
        self,
        helpful: bool,
        satisfaction_score: Optional[float] = None,
        feedback_text: Optional[str] = None
    ) -> None:
        """
        Record explicit user feedback (e.g., "Was that helpful? Yes/No").
        """
        if not self.entries:
            logger.warning("[Alignment] No interaction to apply feedback to")
            return
        
        entry = self.entries[-1]
        
        # Record signal
        if helpful:
            entry.feedback_signal = FeedbackSignal.EXPLICIT_YES
            entry.was_helpful = True
            self.alignment_stats["helpful_count"] += 1
        else:
            entry.feedback_signal = FeedbackSignal.EXPLICIT_NO
            entry.was_helpful = False
        
        entry.feedback_confidence = 0.95  # Explicit feedback is high confidence
        entry.user_satisfaction = satisfaction_score
        
        self.alignment_stats["feedback_received"] += 1
        self.alignment_stats["explicit_signal_count"] += 1
        
        level = "helpful ✓" if helpful else "not helpful ✗"
        logger.info(
            f"[Alignment] Explicit feedback: {level} "
            f"(satisfaction={satisfaction_score})"
        )
    
    def infer_implicit_feedback(self, new_user_input: str) -> None:
        """
        Infer implicit feedback from the user's next action.
        
        - Repeat same question → implicit NO (didn't solve it)
        - Related followup → implicit YES (helped, now building on it)
        - Complete topic change → implicit ABANDON
        """
        if not self.entries:
            return
        
        entry = self.entries[-1]
        is_repeat = self._is_repeat_question(self.last_user_input, new_user_input)
        is_followup = self._is_followup_question(self.last_user_input, new_user_input)
        is_abandon = not is_repeat and not is_followup and len(new_user_input) > 5
        
        if is_repeat:
            entry.feedback_signal = FeedbackSignal.IMPLICIT_REPEAT
            entry.was_helpful = False
            entry.feedback_confidence = 0.7
            self.alignment_stats["implicit_signal_count"] += 1
            logger.info(f"[Alignment] Inferred: User repeated question → NOT HELPFUL")
        
        elif is_followup:
            entry.feedback_signal = FeedbackSignal.IMPLICIT_FOLLOWUP
            entry.was_helpful = True
            entry.feedback_confidence = 0.6
            self.alignment_stats["implicit_signal_count"] += 1
            self.alignment_stats["helpful_count"] += 1
            logger.info(f"[Alignment] Inferred: User asked followup → HELPFUL")
        
        elif is_abandon:
            entry.feedback_signal = FeedbackSignal.IMPLICIT_ABANDON
            entry.was_helpful = False
            entry.feedback_confidence = 0.5
            self.alignment_stats["implicit_signal_count"] += 1
            logger.info(f"[Alignment] Inferred: User abandoned topic → NOT HELPFUL")
    
    def _is_repeat_question(self, last_input: str, new_input: str) -> bool:
        """Detect if user is repeating essentially the same question."""
        last_lower = last_input.lower().strip()
        new_lower = new_input.lower().strip()
        
        # Exact or near-exact repeat
        if last_lower == new_lower:
            return True
        
        # Check if main keywords appear in both
        last_words = set(last_lower.split())
        new_words = set(new_lower.split())
        
        common_words = last_words & new_words
        stop_words = {"the", "a", "an", "is", "are", "to", "of", "for"}
        meaningful_common = common_words - stop_words
        
        # If 70%+ of keywords overlap, it's a repeat
        if len(last_words) > 0 and len(meaningful_common) / len(last_words) > 0.7:
            return True
        
        return False
    
    def _is_followup_question(self, last_input: str, new_input: str) -> bool:
        """Detect if user is building on previous question."""
        followup_markers = [
            "what about",
            "how about",
            "what if",
            "can you also",
            "also",
            "additionally",
            "furthermore",
            "next",
            "then",
        ]
        
        new_lower = new_input.lower()
        
        # Check for explicit followup markers
        if any(marker in new_lower for marker in followup_markers):
            return True
        
        # Check if new input uses pronouns (implicit reference to previous topic)
        pronouns = ["it", "that", "them", "this", "these"]
        if any(f" {p} " in f" {new_lower} " for p in pronouns):
            return True
        
        # Check for common related questions after code/explanation
        if any(phrase in new_lower for phrase in ["how does", "why does", "what makes", "explain that"]):
            return True
        
        return False
    
    def get_alignment_for_goal(self, goal: str) -> Dict[str, Any]:
        """Get alignment statistics for a specific goal."""
        goal_entries = [e for e in self.entries if e.goal == goal]
        
        if not goal_entries:
            return {}
        
        helpful_count = sum(1 for e in goal_entries if e.was_helpful)
        
        return {
            "goal": goal,
            "total_interactions": len(goal_entries),
            "helpful_count": helpful_count,
            "success_rate": helpful_count / len(goal_entries) if goal_entries else 0.0,
            "recent_entries": goal_entries[-3:],  # Last 3
        }
    
    def get_routing_effectiveness(self, routing_path: str) -> Dict[str, Any]:
        """Evaluate how effective a routing path is."""
        path_entries = [e for e in self.entries if e.routing_path == routing_path]
        
        if not path_entries:
            return {}
        
        helpful_count = sum(1 for e in path_entries if e.was_helpful)
        
        return {
            "routing_path": routing_path,
            "total_uses": len(path_entries),
            "successful": helpful_count,
            "effectiveness_rate": helpful_count / len(path_entries) if path_entries else 0.0,
        }
    
    def get_problem_areas(self) -> List[Dict[str, Any]]:
        """Identify recurring goals where ALICE is struggling."""
        goal_stats = {}
        
        for entry in self.entries:
            if entry.goal not in goal_stats:
                goal_stats[entry.goal] = {"total": 0, "helpful": 0}
            
            goal_stats[entry.goal]["total"] += 1
            if entry.was_helpful:
                goal_stats[entry.goal]["helpful"] += 1
        
        # Find goals with < 60% success rate (problem areas)
        problems = []
        for goal, stats in goal_stats.items():
            if stats["total"] >= 2:  # Only count if seen multiple times
                success_rate = stats["helpful"] / stats["total"]
                if success_rate < 0.6:
                    problems.append({
                        "goal": goal,
                        "success_rate": success_rate,
                        "attempts": stats["total"],
                        "failures": stats["total"] - stats["helpful"],
                    })
        
        # Sort by worst first
        return sorted(problems, key=lambda x: x["success_rate"])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall alignment statistics."""
        total = self.alignment_stats["total_interactions"]
        helpful = self.alignment_stats["helpful_count"]
        
        return {
            **self.alignment_stats,
            "overall_success_rate": helpful / total if total > 0 else 0.0,
            "feedback_rate": self.alignment_stats["feedback_received"] / total if total > 0 else 0.0,
        }
    
    def get_recent_feedback(self, n: int = 5) -> List[AlignmentEntry]:
        """Get most recent entries with feedback."""
        with_feedback = [e for e in self.entries if e.feedback_signal != FeedbackSignal.NEUTRAL]
        return with_feedback[-n:]
