"""
Pattern Learning System for A.L.I.C.E
Learns user behavior patterns to enable anticipatory suggestions
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json
import logging

from ai.infrastructure.event_bus import get_event_bus, EventType, EventPriority

logger = logging.getLogger(__name__)


@dataclass
class UserPattern:
    """A learned behavioral pattern"""
    pattern_id: str
    pattern_type: str  # "temporal", "sequential", "contextual"
    description: str
    
    # Pattern data
    trigger_conditions: Dict[str, Any]  # What triggers this pattern
    expected_action: str  # What the user typically does
    confidence: float  # 0.0 to 1.0
    
    # Statistics
    occurrences: int = 0
    last_seen: Optional[datetime] = None
    last_suggested: Optional[datetime] = None
    suggestion_accepted: int = 0
    suggestion_rejected: int = 0
    
    @property
    def acceptance_rate(self) -> float:
        """Calculate suggestion acceptance rate"""
        total = self.suggestion_accepted + self.suggestion_rejected
        return self.suggestion_accepted / total if total > 0 else 0.0
    
    @property
    def is_reliable(self) -> bool:
        """Is this pattern reliable enough to suggest?"""
        return self.confidence > 0.6 and self.acceptance_rate > 0.5 and self.occurrences >= 3


class PatternLearner:
    """
    Learns user behavior patterns for anticipatory suggestions
    
    Pattern types:
    - Temporal: "Every Sunday at 3pm, user reviews finance notes"
    - Sequential: "After checking email, user usually opens calendar"
    - Contextual: "When system is idle for >10min, user takes a break"
    """
    
    def __init__(self, data_file: str = "data/patterns.json"):
        self.data_file = data_file
        self.patterns: Dict[str, UserPattern] = {}
        self.event_bus = get_event_bus()
        
        # Temporal pattern tracking
        self._temporal_actions = defaultdict(list)  # day_hour -> actions
        
        # Sequential pattern tracking
        self._action_sequences = []  # Recent action sequence
        self._sequence_window = 5  # Track last N actions
        
        # Contextual pattern tracking
        self._context_actions = defaultdict(list)  # context_state -> actions
        
        self._load_patterns()
    
    def observe_action(self, action: str, context: Dict[str, Any] = None):
        """
        Observe a user action to learn patterns
        
        Args:
            action: The action taken (e.g., "review_notes:finance")
            context: Current context (time, day, system state, etc.)
        """
        now = datetime.now()
        context = context or {}
        
        # Learn temporal patterns (day + hour)
        day_of_week = now.strftime("%A")
        hour = now.hour
        temporal_key = f"{day_of_week}_{hour}"
        
        self._temporal_actions[temporal_key].append({
            'action': action,
            'timestamp': now.isoformat(),
            'context': context
        })
        
        # Learn sequential patterns
        self._action_sequences.append({
            'action': action,
            'timestamp': now.isoformat()
        })
        
        # Keep only recent sequence
        if len(self._action_sequences) > self._sequence_window:
            self._action_sequences.pop(0)
        
        # Learn contextual patterns
        system_state = context.get('system_state', 'unknown')
        self._context_actions[system_state].append({
            'action': action,
            'timestamp': now.isoformat()
        })
        
        # Update existing patterns or create new ones
        self._update_patterns(action, temporal_key, context)
    
    def _update_patterns(self, action: str, temporal_key: str, context: Dict[str, Any]):
        """Update pattern database based on observed action"""
        now = datetime.now()
        
        # Check temporal patterns
        temporal_pattern_id = f"temporal_{temporal_key}_{action}"
        
        if temporal_pattern_id in self.patterns:
            pattern = self.patterns[temporal_pattern_id]
            pattern.occurrences += 1
            pattern.last_seen = now
            pattern.confidence = min(1.0, pattern.confidence + 0.05)
        else:
            # Check if we've seen this action at this time before
            actions_at_time = self._temporal_actions[temporal_key]
            matching = [a for a in actions_at_time if a['action'] == action]
            
            if len(matching) >= 2:  # Seen at least twice
                day, hour = temporal_key.split('_')
                
                self.patterns[temporal_pattern_id] = UserPattern(
                    pattern_id=temporal_pattern_id,
                    pattern_type="temporal",
                    description=f"User typically does '{action}' on {day} around {hour}:00",
                    trigger_conditions={
                        'day_of_week': day,
                        'hour': int(hour)
                    },
                    expected_action=action,
                    confidence=0.3,
                    occurrences=len(matching),
                    last_seen=now
                )
        
        # Check sequential patterns
        if len(self._action_sequences) >= 2:
            prev_action = self._action_sequences[-2]['action']
            current_action = action
            
            seq_pattern_id = f"sequential_{prev_action}→{current_action}"
            
            if seq_pattern_id in self.patterns:
                pattern = self.patterns[seq_pattern_id]
                pattern.occurrences += 1
                pattern.last_seen = now
                pattern.confidence = min(1.0, pattern.confidence + 0.05)
            else:
                # Check history for this sequence
                count = 0
                for i in range(len(self._action_sequences) - 1):
                    if (self._action_sequences[i]['action'] == prev_action and 
                        self._action_sequences[i + 1]['action'] == current_action):
                        count += 1
                
                if count >= 2:
                    self.patterns[seq_pattern_id] = UserPattern(
                        pattern_id=seq_pattern_id,
                        pattern_type="sequential",
                        description=f"After '{prev_action}', user typically does '{current_action}'",
                        trigger_conditions={
                            'previous_action': prev_action
                        },
                        expected_action=current_action,
                        confidence=0.3,
                        occurrences=count,
                        last_seen=now
                    )
    
    def get_suggestions(self, context: Dict[str, Any] = None) -> List[Tuple[UserPattern, str]]:
        """
        Get proactive suggestions based on learned patterns
        
        Args:
            context: Current context (time, recent actions, system state)
        
        Returns:
            List of (pattern, suggestion_text) tuples
        """
        context = context or {}
        now = datetime.now()
        suggestions = []
        
        # Check temporal patterns
        day_of_week = now.strftime("%A")
        hour = now.hour
        
        for pattern in self.patterns.values():
            if not pattern.is_reliable:
                continue
            
            # Skip if suggested recently
            if pattern.last_suggested:
                time_since_suggestion = now - pattern.last_suggested
                if time_since_suggestion < timedelta(hours=1):
                    continue
            
            # Check temporal patterns
            if pattern.pattern_type == "temporal":
                trigger = pattern.trigger_conditions
                
                if (trigger.get('day_of_week') == day_of_week and 
                    trigger.get('hour') == hour):
                    
                    suggestion = self._format_suggestion(pattern)
                    suggestions.append((pattern, suggestion))
            
            # Check sequential patterns
            elif pattern.pattern_type == "sequential":
                if len(self._action_sequences) > 0:
                    last_action = self._action_sequences[-1]['action']
                    
                    if pattern.trigger_conditions.get('previous_action') == last_action:
                        suggestion = self._format_suggestion(pattern)
                        suggestions.append((pattern, suggestion))
            
            # Check contextual patterns
            elif pattern.pattern_type == "contextual":
                system_state = context.get('system_state')
                
                if pattern.trigger_conditions.get('system_state') == system_state:
                    suggestion = self._format_suggestion(pattern)
                    suggestions.append((pattern, suggestion))
        
        # Sort by confidence
        suggestions.sort(key=lambda x: x[0].confidence, reverse=True)
        
        return suggestions[:3]  # Return top 3
    
    def _format_suggestion(self, pattern: UserPattern) -> str:
        """Format a pattern into a suggestion text"""
        action = pattern.expected_action
        
        # Parse action
        if ':' in action:
            action_type, action_target = action.split(':', 1)
            
            if action_type == "review_notes":
                return f"You usually review {action_target} notes around this time. Want me to prepare a summary?"
            elif action_type == "check_calendar":
                return f"Would you like me to show your upcoming events?"
            elif action_type == "check_email":
                return f"Ready to check your email? I can summarize new messages."
            else:
                return f"You typically {action_type} {action_target} now. Need help with that?"
        else:
            return f"You usually {action} around this time. Want assistance?"
    
    def mark_suggestion_result(self, pattern_id: str, accepted: bool):
        """
        Record whether user accepted or rejected a suggestion
        
        Args:
            pattern_id: ID of the pattern that was suggested
            accepted: Whether user accepted the suggestion
        """
        if pattern_id in self.patterns:
            pattern = self.patterns[pattern_id]
            pattern.last_suggested = datetime.now()
            
            if accepted:
                pattern.suggestion_accepted += 1
                pattern.confidence = min(1.0, pattern.confidence + 0.1)
            else:
                pattern.suggestion_rejected += 1
                pattern.confidence = max(0.0, pattern.confidence - 0.05)
            
            self._save_patterns()
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get statistics about learned patterns"""
        total_patterns = len(self.patterns)
        reliable_patterns = sum(1 for p in self.patterns.values() if p.is_reliable)
        
        by_type = defaultdict(int)
        for pattern in self.patterns.values():
            by_type[pattern.pattern_type] += 1
        
        total_suggestions = sum(
            p.suggestion_accepted + p.suggestion_rejected 
            for p in self.patterns.values()
        )
        
        total_accepted = sum(p.suggestion_accepted for p in self.patterns.values())
        
        return {
            'total_patterns': total_patterns,
            'reliable_patterns': reliable_patterns,
            'patterns_by_type': dict(by_type),
            'total_suggestions_made': total_suggestions,
            'total_accepted': total_accepted,
            'overall_acceptance_rate': total_accepted / total_suggestions if total_suggestions > 0 else 0.0
        }
    
    def _save_patterns(self):
        """Save patterns to disk"""
        try:
            import os
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            
            data = {
                pattern_id: {
                    'pattern_type': p.pattern_type,
                    'description': p.description,
                    'trigger_conditions': p.trigger_conditions,
                    'expected_action': p.expected_action,
                    'confidence': p.confidence,
                    'occurrences': p.occurrences,
                    'last_seen': p.last_seen.isoformat() if p.last_seen else None,
                    'last_suggested': p.last_suggested.isoformat() if p.last_suggested else None,
                    'suggestion_accepted': p.suggestion_accepted,
                    'suggestion_rejected': p.suggestion_rejected
                }
                for pattern_id, p in self.patterns.items()
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save patterns: {e}")
    
    def _load_patterns(self):
        """Load patterns from disk"""
        try:
            with open(self.data_file, 'r') as f:
                data = json.load(f)
            
            for pattern_id, p in data.items():
                self.patterns[pattern_id] = UserPattern(
                    pattern_id=pattern_id,
                    pattern_type=p['pattern_type'],
                    description=p['description'],
                    trigger_conditions=p['trigger_conditions'],
                    expected_action=p['expected_action'],
                    confidence=p['confidence'],
                    occurrences=p['occurrences'],
                    last_seen=datetime.fromisoformat(p['last_seen']) if p['last_seen'] else None,
                    last_suggested=datetime.fromisoformat(p['last_suggested']) if p['last_suggested'] else None,
                    suggestion_accepted=p['suggestion_accepted'],
                    suggestion_rejected=p['suggestion_rejected']
                )
            
            logger.info(f"Loaded {len(self.patterns)} patterns from {self.data_file}")
            
        except FileNotFoundError:
            logger.info("No existing patterns file, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")


# Global instance
_pattern_learner = None


def get_pattern_learner() -> PatternLearner:
    """Get the global pattern learner instance"""
    global _pattern_learner
    if _pattern_learner is None:
        _pattern_learner = PatternLearner()
    return _pattern_learner
