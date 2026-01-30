"""
Predictive Prefetcher for A.L.I.C.E
Anticipates likely next actions and prefetches data.
Makes A.L.I.C.E feel instant and responsive.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A prediction about likely next action"""
    action: str
    confidence: float
    context: Dict[str, Any]
    prefetch_data: Optional[Any] = None


class PredictivePrefetcher:
    """
    Predicts likely next actions based on:
    - Current action patterns
    - Time of day patterns
    - Sequential patterns (after X, user often does Y)
    - Context patterns
    """
    
    def __init__(self, world_state=None):
        self.world_state = world_state
        self.action_sequences: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.time_patterns: Dict[int, List[str]] = defaultdict(list)  # hour -> actions
        self.sequential_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.prefetched_data: Dict[str, Any] = {}
    
    def record_action(self, intent: str, entities: Dict, success: bool):
        """Record an action for pattern learning"""
        if not success:
            return
        
        hour = datetime.now().hour
        self.time_patterns[hour].append(intent)
        
        # Track sequences
        if self.world_state and self.world_state.turns:
            last_intent = self.world_state.last_intent
            if last_intent:
                self.sequential_patterns[last_intent][intent] += 1
    
    def predict_next_actions(self, current_intent: str, entities: Dict) -> List[Prediction]:
        """Predict likely next actions"""
        predictions = []
        
        # Sequential patterns: after current_intent, what's next?
        if current_intent in self.sequential_patterns:
            next_actions = self.sequential_patterns[current_intent]
            total = sum(next_actions.values())
            if total > 0:
                for action, count in sorted(next_actions.items(), key=lambda x: x[1], reverse=True)[:3]:
                    confidence = count / total
                    if confidence > 0.3:  # Only if significant
                        predictions.append(Prediction(
                            action=action,
                            confidence=confidence,
                            context={"pattern": "sequential", "from": current_intent}
                        ))
        
        # Time-based patterns
        hour = datetime.now().hour
        if hour in self.time_patterns:
            recent_actions = self.time_patterns[hour][-5:]
            if recent_actions:
                from collections import Counter
                common = Counter(recent_actions).most_common(2)
                for action, count in common:
                    if action != current_intent:
                        predictions.append(Prediction(
                            action=action,
                            confidence=count / len(recent_actions),
                            context={"pattern": "time_based", "hour": hour}
                        ))
        
        return sorted(predictions, key=lambda x: x.confidence, reverse=True)
    
    def prefetch_for_prediction(self, prediction: Prediction):
        """Prefetch data for a predicted action"""
        # This would prefetch plugin data, but for now just mark it
        self.prefetched_data[prediction.action] = {
            "predicted_at": datetime.now(),
            "confidence": prediction.confidence
        }
        logger.debug(f"[Prefetcher] Prefetched for: {prediction.action} (confidence: {prediction.confidence:.2f})")
    
    def get_prefetched(self, action: str) -> Optional[Any]:
        """Get prefetched data for an action"""
        return self.prefetched_data.get(action)


_prefetcher: Optional[PredictivePrefetcher] = None


def get_prefetcher(world_state=None) -> PredictivePrefetcher:
    global _prefetcher
    if _prefetcher is None:
        _prefetcher = PredictivePrefetcher(world_state=world_state)
    return _prefetcher
