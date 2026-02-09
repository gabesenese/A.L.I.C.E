"""
Teacher Loop for A.L.I.C.E
Analyzes LLM fallbacks and suggests patterns to learn

When Alice uses the LLM (last resort), the teacher loop:
1. Logs the interaction to data/training/logged_interactions.jsonl
2. Identifies patterns in frequent LLM calls
3. Suggests converting them to learned patterns
4. Optionally auto-creates patterns after validation
"""

import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)


LOGGED_INTERACTIONS_PATH = "data/training/logged_interactions.jsonl"
LEARNED_PATTERNS_PATH = "memory/learning_patterns.json"
TEACHER_SUGGESTIONS_PATH = "data/training/teacher_suggestions.json"


@dataclass
class PatternSuggestion:
    """A suggested pattern to learn from LLM fallbacks"""
    user_input_pattern: str
    example_inputs: List[str]
    common_response: str
    frequency: int
    confidence: float
    intent: str
    entities: Dict[str, Any]
    should_auto_learn: bool = False


class TeacherLoop:
    """
    Analyzes LLM fallback calls and suggests patterns
    
    The teacher loop helps Alice become more independent by:
    - Identifying questions that repeatedly need LLM
    - Suggesting patterns that should be learned
    - Auto-creating patterns when confidence is high
    - Reducing LLM dependency over time
    """
    
    def __init__(
        self,
        min_frequency: int = 3,
        auto_learn_threshold: int = 5,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize teacher loop
        
        Args:
            min_frequency: Minimum times a pattern must appear to suggest learning
            auto_learn_threshold: Frequency at which to auto-learn (no approval needed)
            similarity_threshold: How similar inputs must be to group (0-1)
        """
        self.min_frequency = min_frequency
        self.auto_learn_threshold = auto_learn_threshold
        self.similarity_threshold = similarity_threshold
        
        self.suggestions: List[PatternSuggestion] = []
        self.learned_count = 0
    
    def analyze_fallbacks(
        self,
        lookback_hours: int = 24,
        max_interactions: int = 1000
    ) -> List[PatternSuggestion]:
        """
        Analyze logged LLM fallbacks and suggest patterns to learn
        
        Args:
            lookback_hours: How far back to analyze (hours)
            max_interactions: Maximum interactions to analyze
            
        Returns:
            List of pattern suggestions
        """
        if not os.path.exists(LOGGED_INTERACTIONS_PATH):
            logger.info("[TeacherLoop] No logged interactions yet")
            return []
        
        # Load logged interactions
        interactions = self._load_interactions(lookback_hours, max_interactions)
        
        if not interactions:
            logger.info("[TeacherLoop] No recent LLM fallbacks to analyze")
            return []
        
        logger.info(f"[TeacherLoop] Analyzing {len(interactions)} LLM fallbacks")
        
        # Group similar interactions
        groups = self._group_similar_interactions(interactions)
        
        # Generate suggestions from groups
        suggestions = []
        for group in groups:
            if len(group) >= self.min_frequency:
                suggestion = self._create_suggestion(group)
                if suggestion:
                    suggestions.append(suggestion)
        
        # Sort by frequency (most frequent first)
        suggestions.sort(key=lambda s: s.frequency, reverse=True)
        
        self.suggestions = suggestions
        
        # Save suggestions to file
        self._save_suggestions(suggestions)
        
        logger.info(f"[TeacherLoop] Generated {len(suggestions)} pattern suggestions")
        
        return suggestions
    
    def _load_interactions(
        self,
        lookback_hours: int,
        max_interactions: int
    ) -> List[Dict[str, Any]]:
        """Load recent LLM fallback interactions"""
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        
        interactions = []
        try:
            with open(LOGGED_INTERACTIONS_PATH, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        entry = json.loads(line)
                        
                        # Check timestamp
                        timestamp = datetime.fromisoformat(entry.get('timestamp', ''))
                        if timestamp < cutoff_time:
                            continue
                        
                        interactions.append(entry)
                        
                        if len(interactions) >= max_interactions:
                            break
                    
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.debug(f"[TeacherLoop] Skipping invalid entry: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"[TeacherLoop] Error loading interactions: {e}")
        
        return interactions
    
    def _group_similar_interactions(
        self,
        interactions: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Group similar interactions by user input pattern"""
        # Simple grouping by normalized user input
        # TODO: Use semantic similarity for better grouping
        
        groups_dict = defaultdict(list)
        
        for interaction in interactions:
            user_input = interaction.get('user_input', '').lower().strip()
            
            # Normalize: remove punctuation, extra spaces
            normalized = ' '.join(user_input.split())
            
            # Group by first 5 words (captures the main question pattern)
            words = normalized.split()
            key = ' '.join(words[:5]) if len(words) >= 5 else normalized
            
            groups_dict[key].append(interaction)
        
        # Convert to list of groups
        groups = list(groups_dict.values())
        
        return groups
    
    def _create_suggestion(
        self,
        group: List[Dict[str, Any]]
    ) -> Optional[PatternSuggestion]:
        """Create a pattern suggestion from a group of similar interactions"""
        if not group:
            return None
        
        # Collect examples
        example_inputs = [i.get('user_input', '') for i in group[:5]]
        
        # Find most common response
        responses = [i.get('llm_response', '') for i in group if i.get('llm_response')]
        if not responses:
            return None
        
        response_counts = Counter(responses)
        common_response = response_counts.most_common(1)[0][0]
        
        # Get intent and entities from first example
        first = group[0]
        intent = first.get('intent', 'unknown')
        entities = first.get('entities', {})
        
        # Calculate confidence based on response consistency
        consistency = response_counts[common_response] / len(responses)
        
        # Determine if should auto-learn
        should_auto_learn = (
            len(group) >= self.auto_learn_threshold and
            consistency >= 0.8
        )
        
        return PatternSuggestion(
            user_input_pattern=group[0].get('user_input', ''),
            example_inputs=example_inputs,
            common_response=common_response,
            frequency=len(group),
            confidence=consistency,
            intent=intent,
            entities=entities,
            should_auto_learn=should_auto_learn
        )
    
    def _save_suggestions(self, suggestions: List[PatternSuggestion]):
        """Save suggestions to file for review"""
        try:
            os.makedirs(os.path.dirname(TEACHER_SUGGESTIONS_PATH), exist_ok=True)
            
            data = {
                'timestamp': datetime.now().isoformat(),
                'total_suggestions': len(suggestions),
                'auto_learn_ready': sum(1 for s in suggestions if s.should_auto_learn),
                'suggestions': [
                    {
                        'pattern': s.user_input_pattern,
                        'examples': s.example_inputs,
                        'response': s.common_response,
                        'frequency': s.frequency,
                        'confidence': s.confidence,
                        'intent': s.intent,
                        'auto_learn': s.should_auto_learn
                    }
                    for s in suggestions
                ]
            }
            
            with open(TEACHER_SUGGESTIONS_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"[TeacherLoop] Saved suggestions to {TEACHER_SUGGESTIONS_PATH}")
        
        except Exception as e:
            logger.error(f"[TeacherLoop] Error saving suggestions: {e}")
    
    def learn_pattern(
        self,
        suggestion: PatternSuggestion,
        conversational_engine=None
    ) -> bool:
        """
        Learn a pattern (add to curated_patterns.json or training data)
        
        Args:
            suggestion: Pattern suggestion to learn
            conversational_engine: Optional engine to update
            
        Returns:
            True if successfully learned
        """
        try:
            # Add to conversational engine's learned responses
            if conversational_engine:
                intent = suggestion.intent
                if intent not in conversational_engine.learned_responses:
                    conversational_engine.learned_responses[intent] = []
                
                if suggestion.common_response not in conversational_engine.learned_responses[intent]:
                    conversational_engine.learned_responses[intent].append(suggestion.common_response)
                    logger.info(f"[TeacherLoop] Learned pattern for intent '{intent}'")
            
            # Also add to training data for persistence
            self._add_to_training_data(suggestion)
            
            self.learned_count += 1
            return True
        
        except Exception as e:
            logger.error(f"[TeacherLoop] Error learning pattern: {e}")
            return False
    
    def _add_to_training_data(self, suggestion: PatternSuggestion):
        """Add learned pattern to training data"""
        try:
            training_path = "data/training/example_training_data.jsonl"
            
            # Add entry for this learned pattern
            entry = {
                'user_input': suggestion.user_input_pattern,
                'intent': suggestion.intent,
                'entities': suggestion.entities,
                'response': suggestion.common_response,
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'learned_from_teacher': True,
                'frequency': suggestion.frequency
            }
            
            with open(training_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
            
            logger.info(f"[TeacherLoop] Added pattern to training data")
        
        except Exception as e:
            logger.error(f"[TeacherLoop] Error adding to training data: {e}")
    
    def auto_learn_high_confidence(
        self,
        conversational_engine=None
    ) -> int:
        """
        Auto-learn all high-confidence suggestions
        
        Returns:
            Number of patterns learned
        """
        learned = 0
        
        for suggestion in self.suggestions:
            if suggestion.should_auto_learn:
                if self.learn_pattern(suggestion, conversational_engine):
                    learned += 1
                    logger.info(
                        f"[TeacherLoop] Auto-learned: '{suggestion.user_input_pattern}' "
                        f"(freq={suggestion.frequency}, conf={suggestion.confidence:.2f})"
                    )
        
        return learned
    
    def get_report(self) -> Dict[str, Any]:
        """Get teacher loop report"""
        return {
            'total_suggestions': len(self.suggestions),
            'auto_learn_ready': sum(1 for s in self.suggestions if s.should_auto_learn),
            'patterns_learned': self.learned_count,
            'top_suggestions': [
                {
                    'pattern': s.user_input_pattern,
                    'frequency': s.frequency,
                    'confidence': s.confidence,
                    'auto_learn': s.should_auto_learn
                }
                for s in self.suggestions[:5]
            ]
        }


# Singleton instance
_teacher_loop: Optional[TeacherLoop] = None


def get_teacher_loop(
    min_frequency: int = 3,
    auto_learn_threshold: int = 5
) -> TeacherLoop:
    """Get singleton teacher loop instance"""
    global _teacher_loop
    if _teacher_loop is None:
        _teacher_loop = TeacherLoop(
            min_frequency=min_frequency,
            auto_learn_threshold=auto_learn_threshold
        )
    return _teacher_loop
