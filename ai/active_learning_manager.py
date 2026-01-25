"""
Active Learning Manager for A.L.I.C.E

This module implements active learning capabilities, allowing A.L.I.C.E to learn
and improve from user corrections and feedback over time.

Key Features:
- Track user corrections for intent classification, entity extraction, and response quality
- Store learning data with confidence scores and metadata
- Analyze patterns in corrections to improve future performance
- Provide feedback mechanisms for users to teach A.L.I.C.E
- Integrate with existing NLP, LLM, and memory systems

Author: A.L.I.C.E Development Team
Date: January 2026
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import re
from enum import Enum

class CorrectionType(Enum):
    """Types of corrections that can be made"""
    INTENT_CLASSIFICATION = "intent_classification"
    ENTITY_EXTRACTION = "entity_extraction"
    RESPONSE_QUALITY = "response_quality"
    FACTUAL_ERROR = "factual_error"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    CONVERSATION_CONTEXT = "conversation_context"
    RELATIONSHIP_EXTRACTION = "relationship_extraction"

class FeedbackType(Enum):
    """Types of feedback users can provide"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    CORRECTION = "correction"
    SUGGESTION = "suggestion"
    CLARIFICATION = "clarification"

@dataclass
class Correction:
    """Represents a user correction or feedback"""
    id: str
    timestamp: str
    correction_type: str
    original_input: str
    original_output: Any
    corrected_output: Any
    user_feedback: str
    confidence_score: float
    context: Dict[str, Any]
    applied: bool = False
    validation_count: int = 0

@dataclass
class LearningPattern:
    """Represents a learned pattern from corrections"""
    pattern_id: str
    pattern_type: str
    trigger_conditions: Dict[str, Any]
    correction_action: Dict[str, Any]
    confidence: float
    usage_count: int
    success_rate: float
    last_updated: str

@dataclass
class FeedbackEntry:
    """Represents user feedback on A.L.I.C.E's performance"""
    feedback_id: str
    timestamp: str
    feedback_type: str
    user_input: str
    alice_response: str
    user_rating: int  # 1-5 scale
    user_comment: str
    improvement_suggestion: str
    context: Dict[str, Any]

class ActiveLearningManager:
    """Manages active learning from user corrections and feedback"""
    
    def __init__(self, data_dir: str = "memory"):
        self.data_dir = data_dir
        self.corrections_file = os.path.join(data_dir, "corrections.json")
        self.patterns_file = os.path.join(data_dir, "learning_patterns.json")
        self.feedback_file = os.path.join(data_dir, "user_feedback.json")
        
        # Learning data
        self.corrections: List[Correction] = []
        self.learning_patterns: List[LearningPattern] = []
        self.feedback_entries: List[FeedbackEntry] = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_corrections': 0,
            'applied_corrections': 0,
            'accuracy_improvement': 0.0,
            'user_satisfaction': 0.0
        }
        
        # Load existing data
        self._load_data()
        
    def _load_data(self):
        """Load existing learning data from disk"""
        try:
            # Load corrections
            if os.path.exists(self.corrections_file):
                with open(self.corrections_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.corrections = [Correction(**item) for item in data]
            
            # Load patterns
            if os.path.exists(self.patterns_file):
                with open(self.patterns_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.learning_patterns = [LearningPattern(**item) for item in data]
            
            # Load feedback
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.feedback_entries = [FeedbackEntry(**item) for item in data]
                    
        except Exception as e:
            print(f"Warning: Could not load active learning data: {e}")
    
    def _save_data(self):
        """Save learning data to disk"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Save corrections
            with open(self.corrections_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(c) for c in self.corrections], f, indent=2, ensure_ascii=False)
            
            # Save patterns
            with open(self.patterns_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(p) for p in self.learning_patterns], f, indent=2, ensure_ascii=False)
            
            # Save feedback
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(f) for f in self.feedback_entries], f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving active learning data: {e}")
    
    def record_correction(self, 
                         correction_type: CorrectionType,
                         original_input: str,
                         original_output: Any,
                         corrected_output: Any,
                         user_feedback: str,
                         context: Dict[str, Any] = None) -> str:
        """Record a user correction"""
        correction_id = f"corr_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.corrections)}"
        
        correction = Correction(
            id=correction_id,
            timestamp=datetime.now().isoformat(),
            correction_type=correction_type.value,
            original_input=original_input,
            original_output=original_output,
            corrected_output=corrected_output,
            user_feedback=user_feedback,
            confidence_score=1.0,  # High confidence for direct user corrections
            context=context or {},
            applied=False,
            validation_count=1
        )
        
        self.corrections.append(correction)
        self.performance_metrics['total_corrections'] += 1
        
        # Try to learn a pattern immediately
        self._analyze_new_correction(correction)
        
        self._save_data()
        return correction_id
    
    def record_feedback(self,
                       feedback_type: FeedbackType,
                       user_input: str,
                       alice_response: str,
                       user_rating: int,
                       user_comment: str = "",
                       improvement_suggestion: str = "",
                       context: Dict[str, Any] = None) -> str:
        """Record user feedback on A.L.I.C.E's performance"""
        feedback_id = f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.feedback_entries)}"
        
        feedback = FeedbackEntry(
            feedback_id=feedback_id,
            timestamp=datetime.now().isoformat(),
            feedback_type=feedback_type.value,
            user_input=user_input,
            alice_response=alice_response,
            user_rating=max(1, min(5, user_rating)),
            user_comment=user_comment,
            improvement_suggestion=improvement_suggestion,
            context=context or {}
        )
        
        self.feedback_entries.append(feedback)
        
        # Update satisfaction metric
        self._update_satisfaction_metric()
        
        self._save_data()
        return feedback_id
    
    def _analyze_new_correction(self, correction: Correction):
        """Analyze a new correction to potentially create learning patterns"""
        if correction.correction_type == CorrectionType.INTENT_CLASSIFICATION.value:
            self._analyze_intent_correction(correction)
        elif correction.correction_type == CorrectionType.ENTITY_EXTRACTION.value:
            self._analyze_entity_correction(correction)
        elif correction.correction_type == CorrectionType.RESPONSE_QUALITY.value:
            self._analyze_response_correction(correction)
    
    def _analyze_intent_correction(self, correction: Correction):
        """Analyze intent classification corrections"""
        original_intent = correction.original_output
        correct_intent = correction.corrected_output
        text = correction.original_input.lower()
        
        # Look for keyword patterns
        words = text.split()
        
        # Create a learning pattern
        pattern_id = f"intent_{original_intent}_to_{correct_intent}_{len(self.learning_patterns)}"
        
        pattern = LearningPattern(
            pattern_id=pattern_id,
            pattern_type="intent_classification",
            trigger_conditions={
                "contains_words": words[:5],  # First 5 words as triggers
                "original_intent": original_intent,
                "text_length_range": [len(text) - 10, len(text) + 10]
            },
            correction_action={
                "correct_intent": correct_intent,
                "confidence_boost": 0.2
            },
            confidence=0.8,
            usage_count=0,
            success_rate=1.0,
            last_updated=datetime.now().isoformat()
        )
        
        self.learning_patterns.append(pattern)
    
    def _analyze_entity_correction(self, correction: Correction):
        """Analyze entity extraction corrections"""
        original_entities = correction.original_output
        correct_entities = correction.corrected_output
        text = correction.original_input.lower()
        
        # Find patterns in missed entities
        if isinstance(correct_entities, dict):
            for entity_type, entities in correct_entities.items():
                if entity_type not in original_entities or len(entities) > len(original_entities.get(entity_type, [])):
                    # Create pattern for missed entities
                    pattern_id = f"entity_{entity_type}_{len(self.learning_patterns)}"
                    
                    pattern = LearningPattern(
                        pattern_id=pattern_id,
                        pattern_type="entity_extraction",
                        trigger_conditions={
                            "entity_type": entity_type,
                            "context_words": text.split()[:10],
                            "text_contains": [str(e) for e in entities if isinstance(e, str)][:3]
                        },
                        correction_action={
                            "add_entities": {entity_type: entities},
                            "confidence_boost": 0.15
                        },
                        confidence=0.7,
                        usage_count=0,
                        success_rate=1.0,
                        last_updated=datetime.now().isoformat()
                    )
                    
                    self.learning_patterns.append(pattern)
    
    def _analyze_response_correction(self, correction: Correction):
        """Analyze response quality corrections"""
        original_response = correction.original_output
        correct_response = correction.corrected_output
        user_input = correction.original_input.lower()
        
        # Analyze patterns in improved responses
        original_words = set(original_response.lower().split()) if isinstance(original_response, str) else set()
        correct_words = set(correct_response.lower().split()) if isinstance(correct_response, str) else set()
        
        # Find words that should be added or avoided
        words_to_add = correct_words - original_words
        words_to_avoid = original_words - correct_words
        
        if len(words_to_add) > 0 or len(words_to_avoid) > 0:
            pattern_id = f"response_quality_{len(self.learning_patterns)}"
            
            pattern = LearningPattern(
                pattern_id=pattern_id,
                pattern_type="response_quality",
                trigger_conditions={
                    "input_keywords": user_input.split()[:5],
                    "context_length": len(user_input),
                    "original_response_length": len(original_response) if isinstance(original_response, str) else 0
                },
                correction_action={
                    "preferred_words": list(words_to_add)[:10],
                    "avoid_words": list(words_to_avoid)[:10],
                    "style_improvement": correction.user_feedback
                },
                confidence=0.6,
                usage_count=0,
                success_rate=1.0,
                last_updated=datetime.now().isoformat()
            )
            
            self.learning_patterns.append(pattern)
    
    def apply_learning(self, user_input: str, current_output: Dict[str, Any]) -> Dict[str, Any]:
        """Apply learned patterns to improve current output"""
        modified_output = current_output.copy()
        
        for pattern in self.learning_patterns:
            if self._pattern_matches(pattern, user_input, current_output):
                modified_output = self._apply_pattern(pattern, modified_output)
                pattern.usage_count += 1
                pattern.last_updated = datetime.now().isoformat()
        
        return modified_output
    
    def _pattern_matches(self, pattern: LearningPattern, user_input: str, current_output: Dict[str, Any]) -> bool:
        """Check if a learning pattern matches the current input"""
        text = user_input.lower()
        conditions = pattern.trigger_conditions
        
        if pattern.pattern_type == "intent_classification":
            # Check word matches
            if "contains_words" in conditions:
                required_words = conditions["contains_words"]
                if not any(word in text for word in required_words):
                    return False
            
            # Check original intent
            if "original_intent" in conditions:
                if current_output.get("intent") != conditions["original_intent"]:
                    return False
            
            # Check text length
            if "text_length_range" in conditions:
                min_len, max_len = conditions["text_length_range"]
                if not (min_len <= len(text) <= max_len):
                    return False
            
            return True
        
        elif pattern.pattern_type == "entity_extraction":
            # Check entity type relevance
            if "entity_type" in conditions:
                # This pattern is for specific entity type
                pass
            
            # Check context words
            if "context_words" in conditions:
                context_words = conditions["context_words"]
                if not any(word in text for word in context_words):
                    return False
            
            # Check if text contains target entities
            if "text_contains" in conditions:
                target_entities = conditions["text_contains"]
                if not any(entity.lower() in text for entity in target_entities):
                    return False
            
            return True
        
        elif pattern.pattern_type == "response_quality":
            # Check input keywords
            if "input_keywords" in conditions:
                keywords = conditions["input_keywords"]
                if not any(keyword in text for keyword in keywords):
                    return False
            
            # Check context length similarity
            if "context_length" in conditions:
                expected_length = conditions["context_length"]
                length_diff = abs(len(text) - expected_length)
                if length_diff > expected_length * 0.5:  # Allow 50% variance
                    return False
            
            return True
        
        return False
    
    def _apply_pattern(self, pattern: LearningPattern, output: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a learning pattern to modify output"""
        if pattern.pattern_type == "intent_classification":
            action = pattern.correction_action
            if "correct_intent" in action:
                output["intent"] = action["correct_intent"]
                # Boost confidence if specified
                if "confidence_boost" in action:
                    current_confidence = output.get("confidence", 0.5)
                    output["confidence"] = min(1.0, current_confidence + action["confidence_boost"])
        
        elif pattern.pattern_type == "entity_extraction":
            action = pattern.correction_action
            if "add_entities" in action:
                entities_to_add = action["add_entities"]
                current_entities = output.get("entities", {})
                
                # Merge new entities
                for entity_type, new_entities in entities_to_add.items():
                    if entity_type not in current_entities:
                        current_entities[entity_type] = []
                    
                    # Add new entities if not already present
                    for entity in new_entities:
                        if entity not in current_entities[entity_type]:
                            current_entities[entity_type].append(entity)
                
                output["entities"] = current_entities
                
                # Boost confidence if specified
                if "confidence_boost" in action:
                    current_confidence = output.get("confidence", 0.5)
                    output["confidence"] = min(1.0, current_confidence + action["confidence_boost"])
        
        elif pattern.pattern_type == "response_quality":
            # Store pattern for use in LLM response generation
            action = pattern.correction_action
            output["learning_guidance"] = {
                "preferred_words": action.get("preferred_words", []),
                "avoid_words": action.get("avoid_words", []),
                "style_improvement": action.get("style_improvement", "")
            }
        
        return output
    
    def _update_satisfaction_metric(self):
        """Update user satisfaction metric based on feedback"""
        if not self.feedback_entries:
            return
        
        recent_feedback = [f for f in self.feedback_entries 
                          if datetime.fromisoformat(f.timestamp) > datetime.now() - timedelta(days=30)]
        
        if recent_feedback:
            avg_rating = sum(f.user_rating for f in recent_feedback) / len(recent_feedback)
            self.performance_metrics['user_satisfaction'] = avg_rating / 5.0
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get statistics about learning progress"""
        return {
            "total_corrections": len(self.corrections),
            "total_patterns": len(self.learning_patterns),
            "total_feedback": len(self.feedback_entries),
            "applied_patterns": len([p for p in self.learning_patterns if p.usage_count > 0]),
            "recent_corrections": len([c for c in self.corrections 
                                     if datetime.fromisoformat(c.timestamp) > datetime.now() - timedelta(days=7)]),
            "average_user_rating": self.performance_metrics.get('user_satisfaction', 0.0) * 5,
            "correction_types": Counter([c.correction_type for c in self.corrections]),
            "pattern_success_rates": {p.pattern_id: p.success_rate for p in self.learning_patterns}
        }
    
    def suggest_improvements(self) -> List[str]:
        """Suggest improvements based on learning data"""
        suggestions = []
        
        # Analyze common correction types
        correction_counts = Counter([c.correction_type for c in self.corrections])
        
        if correction_counts.get(CorrectionType.INTENT_CLASSIFICATION.value, 0) > 5:
            suggestions.append("Consider updating intent classification keywords and patterns")
        
        if correction_counts.get(CorrectionType.RESPONSE_QUALITY.value, 0) > 3:
            suggestions.append("Review response templates and improve LLM prompting")
        
        # Analyze user feedback
        low_rating_feedback = [f for f in self.feedback_entries if f.user_rating <= 2]
        if len(low_rating_feedback) > 2:
            suggestions.append("Address recurring issues mentioned in low-rating feedback")
        
        return suggestions
    
    def export_learning_data(self, filename: str = None) -> str:
        """Export learning data for analysis or backup"""
        if filename is None:
            filename = f"alice_learning_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "corrections": [asdict(c) for c in self.corrections],
            "patterns": [asdict(p) for p in self.learning_patterns],
            "feedback": [asdict(f) for f in self.feedback_entries],
            "statistics": self.get_learning_stats(),
            "export_timestamp": datetime.now().isoformat()
        }
        
        export_path = os.path.join(self.data_dir, filename)
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return export_path