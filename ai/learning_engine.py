"""
ALICE Learning Engine - Unified learning system
Consolidates: training_system, ml_learner, response_generator

Responsibilities:
- Collect interactions
- Learn patterns (TF-IDF + sklearn)
- Generate responses from learned patterns
- Track quality and user feedback
- Provide training data for fine-tuning
"""

import os
import json
import logging
import pickle
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict, field
import threading

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Optional deep learning imports
try:
    from sentence_transformers import SentenceTransformer
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    logger.warning("[Learning] sentence-transformers not available, using traditional ML")


@dataclass
class TrainingExample:
    """A single training example"""
    user_input: str
    assistant_response: str
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0
    user_rating: Optional[int] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class LearningEngine:
    """
    Unified learning system for ALICE
    - Collects interactions
    - Learns patterns
    - Generates responses from learned knowledge
    - Tracks quality and user preferences
    """
    
    def __init__(self, data_dir: str = "data/training"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage
        self.training_file = self.data_dir / "training_data.jsonl"
        self.stats_file = self.data_dir / "training_stats.json"
        self.patterns_file = self.data_dir / "learned_patterns.pkl"
        self.embeddings_file = self.data_dir / "embeddings.pkl"
        
        # In-memory
        self.examples = []  # All examples
        self.patterns = {}  # input_hash → learned response
        self.embeddings = None
        self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=200)
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.stats = self._load_stats()
        
        self._lock = threading.RLock()
        
        # Load existing data
        self._load_patterns()
        self._load_examples()
        
        logger.info(f"[Learning] Engine initialized with {len(self.examples)} examples")
    
    def collect_interaction(
        self,
        user_input: str,
        assistant_response: str,
        intent: Optional[str] = None,
        entities: Optional[Dict] = None,
        quality_score: float = 1.0,
        user_rating: Optional[int] = None
    ):
        """Record an interaction for learning"""
        with self._lock:
            example = TrainingExample(
                user_input=user_input,
                assistant_response=assistant_response,
                intent=intent,
                entities=entities or {},
                quality_score=quality_score,
                user_rating=user_rating
            )
            
            self.examples.append(example)
            
            # Save incrementally
            self._save_example(example)
            
            # Update statistics
            self.stats['total_examples'] = len(self.examples)
            if intent:
                self.stats['examples_by_intent'][intent] = self.stats['examples_by_intent'].get(intent, 0) + 1
            
            # Quality tracking
            if quality_score >= 0.8:
                self.stats['quality_distribution']['high'] += 1
            elif quality_score >= 0.6:
                self.stats['quality_distribution']['medium'] += 1
            else:
                self.stats['quality_distribution']['low'] += 1
            
            self.stats['last_training'] = datetime.now().isoformat()
            self._save_stats()
            
            # Learn pattern immediately
            self._learn_pattern(example)
    
    def _learn_pattern(self, example: TrainingExample):
        """Learn a single pattern"""
        input_hash = hash(example.user_input) % (10**8)
        self.patterns[input_hash] = {
            'input': example.user_input,
            'response': example.assistant_response,
            'intent': example.intent,
            'quality': example.quality_score,
            'rating': example.user_rating
        }
    
    def get_similar_examples(self, user_input: str, top_k: int = 3) -> List[TrainingExample]:
        """Find similar examples using TF-IDF similarity"""
        if len(self.examples) < 2:
            return []
        
        try:
            # Vectorize all examples
            all_texts = [ex.user_input for ex in self.examples] + [user_input]
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=200)
            vectors = vectorizer.fit_transform(all_texts)
            
            # Find similarities
            similarities = cosine_similarity([vectors[-1]], vectors[:-1])[0]
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            similar = []
            for idx in top_indices:
                if similarities[idx] > 0.4:  # Minimum similarity threshold
                    similar.append(self.examples[idx])
            
            return similar
        except Exception as e:
            logger.warning(f"[Learning] Similarity search failed: {e}")
            return []
    
    def generate_response_from_learned(self, user_input: str) -> Optional[str]:
        """
        Generate response using learned patterns (no LLM needed)
        Returns None if no good pattern found
        """
        similar = self.get_similar_examples(user_input, top_k=1)
        
        if similar:
            example = similar[0]
            # Only use if high quality
            if example.quality_score >= 0.7:
                return example.assistant_response
        
        return None
    
    def get_high_quality_examples(self, min_examples: int = 50) -> Optional[List[Dict]]:
        """
        Get high-quality examples for fine-tuning
        Returns None if not enough examples
        """
        high_quality = [
            ex for ex in self.examples
            if ex.quality_score >= 0.8 and (ex.user_rating is None or ex.user_rating >= 3)
        ]
        
        if len(high_quality) < min_examples:
            return None
        
        return [
            {
                'input': ex.user_input,
                'output': ex.assistant_response,
                'intent': ex.intent
            }
            for ex in high_quality
        ]
    
    def should_finetune(self, min_examples: int = 50) -> bool:
        """Check if we have enough data to fine-tune"""
        return self.get_high_quality_examples(min_examples) is not None
    
    def get_training_data(
        self,
        min_quality: float = 0.7,
        max_examples: Optional[int] = None
    ) -> List[TrainingExample]:
        """Get training examples above quality threshold"""
        filtered = [ex for ex in self.examples if ex.quality_score >= min_quality]
        
        if max_examples:
            filtered = filtered[:max_examples]
        
        return filtered
    
    def record_feedback(self, user_input: str, response: str, rating: int, feedback_text: str = ""):
        """Record user feedback on a response"""
        # Find matching example
        for ex in self.examples:
            if ex.user_input.lower() == user_input.lower() and ex.assistant_response == response:
                ex.user_rating = rating
                self._save_stats()
                logger.info(f"[Learning] Feedback recorded: {user_input[:50]}... → rating {rating}")
                break
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics"""
        high_quality = len([ex for ex in self.examples if ex.quality_score >= 0.8])
        user_rated = len([ex for ex in self.examples if ex.user_rating is not None])
        avg_rating = np.mean([ex.user_rating for ex in self.examples if ex.user_rating]) if user_rated > 0 else 0
        
        return {
            'total_examples': len(self.examples),
            'high_quality': high_quality,
            'user_rated': user_rated,
            'average_rating': float(avg_rating),
            'ready_for_finetuning': self.should_finetune(),
            'by_intent': self.stats.get('examples_by_intent', {}),
            'quality_distribution': self.stats.get('quality_distribution', {}),
            'llm_calls_logged': self.stats.get('llm_calls_logged', 0)
        }
    
    def log_llm_call(
        self,
        user_input: str,
        llm_response: str,
        intent: Optional[str] = None,
        context: Dict[str, Any] = None
    ):
        """
        Log LLM call for pattern learning (gatekeeper function)
        
        This is called EVERY time LLM is used, so we can:
        1. Track unhandled → LLM patterns
        2. Auto-create deterministic patterns when threshold met
        3. Suggest learning opportunities
        """
        # Collect as normal interaction
        self.collect_interaction(
            user_input=user_input,
            assistant_response=llm_response,
            intent=intent or 'llm_fallback',
            quality_score=0.8  # LLM responses are generally good
        )
        
        # Track LLM usage
        self.stats['llm_calls_logged'] = self.stats.get('llm_calls_logged', 0) + 1
        
        # Check if this input has been asked before
        similar_count = self._count_similar_inputs(user_input)
        
        if similar_count >= 3:
            logger.info(f"Pattern opportunity: '{user_input[:50]}...' asked {similar_count} times - consider creating pattern")
            return {
                'should_create_pattern': True,
                'similar_count': similar_count,
                'suggested_response': llm_response
            }
        
        return {
            'should_create_pattern': False,
            'similar_count': similar_count
        }
    
    def _count_similar_inputs(self, user_input: str, threshold: float = 0.85) -> int:
        """Count how many times similar input was asked"""
        user_input_lower = user_input.lower().strip()
        count = 0
        
        for ex in self.examples:
            # Simple similarity check
            ex_input_lower = ex.user_input.lower().strip()
            
            # Exact match
            if ex_input_lower == user_input_lower:
                count += 1
            # Fuzzy match (simple word overlap)
            elif self._fuzzy_match(ex_input_lower, user_input_lower) > threshold:
                count += 1
        
        return count
    
    def _fuzzy_match(self, str1: str, str2: str) -> float:
        """Simple fuzzy matching based on word overlap"""
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def suggest_pattern_creation(self, min_occurrences: int = 3) -> List[Dict[str, Any]]:
        """
        Suggest patterns that should be created based on repeated LLM calls
        
        Returns:
            List of pattern suggestions with user_input, response, count
        """
        from collections import defaultdict
        
        # Group similar inputs
        input_groups = defaultdict(list)
        
        for ex in self.examples:
            # Normalize input
            normalized = ex.user_input.lower().strip()
            
            # Find similar group
            found_group = False
            for key in input_groups:
                if self._fuzzy_match(normalized, key) > 0.85:
                    input_groups[key].append(ex)
                    found_group = True
                    break
            
            if not found_group:
                input_groups[normalized].append(ex)
        
        # Find groups with enough occurrences
        suggestions = []
        for normalized_input, examples in input_groups.items():
            if len(examples) >= min_occurrences:
                # Get most common response
                responses = [ex.assistant_response for ex in examples]
                most_common_response = max(set(responses), key=responses.count)
                
                suggestions.append({
                    'user_input': examples[0].user_input,  # Use first original form
                    'suggested_response': most_common_response,
                    'occurrence_count': len(examples),
                    'should_learn': True
                })
        
        return sorted(suggestions, key=lambda x: x['occurrence_count'], reverse=True)
    
    # --- Internal methods ---
    
    def _save_example(self, example: TrainingExample):
        """Append example to training file"""
        try:
            with open(self.training_file, 'a') as f:
                f.write(json.dumps(asdict(example)) + '\n')
        except Exception as e:
            logger.error(f"[Learning] Error saving example: {e}")
    
    def _load_examples(self):
        """Load all examples from disk"""
        if self.training_file.exists():
            try:
                with open(self.training_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            example = TrainingExample(**data)
                            self.examples.append(example)
            except Exception as e:
                logger.warning(f"[Learning] Error loading examples: {e}")
    
    def _save_patterns(self):
        """Save learned patterns to disk"""
        try:
            with open(self.patterns_file, 'wb') as f:
                pickle.dump(self.patterns, f)
        except Exception as e:
            logger.error(f"[Learning] Error saving patterns: {e}")
    
    def _load_patterns(self):
        """Load learned patterns from disk"""
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, 'rb') as f:
                    self.patterns = pickle.load(f)
            except Exception as e:
                logger.warning(f"[Learning] Error loading patterns: {e}")
    
    def _save_stats(self):
        """Save statistics"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"[Learning] Error saving stats: {e}")
    
    def _load_stats(self) -> Dict[str, Any]:
        """Load statistics"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return {
            'total_examples': 0,
            'examples_by_intent': {},
            'quality_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'last_training': None
        }


# Singleton
_learning_engine: Optional[LearningEngine] = None


def get_learning_engine(data_dir: str = "data/training") -> LearningEngine:
    """Get singleton learning engine"""
    global _learning_engine
    if _learning_engine is None:
        _learning_engine = LearningEngine(data_dir)
    return _learning_engine
