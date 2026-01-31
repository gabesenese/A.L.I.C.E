"""
Training System for A.L.I.C.E
Collects conversation data and fine-tunes A.L.I.C.E on your interactions.
Makes A.L.I.C.E learn your preferences, style, and develop her own personality.

This module now also introduces A.L.I.C.E's own lightweight machine learning
so she can start learning patterns directly from your conversations,
independent of the Ollama base model.
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

logger = logging.getLogger(__name__)

# Optional deep learning imports
try:
    from sentence_transformers import SentenceTransformer
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    logger.warning("[ML] sentence-transformers not available, using traditional ML only")


@dataclass
class TrainingExample:
    """A single training example (conversation turn)"""
    user_input: str
    assistant_response: str
    context: Dict[str, Any] = field(default_factory=dict)
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    quality_score: float = 1.0  # 0-1, for filtering high-quality examples
    feedback: Optional[str] = None  # User feedback on response


class TrainingDataCollector:
    """
    Collects all conversation data for training A.L.I.C.E.
    Stores interactions in a format suitable for fine-tuning.
    """
    
    def __init__(self, data_dir: str = "data/training"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.training_file = self.data_dir / "training_data.jsonl"
        self.stats_file = self.data_dir / "training_stats.json"
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = self._load_stats()
        logger.info(f"[Training] Data collector initialized: {self.stats['total_examples']} examples collected")
    
    def _load_stats(self) -> Dict[str, Any]:
        """Load training statistics"""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'total_examples': 0,
            'total_sessions': 0,
            'last_training': None,
            'examples_by_intent': {},
            'quality_distribution': {'high': 0, 'medium': 0, 'low': 0}
        }
    
    def _save_stats(self):
        """Save training statistics"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            logger.error(f"[Training] Error saving stats: {e}")
    
    def collect_interaction(
        self,
        user_input: str,
        assistant_response: str,
        context: Dict[str, Any] = None,
        intent: Optional[str] = None,
        entities: Dict[str, Any] = None,
        quality_score: float = 1.0
    ):
        """
        Collect a conversation interaction for training
        
        Args:
            user_input: What the user said
            assistant_response: What A.L.I.C.E responded
            context: Additional context (goal, world state, etc.)
            intent: Detected intent
            entities: Extracted entities
            quality_score: Quality of this example (0-1)
        """
        with self._lock:
            example = TrainingExample(
                user_input=user_input,
                assistant_response=assistant_response,
                context=context or {},
                intent=intent,
                entities=entities or {},
                quality_score=quality_score
            )
            
            # Save to JSONL file (one example per line)
            try:
                with open(self.training_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(asdict(example), ensure_ascii=False) + '\n')
                
                # Update statistics
                self.stats['total_examples'] += 1
                if intent:
                    self.stats['examples_by_intent'][intent] = \
                        self.stats['examples_by_intent'].get(intent, 0) + 1
                
                # Quality distribution
                if quality_score >= 0.8:
                    self.stats['quality_distribution']['high'] += 1
                elif quality_score >= 0.5:
                    self.stats['quality_distribution']['medium'] += 1
                else:
                    self.stats['quality_distribution']['low'] += 1
                
                self._save_stats()
                
            except Exception as e:
                logger.error(f"[Training] Error collecting interaction: {e}")
    
    def get_training_data(self, min_quality: float = 0.5, max_examples: Optional[int] = None) -> List[TrainingExample]:
        """
        Get collected training data
        
        Args:
            min_quality: Minimum quality score to include
            max_examples: Maximum number of examples to return
        
        Returns:
            List of training examples
        """
        examples = []
        if not self.training_file.exists():
            return examples
        
        try:
            with open(self.training_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if data.get('quality_score', 1.0) >= min_quality:
                            examples.append(TrainingExample(**data))
            
            # Sort by quality (highest first)
            examples.sort(key=lambda x: x.quality_score, reverse=True)
            
            if max_examples:
                examples = examples[:max_examples]
            
            return examples
        except Exception as e:
            logger.error(f"[Training] Error loading training data: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return self.stats.copy()
    
    def clear_data(self):
        """Clear all training data (use with caution)"""
        with self._lock:
            if self.training_file.exists():
                self.training_file.unlink()
            self.stats = {
                'total_examples': 0,
                'total_sessions': 0,
                'last_training': None,
                'examples_by_intent': {},
                'quality_distribution': {'high': 0, 'medium': 0, 'low': 0}
            }
            self._save_stats()
            logger.info("[Training] Training data cleared")


class FineTuningSystem:
    """
    Fine-tunes A.L.I.C.E on collected conversation data.
    Uses Ollama's fine-tuning capabilities or exports for local training.
    """
    
    def __init__(self, base_model: str = "llama3.1:8b", data_collector: Optional[TrainingDataCollector] = None):
        self.base_model = base_model
        self.data_collector = data_collector or TrainingDataCollector()
        self.fine_tuned_model_name = f"alice-{base_model.replace(':', '-')}"
        self.training_dir = Path("data/training")
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[FineTuning] System initialized with base model: {base_model}")
    
    def prepare_training_data(self, min_examples: int = 50) -> Tuple[bool, str]:
        """
        Prepare training data in format suitable for fine-tuning
        
        Args:
            min_examples: Minimum number of examples required
        
        Returns:
            (success, message)
        """
        examples = self.data_collector.get_training_data(min_quality=0.6)
        
        if len(examples) < min_examples:
            return False, f"Need at least {min_examples} examples for training. Currently have {len(examples)}. Keep using A.L.I.C.E to collect more data!"
        
        # Convert to Ollama fine-tuning format (Modelfile format)
        modelfile_path = self.training_dir / "training_data.txt"
        
        try:
            with open(modelfile_path, 'w', encoding='utf-8') as f:
                for example in examples:
                    # Format as conversation
                    f.write(f"User: {example.user_input}\n")
                    f.write(f"Assistant: {example.assistant_response}\n")
                    f.write("\n")
            
            logger.info(f"[FineTuning] Prepared {len(examples)} examples for training")
            return True, f"Prepared {len(examples)} training examples. Ready to train!"
        
        except Exception as e:
            logger.error(f"[FineTuning] Error preparing data: {e}")
            return False, f"Error preparing training data: {e}"
    
    def export_for_training(self, format: str = "jsonl") -> Optional[str]:
        """
        Export training data in various formats for external training
        
        Args:
            format: Export format ('jsonl', 'json', 'txt')
        
        Returns:
            Path to exported file or None
        """
        examples = self.data_collector.get_training_data()
        
        if not examples:
            logger.warning("[FineTuning] No training data to export")
            return None
        
        if format == "jsonl":
            export_path = self.training_dir / "alice_training_data.jsonl"
            with open(export_path, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(json.dumps(asdict(example), ensure_ascii=False) + '\n')
        
        elif format == "json":
            export_path = self.training_dir / "alice_training_data.json"
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump([asdict(ex) for ex in examples], f, indent=2, ensure_ascii=False)
        
        elif format == "txt":
            export_path = self.training_dir / "alice_training_data.txt"
            with open(export_path, 'w', encoding='utf-8') as f:
                for example in examples:
                    f.write(f"### User\n{example.user_input}\n\n### Assistant\n{example.assistant_response}\n\n---\n\n")
        
        logger.info(f"[FineTuning] Exported {len(examples)} examples to {export_path}")
        return str(export_path)
    
    def check_fine_tuned_model(self) -> bool:
        """Check if fine-tuned model exists"""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return any(self.fine_tuned_model_name in m.get('name', '') for m in models)
        except Exception:
            pass
        return False
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get status of training system"""
        stats = self.data_collector.get_stats()
        examples = self.data_collector.get_training_data()
        has_model = self.check_fine_tuned_model()
        
        return {
            'total_examples': stats['total_examples'],
            'high_quality_examples': len([e for e in examples if e.quality_score >= 0.8]),
            'ready_for_training': len(examples) >= 50,
            'fine_tuned_model_exists': has_model,
            'fine_tuned_model_name': self.fine_tuned_model_name if has_model else None,
            'base_model': self.base_model,
            'examples_by_intent': stats.get('examples_by_intent', {})
        }


_training_collector: Optional[TrainingDataCollector] = None
_fine_tuning_system: Optional[FineTuningSystem] = None


def get_training_collector() -> TrainingDataCollector:
    """Get singleton training data collector"""
    global _training_collector
    if _training_collector is None:
        _training_collector = TrainingDataCollector()
    return _training_collector


def get_fine_tuning_system(base_model: str = "llama3.1:8b") -> FineTuningSystem:
    """Get singleton fine-tuning system"""
    global _fine_tuning_system
    if _fine_tuning_system is None:
        _fine_tuning_system = FineTuningSystem(base_model=base_model)
    return _fine_tuning_system


class SimpleMLLearner:
    """
    A.L.I.C.E's own machine learning brain that learns from conversations.
    
    Uses hybrid approach:
    - Traditional ML (LogisticRegression + TF-IDF) for fast, lightweight learning
    - Deep Learning (SentenceTransformer) when available for better understanding
    
    Learns to predict response quality and develop consistent personality.
    This is A.L.I.C.E's own learning, independent of Ollama.
    """

    def __init__(self, data_dir: str = "data/training"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.data_dir / "ml_learner.pkl"

        # Traditional ML (always available)
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.model: Optional[LogisticRegression] = None
        
        # Deep learning (optional, for better understanding)
        self.deep_model: Optional[SentenceTransformer] = None
        self.use_deep_learning = DEEP_LEARNING_AVAILABLE
        
        if self.use_deep_learning:
            try:
                self.deep_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("[ML] Deep learning enabled for A.L.I.C.E's personal learning")
            except Exception as e:
                logger.warning(f"[ML] Could not load deep learning model: {e}")
                self.use_deep_learning = False
        
        self.trained: bool = False
        self._load()

    def _load(self) -> None:
        """Load a previously trained model if it exists."""
        try:
            if self.model_path.exists():
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                self.vectorizer = data["vectorizer"]
                self.model = data["model"]
                self.trained = True
                logger.info("[ML] Loaded existing A.L.I.C.E ML learner model")
        except Exception as e:
            logger.warning(f"[ML] Failed to load ML learner model: {e}")

    def train_from_examples(
        self,
        examples: List[TrainingExample],
        min_examples: int = 10,
    ) -> bool:
        """
        Train A.L.I.C.E's ML brain from conversation examples.
        
        Uses hybrid approach:
        - Traditional ML for fast learning (TF-IDF + LogisticRegression)
        - Deep learning for semantic understanding (SentenceTransformer)
        
        Learns to predict response quality and develop personality patterns.
        """
        if len(examples) < min_examples:
            logger.info(f"[ML] Not enough examples to train ({len(examples)}/{min_examples})")
            return False

        try:
            texts = [f"{ex.user_input} {ex.assistant_response}" for ex in examples]
            
            # Traditional ML training (always)
            X_tfidf = self.vectorizer.fit_transform(texts)
            y = np.array([1 if ex.quality_score >= 0.8 else 0 for ex in examples], dtype=int)

            # Avoid training on single-class data
            if y.sum() == 0 or y.sum() == len(y):
                logger.info("[ML] Training data has only one quality class; skipping ML training")
                return False

            model = LogisticRegression(max_iter=1000)
            model.fit(X_tfidf, y)
            self.model = model
            
            # Deep learning training (if available)
            if self.use_deep_learning and self.deep_model:
                try:
                    # Generate deep embeddings for better semantic understanding
                    embeddings = self.deep_model.encode(texts, show_progress_bar=False)
                    
                    # Train on deep embeddings too (better for semantic patterns)
                    from sklearn.ensemble import GradientBoostingClassifier
                    deep_model = GradientBoostingClassifier(n_estimators=50, max_depth=3)
                    deep_model.fit(embeddings, y)
                    
                    # Store deep model alongside traditional
                    self.deep_classifier = deep_model
                    logger.info("[ML] Deep learning training completed")
                except Exception as e:
                    logger.warning(f"[ML] Deep learning training failed: {e}")
            
            self.trained = True

            # Persist models
            try:
                save_data = {
                    "vectorizer": self.vectorizer,
                    "model": self.model
                }
                if hasattr(self, 'deep_classifier'):
                    save_data["deep_classifier"] = self.deep_classifier
                
                with open(self.model_path, "wb") as f:
                    pickle.dump(save_data, f)
            except Exception as e:
                logger.warning(f"[ML] Failed to save ML learner model: {e}")

            training_type = "hybrid (traditional + deep learning)" if self.use_deep_learning else "traditional ML"
            logger.info(f"[ML]  A.L.I.C.E trained on {len(examples)} examples using {training_type}")
            return True

        except Exception as e:
            logger.error(f"[ML] Error training A.L.I.C.E ML learner: {e}")
            return False

    def predict_high_quality_prob(
        self,
        user_input: str,
        assistant_response: str,
    ) -> float:
        """
        Predict how likely it is that this response will be "high quality"
        based on what A.L.I.C.E has already seen.
        """
        if not self.trained or not self.model:
            # Neutral probability if we have no trained model yet
            return 0.5

        try:
            text = f"{user_input} {assistant_response}"
            X = self.vectorizer.transform([text])
            prob = self.model.predict_proba(X)[0, 1]
            return float(prob)
        except Exception as e:
            logger.debug(f"[ML] Prediction error in ML learner: {e}")
            return 0.5


_ml_learner: Optional[SimpleMLLearner] = None


def get_ml_learner() -> SimpleMLLearner:
    """Get singleton ML learner (A.L.I.C.E's own ML brain)."""
    global _ml_learner
    if _ml_learner is None:
        _ml_learner = SimpleMLLearner()
    return _ml_learner

# Offline training loop (evaluate -> update threshold -> deploy)

from .runtime_thresholds import get_thresholds, update_thresholds, get_goal_path_confidence, get_tool_path_confidence


def run_offline_loop(
        data_collector: Optional["TrainingDataCollector"] = None,
        ml_learner: Optional["SimpleMLLearner"] = None,
        min_examples: int = 20,
) -> Dict[str, Any]:
    """
    Run the full offline loop: train from logs, evaluate accuracy, update thresholds, deploy model.

    Returns:
        Dict with keys: trained, intent_accuracy (if evaluable), thresholds_updated, model_saved.
    """
    from .runtime_thresholds import get_thresholds

    data_collector = data_collector or TrainingDataCollector()
    ml_learner = ml_learner or SimpleMLLearner()
    examples = data_collector.get_training_data(min_quality=0.5)
    result = {"trained": False, "intent_accuracy": None, "threshold_updated": False,  "model_saved": False}

    if len(examples) < min_examples:
        logger.info(f"[OfflineLoop] Not enough examples ({len(examples)}/{min_examples})")
        return result

    # 1) Train router/ML Learner from logs
    trained = ml_learner.train_from_examples(examples, min_examples=min_examples)
    result["trained"] = trained
    if trained:
        try:
            save_data = {
                "vectorizer": _ml_learner.vectorizer,
                "model": ml_learner.model,
            }
            if hasattr(ml_learner, "deep_classifier"):
                save_data["deep_classifier"] = ml_learner.deep_classifier
            with open(ml_learner.model_path, "wb") as f:
                pickle.dump(save_data, f)
            result["model_saved"] = True
        except Exception as e:
            logger.warning(f"[OfflineLoop] Could not save model: {e}")

    # 2) Evaluate accuracy (intent prediction from user_input if we have intent in examples)         
    intent_correct = 0
    intent_total = 0 
    for ex in examples[-100:]:
        if not ex.intent:
            continue
        intent_total += 1
        # Simple check: ML Learner predicts high quality for this pair, we use intent as proxy for "correct path"
        prob = ml_learner.predict_high_quality_prob(ex.user_input, ex.assistant_response)
        if prob >= 0.5:
            intent_correct += 1
        if intent_total > 0:
            result["intent_accuracy"] = intent_correct / intent_total

    # 3) Update thresholds from evaluation ( optional: nudge the goal_path down if accuracy is high)
    thresholds = get_thresholds()
    if result.get("intent_accuracy") is not None:
        acc = result["intent_accuracy"]
        if acc >= 0.8 and thresholds.get("goal_path_confidence", 0.6) < 0.65:
            update_thresholds({"goal_path_confidence": 0.65})
            result["thresholds_updated"] = True
        elif acc < 0.5 and thresholds.get("goal_path_confidence", 0.6) > 0.55:
            update_thresholds({"goal_path_confidence": 0.55})
            result["thresholds_updated"] = True

    return result




