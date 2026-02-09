#!/usr/bin/env python3
"""
Machine Learning Models for A.L.I.C.E

Three small, targeted ML models:
1. Router Classifier - Predict best route from features
2. Intent Refiner - Predict corrected intent from embeddings  
3. Pattern Clusterer - Find similar patterns for auto-suggestion
"""

import json
import logging
import pickle
import importlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

np = None
TfidfVectorizer = None
LogisticRegression = None
KMeans = None
StandardScaler = None
_ML_DEPS_LOADED = False

logger = logging.getLogger(__name__)


def _load_ml_deps() -> None:
    global _ML_DEPS_LOADED, np, TfidfVectorizer, LogisticRegression, KMeans, StandardScaler
    if _ML_DEPS_LOADED:
        return
    try:
        np = importlib.import_module("numpy")
    except ImportError:  # pragma: no cover
        np = None
    try:
        sklearn_text = importlib.import_module("sklearn.feature_extraction.text")
        sklearn_linear = importlib.import_module("sklearn.linear_model")
        sklearn_cluster = importlib.import_module("sklearn.cluster")
        sklearn_preproc = importlib.import_module("sklearn.preprocessing")
        TfidfVectorizer = sklearn_text.TfidfVectorizer
        LogisticRegression = sklearn_linear.LogisticRegression
        KMeans = sklearn_cluster.KMeans
        StandardScaler = sklearn_preproc.StandardScaler
    except ImportError:  # pragma: no cover
        TfidfVectorizer = None
        LogisticRegression = None
        KMeans = None
        StandardScaler = None
    _ML_DEPS_LOADED = True


def _missing_router_deps() -> List[str]:
    _load_ml_deps()
    missing = []
    if np is None:
        missing.append("numpy")
    if LogisticRegression is None or StandardScaler is None:
        missing.append("scikit-learn")
    return missing


def _missing_cluster_deps() -> List[str]:
    _load_ml_deps()
    missing = []
    if np is None:
        missing.append("numpy")
    if TfidfVectorizer is None or KMeans is None:
        missing.append("scikit-learn")
    return missing


class RouterClassifier:
    """
    Predicts best route (CONVERSATIONAL, TOOL, CLARIFICATION, LLM_FALLBACK)
    based on features from NLP and context.
    
    Features:
    - Intent confidence score
    - Presence of domain keywords
    - Intent type (vague, specific, tool, conversational)
    - Text length
    - Repetition patterns
    """
    
    ROUTES = {
        'CONVERSATIONAL': 0,
        'TOOL': 1,
        'CLARIFICATION': 2,
        'LLM_FALLBACK': 3
    }
    
    ROUTE_NAMES = {v: k for k, v in ROUTES.items()}
    
    def __init__(self, data_dir: str = "data/training"):
        self.data_dir = Path(data_dir)
        self.model_file = self.data_dir / "router_classifier.pkl"
        self.scaler_file = self.data_dir / "router_scaler.pkl"
        
        self.model = None
        self.scaler = None
        self.is_trained = False
        
        self._load_model()
    
    def _load_model(self):
        """Load trained model from disk"""
        if self.model_file.exists() and self.scaler_file.exists():
            try:
                with open(self.model_file, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                logger.info("[RouterClassifier] Loaded trained model")
            except Exception as e:
                logger.warning(f"[RouterClassifier] Error loading model: {e}")
    
    def extract_features(self, 
                        user_input: str,
                        intent: str,
                        confidence: float,
                        has_domain_keywords: bool = False,
                        has_vague_pattern: bool = False) -> Any:
        """
        Extract features for routing prediction
        
        Returns:
            Feature vector (8 features)
        """
        features = []
        
        # 1. Intent confidence (0-1)
        features.append(float(confidence))
        
        # 2. Has domain keywords (0-1)
        features.append(1.0 if has_domain_keywords else 0.0)
        
        # 3. Has vague pattern (0-1)
        features.append(1.0 if has_vague_pattern else 0.0)
        
        # 4. Text length (normalized 0-1, assuming max 200 chars)
        text_len = min(len(user_input), 200) / 200.0
        features.append(text_len)
        
        # 5. Intent type: is tool intent (0-1)
        tool_intents = {'email', 'notes', 'weather', 'time', 'calendar', 'file', 'music'}
        is_tool = 1.0 if any(t in intent.lower() for t in tool_intents) else 0.0
        features.append(is_tool)
        
        # 6. Intent type: is conversational (0-1)
        conv_intents = {'greeting', 'farewell', 'thanks', 'help', 'status', 'small_talk'}
        is_conv = 1.0 if any(c in intent.lower() for c in conv_intents) else 0.0
        features.append(is_conv)
        
        # 7. Intent type: is clarification (0-1)
        clarif_intents = {'vague', 'unclear', 'ambiguous', 'question_about'}
        is_clarif = 1.0 if any(c in intent.lower() for c in clarif_intents) else 0.0
        features.append(is_clarif)
        
        # 8. Word count (normalized 0-1, assuming max 20 words)
        word_count = min(len(user_input.split()), 20) / 20.0
        features.append(word_count)
        
        if np is None:
            return [features]
        return np.array([features])
    
    def predict(self, features: Any) -> Tuple[str, float]:
        """
        Predict route for given features
        
        Returns:
            (route_name, confidence)
        """
        if _missing_router_deps():
            return 'LLM_FALLBACK', 0.0
        if not self.is_trained or self.model is None:
            return 'LLM_FALLBACK', 0.0
        
        try:
            # Scale features
            scaled = self.scaler.transform(features)
            
            # Predict
            route_idx = self.model.predict(scaled)[0]
            proba = self.model.predict_proba(scaled)[0]
            confidence = float(np.max(proba))
            
            route_name = self.ROUTE_NAMES.get(route_idx, 'LLM_FALLBACK')
            
            return route_name, confidence
        except Exception as e:
            logger.warning(f"[RouterClassifier] Prediction error: {e}")
            return 'LLM_FALLBACK', 0.0
    
    def train(self, training_data: List[Dict[str, Any]]):
        """
        Train classifier on routing logs
        
        Expected format:
        [
            {
                'user_input': '...',
                'intent': '...',
                'confidence': 0.8,
                'has_domain_keywords': True,
                'has_vague_pattern': False,
                'actual_route': 'TOOL'
            },
            ...
        ]
        """
        if _missing_router_deps():
            logger.warning("[RouterClassifier] Missing dependencies (numpy/scikit-learn)")
            return False
        if not training_data or len(training_data) < 10:
            logger.warning("[RouterClassifier] Not enough training data")
            return False
        
        try:
            X = []
            y = []
            
            for entry in training_data:
                # Extract features
                features = self.extract_features(
                    user_input=entry.get('user_input', ''),
                    intent=entry.get('intent', ''),
                    confidence=float(entry.get('confidence', 0.5)),
                    has_domain_keywords=entry.get('has_domain_keywords', False),
                    has_vague_pattern=entry.get('has_vague_pattern', False)
                )
                
                # Get label
                route = entry.get('actual_route', 'LLM_FALLBACK')
                label = self.ROUTES.get(route, 3)  # Default to LLM_FALLBACK
                
                X.append(features[0])
                y.append(label)
            
            X = np.array(X)
            y = np.array(y)
            
            if len(set(y.tolist())) < 2:
                logger.warning("[RouterClassifier] Need at least 2 classes to train")
                return False
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = LogisticRegression(max_iter=1000, random_state=42)
            self.model.fit(X_scaled, y)
            
            self.is_trained = True
            
            # Save model
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            logger.info(f"[RouterClassifier] Trained on {len(training_data)} examples")
            return True
        
        except Exception as e:
            logger.error(f"[RouterClassifier] Training error: {e}")
            return False


class IntentRefiner:
    """
    Predicts corrected intent from embeddings and context.
    Uses scenario/correction data to learn intent mappings.
    
    Example:
    - Input: "tell me about the sun" (detected: weather_query, confidence: 0.65)
    - Output: "vague_question" (corrected)
    """
    
    def __init__(self, data_dir: str = "data/training"):
        self.data_dir = Path(data_dir)
        self.model_file = self.data_dir / "intent_refiner.pkl"
        self.intent_map_file = self.data_dir / "intent_corrections.json"
        
        self.model = None
        self.intent_corrections = {}
        self.is_trained = False
        
        self._load_model()
        self._load_intent_map()
    
    def _load_model(self):
        """Load trained refiner from disk"""
        if self.model_file.exists():
            try:
                with open(self.model_file, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                logger.info("[IntentRefiner] Loaded trained model")
            except Exception as e:
                logger.warning(f"[IntentRefiner] Error loading model: {e}")
    
    def _load_intent_map(self):
        """Load intent correction mappings"""
        if self.intent_map_file.exists():
            try:
                with open(self.intent_map_file, 'r', encoding='utf-8', errors='ignore') as f:
                    self.intent_corrections = json.load(f)
            except Exception as e:
                logger.warning(f"[IntentRefiner] Error loading intent map: {e}")
    
    def refine_intent(self, 
                     user_input: str,
                     detected_intent: str,
                     confidence: float) -> Tuple[str, float, str]:
        """
        Refine detected intent based on input and confidence
        
        Returns:
            (refined_intent, refinement_confidence, reason)
        """
        reason = "rule_based"
        
        # Check correction map first (learned from scenarios/corrections)
        map_key = f"{detected_intent}_{confidence:.1f}"
        if map_key in self.intent_corrections:
            correction = self.intent_corrections[map_key]
            refined = correction.get('corrected_intent', detected_intent)
            conf = correction.get('confidence', 0.7)
            reason = "mapped"
            return refined, conf, reason
        
        # Use ML model if available and low confidence
        if self.is_trained and confidence < 0.7:
            # Try ML prediction
            try:
                # Simple heuristic: use model if available
                refined, ml_conf = self._ml_predict(user_input, detected_intent)
                if ml_conf > 0.6:
                    reason = "ml_model"
                    return refined, ml_conf, reason
            except Exception as e:
                logger.debug(f"[IntentRefiner] ML prediction failed: {e}")
        
        # Return original if no refinement
        return detected_intent, confidence, reason
    
    def _ml_predict(self, user_input: str, detected_intent: str) -> Tuple[str, float]:
        """Simple ML-based intent refinement"""
        # Placeholder for actual ML prediction
        # Would use embeddings + model to predict intent
        return detected_intent, 0.0
    
    def train(self, correction_data: List[Dict[str, Any]]):
        """
        Train refiner on correction data
        
        Expected format:
        [
            {
                'user_input': '...',
                'original_intent': 'weather_query',
                'corrected_intent': 'vague_question',
                'confidence_change': 0.15
            },
            ...
        ]
        """
        if not correction_data:
            return False
        
        try:
            # Build intent correction map
            for entry in correction_data:
                original = entry.get('original_intent', '')
                corrected = entry.get('corrected_intent', '')
                conf = entry.get('confidence', 0.7)
                
                key = f"{original}_{conf:.1f}"
                
                self.intent_corrections[key] = {
                    'corrected_intent': corrected,
                    'confidence': conf + 0.1,  # Boost confidence for corrected intent
                    'count': self.intent_corrections.get(key, {}).get('count', 0) + 1
                }
            
            # Save map
            with open(self.intent_map_file, 'w') as f:
                json.dump(self.intent_corrections, f, indent=2)
            
            logger.info(f"[IntentRefiner] Trained on {len(correction_data)} corrections")
            return True
        
        except Exception as e:
            logger.error(f"[IntentRefiner] Training error: {e}")
            return False


class PatternClusterer:
    """
    Clusters similar LLM_FALLBACK interactions per domain
    to suggest new patterns.
    
    Uses KMeans clustering on TF-IDF vectors.
    """
    
    def __init__(self, data_dir: str = "data/training"):
        self.data_dir = Path(data_dir)
        self.vectorizer_file = self.data_dir / "pattern_vectorizer.pkl"
        self.kmeans_file = self.data_dir / "pattern_kmeans.pkl"
        
        self.vectorizer = None
        self.kmeans = None
        self.is_trained = False
        
        self._load_model()
    
    def _load_model(self):
        """Load trained clusterer from disk"""
        if self.vectorizer_file.exists() and self.kmeans_file.exists():
            try:
                with open(self.vectorizer_file, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                with open(self.kmeans_file, 'rb') as f:
                    self.kmeans = pickle.load(f)
                self.is_trained = True
                logger.info("[PatternClusterer] Loaded trained model")
            except Exception as e:
                logger.warning(f"[PatternClusterer] Error loading model: {e}")
    
    def find_pattern_candidates(self, 
                               fallback_interactions: List[Dict[str, Any]],
                               min_cluster_size: int = 3) -> List[Dict[str, Any]]:
        """
        Find similar interactions that should become patterns
        
        Returns:
            List of pattern candidates with cluster info
        """
        if not fallback_interactions or len(fallback_interactions) < min_cluster_size:
            return []

        if _missing_cluster_deps():
            logger.warning("[PatternClusterer] Missing dependencies (numpy/scikit-learn)")
            return []
        
        try:
            # Group by domain
            by_domain = defaultdict(list)
            for interaction in fallback_interactions:
                domain = interaction.get('domain', 'unknown')
                by_domain[domain].append(interaction)
            
            candidates = []
            
            # Cluster within each domain
            for domain, interactions in by_domain.items():
                if len(interactions) < min_cluster_size:
                    continue
                
                # Extract texts
                texts = [i.get('user_input', '') for i in interactions]
                
                # Vectorize
                if not self.is_trained:
                    self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=100)
                
                vectors = self.vectorizer.fit_transform(texts)
                
                # Cluster
                n_clusters = max(1, len(interactions) // 3)  # Rough heuristic
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(vectors)
                
                # Find clusters with multiple items
                clusters = defaultdict(list)
                for idx, label in enumerate(labels):
                    clusters[label].append(interactions[idx])
                
                # Extract candidates from clusters
                for cluster_id, cluster_items in clusters.items():
                    if len(cluster_items) >= min_cluster_size:
                        # Get most common response
                        responses = [i.get('assistant_response', '') for i in cluster_items]
                        most_common = max(set(responses), key=responses.count) if responses else ""
                        
                        avg_quality = 0.7
                        if np is not None:
                            avg_quality = float(np.mean([i.get('quality_score', 0.7) for i in cluster_items]))
                        else:
                            qualities = [i.get('quality_score', 0.7) for i in cluster_items]
                            avg_quality = sum(qualities) / max(len(qualities), 1)

                        candidate = {
                            'domain': domain,
                            'cluster_id': cluster_id,
                            'size': len(cluster_items),
                            'examples': cluster_items,
                            'suggested_response': most_common,
                            'avg_quality': avg_quality
                        }
                        candidates.append(candidate)
            
            return candidates
        
        except Exception as e:
            logger.warning(f"[PatternClusterer] Clustering error: {e}")
            return []
    
    def train(self, fallback_data: List[Dict[str, Any]]):
        """Train clusterer on fallback interactions"""
        if not fallback_data:
            return False

        if _missing_cluster_deps():
            logger.warning("[PatternClusterer] Missing dependencies (numpy/scikit-learn)")
            return False
        
        try:
            texts = [d.get('user_input', '') for d in fallback_data]
            
            self.vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=100)
            vectors = self.vectorizer.fit_transform(texts)
            
            n_clusters = max(1, len(fallback_data) // 5)
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.kmeans.fit(vectors)
            
            self.is_trained = True
            
            # Save model
            with open(self.vectorizer_file, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            with open(self.kmeans_file, 'wb') as f:
                pickle.dump(self.kmeans, f)
            
            logger.info(f"[PatternClusterer] Trained on {len(fallback_data)} interactions")
            return True
        
        except Exception as e:
            logger.error(f"[PatternClusterer] Training error: {e}")
            return False


# Singletons
_router_classifier: Optional[RouterClassifier] = None
_intent_refiner: Optional[IntentRefiner] = None
_pattern_clusterer: Optional[PatternClusterer] = None


def get_router_classifier(data_dir: str = "data/training") -> RouterClassifier:
    """Get singleton router classifier"""
    global _router_classifier
    if _router_classifier is None:
        _router_classifier = RouterClassifier(data_dir)
    return _router_classifier


def get_intent_refiner(data_dir: str = "data/training") -> IntentRefiner:
    """Get singleton intent refiner"""
    global _intent_refiner
    if _intent_refiner is None:
        _intent_refiner = IntentRefiner(data_dir)
    return _intent_refiner


def get_pattern_clusterer(data_dir: str = "data/training") -> PatternClusterer:
    """Get singleton pattern clusterer"""
    global _pattern_clusterer
    if _pattern_clusterer is None:
        _pattern_clusterer = PatternClusterer(data_dir)
    return _pattern_clusterer
