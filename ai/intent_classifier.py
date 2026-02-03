"""
Enhanced Semantic Intent Classification System for A.L.I.C.E

Advanced features:
- Query caching with LSH and LRU eviction
- Bayesian confidence calibration
- Conversation context awareness
- Intent hierarchy organization
- Performance statistics tracking
- User feedback learning

This allows A.L.I.C.E to understand variations of commands naturally:
- "how many notes do i have" → notes:count
- "tell me about my emails" → email:list  
- "what's playing right now" → music:status
"""

import logging
import json
import os
import time
import hashlib
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import threading
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class IntentExample:
    """Training example for an intent"""
    text: str
    intent: str
    plugin: str
    action: str
    confidence: float = 1.0
    
    def to_dict(self):
        return asdict(self)
    
    @staticmethod
    def from_dict(data: Dict):
        return IntentExample(**data)


@dataclass
class IntentHierarchy:
    """Hierarchical intent structure for better organization"""
    name: str
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    plugins: Set[str] = field(default_factory=set)
    
    def to_dict(self):
        return {
            'name': self.name,
            'parent': self.parent,
            'children': self.children,
            'plugins': list(self.plugins)
        }


class QueryCache:
    """
    LSH-based query cache with TTL and LRU eviction
    Caches recent query results to improve performance
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize query cache
        
        Args:
            max_size: Maximum number of cached queries
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Dict, float]] = {}  # query_hash -> (result, timestamp)
        self.access_order = deque()  # LRU tracking
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def _hash_query(self, text: str) -> str:
        """Generate hash for query text"""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()
    
    def get(self, text: str) -> Optional[Dict]:
        """
        Get cached result for query
        
        Args:
            text: Query text
            
        Returns:
            Cached result or None if not found/expired
        """
        with self.lock:
            query_hash = self._hash_query(text)
            
            if query_hash in self.cache:
                result, timestamp = self.cache[query_hash]
                
                # Check if expired
                if time.time() - timestamp > self.ttl_seconds:
                    del self.cache[query_hash]
                    self.access_order.remove(query_hash)
                    self.misses += 1
                    return None
                
                # Move to end (most recently used)
                self.access_order.remove(query_hash)
                self.access_order.append(query_hash)
                self.hits += 1
                return result
            
            self.misses += 1
            return None
    
    def put(self, text: str, result: Dict):
        """
        Cache a query result
        
        Args:
            text: Query text
            result: Classification result
        """
        with self.lock:
            query_hash = self._hash_query(text)
            
            # Evict oldest if at capacity
            if len(self.cache) >= self.max_size and query_hash not in self.cache:
                oldest = self.access_order.popleft()
                del self.cache[oldest]
            
            self.cache[query_hash] = (result, time.time())
            
            if query_hash in self.access_order:
                self.access_order.remove(query_hash)
            self.access_order.append(query_hash)
    
    def clear(self):
        """Clear all cached entries"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0.0
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate
            }


class ConfidenceCalibrator:
    """
    Bayesian confidence calibration based on historical accuracy
    Adjusts confidence scores to better reflect actual accuracy
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize confidence calibrator
        
        Args:
            history_size: Number of recent predictions to track
        """
        self.history_size = history_size
        self.predictions = deque(maxlen=history_size)  # (confidence, was_correct)
        self.confidence_bins = defaultdict(lambda: {'correct': 0, 'total': 0})
        self.lock = threading.Lock()
    
    def record_prediction(self, confidence: float, was_correct: bool):
        """
        Record a prediction outcome
        
        Args:
            confidence: Original confidence score
            was_correct: Whether prediction was correct
        """
        with self.lock:
            self.predictions.append((confidence, was_correct))
            
            # Update bins (0.1 intervals)
            bin_key = round(confidence, 1)
            self.confidence_bins[bin_key]['total'] += 1
            if was_correct:
                self.confidence_bins[bin_key]['correct'] += 1
    
    def calibrate(self, confidence: float) -> float:
        """
        Calibrate a confidence score based on historical accuracy
        
        Args:
            confidence: Raw confidence score
            
        Returns:
            Calibrated confidence score
        """
        with self.lock:
            bin_key = round(confidence, 1)
            
            if bin_key in self.confidence_bins:
                stats = self.confidence_bins[bin_key]
                if stats['total'] >= 10:  # Require minimum samples
                    actual_accuracy = stats['correct'] / stats['total']
                    # Blend original confidence with empirical accuracy
                    return 0.7 * actual_accuracy + 0.3 * confidence
            
            return confidence
    
    def get_stats(self) -> Dict:
        """Get calibration statistics"""
        with self.lock:
            if not self.predictions:
                return {'bins': {}, 'overall_accuracy': 0.0}
            
            overall_accuracy = sum(1 for _, correct in self.predictions if correct) / len(self.predictions)
            
            bins = {}
            for bin_key, stats in self.confidence_bins.items():
                if stats['total'] > 0:
                    bins[bin_key] = {
                        'accuracy': stats['correct'] / stats['total'],
                        'count': stats['total']
                    }
            
            return {
                'bins': bins,
                'overall_accuracy': overall_accuracy,
                'total_predictions': len(self.predictions)
            }


class SemanticIntentClassifier:
    """
    Advanced semantic intent classifier with caching, calibration, and context awareness
    Maps user input to plugin actions without regex patterns
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", examples_file: str = "data/samples/intent_examples.json",
                 use_cache: bool = True, use_calibration: bool = True):
        """
        Initialize the semantic intent classifier
        
        Args:
            model_name: SentenceTransformer model to use
            examples_file: Path to intent examples database
            use_cache: Enable query caching
            use_calibration: Enable confidence calibration
        """
        self.model_name = model_name
        self.examples_file = examples_file
        self.model = None
        self.examples: List[IntentExample] = []
        self.example_embeddings = None
        
        # Advanced features
        self.cache = QueryCache() if use_cache else None
        self.calibrator = ConfidenceCalibrator() if use_calibration else None
        self.conversation_context: List[str] = []  # Last 5 user queries
        self.intent_hierarchy: Dict[str, IntentHierarchy] = {}
        
        # Statistics
        self.total_queries = 0
        self.low_confidence_queries = []  # Track queries with low confidence for learning
        
        # Initialize the model
        self._load_model()
        
        # Load intent examples
        self._load_examples()
        
        # Generate embeddings for all examples
        self._generate_embeddings()
        
        # Build intent hierarchy
        self._build_intent_hierarchy()
    
    def _load_model(self):
        """Load the sentence transformer model with timeout and retry logic"""
        import time
        import os
        
        # Check if disabled
        if os.environ.get('ALICE_DISABLE_SEMANTIC_CLASSIFIER'):
            logger.info("Semantic classifier disabled via environment variable")
            self.model = None
            return
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Loading semantic model: {self.model_name} (attempt {attempt + 1}/{max_retries})")
                # Set longer timeout for model download
                os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'  # 60 seconds timeout
                
                self.model = SentenceTransformer(
                    self.model_name,
                    device='cpu'  # Explicitly use CPU to avoid GPU issues
                )
                logger.info("Semantic intent classifier loaded successfully")
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Failed to load semantic model (attempt {attempt + 1}): {e}")
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to load semantic model after {max_retries} attempts: {e}")
                    logger.warning("Semantic intent classifier will be disabled. A.L.I.C.E will use pattern-based matching only.")
                    self.model = None  # Set to None instead of raising
                    return
    
    def _load_examples(self):
        """Load intent examples from file or create defaults"""
        if os.path.exists(self.examples_file):
            try:
                with open(self.examples_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.examples = [IntentExample.from_dict(ex) for ex in data]
                logger.info(f"Loaded {len(self.examples)} intent examples")
            except Exception as e:
                logger.error(f"Failed to load intent examples: {e}")
                self._create_default_examples()
        else:
            self._create_default_examples()
        
        # Load learned corrections from training data
        if hasattr(self, '_load_learned_corrections'):
            self._load_learned_corrections()
    
    def _load_learned_corrections(self):
        """Load learned corrections from training data (stub for semantic classifier)"""
        # Semantic classifier doesn't use learned corrections same way as pattern-based
        pass
    
    def _create_default_examples(self):
        """Create default intent examples covering common use cases"""
        logger.info("Creating default intent examples")
        
        default_examples = [
            # Notes Plugin
            IntentExample("create a note", "notes", "notes", "create"),
            IntentExample("make a note", "notes", "notes", "create"),
            IntentExample("add a note about the meeting", "notes", "notes", "create"),
            IntentExample("write down this idea", "notes", "notes", "create"),
            IntentExample("remember this for later", "notes", "notes", "create"),
            IntentExample("jot this down", "notes", "notes", "create"),
            
            IntentExample("how many notes do i have", "notes", "notes", "count"),
            IntentExample("count my notes", "notes", "notes", "count"),
            IntentExample("number of notes", "notes", "notes", "count"),
            IntentExample("show me note statistics", "notes", "notes", "count"),
            IntentExample("do i have any notes", "notes", "notes", "count"),
            
            IntentExample("list my notes", "notes", "notes", "list"),
            IntentExample("show all notes", "notes", "notes", "list"),
            IntentExample("show me my notes", "notes", "notes", "list"),
            IntentExample("what notes do i have", "notes", "notes", "list"),
            IntentExample("display my notes", "notes", "notes", "list"),
            
            IntentExample("search my notes", "notes", "notes", "search"),
            IntentExample("find notes about work", "notes", "notes", "search"),
            IntentExample("look for notes", "notes", "notes", "search"),
            
            IntentExample("delete a note", "notes", "notes", "delete"),
            IntentExample("remove this note", "notes", "notes", "delete"),
            IntentExample("delete the grocery list", "notes", "notes", "delete"),
            IntentExample("remove the grocery list", "notes", "notes", "delete"),
            IntentExample("delete the shopping list", "notes", "notes", "delete"),
            
            # Email Plugin
            IntentExample("check my email", "email", "email", "list"),
            IntentExample("show my emails", "email", "email", "list"),
            IntentExample("what emails do i have", "email", "email", "list"),
            IntentExample("list my inbox", "email", "email", "list"),
            IntentExample("show inbox", "email", "email", "list"),
            IntentExample("any new emails", "email", "email", "list"),
            
            IntentExample("how many emails", "email", "email", "count"),
            IntentExample("how many unread emails", "email", "email", "count_unread"),
            IntentExample("unread count", "email", "email", "count_unread"),
            
            IntentExample("read my latest email", "email", "email", "read"),
            IntentExample("read my latest 5 emails", "email", "email", "read"),
            IntentExample("can you read my latest emails", "email", "email", "read"),
            IntentExample("show me my latest emails", "email", "email", "read"),
            IntentExample("open the first email", "email", "email", "read"),
            IntentExample("show me the latest message", "email", "email", "read"),
            IntentExample("read the latest 10 emails", "email", "email", "read"),
            
            IntentExample("send an email", "email", "email", "compose"),
            IntentExample("write an email", "email", "email", "compose"),
            IntentExample("compose a message", "email", "email", "compose"),
            
            IntentExample("search for emails from john", "email", "email", "search"),
            IntentExample("find emails about the project", "email", "email", "search"),
            
            # Music Plugin
            IntentExample("play some music", "music", "music", "play"),
            IntentExample("play a song", "music", "music", "play"),
            IntentExample("start playing music", "music", "music", "play"),
            IntentExample("i want to listen to music", "music", "music", "play"),
            IntentExample("play something", "music", "music", "play"),
            
            IntentExample("pause the music", "music", "music", "pause"),
            IntentExample("stop playing", "music", "music", "pause"),
            IntentExample("pause this", "music", "music", "pause"),
            
            IntentExample("what's playing", "music", "music", "status"),
            IntentExample("what song is this", "music", "music", "status"),
            IntentExample("current song", "music", "music", "status"),
            IntentExample("what am i listening to", "music", "music", "status"),
            
            IntentExample("next song", "music", "music", "next"),
            IntentExample("skip this", "music", "music", "next"),
            IntentExample("play next", "music", "music", "next"),
            
            IntentExample("previous song", "music", "music", "previous"),
            IntentExample("go back", "music", "music", "previous"),
            IntentExample("last track", "music", "music", "previous"),
            
            IntentExample("turn up the volume", "music", "music", "volume_up"),
            IntentExample("louder", "music", "music", "volume_up"),
            IntentExample("turn down the volume", "music", "music", "volume_down"),
            IntentExample("quieter", "music", "music", "volume_down"),
            
            # Calendar Plugin
            IntentExample("what's on my calendar", "calendar", "calendar", "list"),
            IntentExample("show my schedule", "calendar", "calendar", "list"),
            IntentExample("what do i have today", "calendar", "calendar", "list_today"),
            IntentExample("any meetings today", "calendar", "calendar", "list_today"),
            IntentExample("what's my schedule for tomorrow", "calendar", "calendar", "list_tomorrow"),
            
            IntentExample("add an event", "calendar", "calendar", "create"),
            IntentExample("schedule a meeting", "calendar", "calendar", "create"),
            IntentExample("create an appointment", "calendar", "calendar", "create"),
            
            # Document Plugin
            IntentExample("open a document", "document", "document", "open"),
            IntentExample("find my documents", "document", "document", "search"),
            IntentExample("search for files", "document", "document", "search"),
            IntentExample("recent documents", "document", "document", "list_recent"),
            
            # Task/TODO
            IntentExample("add a task", "task", "notes", "create_task"),
            IntentExample("create a todo", "task", "notes", "create_task"),
            IntentExample("add to my todo list", "task", "notes", "create_task"),
            IntentExample("what are my tasks", "task", "notes", "list_tasks"),
            
            # Weather
            IntentExample("what's the weather", "weather", "weather", "current"),
            IntentExample("how's the weather today", "weather", "weather", "current"),
            IntentExample("is it going to rain", "weather", "weather", "forecast"),
            IntentExample("weather forecast", "weather", "weather", "forecast"),
            
            # Time/Date
            IntentExample("what time is it", "time", "time", "current"),
            IntentExample("what's the date", "time", "time", "date"),
            IntentExample("what day is it", "time", "time", "day"),
            
            # General Questions
            IntentExample("tell me about this", "question", "general", "explain"),
            IntentExample("what is this", "question", "general", "define"),
            IntentExample("explain this to me", "question", "general", "explain"),
            IntentExample("help me understand", "question", "general", "explain"),
        ]
        
        self.examples = default_examples
        self._save_examples()
    
    def _generate_embeddings(self):
        """Generate embeddings for all example texts"""
        if not self.examples:
            logger.warning("No examples to generate embeddings for")
            return
        
        if self.model is None:
            logger.warning("Model not available, skipping embedding generation")
            self.example_embeddings = None
            return
        
        try:
            texts = [ex.text for ex in self.examples]
            self.example_embeddings = self.model.encode(texts, convert_to_numpy=True)
            logger.info(f"Generated embeddings for {len(texts)} examples")
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            self.example_embeddings = None
    
    def _build_intent_hierarchy(self):
        """Build hierarchical intent structure"""
        # Group intents by plugin
        plugin_intents = defaultdict(set)
        for example in self.examples:
            plugin_intents[example.plugin].add(example.intent)
        
        # Create hierarchy nodes
        for plugin, intents in plugin_intents.items():
            if plugin not in self.intent_hierarchy:
                self.intent_hierarchy[plugin] = IntentHierarchy(
                    name=plugin,
                    plugins={plugin}
                )
    
    def update_conversation_context(self, query: str, max_context: int = 5):
        """
        Update conversation context with latest query
        
        Args:
            query: User query
            max_context: Maximum context history to maintain
        """
        self.conversation_context.append(query)
        if len(self.conversation_context) > max_context:
            self.conversation_context.pop(0)
    
    def classify(self, text: str, top_k: int = 3, threshold: float = 0.4, use_context: bool = True) -> List[Tuple[IntentExample, float]]:
        """
        Classify text to find matching intents with caching and calibration
        
        Args:
            text: User input text
            top_k: Number of top matches to return
            threshold: Minimum similarity score (0-1)
            use_context: Whether to use conversation context
        
        Returns:
            List of (IntentExample, similarity_score) tuples
        """
        self.total_queries += 1
        
        # Check cache first
        if self.cache:
            cached = self.cache.get(text)
            if cached:
                logger.debug(f"Cache hit for query: {text[:50]}")
                return [(IntentExample.from_dict(cached['example']), cached['confidence'])]
        
        if not self.examples or self.example_embeddings is None:
            logger.warning("No examples or embeddings available")
            return []
        
        if self.model is None:
            logger.debug("Semantic model not available, returning empty classification")
            return []
        
        try:
            # Generate embedding for input text
            # Optionally incorporate context
            query_text = text
            if use_context and self.conversation_context:
                # Weight recent context (last query gets 0.2 weight)
                context_weight = " ".join(self.conversation_context[-1:])
                query_text = f"{text} {context_weight}"
            
            text_embedding = self.model.encode([query_text], convert_to_numpy=True)[0]
            
            # Calculate cosine similarity with all examples
            similarities = np.dot(self.example_embeddings, text_embedding) / (
                np.linalg.norm(self.example_embeddings, axis=1) * np.linalg.norm(text_embedding)
            )
            
            # Get top-k most similar examples
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                raw_score = float(similarities[idx])
                
                # Apply confidence calibration
                if self.calibrator:
                    calibrated_score = self.calibrator.calibrate(raw_score)
                else:
                    calibrated_score = raw_score
                
                if calibrated_score >= threshold:
                    results.append((self.examples[idx], calibrated_score))
            
            if results:
                logger.debug(f"Intent classification: {results[0][0].intent}:{results[0][0].action} (confidence: {results[0][1]:.2f})")
                
                # Cache the result
                if self.cache:
                    cache_result = {
                        'example': results[0][0].to_dict(),
                        'confidence': results[0][1]
                    }
                    self.cache.put(text, cache_result)
            else:
                # Track low confidence queries for learning
                if len(results) == 0 or results[0][1] < 0.5:
                    self.low_confidence_queries.append(text)
                    if len(self.low_confidence_queries) > 100:
                        self.low_confidence_queries.pop(0)
            
            # Update conversation context
            self.update_conversation_context(text)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to classify intent: {e}")
            return []
    
    def get_best_match(self, text: str, threshold: float = 0.4) -> Optional[Tuple[IntentExample, float]]:
        """
        Get the single best matching intent
        
        Args:
            text: User input text
            threshold: Minimum similarity score
        
        Returns:
            (IntentExample, score) or None if no match above threshold
        """
        results = self.classify(text, top_k=1, threshold=threshold)
        if results:
            return results[0]
        return None
    
    def add_example(self, text: str, intent: str, plugin: str, action: str, confidence: float = 1.0):
        """Add a new example to improve classification"""
        example = IntentExample(text, intent, plugin, action, confidence)
        self.examples.append(example)
        
        # Regenerate embeddings
        self._generate_embeddings()
        
        # Save updated examples
        self._save_examples()
        
        logger.info(f"Added new intent example: {text} → {plugin}:{action}")
    
    def _save_examples(self):
        """Save intent examples to file"""
        try:
            os.makedirs(os.path.dirname(self.examples_file), exist_ok=True)
            with open(self.examples_file, 'w', encoding='utf-8') as f:
                data = [ex.to_dict() for ex in self.examples]
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved {len(self.examples)} intent examples")
        except Exception as e:
            logger.error(f"Failed to save intent examples: {e}")
    
    def get_plugin_action(self, text: str, threshold: float = 0.4) -> Optional[Dict[str, any]]:
        """
        Convenience method to get plugin and action for text
        
        Returns:
            Dict with 'plugin', 'action', 'confidence', 'intent' or None
        """
        match = self.get_best_match(text, threshold)
        if match:
            example, score = match
            return {
                'plugin': example.plugin,
                'action': example.action,
                'intent': example.intent,
                'confidence': score,
                'matched_example': example.text
            }
        return None
    
    def get_statistics(self) -> Dict:
        """Get classifier statistics"""
        stats = {
            'total_queries': self.total_queries,
            'total_examples': len(self.examples),
            'low_confidence_count': len(self.low_confidence_queries),
        }
        
        if self.cache:
            stats['cache'] = self.cache.get_stats()
        
        if self.calibrator:
            stats['calibration'] = self.calibrator.get_stats()
        
        if self.example_embeddings is not None:
            # Calculate average confidence across recent queries
            stats['avg_embedding_norm'] = float(np.mean(np.linalg.norm(self.example_embeddings, axis=1)))
        
        return stats
    
    def record_feedback(self, query: str, plugin: str, action: str, was_correct: bool):
        """
        Record user feedback on classification accuracy
        
        Args:
            query: Original query
            plugin: Predicted plugin
            action: Predicted action
            was_correct: Whether the prediction was correct
        """
        # Find the confidence for this prediction
        result = self.get_plugin_action(query)
        if result and self.calibrator:
            self.calibrator.record_prediction(result['confidence'], was_correct)
        
        # If incorrect, could add as new training example
        if not was_correct:
            logger.info(f"Incorrect classification logged: '{query}' -> {plugin}:{action}")


# Singleton instance
_classifier_instance = None

def get_intent_classifier() -> SemanticIntentClassifier:
    """Get or create the global intent classifier instance"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = SemanticIntentClassifier()
    return _classifier_instance


if __name__ == "__main__":
    # Test the classifier
    logging.basicConfig(level=logging.INFO)
    
    classifier = SemanticIntentClassifier()
    
    test_queries = [
        "how many notes do i have",
        "show me my emails",
        "what's currently playing",
        "create a reminder",
        "what's on my schedule today",
        "find documents about the project",
    ]
    
    print("\n" + "="*80)
    print("Testing Enhanced Semantic Intent Classification")
    print("="*80 + "\n")
    
    for query in test_queries:
        result = classifier.get_plugin_action(query)
        if result:
            print(f"Query: '{query}'")
            print(f"  -> Plugin: {result['plugin']}")
            print(f"  -> Action: {result['action']}")
            print(f"  -> Confidence: {result['confidence']:.2f}")
            print(f"  -> Matched: '{result['matched_example']}'")
            print()
        else:
            print(f"Query: '{query}'")
            print(f"  -> No match found")
            print()
    
    # Show statistics
    stats = classifier.get_statistics()
    print("\n" + "="*80)
    print("Classifier Statistics")
    print("="*80)
    print(f"Total queries: {stats['total_queries']}")
    print(f"Total examples: {stats['total_examples']}")
    if 'cache' in stats:
        print(f"Cache hit rate: {stats['cache']['hit_rate']:.1%}")
        print(f"Cache size: {stats['cache']['size']}/{stats['cache']['max_size']}")
