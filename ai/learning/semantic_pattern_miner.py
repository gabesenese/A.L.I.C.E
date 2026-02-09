"""
Enhanced Pattern Miner with Semantic Clustering

Extends the basic pattern miner with:
- Semantic similarity using TF-IDF vectorization or simple embeddings
- Smarter clustering that finds semantically similar intents
- Quality scoring based on cluster cohesion
"""

import json
import os
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict, Counter
from datetime import datetime
import math
import logging

logger = logging.getLogger(__name__)


class SimpleEmbedding:
    """Simple TF-IDF style embedding for semantic similarity"""
    
    @staticmethod
    def vectorize(text: str) -> Dict[str, float]:
        """
        Convert text to simple word-frequency vector.
        Returns dict of word -> frequency
        """
        # Normalize and tokenize
        words = re.findall(r'\b[a-z]+\b', text.lower())
        
        # Count frequencies
        freq = Counter(words)
        total = sum(freq.values())
        
        # Return normalized frequencies
        return {word: count / total for word, count in freq.items()}
    
    @staticmethod
    def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        Returns 0-1 score.
        """
        # Get all keys
        all_keys = set(vec1.keys()) | set(vec2.keys())
        
        if not all_keys:
            return 0.0
        
        # Calculate dot product
        dot_product = sum(vec1.get(key, 0) * vec2.get(key, 0) for key in all_keys)
        
        # Calculate magnitudes
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)


class SemanticPatternMiner:
    """
    Advanced pattern miner using semantic clustering.
    
    Detects not just similar strings but semantically similar intents.
    Example: "email me" and "send me a message" map to email_send intent.
    """
    
    def __init__(self, logged_interactions_path: str = "data/training/logged_interactions.jsonl",
                 semantic_patterns_path: str = "data/training/semantic_patterns.json",
                 clustering_threshold: float = 0.6):
        """
        Initialize semantic pattern miner.
        
        Args:
            logged_interactions_path: Path to logged interactions
            semantic_patterns_path: Path to save semantic patterns
            clustering_threshold: Similarity threshold for clustering (0-1)
        """
        self.logged_interactions_path = logged_interactions_path
        self.semantic_patterns_path = semantic_patterns_path
        self.clustering_threshold = clustering_threshold
        self.embedding_cache = {}
        self.semantic_patterns = self._load_semantic_patterns()
    
    def _load_semantic_patterns(self) -> Dict[str, Any]:
        """Load existing semantic patterns"""
        if os.path.exists(self.semantic_patterns_path):
            with open(self.semantic_patterns_path, 'r') as f:
                return json.load(f)
        return {
            "patterns": [],
            "intent_clusters": {},
            "metadata": {
                "total_patterns": 0,
                "total_interactions_analyzed": 0,
                "last_updated": datetime.now().isoformat()
            }
        }
    
    def _save_semantic_patterns(self):
        """Save semantic patterns to disk"""
        os.makedirs(os.path.dirname(self.semantic_patterns_path) or '.', exist_ok=True)
        with open(self.semantic_patterns_path, 'w') as f:
            json.dump(self.semantic_patterns, f, indent=2)
    
    def _load_interactions(self) -> List[Dict[str, Any]]:
        """Load logged interactions"""
        interactions = []
        if os.path.exists(self.logged_interactions_path):
            with open(self.logged_interactions_path, 'r') as f:
                for line in f:
                    if line.strip():
                        interactions.append(json.loads(line))
        return interactions
    
    def _get_embedding(self, text: str) -> Dict[str, float]:
        """Get cached embedding for text"""
        if text not in self.embedding_cache:
            self.embedding_cache[text] = SimpleEmbedding.vectorize(text)
        return self.embedding_cache[text]
    
    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        vec1 = self._get_embedding(text1)
        vec2 = self._get_embedding(text2)
        return SimpleEmbedding.cosine_similarity(vec1, vec2)
    
    def cluster_by_semantics(self, interactions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Cluster interactions by semantic similarity of user inputs.
        
        Returns:
            Dict mapping cluster ID to list of interactions
        """
        if not interactions:
            return {}
        
        clusters = {}
        cluster_id = 0
        unclustered = interactions.copy()
        
        while unclustered:
            # Start new cluster with first unclustered item
            seed = unclustered.pop(0)
            cluster_key = f"cluster_{cluster_id}"
            cluster = [seed]
            
            # Find all similar items
            remaining = []
            seed_text = seed.get('user_input', '')
            
            for interaction in unclustered:
                interaction_text = interaction.get('user_input', '')
                similarity = self._similarity(seed_text, interaction_text)
                
                if similarity >= self.clustering_threshold:
                    cluster.append(interaction)
                else:
                    remaining.append(interaction)
            
            unclustered = remaining
            
            # Store cluster if it has enough items
            if len(cluster) >= 2:  # Only store clusters with 2+ items
                clusters[cluster_key] = cluster
                cluster_id += 1
        
        return clusters
    
    def extract_pattern_from_cluster(self, cluster_key: str, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract a pattern from a semantic cluster.
        
        Args:
            cluster_key: Cluster identifier
            interactions: List of interactions in cluster
            
        Returns:
            Pattern definition
        """
        # Extract common elements
        intents = [i.get('intent', 'unknown') for i in interactions]
        intent_counter = Counter(intents)
        most_common_intent = intent_counter.most_common(1)[0][0]
        
        user_inputs = [i.get('user_input', '') for i in interactions]
        responses = [i.get('response', '') for i in interactions]
        tools = [i.get('tool') for i in interactions if i.get('tool')]
        
        # Calculate pattern quality
        intent_agreement = intent_counter.most_common(1)[0][1] / len(intents)
        cluster_cohesion = sum(
            self._similarity(user_inputs[0], inp) 
            for inp in user_inputs[1:]
        ) / max(len(user_inputs) - 1, 1)
        
        quality_score = (intent_agreement + cluster_cohesion) / 2
        
        pattern = {
            'id': cluster_key,
            'intent': most_common_intent,
            'cluster_size': len(interactions),
            'examples': user_inputs[:3],
            'template_response': responses[0] if responses else "",
            'primary_tool': Counter(tools).most_common(1)[0][0] if tools else None,
            'intent_agreement': intent_agreement,
            'cluster_cohesion': cluster_cohesion,
            'quality_score': quality_score,
            'created_at': datetime.now().isoformat()
        }
        
        return pattern
    
    def mine_semantic_patterns(self) -> List[Dict[str, Any]]:
        """
        Mine semantic patterns from logged interactions.
        
        Returns:
            List of discovered patterns
        """
        interactions = self._load_interactions()
        if not interactions:
            logger.warning("No interactions to mine patterns from")
            return []
        
        logger.info(f"Mining patterns from {len(interactions)} interactions")
        
        # Cluster by semantics
        clusters = self.cluster_by_semantics(interactions)
        logger.info(f"Found {len(clusters)} semantic clusters")
        
        # Extract patterns from clusters
        patterns = []
        for cluster_key, cluster_items in clusters.items():
            pattern = self.extract_pattern_from_cluster(cluster_key, cluster_items)
            patterns.append(pattern)
        
        # Sort by quality
        patterns.sort(key=lambda p: p['quality_score'], reverse=True)
        
        return patterns
    
    def update_patterns(self, min_quality: float = 0.6):
        """
        Update semantic patterns from logged interactions.
        
        Args:
            min_quality: Minimum quality score to save pattern
        """
        patterns = self.mine_semantic_patterns()
        
        # Filter by quality
        high_quality_patterns = [p for p in patterns if p['quality_score'] >= min_quality]
        
        # Update metadata
        self.semantic_patterns['patterns'] = high_quality_patterns
        self.semantic_patterns['metadata']['total_patterns'] = len(high_quality_patterns)
        self.semantic_patterns['metadata']['total_interactions_analyzed'] = len(self._load_interactions())
        self.semantic_patterns['metadata']['last_updated'] = datetime.now().isoformat()
        
        # Build intent clusters
        intent_clusters = defaultdict(list)
        for pattern in high_quality_patterns:
            intent_clusters[pattern['intent']].append(pattern)
        self.semantic_patterns['intent_clusters'] = dict(intent_clusters)
        
        self._save_semantic_patterns()
        
        logger.info(f"Updated {len(high_quality_patterns)} high-quality semantic patterns")
        return high_quality_patterns
    
    def get_intent_patterns(self, intent: str) -> List[Dict[str, Any]]:
        """Get all patterns for a specific intent"""
        return self.semantic_patterns['intent_clusters'].get(intent, [])
    
    def get_similar_patterns(self, user_input: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Find k most similar patterns to user input.
        
        Args:
            user_input: User input text
            k: Number of patterns to return
            
        Returns:
            List of similar patterns ranked by similarity
        """
        input_vec = self._get_embedding(user_input)
        
        similarities = []
        for pattern in self.semantic_patterns['patterns']:
            for example in pattern.get('examples', []):
                sim = SimpleEmbedding.cosine_similarity(
                    input_vec,
                    self._get_embedding(example)
                )
                similarities.append((pattern, sim))
        
        # Return top-k unique patterns
        seen = set()
        results = []
        for pattern, sim in sorted(similarities, key=lambda x: x[1], reverse=True):
            if pattern['id'] not in seen:
                results.append(pattern)
                seen.add(pattern['id'])
                if len(results) >= k:
                    break
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get mining statistics"""
        patterns = self.semantic_patterns['patterns']
        intents = Counter(p['intent'] for p in patterns)
        
        return {
            'total_patterns': len(patterns),
            'average_quality': sum(p['quality_score'] for p in patterns) / len(patterns) if patterns else 0,
            'intent_distribution': dict(intents),
            'top_intents': intents.most_common(5),
            'last_updated': self.semantic_patterns['metadata'].get('last_updated')
        }
