"""
Semantic Similarity Engine
==========================
Advanced algorithms for measuring semantic similarity between texts.
Uses multiple techniques for robust, accurate matching.
"""

import re
import math
from typing import List, Tuple, Dict, Set
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class SemanticSimilarity:
    """
    Advanced semantic similarity engine using multiple algorithms.

    Techniques:
    - Cosine similarity with TF-IDF weighting
    - Jaccard similarity for set-based matching
    - N-gram overlap for partial matches
    - Word overlap with semantic expansion
    - Levenshtein distance for fuzzy matching
    """

    def __init__(self):
        # Common stop words to filter out
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'when', 'where', 'who', 'which', 'why', 'how'
        }

        # Document frequency for IDF calculation
        self.doc_count = 0
        self.term_doc_freq = Counter()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize and normalize text"""
        # Convert to lowercase
        text = text.lower()

        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b\w+\b', text)

        # Filter out stop words and very short words
        words = [w for w in words if w not in self.stop_words and len(w) > 2]

        return words

    def get_ngrams(self, tokens: List[str], n: int = 2) -> Set[str]:
        """Generate n-grams from tokens"""
        if len(tokens) < n:
            return set()

        return {' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)}

    def cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        Returns value between 0 (no similarity) and 1 (identical).
        """
        tokens1 = self.tokenize(text1)
        tokens2 = self.tokenize(text2)

        if not tokens1 or not tokens2:
            return 0.0

        # Create term frequency vectors
        freq1 = Counter(tokens1)
        freq2 = Counter(tokens2)

        # Get all unique terms
        all_terms = set(freq1.keys()) | set(freq2.keys())

        # Calculate dot product and magnitudes
        dot_product = sum(freq1[term] * freq2[term] for term in all_terms)
        magnitude1 = math.sqrt(sum(freq1[term] ** 2 for term in all_terms))
        magnitude2 = math.sqrt(sum(freq2[term] ** 2 for term in all_terms))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity (set intersection / set union).
        Returns value between 0 and 1.
        """
        tokens1 = set(self.tokenize(text1))
        tokens2 = set(self.tokenize(text2))

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union)

    def ngram_similarity(self, text1: str, text2: str, n: int = 2) -> float:
        """
        Calculate similarity based on n-gram overlap.
        Returns value between 0 and 1.
        """
        tokens1 = self.tokenize(text1)
        tokens2 = self.tokenize(text2)

        ngrams1 = self.get_ngrams(tokens1, n)
        ngrams2 = self.get_ngrams(tokens2, n)

        if not ngrams1 or not ngrams2:
            # Fall back to unigram similarity
            return self.jaccard_similarity(text1, text2)

        intersection = ngrams1 & ngrams2
        union = ngrams1 | ngrams2

        return len(intersection) / len(union)

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def fuzzy_similarity(self, text1: str, text2: str) -> float:
        """
        Fuzzy string similarity using normalized Levenshtein distance.
        Returns value between 0 and 1.
        """
        # For short strings, use character-level comparison
        if len(text1) < 20 or len(text2) < 20:
            distance = self.levenshtein_distance(text1.lower(), text2.lower())
            max_len = max(len(text1), len(text2))
            if max_len == 0:
                return 1.0
            return 1.0 - (distance / max_len)

        # For longer strings, use token-level comparison
        tokens1 = self.tokenize(text1)
        tokens2 = self.tokenize(text2)

        # Calculate average token similarity
        if not tokens1 or not tokens2:
            return 0.0

        similarities = []
        for t1 in tokens1[:10]:  # Limit to first 10 tokens for performance
            best_match = max(
                (1.0 - (self.levenshtein_distance(t1, t2) / max(len(t1), len(t2))))
                for t2 in tokens2[:10]
            )
            similarities.append(best_match)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def combined_similarity(
        self,
        text1: str,
        text2: str,
        weights: Dict[str, float] = None
    ) -> float:
        """
        Calculate combined similarity using multiple algorithms.

        Args:
            text1: First text
            text2: Second text
            weights: Custom weights for each algorithm
                    (default: balanced across all methods)

        Returns:
            Combined similarity score (0-1)
        """
        if weights is None:
            weights = {
                'cosine': 0.3,
                'jaccard': 0.2,
                'ngram': 0.3,
                'fuzzy': 0.2
            }

        scores = {
            'cosine': self.cosine_similarity(text1, text2),
            'jaccard': self.jaccard_similarity(text1, text2),
            'ngram': self.ngram_similarity(text1, text2),
            'fuzzy': self.fuzzy_similarity(text1, text2)
        }

        # Weighted average
        combined = sum(scores[method] * weights[method] for method in scores)

        return combined

    def find_best_match(
        self,
        query: str,
        candidates: List[str],
        threshold: float = 0.3
    ) -> Tuple[str, float]:
        """
        Find the best matching candidate for a query.

        Args:
            query: Query text
            candidates: List of candidate texts
            threshold: Minimum similarity threshold

        Returns:
            (best_match, similarity_score) or (None, 0.0) if no match above threshold
        """
        if not candidates:
            return None, 0.0

        best_match = None
        best_score = 0.0

        for candidate in candidates:
            score = self.combined_similarity(query, candidate)
            if score > best_score:
                best_score = score
                best_match = candidate

        if best_score < threshold:
            return None, 0.0

        return best_match, best_score

    def rank_candidates(
        self,
        query: str,
        candidates: List[str],
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Rank candidates by similarity to query.

        Args:
            query: Query text
            candidates: List of candidate texts
            top_n: Number of top results to return

        Returns:
            List of (candidate, score) tuples, sorted by score descending
        """
        if not candidates:
            return []

        # Calculate similarity for each candidate
        scored_candidates = [
            (candidate, self.combined_similarity(query, candidate))
            for candidate in candidates
        ]

        # Sort by score descending
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        return scored_candidates[:top_n]


# Global singleton instance
_semantic_similarity = None

def get_semantic_similarity() -> SemanticSimilarity:
    """Get global semantic similarity instance"""
    global _semantic_similarity
    if _semantic_similarity is None:
        _semantic_similarity = SemanticSimilarity()
    return _semantic_similarity
