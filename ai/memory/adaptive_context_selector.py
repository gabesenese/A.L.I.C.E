"""
Adaptive Context Selector for A.L.I.C.E
Intelligently selects only relevant context to send to LLM.
Avoids context overload and improves response quality.

Learns from feedback: which context selections lead to good/bad responses.
"""

import logging
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ContextRelevance:
    """Relevance score for a context piece"""
    context_type: str
    content: str
    relevance_score: float
    reason: str
    selection_id: Optional[str] = None  # Track for feedback


@dataclass
class SelectionFeedback:
    """User feedback on a context selection"""
    selection_id: str
    context_type: str
    user_input: str
    intent: str
    success: bool  # True = good response, False = bad response
    rating: int  # 1-5 scale
    timestamp: str
    correction: Optional[str] = None  # User's correction if response was wrong


class AdaptiveContextSelector:
    """
    Selects only the most relevant context pieces based on:
    - Query intent and entities
    - Recent conversation topics
    - Context type (memory vs conversation vs capabilities)
    - Recency and frequency
    - LEARNED patterns: which context selections historically improved response quality
    """
    
    def __init__(self, feedback_path: str = "data/context/selection_feedback.jsonl"):
        self.max_context_length = 2000  # Max chars to send to LLM
        self.min_relevance = 0.3  # Minimum relevance to include
        self.feedback_path = feedback_path
        
        # Learning state: (context_type, intent) → (successes, failures, avg_rating)
        self.context_success_rates: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
            lambda: {"successes": 0, "failures": 0, "total_rating": 0, "count": 0}
        )
        
        # Learning state: (intent, entity_type) → success_rate
        self.entity_context_success: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(
            lambda: {"successes": 0, "failures": 0, "count": 0}
        )
        
        # Selection history: map selection_id → SelectionFeedback
        self.selection_history: Dict[str, SelectionFeedback] = {}
        
        # Load existing feedback
        self._load_feedback()
    
    def _load_feedback(self):
        """Load historical feedback to rebuild learning state"""
        if not os.path.exists(self.feedback_path):
            return
        
        try:
            with open(self.feedback_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    feedback = SelectionFeedback(**data)
                    self.selection_history[feedback.selection_id] = feedback
                    
                    # Update learning state
                    key = (feedback.context_type, feedback.intent)
                    if feedback.success:
                        self.context_success_rates[key]["successes"] += 1
                    else:
                        self.context_success_rates[key]["failures"] += 1
                    self.context_success_rates[key]["total_rating"] += feedback.rating
                    self.context_success_rates[key]["count"] += 1
        except Exception as e:
            logger.warning(f"Failed to load context feedback: {e}")
    
    def _get_selection_id(self, intent: str, context_types: List[str]) -> str:
        """Generate unique ID for this selection for later feedback"""
        import hashlib
        import time
        seed = f"{intent}-{','.join(context_types)}-{time.time()}"
        return hashlib.md5(seed.encode()).hexdigest()[:16]
    
    def record_feedback(
        self,
        selection_id: str,
        context_type: str,
        user_input: str,
        intent: str,
        success: bool,
        rating: int,
        correction: Optional[str] = None
    ):
        """Record user feedback on a context selection"""
        feedback = SelectionFeedback(
            selection_id=selection_id,
            context_type=context_type,
            user_input=user_input,
            intent=intent,
            success=success,
            rating=rating,
            timestamp=datetime.now().isoformat(),
            correction=correction
        )
        
        self.selection_history[selection_id] = feedback
        
        # Update learning state
        key = (context_type, intent)
        if success:
            self.context_success_rates[key]["successes"] += 1
        else:
            self.context_success_rates[key]["failures"] += 1
        self.context_success_rates[key]["total_rating"] += rating
        self.context_success_rates[key]["count"] += 1
        
        # Persist feedback
        self._save_feedback(feedback)
        
        logger.info(f"Recorded feedback for selection {selection_id}: success={success}, rating={rating}")
    
    def _save_feedback(self, feedback: SelectionFeedback):
        """Save feedback to disk"""
        os.makedirs(os.path.dirname(self.feedback_path), exist_ok=True)
        try:
            with open(self.feedback_path, 'a') as f:
                f.write(json.dumps(asdict(feedback)) + '\n')
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
    
    def _get_learned_relevance_boost(self, context_type: str, intent: str) -> float:
        """Return relevance boost based on learned success patterns"""
        key = (context_type, intent)
        stats = self.context_success_rates[key]
        
        if stats["count"] == 0:
            return 0.0  # No history
        
        # Calculate success rate (0-1)
        total = stats["successes"] + stats["failures"]
        if total == 0:
            return 0.0
        
        success_rate = stats["successes"] / total
        
        # Calculate average rating (1-5 → 0-0.2 boost)
        avg_rating = stats["total_rating"] / stats["count"] if stats["count"] > 0 else 3.0
        rating_boost = (avg_rating - 1.0) / 20.0  # Normalize 1-5 to 0-0.2
        
        # Combine: higher success rate and ratings boost relevance
        boost = success_rate * 0.3 + rating_boost * 0.1
        return boost
    
    def select_relevant_context(
        self,
        user_input: str,
        intent: str,
        entities: Dict[str, Any],
        all_context_parts: List[str],
        context_types: List[str] = None
    ) -> Tuple[str, str]:
        """
        Select only relevant context pieces based on query.
        Returns (optimized_context_string, selection_id) for later feedback tracking.
        """
        if not all_context_parts:
            return "", self._get_selection_id(intent, context_types or [])
        
        selection_id = self._get_selection_id(intent, context_types or [])
        
        # Score each context piece
        scored: List[ContextRelevance] = []
        input_lower = user_input.lower()
        
        for i, context_part in enumerate(all_context_parts):
            context_type = context_types[i] if context_types and i < len(context_types) else "general"
            score, reason = self._score_relevance(
                context_part, input_lower, intent, entities, context_type
            )
            
            # Apply learned relevance boost
            learned_boost = self._get_learned_relevance_boost(context_type, intent)
            score += learned_boost
            score = min(score, 1.0)  # Cap at 1.0
            
            if score >= self.min_relevance:
                scored.append(ContextRelevance(
                    context_type=context_type,
                    content=context_part,
                    relevance_score=score,
                    reason=reason,
                    selection_id=selection_id
                ))
        
        # Sort by relevance
        scored.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Build optimized context, respecting max length
        selected = []
        total_length = 0
        
        for item in scored:
            if total_length + len(item.content) <= self.max_context_length:
                selected.append(item.content)
                total_length += len(item.content)
            else:
                # Try to fit a truncated version
                remaining = self.max_context_length - total_length
                if remaining > 100:  # Only if meaningful space left
                    selected.append(item.content[:remaining] + "...")
                break
        
        context_str = "\n\n".join(selected)
        logger.debug(f"Selected context (ID: {selection_id}): {len(scored)} pieces, {total_length} chars")
        return context_str, selection_id
    
    def _score_relevance(
        self,
        context: str,
        query_lower: str,
        intent: str,
        entities: Dict,
        context_type: str
    ) -> Tuple[float, str]:
        """Score how relevant a context piece is to the current query"""
        score = 0.0
        reasons = []
        context_lower = context.lower()
        
        # Intent matching
        if intent and intent in context_lower:
            score += 0.4
            reasons.append("intent_match")
        
        # Entity matching
        if entities:
            for key, value in list(entities.items())[:3]:
                if isinstance(value, list) and value:
                    val_str = str(value[0]).lower()
                else:
                    val_str = str(value).lower()
                if val_str in context_lower and len(val_str) > 2:
                    score += 0.3
                    reasons.append(f"entity_{key}")
        
        # Keyword overlap
        query_words = set(query_lower.split())
        context_words = set(context_lower.split())
        common = query_words & context_words
        if common:
            overlap = len(common) / max(len(query_words), 1)
            score += overlap * 0.2
            if overlap > 0.2:
                reasons.append("keyword_overlap")
        
        # Context type priority
        type_priority = {
            "personalization": 0.1,  # Always include
            "conversation": 0.3,  # High priority
            "memory": 0.25,  # High priority
            "capabilities": 0.1,  # Lower priority
            "general": 0.15
        }
        score += type_priority.get(context_type, 0.1)
        
        # Recency boost (if context mentions "recent" or "today")
        if any(word in context_lower for word in ["recent", "today", "now", "current"]):
            score += 0.15
            reasons.append("recent")
        
        return min(score, 1.0), ", ".join(reasons)
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Return statistics about learned context selection patterns"""
        stats = {
            "total_feedback_samples": len(self.selection_history),
            "context_type_patterns": {}
        }
        
        # Calculate success rates per context type
        for (context_type, intent), data in self.context_success_rates.items():
            if data["count"] == 0:
                continue
            
            total = data["successes"] + data["failures"]
            success_rate = data["successes"] / total if total > 0 else 0
            avg_rating = data["total_rating"] / data["count"] if data["count"] > 0 else 0
            
            key = f"{context_type}:{intent}"
            stats["context_type_patterns"][key] = {
                "success_rate": round(success_rate, 2),
                "avg_rating": round(avg_rating, 2),
                "samples": data["count"]
            }
        
        return stats


_selector: Optional[AdaptiveContextSelector] = None


def get_context_selector() -> AdaptiveContextSelector:
    global _selector
    if _selector is None:
        _selector = AdaptiveContextSelector()
    return _selector
