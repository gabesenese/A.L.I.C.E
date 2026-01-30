"""
Adaptive Context Selector for A.L.I.C.E
Intelligently selects only relevant context to send to LLM.
Avoids context overload and improves response quality.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ContextRelevance:
    """Relevance score for a context piece"""
    context_type: str
    content: str
    relevance_score: float
    reason: str


class AdaptiveContextSelector:
    """
    Selects only the most relevant context pieces based on:
    - Query intent and entities
    - Recent conversation topics
    - Context type (memory vs conversation vs capabilities)
    - Recency and frequency
    """
    
    def __init__(self):
        self.max_context_length = 2000  # Max chars to send to LLM
        self.min_relevance = 0.3  # Minimum relevance to include
    
    def select_relevant_context(
        self,
        user_input: str,
        intent: str,
        entities: Dict[str, Any],
        all_context_parts: List[str],
        context_types: List[str] = None
    ) -> str:
        """
        Select only relevant context pieces based on query.
        Returns optimized context string.
        """
        if not all_context_parts:
            return ""
        
        # Score each context piece
        scored: List[ContextRelevance] = []
        input_lower = user_input.lower()
        
        for i, context_part in enumerate(all_context_parts):
            context_type = context_types[i] if context_types and i < len(context_types) else "general"
            score, reason = self._score_relevance(
                context_part, input_lower, intent, entities, context_type
            )
            
            if score >= self.min_relevance:
                scored.append(ContextRelevance(
                    context_type=context_type,
                    content=context_part,
                    relevance_score=score,
                    reason=reason
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
        
        return "\n\n".join(selected)
    
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


_selector: Optional[AdaptiveContextSelector] = None


def get_context_selector() -> AdaptiveContextSelector:
    global _selector
    if _selector is None:
        _selector = AdaptiveContextSelector()
    return _selector
