"""
Response Optimizer for A.L.I.C.E
Optimizes responses for clarity, relevance, and user preference.
Makes A.L.I.C.E's responses more natural and helpful.
"""

import logging
import re
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class ResponseOptimizer:
    """
    Optimizes responses by:
    - Removing redundant information
    - Ensuring clarity and conciseness
    - Adapting to user preferences
    - Fixing common response issues
    """
    
    def __init__(self, world_state=None):
        self.world_state = world_state
        self.user_preferences = {
            "verbosity": "medium",  # low, medium, high
            "formality": "casual",  # casual, formal
            "detail_level": "balanced"  # minimal, balanced, detailed
        }
    
    def optimize(self, response: str, intent: str, context: Dict = None) -> str:
        """Optimize a response for clarity and user preference"""
        if not response:
            return response
        
        optimized = response
        
        # Remove redundant phrases
        redundant_patterns = [
            (r"I can help you with that\.\s*", ""),
            (r"I'll be happy to help\.\s*", ""),
            (r"Let me help you with that\.\s*", ""),
            (r"Sure, I can do that\.\s*", ""),
        ]
        
        for pattern, replacement in redundant_patterns:
            optimized = re.sub(pattern, replacement, optimized, flags=re.IGNORECASE)
        
        # Fix common issues
        # Multiple newlines -> single
        optimized = re.sub(r'\n{3,}', '\n\n', optimized)
        
        # Remove trailing whitespace
        optimized = optimized.strip()
        
        # Ensure response ends properly (skip for multi-line/structured responses)
        if optimized and not optimized[-1] in '.!?' and '\n' not in optimized:
            if '?' in optimized:
                pass  # Question, keep as is
            elif len(optimized.split()) > 10:
                optimized += "."
        
        # Adapt to verbosity preference
        if self.user_preferences["verbosity"] == "low" and len(optimized) > 200:
            # Truncate long responses
            sentences = optimized.split('. ')
            if len(sentences) > 2:
                optimized = '. '.join(sentences[:2]) + '.'
        
        return optimized


_response_optimizer: Optional[ResponseOptimizer] = None


def get_response_optimizer(world_state=None) -> ResponseOptimizer:
    global _response_optimizer
    if _response_optimizer is None:
        _response_optimizer = ResponseOptimizer(world_state=world_state)
    return _response_optimizer
