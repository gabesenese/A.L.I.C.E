"""
Ollama Teacher Mode for A.L.I.C.E

Provides "ideal" responses from Ollama to compare against Alice's actual responses,
enabling automated learning from discrepancies.
"""

import json
import logging
from typing import Optional, Dict, Any
from ai.llm_engine import LocalLLMEngine, LLMConfig

logger = logging.getLogger(__name__)


TEACHER_SYSTEM_PROMPT = """You are defining the ideal assistant behavior for ALICE, an intelligent personal assistant.

Your role is to provide the PERFECT response that ALICE should give to the user's input.

Guidelines:
1. Be concise, helpful, and direct
2. For tool/action requests, describe what ALICE should do (e.g., "I'll check your emails now")
3. For conversational inputs, provide warm, natural responses
4. For vague/ambiguous inputs, suggest clarifying questions
5. Match the user's tone and formality level
6. Focus on being helpful without being verbose

Respond ONLY with the ideal assistant response - no explanations, no meta-commentary.
"""


class TeacherMode:
    """
    Ollama-based teacher that provides ideal responses for comparison
    """
    
    def __init__(self, model: str = "llama3.1:8b"):
        """
        Initialize teacher mode
        
        Args:
            model: Ollama model to use for teacher responses
        """
        self.model = model
        self.llm_config = LLMConfig(model=model)
        self.llm = LocalLLMEngine(self.llm_config)
        logger.info(f"Teacher mode initialized with model: {model}")
    
    def get_ideal_response(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Get ideal response from teacher for a user input
        
        Args:
            user_input: User's input to get ideal response for
            context: Optional context (previous messages, domain, etc.)
        
        Returns:
            Ideal response from teacher, or None if error
        """
        try:
            # Build prompt with context if provided
            prompt = f"{TEACHER_SYSTEM_PROMPT}\n\nUser input: {user_input}"
            
            if context:
                if "domain" in context:
                    prompt = f"Domain: {context['domain']}\n{prompt}"
                if "previous_exchange" in context:
                    prompt = f"Context: {context['previous_exchange']}\n{prompt}"
            
            # Get teacher response using correct API
            response = self.llm.chat(user_input=prompt, use_history=False)
            
            if response:
                return response.strip()
            
            return None
            
        except Exception as e:
            logger.error(f"Teacher mode error: {e}")
            return None
    
    def compare_responses(
        self,
        user_input: str,
        alice_response: str,
        expected_route: str
    ) -> Dict[str, Any]:
        """
        Compare Alice's response to ideal teacher response
        
        Args:
            user_input: Original user input
            alice_response: Alice's actual response
            expected_route: Expected routing decision
        
        Returns:
            Comparison result with teacher response and deviation flags
        """
        teacher_response = self.get_ideal_response(
            user_input,
            context={"expected_route": expected_route}
        )
        
        if not teacher_response:
            return {
                "teacher_response": None,
                "needs_learning": False,
                "deviation_type": None
            }
        
        # Simple heuristic for deviation detection
        # TODO: Could use semantic similarity or LLM-based comparison
        alice_lower = alice_response.lower()
        teacher_lower = teacher_response.lower()
        
        # Check for major deviations
        needs_learning = False
        deviation_type = None
        
        # Length deviation (one is 2x+ longer than the other)
        if len(alice_lower) > 2 * len(teacher_lower) or len(teacher_lower) > 2 * len(alice_lower):
            needs_learning = True
            deviation_type = "length_mismatch"
        
        # Tone deviation (teacher asks question, Alice doesn't, or vice versa)
        teacher_asks = "?" in teacher_response
        alice_asks = "?" in alice_response
        if teacher_asks != alice_asks:
            needs_learning = True
            deviation_type = "tone_mismatch"
        
        # Key word overlap check
        teacher_words = set(teacher_lower.split())
        alice_words = set(alice_lower.split())
        common_words = teacher_words & alice_words
        
        # If less than 30% word overlap, flag it
        if common_words:
            overlap_ratio = len(common_words) / max(len(teacher_words), len(alice_words))
            if overlap_ratio < 0.3:
                needs_learning = True
                if not deviation_type:
                    deviation_type = "content_mismatch"
        
        return {
            "teacher_response": teacher_response,
            "needs_learning": needs_learning,
            "deviation_type": deviation_type,
            "alice_response": alice_response
        }
    
    def should_flag_for_learning(
        self,
        alice_route: str,
        expected_route: str,
        comparison: Dict[str, Any]
    ) -> bool:
        """
        Determine if this interaction should be flagged for learning
        
        Args:
            alice_route: Alice's actual routing decision
            expected_route: Expected routing decision
            comparison: Result from compare_responses
        
        Returns:
            True if should be flagged for learning
        """
        # Route mismatch is always a learning opportunity
        if alice_route != expected_route:
            return True
        
        # Teacher comparison flagged a deviation
        if comparison.get("needs_learning", False):
            return True
        
        return False
