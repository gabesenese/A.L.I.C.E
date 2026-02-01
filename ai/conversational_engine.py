"""
Conversational Engine for A.L.I.C.E
A.L.I.C.E's own conversational reasoning and response generation.
Does NOT use Ollama - this is A.L.I.C.E thinking and responding on her own.
"""

import logging
import random
import json
import os
from typing import Dict, Optional, Any, List
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class ConversationalContext:
    """Context for conversational reasoning"""
    user_input: str
    intent: str
    entities: Dict[str, Any]
    recent_topics: List[str]
    active_goal: Optional[str]
    world_state: Optional[Any]


class ConversationalEngine:
    """
    A.L.I.C.E's own conversational reasoning engine.
    Generates responses based on:
    - Training data patterns
    - Memory and context
    - Her own logic
    NO Ollama calls - this is pure A.L.I.C.E.
    """
    
    def __init__(self, memory_system=None, training_collector=None, world_state=None):
        self.memory = memory_system
        self.training_collector = training_collector
        self.world_state = world_state
        
        # Learned conversational patterns
        self.learned_greetings = []
        self.learned_responses = {}
        self.conversation_style = {
            'length': 'medium',
            'tone': 'neutral',
            'uses_questions': False
        }
        
        # Track recent responses to avoid repetition
        self.recent_responses = []
        self.max_recent = 10
        
        # Load patterns from training data
        self._load_patterns()
    
    def _load_patterns(self):
        """Load conversational patterns from training data and curated patterns"""
        # First, load curated patterns
        self._load_curated_patterns()
        
        # Then augment with learned patterns from training data
        if not self.training_collector:
            return
        
        try:
            examples = self.training_collector.get_training_data(min_quality=0.7)
            
            # Extract greetings
            greeting_keywords = ['hi', 'hey', 'hello', 'yo', 'sup']
            plugin_response_markers = ['°c', '°f', 'overcast', 'sunny', 'rain', 'temperature', ' in ', 'kitchener', 'toronto']
            for ex in examples:
                inp = ex.user_input.lower().strip()
                if len(inp.split()) > 4:
                    continue
                if not any(kw in inp for kw in greeting_keywords):
                    continue
                if not ex.assistant_response or len(ex.assistant_response) >= 80:
                    continue
                resp_lower = ex.assistant_response.lower()
                if any(m in resp_lower for m in plugin_response_markers):
                    continue
                # Only add if not already in learned greetings (from curated)
                if ex.assistant_response not in self.learned_greetings:
                    self.learned_greetings.append(ex.assistant_response)
        except Exception as e:
            logger.debug(f"Could not load training patterns: {e}")
    
    def _load_curated_patterns(self):
        """Load curated conversational patterns"""
        curated_path = "memory/curated_patterns.json"
        if not os.path.exists(curated_path):
            logger.debug("No curated patterns found")
            return
        
        try:
            with open(curated_path, 'r') as f:
                data = json.load(f)
            
            for pattern in data.get('patterns', []):
                pattern_type = pattern.get('pattern_type')
                response = pattern.get('response')
                intent = pattern.get('intent', pattern_type)
                
                # Store responses by pattern type
                if pattern_type == 'greeting' and response:
                    self.learned_greetings.append(response)
                
                # Store all patterns in learned_responses by intent
                if intent and response:
                    if intent not in self.learned_responses:
                        self.learned_responses[intent] = []
                    self.learned_responses[intent].append(response)
            
            logger.info(f"Loaded {len(data.get('patterns', []))} curated patterns")
        except Exception as e:
            logger.error(f"Failed to load curated patterns: {e}")
            
            # Learn conversation style
            if examples:
                all_responses = [ex.assistant_response for ex in examples if ex.assistant_response]
                if all_responses:
                    avg_len = sum(len(r) for r in all_responses) / len(all_responses)
                    self.conversation_style['length'] = 'short' if avg_len < 40 else 'long' if avg_len > 100 else 'medium'
                    self.conversation_style['uses_questions'] = sum(1 for r in all_responses if '?' in r) > len(all_responses) * 0.3
            
            logger.info(f"[ConvEngine] Loaded {len(self.learned_greetings)} learned greetings")
            
        except Exception as e:
            logger.debug(f"[ConvEngine] Could not load patterns: {e}")


    def _pick_non_repeating(self, candidates: List[str]) -> str:
        """Pick a response from candidates, preferring one not in recent_responses."""
        if not candidates:
            return ""
        recent = getattr(self, "recent_responses", ()) or []
        max_recent = getattr(self, "max_recent", 10)
        not_recent = [c for c in candidates if c not in recent]
        choice = random.choice(not_recent) if not_recent else random.choice(candidates)
        recent.insert(0, choice)
        self.recent_responses = recent[:max_recent]
        return choice
    
    def can_handle(self, user_input: str, intent: str, context: ConversationalContext) -> bool:
        """
        Check if A.L.I.C.E can handle this conversationally without Ollama.
        Returns True ONLY if she has learned patterns or memories.
        """
        input_lower = user_input.lower()
        
        # Simple greetings - only if we have learned patterns
        if any(word in input_lower for word in ['hi', 'hey', 'hello', 'yo', 'sup']) and len(input_lower.split()) <= 4:
            return len(self.learned_greetings) > 0  # Only if learned
        
        # Thanks/appreciation - only if we have learned patterns
        if 'thank' in input_lower or input_lower in ('thanks', 'thx', 'ty', 'thank you'):
            if self.training_collector:
                try:
                    examples = self.training_collector.get_training_data(min_quality=0.7, max_examples=30)
                    has_thanks_patterns = any('thank' in ex.user_input.lower() for ex in examples)
                    return has_thanks_patterns
                except:
                    pass
            return False
        
        # Identity questions - check if we have learned answers
        if 'who created' in input_lower or 'who made' in input_lower or ('who are you' in input_lower or 'what are you' in input_lower):
            if self.training_collector:
                try:
                    examples = self.training_collector.get_training_data(min_quality=0.6, max_examples=50)
                    has_identity_patterns = any(
                        'who created' in ex.user_input.lower() or 
                        'who are you' in ex.user_input.lower() or
                        'what are you' in ex.user_input.lower()
                        for ex in examples
                    )
                    if has_identity_patterns:
                        return True  # Use learned answer
                except:
                    pass
            return False
        
        # Check memory for similar conversations
        if self.memory:
            try:
                memories = self.memory.search_memories(user_input, limit=1)
                if memories and len(memories) > 0:
                    # A.L.I.C.E has relevant memories - can respond from memory
                    return True
            except:
                pass
        
        # Otherwise, needs Ollama for reasoning
        return False
    
    def generate_response(self, context: ConversationalContext) -> Optional[str]:
        """
        Generate A.L.I.C.E's response ONLY from learned patterns or memory.
        NO hardcoded responses - returns None if no learned pattern exists.
        Then Ollama will be used instead.
        """
        user_input = context.user_input
        input_lower = user_input.lower().strip()
        
        # Greetings - Learn style, filter verbose responses
        greeting_words = ['hi', 'hey', 'hello', 'yo', 'sup']
        if any(word in input_lower for word in greeting_words) and len(input_lower.split()) <= 4:
            if self.learned_greetings:
                # Filter out verbose Ollama-style responses (> 50 chars)
                brief_greetings = [g for g in self.learned_greetings if len(g) < 50]
                
                if brief_greetings:
                    return self._pick_non_repeating(brief_greetings)
                
                # All learned greetings are verbose - use Ollama to generate fresh one
                return None
            # No learned patterns - use Ollama
            return None
        
        # Thanks - ONLY learned patterns
        if 'thank' in input_lower or input_lower in ('thanks', 'thx', 'ty', 'thank you'):
            if self.training_collector:
                try:
                    examples = self.training_collector.get_training_data(min_quality=0.7, max_examples=30)
                    thanks_responses = []
                    for ex in examples:
                        if any(word in ex.user_input.lower() for word in ['thank', 'thanks']):
                            if ex.assistant_response and len(ex.assistant_response) < 60:
                                thanks_responses.append(ex.assistant_response)
                    if thanks_responses:
                        return self._pick_non_repeating(thanks_responses)
                except:
                    pass
            # No learned patterns - use Ollama
            return None
        
        # Identity questions - Learn from data OR discover from codebase
        if 'who created' in input_lower or 'who made' in input_lower or 'who are you' in input_lower or 'what are you' in input_lower:
            # First: try learned patterns
            if self.training_collector:
                try:
                    examples = self.training_collector.get_training_data(min_quality=0.6, max_examples=50)
                    identity_responses = []
                    for ex in examples:
                        if any(phrase in ex.user_input.lower() for phrase in ['who created', 'who made', 'who are you', 'what are you']):
                            if ex.assistant_response and len(ex.assistant_response) < 500:
                                # Filter out false/generic responses
                                resp_lower = ex.assistant_response.lower()
                                if not any(false_info in resp_lower for false_info in ['stanford', 'sri international', 'arpa']):
                                    identity_responses.append(ex.assistant_response)
                    if identity_responses:
                        return self._pick_non_repeating(identity_responses)
                except:
                    pass
            
            # No learned patterns - let Ollama handle it (it will learn from codebase context)
            return None
        
        # Check memory for similar past conversations
        if self.memory:
            try:
                memories = self.memory.search_memories(user_input, limit=2)
                if memories and len(memories) > 0:
                    # A.L.I.C.E recalls similar conversation
                    memory = memories[0]
                    return memory.content[:150]  # Use memory as response
            except Exception as e:
                logger.debug(f"Memory search error: {e}")
        
        # No learned pattern, no memory - must use Ollama
        return None


_conversational_engine: Optional[ConversationalEngine] = None


def get_conversational_engine(memory_system=None, training_collector=None, world_state=None) -> ConversationalEngine:
    """Get singleton conversational engine"""
    global _conversational_engine
    if _conversational_engine is None:
        _conversational_engine = ConversationalEngine(
            memory_system=memory_system,
            training_collector=training_collector,
            world_state=world_state
        )
    return _conversational_engine
