"""
Conversational Engine for A.L.I.C.E
A.L.I.C.E's own conversational reasoning and response generation.
Does NOT use Ollama - this is A.L.I.C.E thinking and responding on her own.
"""

import logging
import random
import json
import os
from pathlib import Path
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
        project_root = Path(__file__).resolve().parents[1]
        curated_path = project_root / "memory" / "curated_patterns.json"
        if not curated_path.exists():
            logger.debug("No curated patterns found")
            return

        try:
            with open(curated_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Load greetings
            if 'greetings' in data:
                self.learned_greetings.extend(data['greetings'])

            # Load meta_questions (who created you, what are you, etc.)
            if 'meta_questions' in data:
                for question_type, responses in data['meta_questions'].items():
                    self.learned_responses[f'meta_{question_type}'] = responses

            # Load acknowledgments (birthday, anniversary, etc.)
            if 'acknowledgments' in data:
                for ack_type, responses in data['acknowledgments'].items():
                    self.learned_responses[f'ack_{ack_type}'] = responses

            # Load errors
            if 'errors' in data:
                for error_type, responses in data['errors'].items():
                    self.learned_responses[f'error_{error_type}'] = responses

            logger.info(f"Loaded curated patterns: {len(self.learned_greetings)} greetings, {len(self.learned_responses)} response categories")
        except Exception as e:
            logger.error(f"Failed to load curated patterns: {e}")
    
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
        Returns True if she has curated patterns, learned patterns, or memories.
        """
        input_lower = user_input.lower().strip()

        # Meta questions - ALWAYS handle if we have curated patterns
        if any(phrase in input_lower for phrase in [
            'who created', 'who made', 'who built', 'who are you', 'what are you',
            'how do you work', 'what can you do', 'capabilities'
        ]):
            # Check if we have curated meta responses
            meta_keys = [k for k in self.learned_responses.keys() if k.startswith('meta_')]
            if meta_keys:
                return True  # We have curated meta answers

        # Personal events (birthday, anniversary) - ALWAYS handle
        if any(word in input_lower for word in ['birthday', 'anniversary']):
            if any(k.startswith('ack_') for k in self.learned_responses.keys()):
                return True  # We have acknowledgment patterns

        # Simple greetings - only if we have learned patterns
        if any(word in input_lower for word in ['hi', 'hey', 'hello', 'yo', 'sup']) and len(input_lower.split()) <= 4:
            # Must be primarily a greeting, not a question
            if '?' not in input_lower and len(self.learned_greetings) > 0:
                return True

        # If we have curated/learned responses for this intent, handle it
        if intent in self.learned_responses and self.learned_responses[intent]:
            return True

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

        # Otherwise, needs Ollama for reasoning
        return False
    
    def generate_response(self, context: ConversationalContext) -> Optional[str]:
        """
        Generate A.L.I.C.E's response from curated patterns, learned patterns, or memory.
        Returns None if no pattern exists (then Ollama will be used).
        """
        user_input = context.user_input
        input_lower = user_input.lower().strip()
        intent = context.intent

        # META QUESTIONS - Handle identity/capability questions
        if 'who created' in input_lower or 'who made' in input_lower or 'who built' in input_lower:
            if 'meta_who_created' in self.learned_responses:
                return self._pick_non_repeating(self.learned_responses['meta_who_created'])

        if 'who are you' in input_lower:
            if 'meta_what_are_you' in self.learned_responses:
                return self._pick_non_repeating(self.learned_responses['meta_what_are_you'])

        if 'what are you' in input_lower:
            if 'meta_what_are_you' in self.learned_responses:
                return self._pick_non_repeating(self.learned_responses['meta_what_are_you'])

        if 'how do you work' in input_lower:
            if 'meta_how_do_you_work' in self.learned_responses:
                return self._pick_non_repeating(self.learned_responses['meta_how_do_you_work'])

        if any(phrase in input_lower for phrase in ['what can you do', 'your capabilities', 'what are your capabilities']):
            if 'meta_capabilities' in self.learned_responses:
                return self._pick_non_repeating(self.learned_responses['meta_capabilities'])

        # PERSONAL EVENTS - Detect and acknowledge birthday/anniversary
        if 'birthday' in input_lower:
            # Try to detect and store the event
            try:
                from features.personal_events import PersonalEventsDetector, PersonalEventsStorage
                detector = PersonalEventsDetector(user_name=context.world_state.user_name if context.world_state else "User")
                storage = PersonalEventsStorage()

                events = detector.detect_events(user_input)
                for event in events:
                    storage.add_event(event)
                    logger.info(f"Stored personal event: {event.description} on {event.date}")

                # Respond with acknowledgment
                if 'ack_birthday' in self.learned_responses:
                    return self._pick_non_repeating(self.learned_responses['ack_birthday'])
            except Exception as e:
                logger.error(f"Error detecting birthday: {e}")

        if 'anniversary' in input_lower:
            try:
                from features.personal_events import PersonalEventsDetector, PersonalEventsStorage
                detector = PersonalEventsDetector(user_name=context.world_state.user_name if context.world_state else "User")
                storage = PersonalEventsStorage()

                events = detector.detect_events(user_input)
                for event in events:
                    storage.add_event(event)
                    logger.info(f"Stored personal event: {event.description} on {event.date}")

                if 'ack_anniversary' in self.learned_responses:
                    return self._pick_non_repeating(self.learned_responses['ack_anniversary'])
            except Exception as e:
                logger.error(f"Error detecting anniversary: {e}")

        # Use curated/learned responses by intent
        if intent in self.learned_responses and self.learned_responses[intent]:
            return self._pick_non_repeating(self.learned_responses[intent])

        # GREETINGS - Brief and friendly
        greeting_words = ['hi', 'hey', 'hello', 'yo', 'sup']
        if any(word in input_lower for word in greeting_words) and len(input_lower.split()) <= 4:
            # Don't treat questions as greetings
            if '?' not in input_lower:
                if self.learned_greetings:
                    # Prefer brief greetings
                    brief_greetings = [g for g in self.learned_greetings if len(g) < 60]
                    if brief_greetings:
                        return self._pick_non_repeating(brief_greetings)
                    return self._pick_non_repeating(self.learned_greetings)

        # THANKS - Learned patterns only
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

        # No learned pattern - use Ollama
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
