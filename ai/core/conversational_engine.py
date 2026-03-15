"""
Conversational Engine for A.L.I.C.E
A.L.I.C.E's own conversational reasoning and response generation.
Does NOT use Ollama - this is A.L.I.C.E thinking and responding on her own.
"""

import logging
import random
import json
import os
import re
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
    plugin_data: Optional[Dict[str, Any]] = None


class ConversationalEngine:
    """
    A.L.I.C.E's own conversational reasoning engine.
    Generates responses based on:
    - Training data patterns
    - Memory and context
    - Her own logic
    NO Ollama calls - this is pure A.L.I.C.E.
    """

    def __init__(
        self,
        memory_system: Optional[Any] = None,
        training_collector: Optional[Any] = None,
        world_state: Optional[Any] = None,
    ) -> None:
        self.memory = memory_system
        self.training_collector = training_collector
        self.world_state = world_state

        # Learned conversational patterns
        self.learned_greetings = []
        self.learned_responses = {}
        self.conversation_style = {
            "length": "medium",
            "tone": "neutral",
            "uses_questions": False,
        }

        # Track recent responses to avoid repetition
        self.recent_responses = []
        self.max_recent = 10

        # Load patterns from training data
        self._load_patterns()

    def _unique_candidates(self, candidates: List[str]) -> List[str]:
        """Return normalized unique candidate strings while preserving order."""
        unique: List[str] = []
        seen = set()
        for candidate in candidates or []:
            text = (candidate or "").strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(text)
        return unique

    def _load_patterns(self) -> None:
        """Load conversational patterns from training data and curated patterns"""
        # First, load curated patterns
        self._load_curated_patterns()

        # Then augment with learned patterns from training data
        if not self.training_collector:
            return

        try:
            examples = self.training_collector.get_training_data(min_quality=0.7)

            # Extract greetings
            greeting_keywords = ["hi", "hey", "hello", "yo", "sup"]
            plugin_response_markers = [
                "°c",
                "°f",
                "overcast",
                "sunny",
                "rain",
                "temperature",
                " in ",
                "kitchener",
                "toronto",
            ]
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

    def _load_curated_patterns(self) -> None:
        """Load curated conversational patterns"""
        project_root = Path(__file__).resolve().parents[1]
        curated_path = project_root / "memory" / "curated_patterns.json"
        if not curated_path.exists():
            logger.debug("No curated patterns found")
            return

        try:
            with open(curated_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load greetings
            if "greetings" in data:
                self.learned_greetings.extend(data["greetings"])

            # Load meta_questions (who created you, what are you, etc.)
            if "meta_questions" in data:
                for question_type, responses in data["meta_questions"].items():
                    self.learned_responses[f"meta_{question_type}"] = responses

            # Load acknowledgments (birthday, anniversary, etc.)
            if "acknowledgments" in data:
                for ack_type, responses in data["acknowledgments"].items():
                    self.learned_responses[f"ack_{ack_type}"] = responses

            # Load errors
            if "errors" in data:
                for error_type, responses in data["errors"].items():
                    self.learned_responses[f"error_{error_type}"] = responses

            logger.info(
                f"Loaded curated patterns: {len(self.learned_greetings)} greetings, {len(self.learned_responses)} response categories"
            )
        except Exception as e:
            logger.error(f"Failed to load curated patterns: {e}")

    def _pick_non_repeating(self, candidates: List[str]) -> str:
        """Pick a response from candidates, preferring one not in recent_responses."""
        normalized_candidates = self._unique_candidates(candidates)
        if not normalized_candidates:
            return ""
        recent = getattr(self, "recent_responses", ()) or []
        max_recent = getattr(self, "max_recent", 10)
        not_recent = [c for c in normalized_candidates if c not in recent]
        choice = (
            random.choice(not_recent)
            if not_recent
            else random.choice(normalized_candidates)
        )
        recent.insert(0, choice)
        self.recent_responses = recent[:max_recent]
        return choice

    @staticmethod
    def _token_set(text: str) -> set:
        """Lowercased token set for robust intent cue checks."""
        return set(re.findall(r"\b[a-z']+\b", (text or "").lower()))

    def can_handle(
        self, user_input: str, intent: str, context: ConversationalContext
    ) -> bool:
        """
        Check if A.L.I.C.E can handle this conversationally without Ollama.
        Returns True ONLY for very simple, repetitive patterns where LLM would be overkill.
        Most questions should go to LLM for natural, contextual responses.
        """
        input_lower = user_input.lower().strip()
        input_tokens = self._token_set(user_input)

        # ONLY handle simple greetings - let LLM handle everything else
        if (
            bool(input_tokens & {"hi", "hey", "hello", "yo", "sup"})
            and len(input_lower.split()) <= 4
        ):
            # Must be primarily a greeting, not a question
            greeting_options = self._unique_candidates(self.learned_greetings)
            if "?" not in input_lower and len(greeting_options) >= 2:
                return True

        # Personal events acknowledgment (birthday, anniversary) - these are simple confirmations
        if bool(input_tokens & {"birthday", "anniversary"}):
            # Only if it's a statement like "my birthday is..." not a question
            if "?" not in input_lower and any(
                word in input_tokens for word in ["my", "is", "on"]
            ):
                if any(k.startswith("ack_") for k in self.learned_responses.keys()):
                    return True

        return False

        # Simple thanks - only if we have learned patterns
        if input_lower in ("thanks", "thx", "ty", "thank you"):
            if self.training_collector:
                try:
                    examples = self.training_collector.get_training_data(
                        min_quality=0.7, max_examples=30
                    )
                    has_thanks_patterns = any(
                        "thank" in ex.user_input.lower() for ex in examples
                    )
                    return has_thanks_patterns
                except:
                    pass
            return False

        # Short conversational acknowledgments — always handled directly, never need LLM
        _ACK_PHRASES = {
            "will do",
            "got it",
            "noted",
            "understood",
            "sure",
            "sure thing",
            "sounds good",
            "sounds great",
            "alright",
            "okay",
            "ok",
            "okie",
            "roger",
            "roger that",
            "copy that",
            "yep",
            "yup",
            "yeah",
            "on it",
            "all good",
            "no problem",
            "perfect",
            "great",
            "awesome",
            "nice",
            "good idea",
            "great idea",
            "nice idea",
            "good point",
            "fair point",
            "fair enough",
            "makes sense",
            "that makes sense",
            "good call",
            "true",
            "exactly",
            "absolutely",
            "definitely",
            "of course",
            "i see",
            "right",
            "for sure",
            "no worries",
            "anytime",
        }
        if input_lower in _ACK_PHRASES or (
            intent == "conversation:ack" and len(input_lower.split()) <= 4
        ):
            return True

    def generate_response(self, context: ConversationalContext) -> Optional[str]:
        """
        Generate A.L.I.C.E's response from learned patterns.
        Only handles simple, repetitive cases. Returns None for everything else (goes to LLM).
        """
        user_input = context.user_input
        input_lower = user_input.lower().strip()
        input_tokens = self._token_set(user_input)

        # GREETINGS - Brief and friendly
        greeting_words = ["hi", "hey", "hello", "yo", "sup"]
        if (
            bool(input_tokens & set(greeting_words))
            and len(input_lower.split()) <= 4
        ):
            # Don't treat questions as greetings
            if "?" not in input_lower:
                greeting_options = self._unique_candidates(self.learned_greetings)
                if len(greeting_options) >= 2:
                    # Prefer brief greetings
                    brief_greetings = [g for g in greeting_options if len(g) < 60]
                    if brief_greetings:
                        return self._pick_non_repeating(brief_greetings)
                    return self._pick_non_repeating(greeting_options)

        # PERSONAL EVENTS - Detect and acknowledge birthday/anniversary statements
        if "birthday" in input_lower and "?" not in input_lower:
            # Try to detect and store the event
            try:
                from features.personal_events import (
                    PersonalEventsDetector,
                    PersonalEventsStorage,
                )

                detector = PersonalEventsDetector(
                    user_name=(
                        context.world_state.user_name if context.world_state else "User"
                    )
                )
                storage = PersonalEventsStorage()

                events = detector.detect_events(user_input)
                for event in events:
                    storage.add_event(event)
                    logger.info(
                        f"Stored personal event: {event.description} on {event.date}"
                    )

                # Respond with acknowledgment
                if "ack_birthday" in self.learned_responses:
                    return self._pick_non_repeating(
                        self.learned_responses["ack_birthday"]
                    )
            except Exception as e:
                logger.error(f"Error detecting birthday: {e}")

        if "anniversary" in input_lower and "?" not in input_lower:
            try:
                from features.personal_events import (
                    PersonalEventsDetector,
                    PersonalEventsStorage,
                )

                detector = PersonalEventsDetector(
                    user_name=(
                        context.world_state.user_name if context.world_state else "User"
                    )
                )
                storage = PersonalEventsStorage()

                events = detector.detect_events(user_input)
                for event in events:
                    storage.add_event(event)
                    logger.info(
                        f"Stored personal event: {event.description} on {event.date}"
                    )

                if "ack_anniversary" in self.learned_responses:
                    return self._pick_non_repeating(
                        self.learned_responses["ack_anniversary"]
                    )
            except Exception as e:
                logger.error(f"Error detecting anniversary: {e}")

        # THANKS - Learned patterns only
        if input_lower in ("thanks", "thx", "ty", "thank you"):
            if self.training_collector:
                try:
                    examples = self.training_collector.get_training_data(
                        min_quality=0.7, max_examples=30
                    )
                    thanks_responses = []
                    for ex in examples:
                        if any(
                            word in ex.user_input.lower()
                            for word in ["thank", "thanks"]
                        ):
                            if (
                                ex.assistant_response
                                and len(ex.assistant_response) < 60
                            ):
                                thanks_responses.append(ex.assistant_response)
                    if thanks_responses:
                        return self._pick_non_repeating(thanks_responses)
                except:
                    pass

        # ACK — short built-in responses, never need LLM
        _ACK_REPLIES = [
            "No problem!",
            "Of course.",
            "Happy to help.",
            "Let me know if you need anything else.",
            "Sure.",
            "Glad I could help.",
        ]
        if context.intent == "conversation:ack" or len(input_lower.split()) <= 4:
            return self._pick_non_repeating(_ACK_REPLIES)


_conversational_engine: Optional[ConversationalEngine] = None


def get_conversational_engine(
    memory_system=None, training_collector=None, world_state=None
) -> ConversationalEngine:
    """Get singleton conversational engine"""
    global _conversational_engine
    if _conversational_engine is None:
        _conversational_engine = ConversationalEngine(
            memory_system=memory_system,
            training_collector=training_collector,
            world_state=world_state,
        )
    return _conversational_engine
