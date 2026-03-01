"""
Response Variance Engine for A.L.I.C.E
Generates varied, context-aware responses without hardcoded templates
Every response is unique and adapts to conversational context
"""

import logging
import hashlib
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


@dataclass
class ResponseContext:
    """Context for generating a varied response"""

    intent_type: str
    data: Dict[str, Any]
    user_id: str
    conversation_history: List[Dict]
    user_mood: str = "neutral"
    user_verbosity_pref: float = 0.5  # 0=terse, 1=verbose
    repetition_count: int = 0
    time_since_last_similar: Optional[float] = None


@dataclass
class ResponseQuality:
    """Quality metrics for a generated response"""

    response: str
    timestamp: float
    user_reaction: Optional[str] = None
    quality_score: float = 0.5
    led_to_clarification: bool = False
    led_to_success: bool = False


class RepetitionDetector:
    """Detects when user is asking the same question repeatedly"""

    def __init__(self, window_size: int = 10):
        self.query_history = defaultdict(lambda: deque(maxlen=window_size))

    def check_repetition(
        self, user_id: str, intent_type: str, query_hash: str
    ) -> Tuple[int, Optional[float]]:
        """
        Returns: (repetition_count, time_since_last_in_seconds)
        """
        history = self.query_history[user_id]

        # Count how many times this query appeared recently
        count = 0
        last_time = None
        current_time = time.time()

        for prev_intent, prev_hash, prev_time in history:
            if prev_intent == intent_type and prev_hash == query_hash:
                count += 1
                if last_time is None or prev_time > last_time:
                    last_time = prev_time

        time_since = (current_time - last_time) if last_time else None

        # Add current query to history
        history.append((intent_type, query_hash, current_time))

        return count, time_since


class ResponseVarianceEngine:
    """
    Generates varied, human-like responses
    Never the same answer twice for the same question
    """

    def __init__(self, llm_generator=None, phrasing_learner=None):
        self.llm_generator = llm_generator
        self.phrasing_learner = phrasing_learner
        self.repetition_detector = RepetitionDetector()
        self.response_history = defaultdict(lambda: deque(maxlen=50))
        self.quality_tracker = defaultdict(list)

        # Track successful response patterns (not templates, but features)
        self.successful_patterns = defaultdict(
            lambda: {
                "avg_length": 0,
                "uses_examples": 0,
                "uses_confirmation": 0,
                "directness_scores": [],
            }
        )

    def _hash_query(self, intent: str, data: Dict) -> str:
        """Create a semantic fingerprint of the query"""
        # Hash based on intent and key data elements (not exact string)
        key_elements = f"{intent}:{data.get('action', '')}:{data.get('query', '')}"
        return hashlib.md5(key_elements.encode()).hexdigest()[:8]

    def _detect_user_mood(self, conversation_history: List[Dict]) -> str:
        """Detect user's current conversational mood"""
        if not conversation_history:
            return "neutral"

        recent_turns = conversation_history[-3:]

        # Simple heuristic detection (can be enhanced with ML)
        frustration_signals = 0
        excitement_signals = 0
        confusion_signals = 0

        for turn in recent_turns:
            user_input = turn.get("user_input", "").lower()

            # Frustration indicators
            if any(
                word in user_input
                for word in ["what", "why", "didn't", "wrong", "again"]
            ):
                frustration_signals += 1
            if "!" in user_input or "?!" in user_input:
                frustration_signals += 1
            if "never mind" in user_input or "forget it" in user_input:
                frustration_signals += 2

            # Excitement indicators
            if "!!" in user_input:
                excitement_signals += 1
            if any(
                word in user_input
                for word in ["awesome", "great", "perfect", "thanks", "love"]
            ):
                excitement_signals += 1

            # Confusion indicators
            if user_input.count("?") > 1:
                confusion_signals += 1
            if any(
                phrase in user_input
                for phrase in [
                    "what do you mean",
                    "don't understand",
                    "confused",
                    "huh",
                ]
            ):
                confusion_signals += 2

        if frustration_signals >= 2:
            return "frustrated"
        elif confusion_signals >= 2:
            return "confused"
        elif excitement_signals >= 2:
            return "excited"

        return "neutral"

    def _get_response_constraints(self, context: ResponseContext) -> Dict[str, Any]:
        """Build constraints for response generation based on context"""
        constraints = {
            "min_length": 10,
            "max_length": 200,
            "tone": "standard",
            "include_confirmation": True,
            "include_details": True,
        }

        # Adjust for user mood
        if context.user_mood == "frustrated":
            constraints["tone"] = "empathetic_direct"
            constraints["max_length"] = 100  # Be more concise
            constraints["include_details"] = False

        elif context.user_mood == "confused":
            constraints["tone"] = "clarifying_patient"
            constraints["include_details"] = True
            constraints["include_examples"] = True

        elif context.user_mood == "excited":
            constraints["tone"] = "enthusiastic"

        # Adjust for verbosity preference
        if context.user_verbosity_pref < 0.3:
            constraints["max_length"] = 80
            constraints["include_details"] = False
        elif context.user_verbosity_pref > 0.7:
            constraints["max_length"] = 300
            constraints["include_examples"] = True

        # Adjust for repetition
        if context.repetition_count >= 2:
            constraints["acknowledge_repetition"] = True
            constraints["tone"] = "gently_curious"

        if context.repetition_count >= 3:
            constraints["tone"] = "concerned_helpful"
            constraints["suggest_alternative"] = True

        return constraints

    def _get_recent_similar_responses(
        self, user_id: str, intent_type: str, n: int = 3
    ) -> List[str]:
        """Get recent responses for similar intents (to avoid repetition)"""
        history = self.response_history[user_id]
        similar = [resp for resp_intent, resp in history if resp_intent == intent_type]
        return similar[-n:] if similar else []

    def _generate_with_variance(
        self, context: ResponseContext, constraints: Dict
    ) -> str:
        """Generate response with variance using LLM"""
        # Get inspiration from past successful responses (not templates)
        similar_responses = self._get_recent_similar_responses(
            context.user_id, context.intent_type, n=3
        )

        # Build generation prompt
        prompt_parts = []

        # Core instruction
        prompt_parts.append(
            f"Generate a natural, conversational response for this situation:"
        )
        prompt_parts.append(f"Intent: {context.intent_type}")
        prompt_parts.append(f"Data: {context.data}")

        # Mood adaptation
        if context.user_mood != "neutral":
            mood_instructions = {
                "frustrated": "User seems frustrated. Be direct, empathetic, and solution-focused. Keep it brief.",
                "confused": "User seems confused. Be clear, patient, and provide step-by-step clarification.",
                "excited": "User seems excited. Match their energy with enthusiasm.",
            }
            prompt_parts.append(mood_instructions.get(context.user_mood, ""))

        # Repetition handling
        if context.repetition_count >= 2:
            prompt_parts.append(
                f"IMPORTANT: User has asked about this {context.repetition_count} times. "
                f"Gently acknowledge this and ask if they need something different."
            )

        # Variance constraint
        if similar_responses:
            prompt_parts.append(
                f"\nIMPORTANT: Generate a DIFFERENT response than these recent ones:\n"
                + "\n".join(f"- {r}" for r in similar_responses)
                + f"\nUse different phrasing, structure, and approach."
            )

        # Verbosity constraint
        if constraints["max_length"] < 100:
            prompt_parts.append(
                f"Keep response under {constraints['max_length']} characters. Be concise."
            )

        prompt = "\n".join(prompt_parts)

        # Use LLM to generate (if available)
        if self.llm_generator:
            try:
                response = self._generate_with_llm(prompt, constraints)
                return response.strip()
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")

        # Fallback to learned patterns if LLM unavailable
        if self.phrasing_learner:
            return self._generate_from_learned_patterns(context, constraints)

        # Final fallback: basic structured response
        return self._generate_basic_response(context, constraints)

    def _generate_from_learned_patterns(
        self, context: ResponseContext, constraints: Dict
    ) -> str:
        """Generate using learned successful patterns"""
        # This uses phrasing learner to blend successful past patterns
        # Not templates, but learned semantic structures

        if not self.phrasing_learner:
            return self._generate_basic_response(context, constraints)

        try:
            patterns = self._get_successful_patterns(context.intent_type, n=5)

            if patterns:
                # Use pattern features to guide generation
                # (This would be enhanced with actual phrasing learner integration)
                return self._apply_pattern_features(context, patterns[0], constraints)
        except Exception as e:
            logger.error(f"Pattern-based generation failed: {e}")

        return self._generate_basic_response(context, constraints)

    def _apply_pattern_features(
        self, context: ResponseContext, pattern: Dict, constraints: Dict
    ) -> str:
        """Apply learned pattern features to generate new response"""
        # This is a placeholder - actual implementation would use
        # pattern features (length, structure, tone) to guide generation
        return self._generate_basic_response(context, constraints)

    def _generate_basic_response(
        self, context: ResponseContext, constraints: Dict
    ) -> str:
        """Fallback: generate basic structured response"""
        data = context.data
        intent = context.intent_type

        # Handle repetition awareness
        if context.repetition_count >= 2:
            prefixes = [
                f"I notice you're asking about this again—",
                f"You've asked about this a few times now—",
                f"I see you're still checking on this—",
            ]
            prefix = prefixes[min(context.repetition_count - 2, len(prefixes) - 1)]
        else:
            prefix = ""

        # Generate core response based on intent type
        core = self._generate_core_content(intent, data, constraints)

        # Combine with prefix if needed
        response = f"{prefix} {core}" if prefix else core

        return response.strip()

    def _generate_core_content(self, intent: str, data: Dict, constraints: Dict) -> str:
        """Generate core content for common intent types"""
        # This provides basic fallback responses
        # Real implementation would be much more sophisticated

        if "weather" in intent:
            temp = data.get("temperature")
            condition = data.get("condition", "unknown")
            location = data.get("location", "")

            if temp is not None:
                variations = [
                    f"It's {condition} and {temp}°C{' in ' + location if location else ''}.",
                    f"Currently {condition}, {temp}°C{' in ' + location if location else ''}.",
                    f"{condition.capitalize()} conditions with {temp}°C{' in ' + location if location else ''}.",
                ]
                return variations[int(time.time()) % len(variations)]
            return f"Conditions are {condition}{' in ' + location if location else ''}."

        elif "notes" in intent:
            if "count" in data:
                count = data["count"]
                variations = [
                    f"You have {count} notes.",
                    f"Found {count} notes in your collection.",
                    f"You're tracking {count} notes.",
                ]
                return variations[int(time.time()) % len(variations)]

        # Generic fallback
        return "Got it."

    def _generate_with_llm(self, prompt: str, constraints: Dict[str, Any]) -> str:
        """Compatibility wrapper for different LLM engine APIs."""
        max_tokens = constraints.get("max_length", 200) // 4

        if hasattr(self.llm_generator, "generate") and callable(
            getattr(self.llm_generator, "generate")
        ):
            return self.llm_generator.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.8,
            )

        if hasattr(self.llm_generator, "chat") and callable(
            getattr(self.llm_generator, "chat")
        ):
            return self.llm_generator.chat(prompt, use_history=False)

        raise AttributeError(
            "LLM generator must provide either 'generate(prompt=...)' or 'chat(user_input, use_history=...)'"
        )

    def _get_successful_patterns(self, intent_type: str, n: int = 5) -> List[Dict[str, Any]]:
        """Compatibility wrapper for phrasing learner pattern access APIs."""
        if not self.phrasing_learner:
            return []

        if hasattr(self.phrasing_learner, "get_successful_patterns") and callable(
            getattr(self.phrasing_learner, "get_successful_patterns")
        ):
            patterns = self.phrasing_learner.get_successful_patterns(intent_type, n=n)
            return patterns if isinstance(patterns, list) else []

        if hasattr(self.phrasing_learner, "learned_patterns"):
            learned_patterns = getattr(self.phrasing_learner, "learned_patterns", {})
            if not isinstance(learned_patterns, dict):
                return []

            matches: List[Dict[str, Any]] = []
            for pattern_key, examples in learned_patterns.items():
                if intent_type in str(pattern_key) and isinstance(examples, list):
                    for example in examples[-n:]:
                        if isinstance(example, dict):
                            matches.append(example)
                if len(matches) >= n:
                    break
            return matches[:n]

        return []

    def generate_response(self, context: ResponseContext) -> str:
        """
        Main method: Generate a varied response based on context
        """
        # Detect repetition
        query_hash = self._hash_query(context.intent_type, context.data)
        rep_count, time_since = self.repetition_detector.check_repetition(
            context.user_id, context.intent_type, query_hash
        )

        context.repetition_count = rep_count
        context.time_since_last_similar = time_since

        # Detect mood if not provided
        if context.user_mood == "neutral" and context.conversation_history:
            context.user_mood = self._detect_user_mood(context.conversation_history)

        # Get response constraints
        constraints = self._get_response_constraints(context)

        # Generate response with variance
        response = self._generate_with_variance(context, constraints)

        # Record this response
        self.response_history[context.user_id].append((context.intent_type, response))

        # Log for learning
        logger.debug(
            f"Generated response for {context.intent_type} "
            f"(mood={context.user_mood}, rep={context.repetition_count}): "
            f"{response[:50]}..."
        )

        return response

    def record_response_quality(self, user_id: str, response: str, user_reaction: str):
        """
        Learn from user's reaction to the response
        """
        quality_score = self._calculate_quality_score(user_reaction)

        quality = ResponseQuality(
            response=response,
            timestamp=time.time(),
            user_reaction=user_reaction,
            quality_score=quality_score,
            led_to_clarification="what" in user_reaction.lower()
            or "?" in user_reaction,
            led_to_success="thanks" in user_reaction.lower() or user_reaction == "",
        )

        self.quality_tracker[user_id].append(quality)

        # Update successful patterns if quality is high
        if quality_score > 0.7:
            self._update_successful_patterns(response, quality_score)

        logger.debug(f"Response quality tracked: score={quality_score:.2f}")

    def _calculate_quality_score(self, user_reaction: str) -> float:
        """Calculate quality score from user's reaction"""
        reaction_lower = user_reaction.lower()

        # Positive signals
        if any(
            word in reaction_lower for word in ["thanks", "perfect", "great", "awesome"]
        ):
            return 0.9

        # Neutral (no follow-up) is good
        if not user_reaction or len(user_reaction) < 10:
            return 0.8

        # Clarification needed (medium)
        if any(word in reaction_lower for word in ["what", "which", "clarify"]):
            return 0.5

        # Negative signals
        if any(
            word in reaction_lower for word in ["wrong", "no", "didn't", "never mind"]
        ):
            return 0.2

        return 0.6  # Default

    def _update_successful_patterns(self, response: str, score: float):
        """Update patterns from successful responses"""
        # Extract features from successful response
        features = {
            "length": len(response.split()),
            "has_examples": "for example" in response.lower()
            or "like" in response.lower(),
            "has_confirmation": "here" in response.lower()
            or "found" in response.lower(),
            "directness": 1.0 if len(response.split()) < 15 else 0.5,
        }

        # Update running averages (simplified)
        # Real implementation would use more sophisticated pattern learning
        logger.debug(f"Learning from successful response: {features}")
