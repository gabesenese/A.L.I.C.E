"""
Tone Trajectory Engine - Tier 2: Stable Response Variance

Locks a consistent tone trajectory for the entire session.
Ensures phrasing learner respects this constraint.
"""

import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ToneStyle(Enum):
    """Supported tone styles."""

    FORMAL = "formal"  # Technical, professional, precise
    CASUAL = "casual"  # Conversational, friendly, relaxed
    TECHNICAL = "technical"  # Specialized terminology, detailed
    CONCISE = "concise"  # Short, direct, minimal words
    VERBOSE = "verbose"  # Detailed, explanatory, thorough
    WARM = "warm"  # Empathetic, supportive, personal


@dataclass
class ToneCharacteristics:
    """Characteristics of a tone style."""

    name: str
    style: ToneStyle
    vocabulary_level: str  # simple, intermediate, technical
    sentence_length: str  # short, medium, long
    emoji_frequency: float  # 0.0-1.0
    contractions: bool  # Use "don't" vs "do not"
    humor_level: float  # 0.0-1.0
    personhood_claims: bool  # "I think" vs "My analysis suggests"
    explanation_depth: str  # minimal, moderate, detailed
    markers: List[str]  # Characteristic phrases

    def to_dict(self) -> Dict:
        """Convert to dict for LLM prompts."""
        return {
            "name": self.name,
            "style": self.style.value,
            "vocabulary_level": self.vocabulary_level,
            "sentence_length": self.sentence_length,
            "emoji_frequency": self.emoji_frequency,
            "contractions": self.contractions,
            "humor_level": self.humor_level,
            "personhood_claims": self.personhood_claims,
            "explanation_depth": self.explanation_depth,
            "markers": self.markers[:5],  # Top 5 markers
        }


class ToneTrajectoryEngine:
    """Manages consistent tone throughout a session."""

    def __init__(self):
        self.tone_styles = self._init_tone_styles()
        self.current_tone: Optional[ToneStyle] = None
        self.session_characteristics: Optional[ToneCharacteristics] = None
        self.tone_stabilization_threshold = 3  # Used N times before locked
        self.tone_usage_count: Dict[str, int] = {}
        self.response_history: List[Dict] = []
        self.tone_consistency_score = 1.0

    def _init_tone_styles(self) -> Dict[ToneStyle, ToneCharacteristics]:
        """Initialize tone style definitions."""
        return {
            ToneStyle.FORMAL: ToneCharacteristics(
                name="Formal",
                style=ToneStyle.FORMAL,
                vocabulary_level="technical",
                sentence_length="medium",
                emoji_frequency=0.0,
                contractions=False,
                humor_level=0.0,
                personhood_claims=False,
                explanation_depth="moderate",
                markers=[
                    "I shall",
                    "I would",
                    "It is beneficial",
                    "Furthermore",
                    "In addition",
                    "One must note",
                ],
            ),
            ToneStyle.CASUAL: ToneCharacteristics(
                name="Casual",
                style=ToneStyle.CASUAL,
                vocabulary_level="simple",
                sentence_length="short",
                emoji_frequency=0.2,
                contractions=True,
                humor_level=0.3,
                personhood_claims=True,
                explanation_depth="moderate",
                markers=[
                    "ha",
                    "yeah",
                    "honestly",
                    "cool",
                    "pretty neat",
                    "you know",
                    "basically",
                    "so like",
                ],
            ),
            ToneStyle.TECHNICAL: ToneCharacteristics(
                name="Technical",
                style=ToneStyle.TECHNICAL,
                vocabulary_level="technical",
                sentence_length="long",
                emoji_frequency=0.0,
                contractions=False,
                humor_level=0.0,
                personhood_claims=False,
                explanation_depth="detailed",
                markers=[
                    "Algorithm",
                    "Implementation",
                    "Abstraction",
                    "Optimization",
                    "Architecture",
                    "Interface",
                ],
            ),
            ToneStyle.CONCISE: ToneCharacteristics(
                name="Concise",
                style=ToneStyle.CONCISE,
                vocabulary_level="simple",
                sentence_length="short",
                emoji_frequency=0.0,
                contractions=True,
                humor_level=0.0,
                personhood_claims=False,
                explanation_depth="minimal",
                markers=[
                    "Done.",
                    "Yes.",
                    "No.",
                    "Check:",
                    "Note:",
                    "Result:",
                    "Fix:",
                    "Next:",
                ],
            ),
            ToneStyle.VERBOSE: ToneCharacteristics(
                name="Verbose",
                style=ToneStyle.VERBOSE,
                vocabulary_level="intermediate",
                sentence_length="long",
                emoji_frequency=0.1,
                contractions=True,
                humor_level=0.1,
                personhood_claims=True,
                explanation_depth="detailed",
                markers=[
                    "Let me explain",
                    "So what's happening here",
                    "The thing about",
                    "To be clear",
                    "What I mean is",
                ],
            ),
            ToneStyle.WARM: ToneCharacteristics(
                name="Warm",
                style=ToneStyle.WARM,
                vocabulary_level="simple",
                sentence_length="medium",
                emoji_frequency=0.3,
                contractions=True,
                humor_level=0.2,
                personhood_claims=True,
                explanation_depth="moderate",
                markers=[
                    "I understand",
                    "I see",
                    "That's tough",
                    "You're doing great",
                    "It's okay",
                    "I've got you",
                ],
            ),
        }

    def set_session_tone(self, tone: ToneStyle) -> None:
        """Lock a tone for the entire session."""
        if tone not in self.tone_styles:
            logger.warning(f"[Tone] Unknown tone: {tone}")
            return

        self.current_tone = tone
        self.session_characteristics = self.tone_styles[tone]
        self.tone_usage_count[tone.value] = 0

        logger.info(f"[Tone] Session tone set to: {tone.value}")

    def detect_user_tone_preference(self, user_input: str) -> Optional[ToneStyle]:
        """
        Detect what tone the user prefers based on their communication style.
        """
        user_lower = user_input.lower()

        # Check for explicit tone requests
        if any(
            phrase in user_lower
            for phrase in ["be formal", "formal tone", "professional"]
        ):
            return ToneStyle.FORMAL

        if any(
            phrase in user_lower for phrase in ["casual", "chill", "relax", "just chat"]
        ):
            return ToneStyle.CASUAL

        if any(
            phrase in user_lower
            for phrase in ["be brief", "short answer", "tldr", "concise"]
        ):
            return ToneStyle.CONCISE

        if any(
            phrase in user_lower
            for phrase in ["explain", "detailed", "full explanation", "deep dive"]
        ):
            return ToneStyle.VERBOSE

        # Infer from communication style
        has_contractions = (
            "'ve" in user_input or "'s" in user_input or "n't" in user_input
        )
        is_technical = any(
            word in user_lower
            for word in ["api", "algorithm", "architecture", "refactor"]
        )
        is_long = len(user_input) > 200
        is_empathetic = any(
            word in user_lower for word in ["feel", "frustrated", "stuck", "hard"]
        )

        if is_empathetic:
            return ToneStyle.WARM

        if is_technical:
            return ToneStyle.TECHNICAL

        if is_long and has_contractions:
            return ToneStyle.CASUAL

        # Default based on message characteristics
        if len(user_input.split()) > 30:
            return ToneStyle.VERBOSE

        return None

    def get_tone_prompt_addendum(self) -> str:
        """
        Get a prompt snippet to enforce the current tone in LLM calls.
        """
        if not self.session_characteristics:
            return ""

        char = self.session_characteristics

        addendum = f"""
TONE CONSTRAINT FOR THIS RESPONSE:
- Style: {char.name}
- Vocabulary level: {char.vocabulary_level}
- Sentence length: {char.sentence_length}
- Depth: {char.explanation_depth}
- Contractions: {"allowed" if char.contractions else "avoid"}
- Personhood claims: {'allowed ("I think...")' if char.personhood_claims else 'avoid ("suggest" instead)'}
- Emoji: {"very occasional" if char.emoji_frequency > 0 else "never"}
- Humor: {"occasional" if char.humor_level > 0.1 else "avoid"}
- Characteristic phrases: {", ".join(char.markers[:3])}

Maintain consistency with this tone throughout.
"""
        return addendum.strip()

    def evaluate_response_consistency(self, response: str) -> float:
        """
        Evaluate how consistent a response is with the session tone (0.0-1.0).
        """
        if not self.session_characteristics:
            return 1.0

        char = self.session_characteristics
        score = 1.0
        issues = []

        # Check for contradiction to contractions setting
        if char.contractions and "do not" in response and "don't" in response is False:
            score -= 0.1  # Should use contractions
            issues.append("uses full forms when contractions expected")

        if not char.contractions and ("don't" in response or "can't" in response):
            score -= 0.1  # Should avoid contractions
            issues.append("uses contractions when they should be avoided")

        # Check sentence length trend
        sentences = response.split(".")
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences)

        if char.sentence_length == "short" and avg_length > 15:
            score -= 0.1
            issues.append("sentences are too long for concise style")
        elif char.sentence_length == "long" and avg_length < 8:
            score -= 0.05
            issues.append("sentences are too short for verbose style")

        # Check for personhood claims consistency
        has_personhood = any(
            phrase in response for phrase in ["I think", "I believe", "I feel"]
        )
        if char.personhood_claims and not has_personhood and len(response) > 100:
            score -= 0.05  # Optional penalty for missing personhood in long response

        if not char.personhood_claims and has_personhood:
            score -= 0.1  # Clear violation
            issues.append("uses personhood claims when style requires neutrality")

        # Check emoji usage
        emoji_present = any(ord(c) > 127 for c in response)
        if char.emoji_frequency == 0.0 and emoji_present:
            score -= 0.05
            issues.append("uses emoji when style forbids it")

        if issues:
            logger.debug(f"[Tone] Consistency issues: {issues}")

        return max(0.0, min(1.0, score))

    def record_response(self, user_input: str, alice_response: str) -> None:
        """Record a response for tone analysis."""
        consistency = self.evaluate_response_consistency(alice_response)

        self.response_history.append(
            {
                "user_input": user_input,
                "alice_response": alice_response,
                "timestamp": str(__import__("datetime").datetime.now().isoformat()),
                "consistency_score": consistency,
            }
        )

        # Update overall consistency
        if self.response_history:
            scores = [r["consistency_score"] for r in self.response_history]
            self.tone_consistency_score = sum(scores) / len(scores)

    def get_session_tone_report(self) -> Dict:
        """Get a report on session tone consistency."""
        return {
            "current_tone": self.current_tone.value if self.current_tone else None,
            "characteristics": (
                self.session_characteristics.to_dict()
                if self.session_characteristics
                else None
            ),
            "consistency_score": round(self.tone_consistency_score, 3),
            "responses_evaluated": len(self.response_history),
            "is_locked": self.current_tone is not None,
        }
