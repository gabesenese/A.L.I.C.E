"""
Personality Evolution Engine for A.L.I.C.E
Evolves A.L.I.C.E's personality traits based on user interactions
Adapts communication style to match each user's preferences
"""

import logging
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PersonalityTraits:
    """
    A.L.I.C.E's personality dimensions
    All values range from 0.0 to 1.0
    """

    verbosity: float = 0.5  # 0=terse, 1=verbose
    formality: float = 0.3  # 0=casual, 1=formal
    humor: float = 0.4  # 0=serious, 1=playful
    directness: float = 0.6  # 0=gentle/indirect, 1=blunt/direct
    enthusiasm: float = 0.5  # 0=neutral, 1=excitable
    empathy: float = 0.7  # 0=factual, 1=emotionally aware

    def copy(self):
        """Create a copy of these traits"""
        return PersonalityTraits(**asdict(self))

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "PersonalityTraits":
        """Create from dictionary"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class InteractionSignal:
    """Signal from a user interaction that affects personality"""

    user_id: str
    signal_type: str  # 'verbosity', 'formality', 'humor', etc.
    direction: float  # -1.0 to +1.0 (decrease or increase trait)
    strength: float  # 0.0 to 1.0 (how strong the signal is)
    reason: str  # Why this signal was detected


class PersonalityEvolutionEngine:
    """
    Evolves A.L.I.C.E's personality based on interactions
    Maintains per-user personality adaptations
    """

    def __init__(self, data_dir: str = "data/personality"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Base personality (default for all users)
        self.base_traits = PersonalityTraits()

        # Per-user personality adaptations
        self.user_traits: Dict[str, PersonalityTraits] = {}

        # Learning rate (how fast personality adapts)
        self.learning_rate = 0.05

        # Min/max bounds on trait values
        self.min_trait = 0.1
        self.max_trait = 0.9

        # Signal history for analysis
        self.signal_history = defaultdict(list)

        # Load persisted traits
        self._load_traits()

    def _load_traits(self):
        """Load personality traits from disk"""
        traits_file = self.data_dir / "personality_traits.json"
        if traits_file.exists():
            try:
                with open(traits_file, "r") as f:
                    data = json.load(f)

                # Load base traits
                if "base" in data:
                    self.base_traits = PersonalityTraits.from_dict(data["base"])

                # Load per-user traits
                for user_id, traits_data in data.get("users", {}).items():
                    self.user_traits[user_id] = PersonalityTraits.from_dict(traits_data)

                logger.info(
                    f"Loaded personality traits for {len(self.user_traits)} users"
                )
            except Exception as e:
                logger.error(f"Failed to load personality traits: {e}")

    def _save_traits(self):
        """Save personality traits to disk"""
        traits_file = self.data_dir / "personality_traits.json"
        try:
            data = {
                "base": self.base_traits.to_dict(),
                "users": {
                    user_id: traits.to_dict()
                    for user_id, traits in self.user_traits.items()
                },
            }

            with open(traits_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug("Saved personality traits to disk")
        except Exception as e:
            logger.error(f"Failed to save personality traits: {e}")

    def get_traits_for_user(self, user_id: str) -> PersonalityTraits:
        """Get personality traits adapted for specific user"""
        if user_id not in self.user_traits:
            # Initialize with copy of base traits
            self.user_traits[user_id] = self.base_traits.copy()

        return self.user_traits[user_id]

    def detect_signals_from_interaction(
        self,
        user_id: str,
        user_input: str,
        alice_response: str,
        user_reaction: Optional[str] = None,
    ) -> List[InteractionSignal]:
        """
        Detect personality signals from an interaction
        """
        signals = []

        # Analyze user's communication style
        user_signals = self._analyze_user_style(user_id, user_input)
        signals.extend(user_signals)

        # Analyze user's reaction to A.L.I.C.E's response
        if user_reaction:
            reaction_signals = self._analyze_user_reaction(
                user_id, alice_response, user_reaction
            )
            signals.extend(reaction_signals)

        # Store signals in history
        self.signal_history[user_id].extend(signals)

        return signals

    def _analyze_user_style(
        self, user_id: str, user_input: str
    ) -> List[InteractionSignal]:
        """Analyze user's communication style and generate signals"""
        signals = []
        input_lower = user_input.lower()
        word_count = len(user_input.split())

        # Verbosity signal
        if word_count < 5:
            # User is terse, A.L.I.C.E should be too
            signals.append(
                InteractionSignal(
                    user_id=user_id,
                    signal_type="verbosity",
                    direction=-1.0,
                    strength=0.5,
                    reason="User uses brief messages",
                )
            )
        elif word_count > 20:
            # User is verbose, A.L.I.C.E can be too
            signals.append(
                InteractionSignal(
                    user_id=user_id,
                    signal_type="verbosity",
                    direction=+1.0,
                    strength=0.5,
                    reason="User uses detailed messages",
                )
            )

        # Formality signal
        if any(
            word in input_lower
            for word in ["please", "thank you", "kindly", "would you"]
        ):
            signals.append(
                InteractionSignal(
                    user_id=user_id,
                    signal_type="formality",
                    direction=+1.0,
                    strength=0.3,
                    reason="User uses polite/formal language",
                )
            )
        elif any(
            word in input_lower
            for word in ["yo", "hey", "sup", "gimme", "gonna", "wanna"]
        ):
            signals.append(
                InteractionSignal(
                    user_id=user_id,
                    signal_type="formality",
                    direction=-1.0,
                    strength=0.4,
                    reason="User uses casual/informal language",
                )
            )

        # Humor signal
        if any(char in user_input for char in ["😂", "😄", "😊", "🤣"]):
            signals.append(
                InteractionSignal(
                    user_id=user_id,
                    signal_type="humor",
                    direction=+1.0,
                    strength=0.6,
                    reason="User uses humor/emojis",
                )
            )

        # Directness signal
        if any(word in input_lower for word in ["just", "simply", "quick", "brief"]):
            signals.append(
                InteractionSignal(
                    user_id=user_id,
                    signal_type="directness",
                    direction=+1.0,
                    strength=0.5,
                    reason="User requests direct/brief answers",
                )
            )

        return signals

    def _analyze_user_reaction(
        self, user_id: str, alice_response: str, user_reaction: str
    ) -> List[InteractionSignal]:
        """Analyze user's reaction to A.L.I.C.E's response"""
        signals = []
        reaction_lower = user_reaction.lower()

        # Check if user found response too verbose
        if any(
            phrase in reaction_lower
            for phrase in ["too long", "too much", "tldr", "shorter", "brief"]
        ):
            signals.append(
                InteractionSignal(
                    user_id=user_id,
                    signal_type="verbosity",
                    direction=-1.0,
                    strength=0.8,
                    reason="User indicated response was too verbose",
                )
            )

        # Check if user needed more detail
        if any(
            phrase in reaction_lower
            for phrase in [
                "more detail",
                "explain more",
                "elaborate",
                "what do you mean",
            ]
        ):
            signals.append(
                InteractionSignal(
                    user_id=user_id,
                    signal_type="verbosity",
                    direction=+1.0,
                    strength=0.8,
                    reason="User requested more detail",
                )
            )
            signals.append(
                InteractionSignal(
                    user_id=user_id,
                    signal_type="directness",
                    direction=-1.0,
                    strength=0.5,
                    reason="User found response unclear",
                )
            )

        # Check if user appreciated humor/personality
        if any(phrase in reaction_lower for phrase in ["haha", "lol", "funny", "😂"]):
            signals.append(
                InteractionSignal(
                    user_id=user_id,
                    signal_type="humor",
                    direction=+1.0,
                    strength=0.7,
                    reason="User appreciated humor",
                )
            )

        # Check if user found response too casual
        if any(
            phrase in reaction_lower for phrase in ["professional", "formal", "serious"]
        ):
            signals.append(
                InteractionSignal(
                    user_id=user_id,
                    signal_type="formality",
                    direction=+1.0,
                    strength=0.8,
                    reason="User prefers formal tone",
                )
            )

        # Check for frustration (need more empathy)
        if any(
            phrase in reaction_lower
            for phrase in ["frustrated", "annoying", "wrong", "didn't help"]
        ):
            signals.append(
                InteractionSignal(
                    user_id=user_id,
                    signal_type="empathy",
                    direction=+1.0,
                    strength=0.9,
                    reason="User expressed frustration",
                )
            )

        return signals

    def apply_signals(self, signals: List[InteractionSignal], user_id: str):
        """Apply personality signals to update traits"""
        if not signals:
            return

        traits = self.get_traits_for_user(user_id)
        changes_made = False

        for signal in signals:
            # Get current trait value
            current_value = getattr(traits, signal.signal_type)

            # Calculate adjustment
            adjustment = signal.direction * signal.strength * self.learning_rate
            new_value = current_value + adjustment

            # Clamp to bounds
            new_value = max(self.min_trait, min(self.max_trait, new_value))

            # Apply if changed
            if abs(new_value - current_value) > 0.001:
                setattr(traits, signal.signal_type, new_value)
                changes_made = True

                logger.debug(
                    f"Personality evolution for {user_id}: "
                    f"{signal.signal_type} "
                    f"{current_value:.2f} → {new_value:.2f} "
                    f"(reason: {signal.reason})"
                )

        if changes_made:
            self._save_traits()

    def learn_from_interaction(
        self,
        user_id: str,
        user_input: str,
        alice_response: str,
        user_reaction: Optional[str] = None,
    ):
        """
        Main method: Learn from a complete interaction
        """
        # Detect signals
        signals = self.detect_signals_from_interaction(
            user_id, user_input, alice_response, user_reaction
        )

        # Apply signals to evolve personality
        if signals:
            self.apply_signals(signals, user_id)

            logger.info(
                f"Personality updated for {user_id} based on {len(signals)} signals"
            )

    def get_personality_profile(self, user_id: str) -> Dict[str, any]:
        """Get human-readable personality profile for user"""
        traits = self.get_traits_for_user(user_id)

        def describe_trait(value: float, low_label: str, high_label: str) -> str:
            if value < 0.3:
                return f"very {low_label}"
            elif value < 0.45:
                return low_label
            elif value < 0.55:
                return "balanced"
            elif value < 0.7:
                return high_label
            else:
                return f"very {high_label}"

        profile = {
            "user_id": user_id,
            "traits": traits.to_dict(),
            "description": {
                "verbosity": describe_trait(traits.verbosity, "concise", "detailed"),
                "formality": describe_trait(traits.formality, "casual", "formal"),
                "humor": describe_trait(traits.humor, "serious", "playful"),
                "directness": describe_trait(traits.directness, "gentle", "direct"),
                "enthusiasm": describe_trait(traits.enthusiasm, "calm", "enthusiastic"),
                "empathy": describe_trait(traits.empathy, "factual", "empathetic"),
            },
            "sample_count": len(self.signal_history.get(user_id, [])),
        }

        return profile

    def reset_user_traits(self, user_id: str):
        """Reset user's traits to base personality"""
        if user_id in self.user_traits:
            del self.user_traits[user_id]
            self._save_traits()
            logger.info(f"Reset personality traits for {user_id}")
