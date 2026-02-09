"""
User Profile & Preference Engine
==================================
Deep user modeling system that learns preferences, habits, communication style,
and behavioral patterns over time.

Makes Alice personalized and adaptive like Jarvis.
"""

import json
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class UserPreference:
    """A learned user preference"""
    key: str
    value: Any
    confidence: float = 0.5
    evidence_count: int = 1
    last_updated: float = field(default_factory=time.time)
    category: str = 'general'


@dataclass
class BehavioralPattern:
    """A detected behavioral pattern"""
    pattern_id: str
    pattern_type: str
    description: str
    frequency: float = 0.0
    last_observed: float = field(default_factory=time.time)
    times_observed: int = 1
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """Complete user profile"""
    user_id: str
    name: Optional[str] = None
    preferences: Dict[str, UserPreference] = field(default_factory=dict)
    behaviors: Dict[str, BehavioralPattern] = field(default_factory=dict)
    communication_style: Dict[str, float] = field(default_factory=dict)
    schedule: Dict[str, List[str]] = field(default_factory=dict)
    interests: Set[str] = field(default_factory=set)
    expertise_areas: Dict[str, float] = field(default_factory=dict)
    interaction_count: int = 0
    first_interaction: float = field(default_factory=time.time)
    last_interaction: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class UserProfileEngine:
    """
    Advanced user profiling and preference learning engine.

    Learns:
    - Communication preferences (brief vs detailed, formal vs casual)
    - Task preferences (tools, methods, workflows)
    - Behavioral patterns (morning routines, work habits)
    - Schedule and availability patterns
    - Topics of interest and expertise
    - Decision-making patterns
    """

    def __init__(self, storage_path: str = "data/user_profiles"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Current user profile
        self.profile: Optional[UserProfile] = None

        # Learning parameters
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        self.min_evidence = 3

        # Communication style dimensions
        self.style_dimensions = {
            'brevity': 0.5,  # 0=verbose, 1=brief
            'formality': 0.5,  # 0=casual, 1=formal
            'technicality': 0.5,  # 0=simple, 1=technical
            'proactivity': 0.5,  # 0=reactive, 1=proactive
            'detail_orientation': 0.5  # 0=high-level, 1=detailed
        }

    def load_profile(self, user_id: str = "default") -> UserProfile:
        """Load user profile from storage"""
        profile_file = self.storage_path / f"{user_id}.json"

        if profile_file.exists():
            try:
                with open(profile_file, 'r') as f:
                    data = json.load(f)

                # Reconstruct profile
                self.profile = UserProfile(user_id=user_id)
                self.profile.name = data.get('name')
                self.profile.interaction_count = data.get('interaction_count', 0)
                self.profile.first_interaction = data.get('first_interaction', time.time())
                self.profile.last_interaction = data.get('last_interaction', time.time())
                self.profile.interests = set(data.get('interests', []))
                self.profile.expertise_areas = data.get('expertise_areas', {})
                self.profile.communication_style = data.get('communication_style', self.style_dimensions.copy())
                self.profile.schedule = data.get('schedule', {})
                self.profile.metadata = data.get('metadata', {})

                # Reconstruct preferences
                for pref_data in data.get('preferences', []):
                    pref = UserPreference(**pref_data)
                    self.profile.preferences[pref.key] = pref

                # Reconstruct behaviors
                for behavior_data in data.get('behaviors', []):
                    behavior = BehavioralPattern(**behavior_data)
                    self.profile.behaviors[behavior.pattern_id] = behavior

                logger.info(f"Loaded profile for {user_id}: {self.profile.interaction_count} interactions")

            except Exception as e:
                logger.error(f"Error loading profile: {e}")
                self.profile = UserProfile(user_id=user_id)
        else:
            logger.info(f"Creating new profile for {user_id}")
            self.profile = UserProfile(user_id=user_id)

        return self.profile

    def save_profile(self):
        """Save user profile to storage"""
        if not self.profile:
            return

        profile_file = self.storage_path / f"{self.profile.user_id}.json"

        try:
            data = {
                'user_id': self.profile.user_id,
                'name': self.profile.name,
                'interaction_count': self.profile.interaction_count,
                'first_interaction': self.profile.first_interaction,
                'last_interaction': self.profile.last_interaction,
                'interests': list(self.profile.interests),
                'expertise_areas': self.profile.expertise_areas,
                'communication_style': self.profile.communication_style,
                'schedule': self.profile.schedule,
                'metadata': self.profile.metadata,
                'preferences': [asdict(pref) for pref in self.profile.preferences.values()],
                'behaviors': [asdict(behavior) for behavior in self.profile.behaviors.values()]
            }

            with open(profile_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving profile: {e}")

    def record_interaction(
        self,
        user_input: str,
        alice_response: str,
        intent: Optional[str] = None,
        entities: Dict[str, Any] = None,
        feedback: Optional[str] = None
    ):
        """
        Record an interaction and learn from it.

        Args:
            user_input: User's message
            alice_response: Alice's response
            intent: Detected intent
            entities: Extracted entities
            feedback: User feedback (positive/negative/neutral)
        """
        if not self.profile:
            self.load_profile()

        self.profile.interaction_count += 1
        self.profile.last_interaction = time.time()

        # Learn communication style
        self._learn_communication_style(user_input, alice_response, feedback)

        # Learn preferences from entities and intent
        if entities:
            self._learn_from_entities(entities, intent)

        # Detect behavioral patterns
        self._detect_patterns(user_input, intent)

        # Learn interests from topics
        self._learn_interests(user_input, entities)

        # Auto-save periodically
        if self.profile.interaction_count % 10 == 0:
            self.save_profile()

    def _learn_communication_style(
        self,
        user_input: str,
        alice_response: str,
        feedback: Optional[str]
    ):
        """Learn user's preferred communication style"""
        if not self.profile:
            return

        # Analyze user input characteristics
        input_length = len(user_input.split())
        is_question = '?' in user_input
        has_technical_terms = any(term in user_input.lower() for term in ['api', 'function', 'class', 'algorithm'])

        # Brevity preference
        if input_length < 10:
            self._update_style_dimension('brevity', 0.7)  # Prefers brief
        elif input_length > 30:
            self._update_style_dimension('brevity', 0.3)  # Prefers detailed

        # Technicality preference
        if has_technical_terms:
            self._update_style_dimension('technicality', 0.7)

        # Learn from feedback
        if feedback == 'positive':
            response_length = len(alice_response.split())
            if response_length < 50:
                self._update_style_dimension('detail_orientation', 0.4)  # Liked brief response
            else:
                self._update_style_dimension('detail_orientation', 0.6)  # Liked detailed response

    def _update_style_dimension(self, dimension: str, target_value: float):
        """Update a communication style dimension with learning rate"""
        if dimension not in self.profile.communication_style:
            self.profile.communication_style[dimension] = 0.5

        current = self.profile.communication_style[dimension]
        # Exponential moving average
        new_value = current * (1 - self.learning_rate) + target_value * self.learning_rate
        self.profile.communication_style[dimension] = new_value

    def _learn_from_entities(self, entities: Dict[str, Any], intent: Optional[str]):
        """Learn preferences from extracted entities"""
        if not self.profile:
            return

        for entity_name, entity_data in entities.items():
            entity_type = entity_data.get('type', 'general') if isinstance(entity_data, dict) else 'general'

            # Learn tool preferences
            if intent in ['code_analysis', 'file_operations']:
                pref_key = f"preferred_tool_{intent}"
                self._update_preference(pref_key, entity_name, category='tools')

            # Learn topic preferences
            if entity_type == 'concept':
                self._update_preference(f"topic_{entity_name}", True, category='interests')

    def _update_preference(
        self,
        key: str,
        value: Any,
        category: str = 'general',
        confidence_boost: float = 0.1
    ):
        """Update or create a preference"""
        if not self.profile:
            return

        if key in self.profile.preferences:
            pref = self.profile.preferences[key]
            pref.value = value
            pref.evidence_count += 1
            pref.confidence = min(1.0, pref.confidence + confidence_boost)
            pref.last_updated = time.time()
        else:
            pref = UserPreference(
                key=key,
                value=value,
                category=category,
                confidence=0.3 + confidence_boost
            )
            self.profile.preferences[key] = pref

    def _detect_patterns(self, user_input: str, intent: Optional[str]):
        """Detect behavioral patterns"""
        if not self.profile:
            return

        # Time-based patterns
        current_hour = datetime.now().hour

        if 6 <= current_hour < 9:
            self._record_pattern('morning_routine', 'time_of_day', 'Active in early morning')
        elif 9 <= current_hour < 12:
            self._record_pattern('morning_work', 'time_of_day', 'Active during morning work hours')
        elif 12 <= current_hour < 14:
            self._record_pattern('midday_activity', 'time_of_day', 'Active around midday')
        elif 14 <= current_hour < 18:
            self._record_pattern('afternoon_work', 'time_of_day', 'Active during afternoon')
        elif 18 <= current_hour < 22:
            self._record_pattern('evening_activity', 'time_of_day', 'Active in evening')
        elif 22 <= current_hour or current_hour < 6:
            self._record_pattern('night_owl', 'time_of_day', 'Active late night or early morning')

        # Intent-based patterns
        if intent:
            pattern_id = f"frequent_{intent}"
            self._record_pattern(pattern_id, 'intent_frequency', f"Frequently uses {intent}")

    def _record_pattern(
        self,
        pattern_id: str,
        pattern_type: str,
        description: str
    ):
        """Record observation of a behavioral pattern"""
        if not self.profile:
            return

        if pattern_id in self.profile.behaviors:
            pattern = self.profile.behaviors[pattern_id]
            pattern.times_observed += 1
            pattern.last_observed = time.time()

            # Update confidence
            pattern.confidence = min(1.0, pattern.times_observed / 20.0)

            # Update frequency (observations per day)
            days_active = (time.time() - self.profile.first_interaction) / 86400.0
            if days_active > 0:
                pattern.frequency = pattern.times_observed / days_active

        else:
            pattern = BehavioralPattern(
                pattern_id=pattern_id,
                pattern_type=pattern_type,
                description=description,
                confidence=0.05
            )
            self.profile.behaviors[pattern_id] = pattern

    def _learn_interests(self, user_input: str, entities: Dict[str, Any]):
        """Learn user interests from conversation"""
        if not self.profile or not entities:
            return

        # Extract topics from entities
        for entity_name, entity_data in entities.items():
            if isinstance(entity_data, dict):
                entity_type = entity_data.get('type', '')
                if entity_type in ['concept', 'topic', 'technology']:
                    self.profile.interests.add(entity_name)

    def get_preference(self, key: str) -> Optional[Any]:
        """Get a user preference value"""
        if not self.profile or key not in self.profile.preferences:
            return None

        pref = self.profile.preferences[key]
        if pref.confidence >= self.confidence_threshold:
            return pref.value

        return None

    def get_communication_style(self) -> Dict[str, float]:
        """Get user's communication style preferences"""
        if not self.profile:
            return self.style_dimensions.copy()

        return self.profile.communication_style.copy()

    def should_be_brief(self) -> bool:
        """Check if user prefers brief responses"""
        style = self.get_communication_style()
        return style.get('brevity', 0.5) > 0.6

    def should_be_technical(self) -> bool:
        """Check if user prefers technical language"""
        style = self.get_communication_style()
        return style.get('technicality', 0.5) > 0.6

    def get_active_patterns(self, min_confidence: float = 0.5) -> List[BehavioralPattern]:
        """Get high-confidence behavioral patterns"""
        if not self.profile:
            return []

        return [
            pattern for pattern in self.profile.behaviors.values()
            if pattern.confidence >= min_confidence
        ]

    def get_schedule_prediction(self) -> Dict[str, Any]:
        """Predict user availability based on learned patterns"""
        if not self.profile:
            return {}

        current_hour = datetime.now().hour
        day_of_week = datetime.now().strftime('%A')

        # Check for time-based patterns
        active_patterns = self.get_active_patterns(min_confidence=0.6)
        current_time_patterns = [
            p for p in active_patterns
            if p.pattern_type == 'time_of_day' and p.pattern_id in p.description
        ]

        prediction = {
            'likely_available': len(current_time_patterns) > 0,
            'confidence': max([p.confidence for p in current_time_patterns], default=0.0),
            'typical_activity': current_time_patterns[0].description if current_time_patterns else 'unknown'
        }

        return prediction

    def get_profile_summary(self) -> Dict[str, Any]:
        """Get a summary of the user profile"""
        if not self.profile:
            return {}

        return {
            'user_id': self.profile.user_id,
            'name': self.profile.name,
            'interactions': self.profile.interaction_count,
            'days_active': (time.time() - self.profile.first_interaction) / 86400.0,
            'communication_style': self.get_communication_style(),
            'interests_count': len(self.profile.interests),
            'top_interests': list(self.profile.interests)[:5],
            'learned_preferences': len(self.profile.preferences),
            'behavioral_patterns': len(self.profile.behaviors),
            'high_confidence_preferences': len([
                p for p in self.profile.preferences.values()
                if p.confidence >= self.confidence_threshold
            ])
        }


# Global singleton
_profile_engine = None


def get_profile_engine(storage_path: str = "data/user_profiles") -> UserProfileEngine:
    """Get or create global user profile engine"""
    global _profile_engine
    if _profile_engine is None:
        _profile_engine = UserProfileEngine(storage_path=storage_path)
        _profile_engine.load_profile()  # Load default profile
    return _profile_engine
