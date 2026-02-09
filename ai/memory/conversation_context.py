"""
Conversation Context Manager
==============================
Advanced multi-turn conversation tracking with reference resolution,
topic tracking, and seamless context across messages.

This enables Alice to understand "it", "that", "them" and maintain
coherent multi-turn conversations like Jarvis.
"""

import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation"""
    turn_id: int
    timestamp: float
    user_input: str
    alice_response: str
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)
    topics: List[str] = field(default_factory=list)
    sentiment: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceCandidate:
    """Potential referent for a pronoun"""
    entity: str
    entity_type: str
    mention_turn: int
    recency_score: float
    salience_score: float
    total_score: float


class ConversationContextManager:
    """
    Manages conversation context across multiple turns.

    Capabilities:
    - Multi-turn history tracking
    - Reference resolution (pronouns -> entities)
    - Topic tracking and transitions
    - Context window management
    - Salience tracking (what's important right now)
    - Temporal awareness (recent vs old context)
    """

    def __init__(
        self,
        max_turns: int = 50,
        context_window: int = 10,
        recency_decay: float = 0.5
    ):
        self.max_turns = max_turns
        self.context_window = context_window
        self.recency_decay = recency_decay

        # Conversation history
        self.turns: deque[ConversationTurn] = deque(maxlen=max_turns)
        self.turn_counter = 0

        # Current context state
        self.current_topic: Optional[str] = None
        self.topic_history: List[Tuple[str, int]] = []  # (topic, turn_id)

        # Entity tracking for reference resolution
        self.mentioned_entities: Dict[str, List[int]] = {}  # entity -> [turn_ids where mentioned]
        self.entity_types: Dict[str, str] = {}  # entity -> type (person, file, concept, etc)

        # Salience tracking (what's currently important)
        self.salient_entities: List[str] = []

        # Last referenced items for quick "it/that" resolution
        self.last_file: Optional[str] = None
        self.last_person: Optional[str] = None
        self.last_concept: Optional[str] = None
        self.last_object: Optional[str] = None

    def add_turn(
        self,
        user_input: str,
        alice_response: str,
        intent: Optional[str] = None,
        entities: Dict[str, Any] = None,
        topics: List[str] = None,
        sentiment: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> ConversationTurn:
        """
        Add a new conversation turn.

        Args:
            user_input: User's message
            alice_response: Alice's response
            intent: Detected intent
            entities: Extracted entities
            topics: Detected topics
            sentiment: User sentiment
            metadata: Additional metadata

        Returns:
            The created ConversationTurn
        """
        self.turn_counter += 1

        turn = ConversationTurn(
            turn_id=self.turn_counter,
            timestamp=time.time(),
            user_input=user_input,
            alice_response=alice_response,
            intent=intent,
            entities=entities or {},
            topics=topics or [],
            sentiment=sentiment,
            metadata=metadata or {}
        )

        self.turns.append(turn)

        # Update entity tracking
        self._update_entity_tracking(turn)

        # Update topic tracking
        self._update_topic_tracking(turn)

        # Update salience
        self._update_salience()

        # Update last referenced items
        self._update_last_references(turn)

        return turn

    def _update_entity_tracking(self, turn: ConversationTurn):
        """Track entities mentioned in this turn"""
        for entity_name, entity_data in turn.entities.items():
            if entity_name not in self.mentioned_entities:
                self.mentioned_entities[entity_name] = []

            self.mentioned_entities[entity_name].append(turn.turn_id)

            # Store entity type if available
            if isinstance(entity_data, dict) and 'type' in entity_data:
                self.entity_types[entity_name] = entity_data['type']

    def _update_topic_tracking(self, turn: ConversationTurn):
        """Track topics and detect topic shifts"""
        if not turn.topics:
            return

        # Get primary topic
        primary_topic = turn.topics[0]

        if primary_topic != self.current_topic:
            # Topic shift detected
            if self.current_topic:
                logger.info(f"Topic shift: {self.current_topic} -> {primary_topic}")

            self.current_topic = primary_topic
            self.topic_history.append((primary_topic, turn.turn_id))

            # Limit topic history
            if len(self.topic_history) > 20:
                self.topic_history = self.topic_history[-20:]

    def _update_salience(self):
        """
        Update salience scores for entities.
        Recent mentions are more salient.
        """
        entity_scores = {}

        # Get recent turns (within context window)
        recent_turns = list(self.turns)[-self.context_window:]

        for turn in recent_turns:
            # Calculate recency multiplier
            turns_ago = self.turn_counter - turn.turn_id
            recency_multiplier = self.recency_decay ** turns_ago

            for entity in turn.entities.keys():
                if entity not in entity_scores:
                    entity_scores[entity] = 0.0

                # Add recency-weighted score
                entity_scores[entity] += recency_multiplier

        # Sort by score and take top entities
        sorted_entities = sorted(
            entity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        self.salient_entities = [entity for entity, score in sorted_entities[:5]]

    def _update_last_references(self, turn: ConversationTurn):
        """Update last referenced items by type"""
        for entity_name, entity_data in turn.entities.items():
            entity_type = self.entity_types.get(entity_name, 'object')

            if entity_type == 'file' or '.py' in entity_name or '.txt' in entity_name:
                self.last_file = entity_name
            elif entity_type == 'person':
                self.last_person = entity_name
            elif entity_type == 'concept':
                self.last_concept = entity_name
            else:
                self.last_object = entity_name

    def resolve_reference(
        self,
        reference: str,
        context_turns: int = 3
    ) -> Optional[str]:
        """
        Resolve a pronoun or reference to an entity.

        Args:
            reference: The reference to resolve ("it", "that", "them", etc)
            context_turns: How many recent turns to consider

        Returns:
            The resolved entity, or None if can't resolve
        """
        reference_lower = reference.lower().strip()

        # Simple pronoun resolution
        pronoun_map = {
            'it': self.last_object or self.last_file,
            'that': self.last_object or self.last_concept,
            'this': self.last_object,
            'them': self.last_person,
            'he': self.last_person,
            'she': self.last_person,
            'they': self.last_person,
        }

        if reference_lower in pronoun_map:
            resolved = pronoun_map[reference_lower]
            if resolved:
                logger.info(f"Resolved '{reference}' -> '{resolved}'")
                return resolved

        # More sophisticated resolution using salience
        if reference_lower in ['it', 'that', 'this']:
            if self.salient_entities:
                resolved = self.salient_entities[0]
                logger.info(f"Resolved '{reference}' -> '{resolved}' (most salient)")
                return resolved

        return None

    def get_context_window(
        self,
        turns: int = None
    ) -> List[ConversationTurn]:
        """
        Get recent conversation context.

        Args:
            turns: Number of recent turns (default: self.context_window)

        Returns:
            List of recent ConversationTurns
        """
        if turns is None:
            turns = self.context_window

        return list(self.turns)[-turns:]

    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current conversation context.

        Returns:
            Dictionary with context information
        """
        recent_turns = self.get_context_window()

        return {
            'total_turns': len(self.turns),
            'current_turn': self.turn_counter,
            'current_topic': self.current_topic,
            'recent_topics': [topic for topic, _ in self.topic_history[-3:]],
            'salient_entities': self.salient_entities,
            'last_file': self.last_file,
            'last_person': self.last_person,
            'last_concept': self.last_concept,
            'context_window_size': len(recent_turns),
            'session_duration_minutes': self._get_session_duration()
        }

    def _get_session_duration(self) -> float:
        """Get session duration in minutes"""
        if not self.turns:
            return 0.0

        first_turn = self.turns[0]
        last_turn = self.turns[-1]

        duration_seconds = last_turn.timestamp - first_turn.timestamp
        return duration_seconds / 60.0

    def find_relevant_context(
        self,
        query: str,
        max_turns: int = 5
    ) -> List[ConversationTurn]:
        """
        Find conversation turns relevant to a query.

        Args:
            query: The query to search for
            max_turns: Maximum number of turns to return

        Returns:
            List of relevant ConversationTurns
        """
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))

        # Score each turn by word overlap
        scored_turns = []

        for turn in self.turns:
            user_words = set(re.findall(r'\b\w+\b', turn.user_input.lower()))
            response_words = set(re.findall(r'\b\w+\b', turn.alice_response.lower()))

            all_words = user_words | response_words
            overlap = len(query_words & all_words)

            if overlap > 0:
                # Recency bonus
                turns_ago = self.turn_counter - turn.turn_id
                recency_bonus = self.recency_decay ** turns_ago

                score = overlap + recency_bonus
                scored_turns.append((turn, score))

        # Sort by score and return top results
        scored_turns.sort(key=lambda x: x[1], reverse=True)
        return [turn for turn, score in scored_turns[:max_turns]]

    def detect_repetition(self, user_input: str, threshold: int = 3) -> bool:
        """
        Detect if user is repeating themselves.

        Args:
            user_input: Current user input
            threshold: How many recent turns to check

        Returns:
            True if repetition detected
        """
        if len(self.turns) < threshold:
            return False

        # Normalize input
        normalized = user_input.lower().strip()

        # Check recent turns
        recent = list(self.turns)[-threshold:]

        for turn in recent:
            if turn.user_input.lower().strip() == normalized:
                logger.warning(f"Repetition detected: '{user_input}'")
                return True

        return False

    def get_conversation_flow(self) -> List[str]:
        """
        Get the flow of conversation as a list of intents/topics.

        Returns:
            List of intents/topics in order
        """
        flow = []

        for turn in self.turns:
            if turn.intent:
                flow.append(turn.intent)
            elif turn.topics:
                flow.append(turn.topics[0])
            else:
                flow.append('unknown')

        return flow

    def should_summarize_context(self) -> bool:
        """
        Determine if context should be summarized (too long).

        Returns:
            True if context should be summarized
        """
        return len(self.turns) >= self.max_turns * 0.8

    def clear_context(self, keep_recent: int = 5):
        """
        Clear conversation context, optionally keeping recent turns.

        Args:
            keep_recent: Number of recent turns to keep
        """
        if keep_recent > 0:
            recent_turns = list(self.turns)[-keep_recent:]
            self.turns.clear()
            self.turns.extend(recent_turns)
        else:
            self.turns.clear()

        # Reset tracking
        self.mentioned_entities.clear()
        self.entity_types.clear()
        self.salient_entities.clear()
        self.topic_history.clear()
        self.current_topic = None

        logger.info(f"Context cleared (kept {keep_recent} recent turns)")


# Global singleton
_context_manager = None


def get_context_manager(
    max_turns: int = 50,
    context_window: int = 10
) -> ConversationContextManager:
    """Get or create global conversation context manager"""
    global _context_manager
    if _context_manager is None:
        _context_manager = ConversationContextManager(
            max_turns=max_turns,
            context_window=context_window
        )
    return _context_manager
