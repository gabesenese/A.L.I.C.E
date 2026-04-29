"""
Session Summarizer - Tier 1: Long-Session Coherence

Implements sliding-window memory compression, session summaries (every 5 turns),
and maintains a top-10 semantic facts cache for long conversations.
"""

import json
import logging
import zlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class SessionSummary:
    """Summary of a session for context compression."""

    session_id: str
    created_at: str
    turn_count: int
    main_topics: List[str]
    active_goals: List[str]
    key_decisions: List[str]
    user_preferences: Dict[str, Any]
    compressed_turns: Optional[str] = None  # gzipped turns
    top_10_facts: List[str] = None

    def __post_init__(self):
        if self.top_10_facts is None:
            self.top_10_facts = []


class SessionSummarizer:
    """Manages long-session coherence through compression and summary."""

    def __init__(self, summary_interval: int = 5, keep_recent_turns: int = 10):
        """
        Args:
            summary_interval: Generate summary every N turns
            keep_recent_turns: Always keep last N uncompressed turns in context
        """
        self.summary_interval = max(1, int(summary_interval or 5))
        self.keep_recent_turns = max(1, int(keep_recent_turns or 10))
        self.turn_buffer = deque(maxlen=100)  # Buffer for turn history
        self.summaries: List[SessionSummary] = []
        self.top_10_facts: List[str] = []
        self.session_id = datetime.now().isoformat()
        self.turn_count = 0
        self.active_goals: List[str] = []
        self.main_topics: List[str] = []
        self.key_decisions: List[str] = []
        self.user_preferences: Dict[str, Any] = {}

    def record_turn(
        self,
        user_input: str,
        alice_response: str,
        intent: str = "",
        metadata: Dict[str, Any] = None,
    ) -> None:
        """Record a single turn in the session."""
        metadata = metadata or {}
        self.turn_count += 1

        turn = {
            "turn_number": self.turn_count,
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "alice_response": alice_response,
            "intent": intent,
            "metadata": metadata,
        }
        self.turn_buffer.append(turn)

        # Extract goals, topics, and decisions from metadata
        if "goals" in metadata:
            self.active_goals = metadata["goals"]
        if "topics" in metadata:
            self.main_topics = list(set(self.main_topics + metadata.get("topics", [])))[
                :10
            ]
        if "decision" in metadata:
            self.key_decisions.append(metadata["decision"])
        if "user_preferences" in metadata:
            self.user_preferences.update(metadata["user_preferences"])

        # Check if summary should be generated
        if self.turn_count % self.summary_interval == 0:
            self._generate_summary()

    def _generate_summary(self) -> SessionSummary:
        """Generate a summary of recent turns."""
        recent_turns = list(self.turn_buffer)

        # Extract key information from recent turns
        topics = self._extract_topics(recent_turns)
        decisions = self._extract_decisions(recent_turns)

        summary = SessionSummary(
            session_id=self.session_id,
            created_at=datetime.now().isoformat(),
            turn_count=self.turn_count,
            main_topics=topics[:5],
            active_goals=self.active_goals[:5],
            key_decisions=decisions[:5],
            user_preferences=self.user_preferences,
            top_10_facts=self.top_10_facts[:10],
        )

        # Compress older turns if we have many
        if len(self.turn_buffer) > self.keep_recent_turns:
            old_turns = list(self.turn_buffer)[: -self.keep_recent_turns]
            summary.compressed_turns = self._compress_turns(old_turns)

        self.summaries.append(summary)
        logger.info(
            f"[Session] Generated summary at turn {self.turn_count}: {len(topics)} topics, {len(self.active_goals)} goals"
        )

        return summary

    def _extract_topics(self, turns: List[Dict]) -> List[str]:
        """Extract main topics from recent turns."""
        topics = set()
        for turn in turns:
            intent = turn.get("intent", "").split(":")
            if intent[0]:
                topics.add(intent[0])
        return list(topics)

    def _extract_decisions(self, turns: List[Dict]) -> List[str]:
        """Extract key decisions from recent turns."""
        decisions = []
        for turn in turns:
            if "decision" in turn.get("metadata", {}):
                decisions.append(turn["metadata"]["decision"])
        return decisions[:5]

    def _compress_turns(self, turns: List[Dict]) -> str:
        """Compress older turns using gzip."""
        json_str = json.dumps(turns)
        compressed = zlib.compress(json_str.encode("utf-8"))
        return compressed.hex()

    def update_top_10_facts(self, new_facts: List[str]) -> None:
        """Update the top-10 semantic facts kept in scope."""
        # Preserve existing facts, add new ones, keep top 10
        self.top_10_facts = list(set(self.top_10_facts + new_facts))[:10]
        logger.info(
            f"[Session] Updated top-10 facts: {len(self.top_10_facts)} facts in scope"
        )

    def get_session_context(self) -> Dict[str, Any]:
        """Get current session context for LLM/reasoning."""
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "active_goals": self.active_goals,
            "main_topics": self.main_topics,
            "key_decisions": self.key_decisions[-3:],  # Last 3 decisions
            "top_10_facts": self.top_10_facts,
            "user_preferences": self.user_preferences,
            "recent_turns": list(self.turn_buffer)[-self.keep_recent_turns :],
        }

    def get_recent_turns(self, n: int = 10) -> List[Dict]:
        """Get recent N turns."""
        return list(self.turn_buffer)[-n:]

    def get_compressed_context(self) -> str:
        """Get compressed narrative of session so far."""
        if not self.summaries:
            return ""

        latest = self.summaries[-1]
        context = f"""
Session {latest.session_id[:8]}... (turn {latest.turn_count})
Topics: {", ".join(latest.main_topics)}
Goals: {", ".join(latest.active_goals)}
Key facts: {", ".join(latest.top_10_facts[:5])}
"""
        return context.strip()

    def goal_is_tracked(self, goal: str) -> bool:
        """Check if a goal is currently tracked."""
        return any(
            goal.lower() in g.lower() or g.lower() in goal.lower()
            for g in self.active_goals
        )

    def add_goal(self, goal: str) -> None:
        """Add a goal to active goals."""
        if not self.goal_is_tracked(goal):
            self.active_goals.append(goal)
            logger.info(f"[Session] Tracking goal: {goal}")

    def remove_goal(self, goal: str) -> None:
        """Remove a completed goal."""
        self.active_goals = [g for g in self.active_goals if g.lower() != goal.lower()]
        logger.info(f"[Session] Goal completed: {goal}")
