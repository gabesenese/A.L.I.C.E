"""
Pattern-Based Nudger - Tier 2: Proactive Nudges Based on Patterns

Detects recurring interaction patterns and offers proactive suggestions.
Learns from user behavior to anticipate helpful next steps.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class InteractionPattern:
    """Represents a recurring pattern in user interactions."""

    pattern_name: str
    pattern_type: str  # "sequence", "recurring_intent", "unresolved_issue"
    frequency: int  # Times seen
    suggested_nudge: str  # What to suggest
    confidence: float  # 0.0-1.0
    last_triggered: Optional[str] = None
    trigger_intents: List[str] = None

    def __post_init__(self):
        if self.trigger_intents is None:
            self.trigger_intents = []


@dataclass
class Nudge:
    """A proactive suggestion to the user."""

    nudge_id: str
    user_facing_text: str
    reason: str
    pattern_matched: str
    timing: str  # "immediate", "delayed", "contextual"
    acceptance_likelihood: float  # 0.0-1.0
    delivered: bool = False
    was_accepted: bool = False


class PatternBasedNudger:
    """Detects patterns and generates proactive suggestions."""

    def __init__(self, min_pattern_frequency: int = 3, max_nudges_per_session: int = 5):
        """
        Args:
            min_pattern_frequency: Minimum occurrences to consider a pattern
            max_nudges_per_session: Don't spam more than this many nudges per session
        """
        self.min_pattern_frequency = min_pattern_frequency
        self.max_nudges_per_session = max_nudges_per_session
        self.interaction_history: List[Dict[str, Any]] = []
        self.detected_patterns: List[InteractionPattern] = []
        self.nudges_delivered: List[Nudge] = []
        self.turn_count = 0
        self.pattern_rules = self._init_pattern_rules()

    def _init_pattern_rules(self) -> Dict[str, Dict]:
        """Initialize hard-coded pattern detection rules."""
        return {
            "code_review_ready": {
                "trigger_intents": ["code:analyze", "code:read"],
                "after_n_times": 3,
                "nudge": "I've seen you reviewing code several times. Would you like me to run the full test suite to check for regressions?",
                "reasoning": "After multiple code reviews, testing is typically the next step",
            },
            "weather_umbrella_pattern": {
                "trigger_intents": ["weather:get"],
                "follow_pattern": ["umbrella:ask"],
                "after_n_times": 2,
                "nudge": "Rain expected soon—should I remind you about an umbrella?",
                "reasoning": "If weather forecast asked, umbrella question often follows",
            },
            "time_blocking_pattern": {
                "trigger_intents": ["calendar:check"],
                "after_n_times": 2,
                "nudge": "Setting up regular time blocks for focused work? I can help schedule those.",
                "reasoning": "Repeated calendar checks suggest planning/blocking patterns",
            },
            "debugging_stuck": {
                "trigger_intents": ["error", "debug", "fix"],
                "unresolved_count": 3,
                "nudge": "This bug seems tricky. Want me to check the error logs or suggest debugging strategies?",
                "reasoning": "Multiple failed debugging attempts = stuck user",
            },
            "memory_learning": {
                "trigger_intents": ["note:create", "remember"],
                "after_n_times": 3,
                "nudge": "You're creating lots of notes. Want me to build a summary of key topics?",
                "reasoning": "User documenting suggests knowledge accumulation phase",
            },
            "routine_task": {
                "trigger_intents": ["git:", "file:"],
                "same_sequence_count": 3,
                "nudge": "You're doing this sequence repeatedly. Should I automate it?",
                "reasoning": "Repeated sequences are automation candidates",
            },
        }

    def record_interaction(
        self,
        intent: str,
        success: bool,
        response_type: str,
        user_input: str,
        metadata: Dict[str, Any] = None,
    ) -> None:
        """Record an interaction for pattern detection."""
        self.turn_count += 1
        metadata = metadata or {}

        self.interaction_history.append(
            {
                "turn_number": self.turn_count,
                "timestamp": datetime.now().isoformat(),
                "intent": intent,
                "success": success,
                "response_type": response_type,
                "user_input": user_input,
                "metadata": metadata,
            }
        )

    def detect_patterns(self) -> List[InteractionPattern]:
        """Analyze interaction history to detect recurring patterns."""
        if len(self.interaction_history) < self.min_pattern_frequency:
            return []

        patterns = []

        # 1. Detect repeated intent patterns
        intent_counts = Counter(h["intent"] for h in self.interaction_history)

        for intent, count in intent_counts.most_common():
            if count >= self.min_pattern_frequency:
                for rule_name, rule in self.pattern_rules.items():
                    if intent in rule.get("trigger_intents", []):
                        pattern = InteractionPattern(
                            pattern_name=rule_name,
                            pattern_type="recurring_intent",
                            frequency=count,
                            suggested_nudge=rule["nudge"],
                            confidence=min(1.0, count / 10.0),  # Scale: freq/10
                            trigger_intents=[intent],
                        )
                        patterns.append(pattern)

        # 2. Detect unresolved issues (repeated failures)
        failed_intents = Counter(
            h["intent"] for h in self.interaction_history if not h.get("success", True)
        )

        for intent, count in failed_intents.most_common():
            if count >= 3:  # User has tried 3+ times unsuccessfully
                pattern = InteractionPattern(
                    pattern_name="unresolved_issue",
                    pattern_type="unresolved_issue",
                    frequency=count,
                    suggested_nudge=f"You've tried {intent} a few times without success. Let me help debug this.",
                    confidence=0.8,
                    trigger_intents=[intent],
                )
                patterns.append(pattern)

        # 3. Detect sequence patterns (same set of intents in order)
        recent_intents = [h["intent"] for h in self.interaction_history[-10:]]
        sequences = self._find_repeated_sequences(recent_intents)

        for sequence, count in sequences.items():
            if count >= 3:
                pattern = InteractionPattern(
                    pattern_name=f"sequence_{count}x",
                    pattern_type="sequence",
                    frequency=count,
                    suggested_nudge="I've noticed you do this sequence regularly. Shall I automate it?",
                    confidence=0.7,
                    trigger_intents=list(sequence),
                )
                patterns.append(pattern)

        self.detected_patterns = patterns
        logger.info(
            f"[Nudger] Detected {len(patterns)} patterns from {len(self.interaction_history)} interactions"
        )

        return patterns

    def _find_repeated_sequences(
        self, intents: List[str], seq_length: int = 3
    ) -> Dict[tuple, int]:
        """Find repeated sequences of N intents."""
        sequences = Counter()
        for i in range(len(intents) - seq_length + 1):
            seq = tuple(intents[i : i + seq_length])
            sequences[seq] += 1
        return {seq: count for seq, count in sequences.items() if count > 1}

    def generate_nudge_if_applicable(self) -> Optional[Nudge]:
        """Generate a nudge if current state matches a pattern and hasn't spammed recently."""
        # Don't spam nudges
        if len(self.nudges_delivered) >= self.max_nudges_per_session:
            return None

        # Only nudge after enough interactions
        if len(self.interaction_history) < 5:
            return None

        # Detect current patterns
        patterns = self.detect_patterns()
        if not patterns:
            return None

        # Pick the highest confidence pattern that hasn't been nudged recently
        best_pattern = None
        for pattern in sorted(patterns, key=lambda p: p.confidence, reverse=True):
            if not any(
                n.pattern_matched == pattern.pattern_name
                for n in self.nudges_delivered[-3:]
            ):
                best_pattern = pattern
                break

        if not best_pattern:
            return None

        # Create nudge
        nudge = Nudge(
            nudge_id=f"nudge_{len(self.nudges_delivered)}",
            user_facing_text=best_pattern.suggested_nudge,
            reason=f"Pattern detected: {best_pattern.pattern_name} (seen {best_pattern.frequency}x)",
            pattern_matched=best_pattern.pattern_name,
            timing="contextual",
            acceptance_likelihood=best_pattern.confidence,
        )

        logger.info(
            f"[Nudger] Generated nudge: {best_pattern.pattern_name} "
            f"(confidence={best_pattern.confidence:.2f})"
        )

        return nudge

    def record_nudge_delivery(self, nudge: Nudge, accepted: bool = None) -> None:
        """Record whether a nudge was delivered and accepted."""
        nudge.delivered = True
        nudge.was_accepted = accepted if accepted is not None else False
        self.nudges_delivered.append(nudge)

        status = (
            "✓ accepted"
            if accepted
            else "✗ declined"
            if accepted is False
            else "delivered"
        )
        logger.info(f"[Nudger] Nudge {status}: {nudge.pattern_matched}")

    def get_nudge_effectiveness(self) -> Dict[str, any]:
        """Get statistics on nudge effectiveness."""
        if not self.nudges_delivered:
            return {"total_delivered": 0, "acceptance_rate": 0.0}

        accepted = sum(1 for n in self.nudges_delivered if n.was_accepted)

        return {
            "total_delivered": len(self.nudges_delivered),
            "accepted": accepted,
            "declined": len(self.nudges_delivered) - accepted,
            "acceptance_rate": accepted / len(self.nudges_delivered),
            "avg_confidence": sum(
                n.acceptance_likelihood for n in self.nudges_delivered
            )
            / len(self.nudges_delivered),
        }

    def get_pattern_summary(self) -> str:
        """Get human-readable summary of detected patterns."""
        if not self.detected_patterns:
            return "No patterns detected yet."

        lines = ["Detected interaction patterns:"]
        for pattern in sorted(
            self.detected_patterns, key=lambda p: p.frequency, reverse=True
        )[:3]:
            lines.append(f"  • {pattern.pattern_name}: {pattern.frequency}x detected")

        return "\n".join(lines)
