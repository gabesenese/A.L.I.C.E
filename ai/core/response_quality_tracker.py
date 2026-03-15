"""
Response Quality Tracker.

Tracks measurable response quality per turn and classifies failure types.
Provides degradation signals that the executive and reflection loop can use
to tighten controls when quality drops.

Failure types (response-level, complements ai/core/failure_taxonomy.py
which covers NLP/plugin failures):
    none             – turn was fine
    intent_mismatch  – response domain doesn't match intent domain
    topic_drift      – response barely overlaps with user input
    routing_mistake  – tool intent routed to verbose LLM prose
    weak_knowledge   – multiple uncertainty markers in response
    overgeneralization – multiple generic/vague markers in response
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional


FAILURE_NONE = "none"
FAILURE_INTENT_MISMATCH = "intent_mismatch"
FAILURE_TOPIC_DRIFT = "topic_drift"
FAILURE_ROUTING_MISTAKE = "routing_mistake"
FAILURE_WEAK_KNOWLEDGE = "weak_knowledge"
FAILURE_OVERGENERALIZATION = "overgeneralization"

ALL_FAILURE_TYPES = (
    FAILURE_NONE,
    FAILURE_INTENT_MISMATCH,
    FAILURE_TOPIC_DRIFT,
    FAILURE_ROUTING_MISTAKE,
    FAILURE_WEAK_KNOWLEDGE,
    FAILURE_OVERGENERALIZATION,
)


@dataclass
class TurnQuality:
    relevance: float      # token overlap between input and response (0..1)
    coherence: float      # unique-word ratio in response (0..1)
    verbosity: float      # 0=too short, 0.5=ideal, 1=too long
    alignment: float      # goal alignment score (0..1)
    failure_type: str     # one of ALL_FAILURE_TYPES
    gate_accepted: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "relevance": round(self.relevance, 3),
            "coherence": round(self.coherence, 3),
            "verbosity": round(self.verbosity, 3),
            "alignment": round(self.alignment, 3),
            "failure_type": self.failure_type,
            "gate_accepted": self.gate_accepted,
        }


class ResponseQualityTracker:
    """
    Per-turn quality tracking with failure classification.

    Tracks a rolling window of TurnQuality records and exposes summary
    statistics useful for diagnostics and adaptive control.
    """

    _GENERIC_MARKERS = frozenset((
        "it depends", "in general", "there are many factors",
        "various factors", "cannot be determined",
        "as an ai", "i don't have", "i cannot",
    ))
    _UNCERTAIN_MARKERS = frozenset((
        "i'm not sure", "i am not sure", "maybe",
        "possibly", "i don't know", "not certain",
    ))

    # Tool-oriented intent domains that should NOT produce long prose replies
    _TOOL_DOMAINS = frozenset((
        "notes", "email", "calendar", "file_operations",
        "reminder", "system", "weather", "time",
    ))

    def __init__(self, window: int = 20) -> None:
        self._window = max(5, int(window))
        self._history: Deque[TurnQuality] = deque(maxlen=self._window)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def track_turn(
        self,
        *,
        user_input: str,
        response: str,
        intent: str,
        gate_accepted: bool,
        reflection_score: float,
        goal_alignment: float = 1.0,
    ) -> TurnQuality:
        """Record quality metrics for a completed turn and return them."""
        relevance = self._relevance(user_input, response)
        coherence = self._coherence(response)
        verbosity = self._verbosity(response)
        failure = self.classify_failure(
            user_input=user_input,
            response=response,
            intent=intent,
            gate_accepted=gate_accepted,
            reflection_score=float(reflection_score or 0.0),
            relevance=relevance,
        )
        quality = TurnQuality(
            relevance=relevance,
            coherence=coherence,
            verbosity=verbosity,
            alignment=max(0.0, min(1.0, float(goal_alignment))),
            failure_type=failure,
            gate_accepted=gate_accepted,
        )
        self._history.append(quality)
        return quality

    def classify_failure(
        self,
        *,
        user_input: str,
        response: str,
        intent: str,
        gate_accepted: bool,
        reflection_score: float,
        relevance: Optional[float] = None,
    ) -> str:
        """Classify the failure type for a response without recording it."""
        if gate_accepted and float(reflection_score or 0.0) >= 0.65:
            return FAILURE_NONE

        resp_low = (response or "").lower()
        inp_low = (user_input or "").lower()
        intent_low = (intent or "").lower()

        # Routing mistake: tool intent but the response is verbose prose
        intent_domain = intent_low.split(":")[0] if ":" in intent_low else ""
        if intent_domain in self._TOOL_DOMAINS:
            if not gate_accepted and len(resp_low.split()) > 40:
                return FAILURE_ROUTING_MISTAKE

        # Overgeneralization: too many vague/hedging phrases
        generic_count = sum(1 for m in self._GENERIC_MARKERS if m in resp_low)
        if generic_count >= 2:
            return FAILURE_OVERGENERALIZATION

        # Weak knowledge: too many uncertainty admissions
        uncertain_count = sum(1 for m in self._UNCERTAIN_MARKERS if m in resp_low)
        if uncertain_count >= 2:
            return FAILURE_WEAK_KNOWLEDGE

        # Topic drift: response barely overlaps with user input
        if relevance is None:
            relevance = self._relevance(user_input, response)
        if relevance < 0.10 and not gate_accepted:
            return FAILURE_TOPIC_DRIFT

        # Intent mismatch: intent domain not reflected anywhere in response
        if intent_domain and intent_domain not in resp_low and not gate_accepted:
            return FAILURE_INTENT_MISMATCH

        return FAILURE_NONE

    def get_quality_summary(self) -> Dict[str, Any]:
        """Return statistical summary over the tracking window."""
        if not self._history:
            return {
                "relevance_avg": 0.0,
                "coherence_avg": 0.0,
                "alignment_avg": 0.0,
                "gate_accept_rate": 0.0,
                "failure_counts": {},
                "turns_tracked": 0,
            }
        n = len(self._history)
        rel_avg = sum(t.relevance for t in self._history) / n
        coh_avg = sum(t.coherence for t in self._history) / n
        aln_avg = sum(t.alignment for t in self._history) / n
        gate_rate = sum(1 for t in self._history if t.gate_accepted) / n
        failure_counts: Dict[str, int] = {}
        for t in self._history:
            failure_counts[t.failure_type] = failure_counts.get(t.failure_type, 0) + 1
        return {
            "relevance_avg": round(rel_avg, 3),
            "coherence_avg": round(coh_avg, 3),
            "alignment_avg": round(aln_avg, 3),
            "gate_accept_rate": round(gate_rate, 3),
            "failure_counts": failure_counts,
            "turns_tracked": n,
        }

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------

    def _relevance(self, user_input: str, response: str) -> float:
        """Token overlap between user input and response."""
        tu = set(re.findall(r"[a-z0-9']+", (user_input or "").lower()))
        tr = set(re.findall(r"[a-z0-9']+", (response or "").lower()))
        if not tu:
            return 0.0
        return len(tu.intersection(tr)) / max(len(tu), 1)

    def _coherence(self, response: str) -> float:
        """
        Rough coherence proxy: unique-word ratio.
        Low ratio → repetitive / low-quality output.
        """
        words = re.findall(r"[a-z0-9']+", (response or "").lower())
        if not words:
            return 0.0
        return min(1.0, len(set(words)) / max(len(words), 1))

    def _verbosity(self, response: str) -> float:
        """
        Verbosity score:  0.0 = too short  |  0.5 = ideal  |  1.0 = too long
        Ideal band: 30–300 words.
        """
        wc = len((response or "").split())
        if wc < 10:
            return 0.0
        if wc > 500:
            return 1.0
        if 30 <= wc <= 300:
            return 0.5
        return 0.3


_quality_tracker: ResponseQualityTracker | None = None


def get_response_quality_tracker() -> ResponseQualityTracker:
    global _quality_tracker
    if _quality_tracker is None:
        _quality_tracker = ResponseQualityTracker()
    return _quality_tracker
