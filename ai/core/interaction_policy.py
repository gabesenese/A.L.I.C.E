"""
A.L.I.C.E - Interaction Policy
================================
Translates user mood / sentiment into concrete pipeline settings.

Single place to tune how A.L.I.C.E adjusts tone, response length,
and clarification behaviour based on emotional context — without
touching the underlying model.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class PolicySettings:
    """
    Downstream hints for response generation and clarification gating.

    Fields
    ------
    clarification_threshold:
        Ambiguity score (0–1) above which a clarification is issued.
        Lower = more willing to ask; higher = push through uncertainty.
    response_length:
        Hint for response formulator: 'brief' | 'normal' | 'detailed'
    tone:
        Stylistic hint: 'empathetic' | 'direct' | 'encouraging' | 'neutral'
    skip_clarification:
        When True the pipeline should not stop for clarification even if
        ambiguity is above the threshold (user is frustrated / in a hurry).
    add_empathy_prefix:
        When True the response formulator may prepend an acknowledging
        phrase ("I understand…", "Got it—…") before the main answer.
    """

    clarification_threshold: float
    response_length: str
    tone: str
    skip_clarification: bool
    add_empathy_prefix: bool


class InteractionPolicy:
    """
    Derive a PolicySettings object from the user's inferred mood,
    raw VADER sentiment, and urgency signal.

    Usage
    -----
        policy = InteractionPolicy()
        settings = policy.derive(mood, sentiment, urgency)
        if settings.skip_clarification:
            # bypass the clarification gate
    """

    def derive(
        self,
        mood: str,
        sentiment: Dict[str, Any],
        urgency: str,
    ) -> PolicySettings:
        """
        Parameters
        ----------
        mood:
            One of: 'positive' | 'negative' | 'neutral' | 'frustrated' | 'urgent'
            (as produced by Perception._infer_mood).
        sentiment:
            VADER result dict with at least a 'compound' key (-1 → +1).
        urgency:
            'low' | 'medium' | 'high' (as produced by NLP EmotionDetector).
        """
        compound = (sentiment or {}).get("compound", 0.0)

        if mood == "frustrated":
            # User is annoyed — keep responses short, empathetic,
            # and never trigger another clarification round.
            return PolicySettings(
                clarification_threshold=0.35,
                response_length="brief",
                tone="empathetic",
                skip_clarification=True,
                add_empathy_prefix=True,
            )

        if mood == "urgent" or urgency == "high":
            # User needs a fast answer — be direct and skip friction.
            return PolicySettings(
                clarification_threshold=0.30,
                response_length="brief",
                tone="direct",
                skip_clarification=True,
                add_empathy_prefix=False,
            )

        if mood == "negative" or compound < -0.2:
            # Mild negativity — still clarify when needed, but be warm.
            return PolicySettings(
                clarification_threshold=0.50,
                response_length="normal",
                tone="empathetic",
                skip_clarification=False,
                add_empathy_prefix=True,
            )

        if mood == "positive" or compound > 0.4:
            # User is in a good mood — normal flow, slightly encouraging tone.
            return PolicySettings(
                clarification_threshold=0.60,
                response_length="normal",
                tone="encouraging",
                skip_clarification=False,
                add_empathy_prefix=False,
            )

        # Neutral default
        return PolicySettings(
            clarification_threshold=0.55,
            response_length="normal",
            tone="neutral",
            skip_clarification=False,
            add_empathy_prefix=False,
        )


# ---------------------------------------------------------------------------
# Response Knob Policy  (contextual bandit for per-turn style adjustments)
# ---------------------------------------------------------------------------

@dataclass
class ResponseKnobs:
    """Five continuous style parameters, each in [0.0, 1.0]."""
    length: float = 0.5          # 0 = terse, 1 = verbose
    directness: float = 0.5      # 0 = hedging, 1 = blunt
    technical_depth: float = 0.3 # 0 = plain, 1 = technical
    empathy_level: float = 0.4   # 0 = dry, 1 = high-empathy
    formality: float = 0.5       # 0 = casual, 1 = formal

    def blend(self, other: "ResponseKnobs", weight: float) -> "ResponseKnobs":
        w = max(0.0, min(1.0, weight))
        return ResponseKnobs(
            length=self.length + w * (other.length - self.length),
            directness=self.directness + w * (other.directness - self.directness),
            technical_depth=self.technical_depth + w * (other.technical_depth - self.technical_depth),
            empathy_level=self.empathy_level + w * (other.empathy_level - self.empathy_level),
            formality=self.formality + w * (other.formality - self.formality),
        )

    def clamp(self) -> "ResponseKnobs":
        def _c(v: float) -> float:
            return max(0.0, min(1.0, v))
        return ResponseKnobs(
            length=_c(self.length), directness=_c(self.directness),
            technical_depth=_c(self.technical_depth),
            empathy_level=_c(self.empathy_level), formality=_c(self.formality),
        )

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


_SENTIMENT_DEFAULTS: Dict[str, ResponseKnobs] = {
    "positive":   ResponseKnobs(length=0.5, directness=0.6, empathy_level=0.5, formality=0.4),
    "negative":   ResponseKnobs(length=0.4, directness=0.4, empathy_level=0.8, formality=0.5),
    "frustrated": ResponseKnobs(length=0.3, directness=0.7, empathy_level=0.7, formality=0.4),
    "excited":    ResponseKnobs(length=0.6, directness=0.6, empathy_level=0.5, formality=0.3),
    "neutral":    ResponseKnobs(length=0.5, directness=0.5, empathy_level=0.4, formality=0.5),
}


@dataclass
class KnobArm:
    """Running statistics for one context arm."""
    count: int = 0
    knobs_sum: ResponseKnobs = field(default_factory=ResponseKnobs)
    reward_sum: float = 0.0
    last_reward: float = 0.5

    def update(self, knobs: ResponseKnobs, reward: float) -> None:
        self.count += 1
        self.reward_sum += reward
        self.last_reward = reward
        n = self.count
        def _m(old: float, new: float) -> float:
            return old + (new - old) / n
        self.knobs_sum = ResponseKnobs(
            length=_m(self.knobs_sum.length, knobs.length),
            directness=_m(self.knobs_sum.directness, knobs.directness),
            technical_depth=_m(self.knobs_sum.technical_depth, knobs.technical_depth),
            empathy_level=_m(self.knobs_sum.empathy_level, knobs.empathy_level),
            formality=_m(self.knobs_sum.formality, knobs.formality),
        )

    def mean_knobs(self) -> ResponseKnobs:
        return self.knobs_sum  # already a running mean

    def mean_reward(self) -> float:
        return self.reward_sum / self.count if self.count > 0 else 0.0


class KnobBandit:
    """
    Epsilon-greedy contextual bandit over ResponseKnobs style parameters.

    context_key = (intent_category, sentiment, topic_prefix, hour_bucket)
    Each key maintains a KnobArm with a running mean of ResponseKnobs weighted
    by observed reward.  Call record_outcome() after each reply.
    """

    def __init__(
        self,
        epsilon: float = 0.15,
        epsilon_min: float = 0.03,
        anneal_rate: float = 0.005,
    ) -> None:
        self._epsilon = epsilon
        self._epsilon_min = epsilon_min
        self._anneal_rate = anneal_rate
        self._arms: Dict[str, KnobArm] = {}
        self._turn_count: int = 0
        self._bandit_lock = threading.Lock()
        self._pending: Dict[str, ResponseKnobs] = {}

    def propose(
        self,
        intent: str,
        sentiment: str = "neutral",
        topic: str = "",
        override: Optional[ResponseKnobs] = None,
    ) -> Tuple[str, ResponseKnobs]:
        """Propose ResponseKnobs for the current context. Returns (context_key, knobs)."""
        key = self._context_key(intent, sentiment, topic)
        if override is not None:
            with self._bandit_lock:
                self._pending[key] = override
            return key, override.clamp()
        with self._bandit_lock:
            arm = self._arms.setdefault(key, KnobArm())
            effective_epsilon = max(self._epsilon_min, self._epsilon)
            if arm.count == 0 or random.random() < effective_epsilon:
                base = _SENTIMENT_DEFAULTS.get(sentiment, ResponseKnobs())
                knobs = self._explore(base)
            else:
                knobs = arm.mean_knobs()
            knobs = knobs.clamp()
            self._pending[key] = knobs
            return key, knobs

    def record_outcome(self, context_key: str, reward: float) -> None:
        """Feed back observed reward for the most recent proposal (reward in [-1, 1])."""
        with self._bandit_lock:
            proposed_knobs = self._pending.pop(context_key, None)
            if proposed_knobs is None:
                return
            arm = self._arms.setdefault(context_key, KnobArm())
            arm.update(proposed_knobs, reward)
            self._turn_count += 1
            if self._turn_count % 100 == 0:
                self._epsilon = max(self._epsilon_min, self._epsilon - self._anneal_rate)

    def best_knobs_for(self, intent: str, sentiment: str = "neutral", topic: str = "") -> ResponseKnobs:
        key = self._context_key(intent, sentiment, topic)
        with self._bandit_lock:
            arm = self._arms.get(key)
            if arm and arm.count > 0:
                return arm.mean_knobs().clamp()
        return _SENTIMENT_DEFAULTS.get(sentiment, ResponseKnobs())

    def stats(self) -> Dict[str, dict]:
        with self._bandit_lock:
            return {
                key: {"count": arm.count, "mean_reward": round(arm.mean_reward(), 3),
                      "knobs": arm.mean_knobs().to_dict()}
                for key, arm in self._arms.items() if arm.count > 0
            }

    @staticmethod
    def _context_key(intent: str, sentiment: str, topic: str) -> str:
        category = intent.split(":")[0] if ":" in intent else intent
        hour_bucket = time.localtime().tm_hour // 6
        topic_prefix = topic[:12].lower().replace(" ", "_") if topic else "any"
        sent = sentiment.lower() if sentiment else "neutral"
        return f"{category}|{sent}|{topic_prefix}|h{hour_bucket}"

    @staticmethod
    def _explore(base: ResponseKnobs) -> ResponseKnobs:
        def _n(v: float) -> float:
            return v + random.gauss(0, 0.15)
        return ResponseKnobs(
            length=_n(base.length), directness=_n(base.directness),
            technical_depth=_n(base.technical_depth),
            empathy_level=_n(base.empathy_level), formality=_n(base.formality),
        ).clamp()


_knob_bandit_instance: Optional[KnobBandit] = None
_bandit_lock = threading.Lock()


def get_knob_bandit() -> KnobBandit:
    """Return the process-wide singleton KnobBandit."""
    global _knob_bandit_instance
    if _knob_bandit_instance is None:
        with _bandit_lock:
            if _knob_bandit_instance is None:
                _knob_bandit_instance = KnobBandit()
    return _knob_bandit_instance
