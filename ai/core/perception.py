"""Perception layer for Foundation 2 routing.

Extracted from nlp_processor.py to keep routing concerns modular.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING
import re

if TYPE_CHECKING:
    from ai.core.nlp_processor import ProcessedQuery


@dataclass
class PerceptionResult:
    """Unified perception object produced between NLP output and policy/routing."""

    query: "ProcessedQuery"
    inferred_mood: str
    ambiguity: float
    followup_domain: Optional[str]
    needs_clarification: bool
    clarification_question: Optional[str]
    interaction_hints: Dict[str, Any]

    @property
    def intent(self) -> str:
        return self.query.intent

    @property
    def confidence(self) -> float:
        return self.query.intent_confidence

    @property
    def entities(self) -> Dict[str, Any]:
        return self.query.entities

    @property
    def sentiment(self) -> Dict[str, float]:
        return self.query.sentiment


class Perception:
    """Derives mood/ambiguity/follow-up domain from a ProcessedQuery."""

    AMBIGUITY_CLARIFICATION_THRESHOLD = 0.65
    CLARIFICATION_CONFIDENCE_THRESHOLD = 0.40

    FRUSTRATION_MARKERS: Set[str] = {
        "no",
        "wrong",
        "stop",
        "ugh",
        "again",
        "not",
        "don't",
        "didn't",
        "why",
        "broken",
        "useless",
        "terrible",
        "awful",
        "hate",
        "redo",
        "fix",
        "what",
        "seriously",
        "come on",
    }

    _DOMAIN_SIGNALS: Dict[str, Set[str]] = {
        "weather": {
            "weather",
            "rain",
            "snow",
            "temp",
            "cold",
            "warm",
            "umbrella",
            "wear",
            "coat",
            "jacket",
            "forecast",
            "tomorrow",
            "tonight",
            "humidity",
            "wind",
            "chilly",
            "freezing",
            "hot",
            "sunny",
        },
        "notes": {"note", "task", "todo", "reminder", "list"},
        "email": {"email", "mail", "reply", "inbox", "send", "draft"},
        "calendar": {"event", "meeting", "schedule", "appointment", "calendar"},
        "reminder": {"remind", "reminder", "alert", "notify", "forget", "alarm"},
    }

    def build(
        self,
        query: "ProcessedQuery",
        last_intent: Optional[str] = None,
        conversation_topics: Optional[List[str]] = None,
    ) -> PerceptionResult:
        mood = self._infer_mood(query)
        ambiguity = self._calc_ambiguity(query)
        followup_domain = self._detect_followup_domain(
            query, last_intent, conversation_topics or []
        )
        needs_clarif, clarif_q = self._clarification_need(query, ambiguity)
        hints: Dict[str, Any] = {
            "mood": mood,
            "ambiguity": ambiguity,
            "followup_domain": followup_domain,
            "response_length": (
                "brief" if mood in ("frustrated", "urgent") else "normal"
            ),
            "empathy": mood in ("frustrated", "negative"),
        }
        return PerceptionResult(
            query=query,
            inferred_mood=mood,
            ambiguity=ambiguity,
            followup_domain=followup_domain,
            needs_clarification=needs_clarif,
            clarification_question=clarif_q,
            interaction_hints=hints,
        )

    def _infer_mood(self, query: "ProcessedQuery") -> str:
        sentiment = query.sentiment or {}
        compound = sentiment.get("compound", 0.0)
        lower = query.original_text.lower()
        tokens = set(lower.split())

        if query.urgency_level == "high" or tokens & {
            "asap",
            "urgent",
            "now",
            "immediately",
        }:
            return "urgent"
        if tokens & self.FRUSTRATION_MARKERS and compound < -0.05:
            return "frustrated"
        if compound >= 0.2:
            return "positive"
        if compound <= -0.2:
            return "negative"
        return "neutral"

    def _calc_ambiguity(self, query: "ProcessedQuery") -> float:
        conf_part = max(0.0, 1.0 - query.intent_confidence)
        val_part = 1.0 - getattr(query, "validation_score", 1.0)
        base = conf_part * 0.6 + val_part * 0.4
        intent = query.intent or ""
        if intent.startswith("vague_") or intent == "conversation:clarification_needed":
            base = min(1.0, base + 0.3)
        return round(base, 3)

    def _detect_followup_domain(
        self,
        query: "ProcessedQuery",
        last_intent: Optional[str],
        topics: List[str],
    ) -> Optional[str]:
        recent = last_intent or (topics[-1] if topics else None)
        if not recent:
            return None
        domain = recent.split(":")[0] if ":" in recent else recent
        signals = self._DOMAIN_SIGNALS.get(domain, set())
        if not signals:
            return None

        lower = query.original_text.lower()
        clean_words = set(re.sub(r"[^\w\s]", "", lower).split())

        nlp_domain = query.intent.split(":")[0] if ":" in query.intent else ""
        same_or_conv = (
            nlp_domain == domain
            or nlp_domain in ("conversation", "")
            or query.intent.endswith(":general")
            or query.intent in ("vague_question", "vague_temporal_question")
        )
        if not same_or_conv and query.intent_confidence >= 0.60:
            return None

        if signals & clean_words:
            return domain

        if any(
            word.startswith(sig[:5]) or sig.startswith(word[:5])
            for sig in signals
            for word in clean_words
            if len(word) >= 6 and len(sig) >= 6
        ):
            return domain

        generic_cues = {"and", "also", "that", "this", "it", "then", "what about"}
        if (
            generic_cues & clean_words
            and query.intent_confidence < 0.55
            and len(clean_words) <= 4
        ):
            return domain
        return None

    def _clarification_need(
        self,
        query: "ProcessedQuery",
        ambiguity: float,
    ) -> Tuple[bool, Optional[str]]:
        parsed_cmd = query.parsed_command or {}
        modifiers = (
            parsed_cmd
            if isinstance(parsed_cmd, dict)
            else getattr(parsed_cmd, "modifiers", {})
        )
        disamb = modifiers.get("disambiguation") or modifiers.get("modifiers", {}).get(
            "disambiguation"
        )
        if disamb and disamb.get("needs_clarification"):
            return True, disamb.get("question")
        if (
            ambiguity >= self.AMBIGUITY_CLARIFICATION_THRESHOLD
            and query.intent_confidence < self.CLARIFICATION_CONFIDENCE_THRESHOLD
        ):
            return True, "Could you clarify what you'd like me to do?"
        return False, None
