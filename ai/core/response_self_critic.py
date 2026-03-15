"""
Response self-critique layer for A.L.I.C.E.

Provides a lightweight second pass that validates whether a draft response:
- aligns with intent/domain
- stays on topic
- avoids obvious unsupported citation patterns
- does not contradict known memory snapshot hints
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "if",
    "then",
    "to",
    "for",
    "of",
    "in",
    "on",
    "at",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "we",
    "they",
    "he",
    "she",
    "me",
    "my",
    "your",
    "our",
    "their",
    "help",
    "please",
    "can",
    "could",
    "would",
    "should",
    "do",
    "does",
    "did",
    "with",
    "about",
    "as",
}


@dataclass
class CritiqueResult:
    passed: bool
    fail_reasons: List[str] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    correction_hint: Optional[str] = None


class ResponseSelfCritic:
    """Heuristic self-critic for first-pass response quality checks."""

    _DOMAIN_KEYWORDS: Dict[str, Set[str]] = {
        "weather": {
            "weather",
            "forecast",
            "rain",
            "snow",
            "temperature",
            "wind",
            "outside",
        },
        "notes": {"note", "notes", "list", "title", "content"},
        "email": {"email", "inbox", "message", "subject", "sender"},
        "calendar": {"calendar", "event", "meeting", "schedule", "time"},
        "memory": {"remember", "memory", "recall", "preference"},
        "time": {"time", "clock", "date", "today"},
        "conversation": set(),
    }

    # Signals that the user is correcting Alice or noting something already exists
    _CORRECTION_SIGNALS: Set[str] = {
        "actually",
        "that's wrong",
        "already exists",
        "already have",
        "already has",
        "you already",
        "that already",
        "not correct",
        "incorrect",
        "already implemented",
        "already includes",
        "it exists",
        "already done",
        "already built",
    }
    # Signals that a feature is already present
    _EXISTS_SIGNALS: Set[str] = {
        "already exists",
        "already have",
        "already has",
        "already implemented",
        "already includes",
        "already added",
        "already there",
        "it already",
        "that's already",
        "already built",
    }
    # Questions that presuppose a prior suggestion had merit
    _PRESUPPOSITION_RE = re.compile(
        r"what\s+(?:still|else|other)\s+(?:aspects?|parts?|things?|areas?)"
        r"|which\s+(?:aspect|part|element)\s+would"
        r"|what\s+(?:aspects?|parts?)\s+(?:would|should|could)"
        r"|what\s+would\s+you\s+like\s+(?:me\s+)?to\s+(?:address|fix|improve|refine|focus)",
        re.IGNORECASE,
    )
    # Suggestion-to-add/implement patterns
    _SUGGESTION_RE = re.compile(
        r"(?:should|could|recommend(?:ed)?|suggest(?:ed)?|consider)\s+"
        r"(?:add(?:ing)?|implement(?:ing)?|includ(?:e|ing)|integrat(?:e|ing))",
        re.IGNORECASE,
    )

    def _tokenize(self, text: str) -> Set[str]:
        return {
            token
            for token in re.findall(r"\b[a-z']+\b", (text or "").lower())
            if token not in _STOPWORDS and len(token) > 1
        }

    def _domain_from_intent(self, intent: str) -> str:
        value = (intent or "").lower()
        if ":" in value:
            return value.split(":", 1)[0]
        if "weather" in value:
            return "weather"
        if "note" in value:
            return "notes"
        if "email" in value:
            return "email"
        if "calendar" in value or "meeting" in value:
            return "calendar"
        if "memory" in value:
            return "memory"
        if "time" in value or "date" in value:
            return "time"
        return "conversation"

    def _topic_overlap_score(self, user_input: str, response: str) -> float:
        user_tokens = self._tokenize(user_input)
        if not user_tokens:
            return 1.0
        response_tokens = self._tokenize(response)
        overlap = len(user_tokens & response_tokens)
        return overlap / max(1, len(user_tokens))

    def _check_intent_and_topic(
        self, user_input: str, intent: str, response: str
    ) -> Optional[str]:
        domain = self._domain_from_intent(intent)
        domain_words = self._DOMAIN_KEYWORDS.get(domain, set())

        if domain_words:
            if not (self._tokenize(response) & domain_words):
                return f"intent-domain mismatch ({domain})"

        overlap = self._topic_overlap_score(user_input, response)
        if overlap < 0.08 and domain != "conversation":
            return "topic mismatch"

        if (
            overlap < 0.04
            and domain == "conversation"
            and len(self._tokenize(user_input)) >= 5
        ):
            return "weak topical relevance"

        return None

    def _check_assumption_without_evidence(
        self, user_input: str, response: str
    ) -> Optional[str]:
        """Flag when the response asks a follow-up that presupposes its prior claim had merit."""
        inp_low = (user_input or "").lower()
        if not any(s in inp_low for s in self._CORRECTION_SIGNALS):
            return None
        if self._PRESUPPOSITION_RE.search((response or "").lower()):
            return "assumption-without-evidence: question presupposes prior suggestion had merit"
        return None

    def _check_redundant_suggestion(
        self, user_input: str, response: str
    ) -> Optional[str]:
        """Flag when the response recommends adding something the user confirmed already exists."""
        inp_low = (user_input or "").lower()
        if not any(s in inp_low for s in self._EXISTS_SIGNALS):
            return None
        if self._SUGGESTION_RE.search((response or "").lower()):
            return "redundant-suggestion: recommending a feature that already exists"
        return None

    def _check_hallucination_risk(self, response: str) -> Optional[str]:
        text = (response or "").lower()

        unsupported_patterns = [
            r"\bsource:\s*https?://",
            r"\baccording to\s+\d{4}\s+study",
            r"\[[0-9]+\]",
        ]
        if any(re.search(pattern, text) for pattern in unsupported_patterns):
            return "possible unsupported citation"

        return None

    def _check_memory_contradiction(
        self, response: str, memory_snapshot: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        if not memory_snapshot:
            return None

        text = (response or "").lower()

        user_name = (memory_snapshot.get("user_name") or "").strip()
        if user_name and (
            "i don't know your name" in text or "i do not know your name" in text
        ):
            return "contradicts memory: known user name"

        active_goal = memory_snapshot.get("active_goal") or {}
        goal_desc = (active_goal.get("description") or "").strip().lower()
        if (
            goal_desc
            and "can't help" in text
            and any(word in goal_desc for word in ["help", "learn", "study", "create"])
        ):
            return "contradicts memory: active goal exists"

        return None

    def assess(
        self,
        user_input: str,
        intent: str,
        entities: Dict[str, Any],
        response: str,
        memory_snapshot: Optional[Dict[str, Any]] = None,
    ) -> CritiqueResult:
        """Assess draft response and return pass/fail with reasons."""
        reasons: List[str] = []

        intent_topic_issue = self._check_intent_and_topic(user_input, intent, response)
        if intent_topic_issue:
            reasons.append(intent_topic_issue)

        hallucination_issue = self._check_hallucination_risk(response)
        if hallucination_issue:
            reasons.append(hallucination_issue)

        memory_issue = self._check_memory_contradiction(response, memory_snapshot)
        if memory_issue:
            reasons.append(memory_issue)

        assumption_issue = self._check_assumption_without_evidence(user_input, response)
        if assumption_issue:
            reasons.append(assumption_issue)

        redundant_issue = self._check_redundant_suggestion(user_input, response)
        if redundant_issue:
            reasons.append(redundant_issue)

        # Select a specific correction hint rather than a generic "acknowledge and ask" template
        correction_hint: Optional[str] = None
        if assumption_issue:
            correction_hint = (
                "Premise was unsupported — retract the framing and wait for a real question."
            )
        elif redundant_issue:
            correction_hint = (
                "Suggestion was entirely redundant — the feature already exists; no gap to fill."
            )

        topic_overlap = self._topic_overlap_score(user_input, response)
        return CritiqueResult(
            passed=not reasons,
            fail_reasons=reasons,
            scores={
                "topic_overlap": topic_overlap,
                "entity_count": float(len(entities or {})),
            },
            correction_hint=correction_hint,
        )


_self_critic = None


def get_response_self_critic() -> ResponseSelfCritic:
    global _self_critic
    if _self_critic is None:
        _self_critic = ResponseSelfCritic()
    return _self_critic
