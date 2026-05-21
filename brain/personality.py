"""World-model backed personality drift and prompt shaping."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

from memory.world_model import DEFAULT_PERSONALITY, WorldModel, get_world_model


MIN_WEIGHT = 0.1
MAX_WEIGHT = 0.9

_SHORT_RESPONSE_WORD_LIMIT = 3
_SHORT_RESPONSE_CHAR_LIMIT = 24
_SHORT_RESPONSE_STREAK_THRESHOLD = 2

_WARM_SIGNAL = re.compile(
    r"\b(thanks|thank you|appreciate|nice|great|good work|perfect|lol|haha|love that)\b",
    re.IGNORECASE,
)
_STRESS_SIGNAL = re.compile(
    r"\b(stressed|anxious|overwhelmed|panic|worried|burned out|exhausted|tired)\b",
    re.IGNORECASE,
)
_DIRECTNESS_SIGNAL = re.compile(
    r"\b(brief|short|concise|direct|just tell me|no fluff|straight answer)\b",
    re.IGNORECASE,
)
_LOW_DIRECTNESS_SIGNAL = re.compile(
    r"\b(explain|walk me through|more detail|why|how does)\b",
    re.IGNORECASE,
)
_TOKEN_PATTERN = re.compile(r"\b[a-z][a-z0-9_-]{3,}\b", re.IGNORECASE)
_STOPWORDS = {
    "about",
    "after",
    "again",
    "alice",
    "also",
    "before",
    "build",
    "could",
    "from",
    "have",
    "into",
    "just",
    "like",
    "need",
    "next",
    "phase",
    "please",
    "should",
    "that",
    "this",
    "want",
    "what",
    "when",
    "with",
    "working",
    "would",
}


def _clamp(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = 0.5
    return round(max(MIN_WEIGHT, min(MAX_WEIGHT, numeric)), 3)


def _as_personality(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    personality = dict(DEFAULT_PERSONALITY)
    personality.update(dict(payload or {}))
    for key in (
        "curiosity_weight",
        "directness",
        "humor_threshold",
        "concern_sensitivity",
    ):
        personality[key] = _clamp(personality.get(key))
    personality["interests"] = [
        str(item).strip().lower()
        for item in list(personality.get("interests") or [])
        if str(item).strip()
    ][:30]
    opinions = personality.get("opinions")
    personality["opinions"] = dict(opinions or {}) if isinstance(opinions, dict) else {}
    return personality


def _topic_candidates(text: str, extra_topics: Iterable[str] | None = None) -> List[str]:
    terms: List[str] = []
    for raw in list(extra_topics or []):
        cleaned = str(raw or "").strip().lower()
        if cleaned and len(cleaned) >= 3:
            terms.append(cleaned[:64])

    for token in _TOKEN_PATTERN.findall(str(text or "").lower()):
        if token in _STOPWORDS or token.isdigit():
            continue
        terms.append(token[:64])
    return terms


def _describe_weight(value: float, low: str, mid: str, high: str) -> str:
    if value < 0.35:
        return low
    if value > 0.65:
        return high
    return mid


class PersonalityLayer:
    """Updates and renders ALICE's world-model backed personality weights."""

    def __init__(self, world_model: WorldModel | None = None) -> None:
        self.world_model = world_model or get_world_model()

    def update_after_turn(
        self,
        *,
        user_input: str,
        response_text: str = "",
        conversation_topics: Iterable[str] | None = None,
    ) -> Dict[str, Any]:
        personality = _as_personality(self.world_model.get_personality())
        meta = self.world_model.get_personality_meta()

        user_text = str(user_input or "").strip()
        word_count = len(user_text.split())
        is_short = bool(user_text) and (
            word_count <= _SHORT_RESPONSE_WORD_LIMIT
            or len(user_text) <= _SHORT_RESPONSE_CHAR_LIMIT
        )

        short_streak = int(meta.get("short_response_streak") or 0)
        short_streak = short_streak + 1 if is_short else 0
        meta["short_response_streak"] = short_streak

        if short_streak >= _SHORT_RESPONSE_STREAK_THRESHOLD:
            personality["curiosity_weight"] = _clamp(
                float(personality["curiosity_weight"]) - 0.02
            )

        if _WARM_SIGNAL.search(user_text):
            personality["humor_threshold"] = _clamp(
                float(personality["humor_threshold"]) + 0.015
            )

        if _DIRECTNESS_SIGNAL.search(user_text):
            personality["directness"] = _clamp(float(personality["directness"]) + 0.02)
            personality["curiosity_weight"] = _clamp(
                float(personality["curiosity_weight"]) - 0.01
            )
        elif _LOW_DIRECTNESS_SIGNAL.search(user_text):
            personality["directness"] = _clamp(float(personality["directness"]) - 0.01)

        if _STRESS_SIGNAL.search(user_text):
            personality["concern_sensitivity"] = _clamp(
                float(personality["concern_sensitivity"]) + 0.02
            )

        topic_counts = Counter(dict(meta.get("topic_counts") or {}))
        for topic in _topic_candidates(user_text, conversation_topics):
            topic_counts[topic] += 1

        interests = list(personality.get("interests") or [])
        interest_set = {str(item).lower() for item in interests}
        for topic, count in topic_counts.most_common():
            if count >= 3 and topic not in interest_set:
                interests.append(topic)
                interest_set.add(topic)
            if len(interests) >= 30:
                break

        personality["interests"] = interests[:30]
        meta["topic_counts"] = dict(topic_counts.most_common(120))

        self.world_model.update_personality(personality)
        self.world_model.update_personality_meta(meta)
        return self.world_model.get_personality()

    def build_system_instructions(self) -> str:
        return personality_to_system_instructions(self.world_model.get_personality())


_INTEREST_NOISE: frozenset = frozenset({
    "able", "also", "back", "been", "bill", "body", "call", "case", "check",
    "code", "come", "cool", "does", "done", "even", "feel", "file", "find",
    "from", "give", "going", "good", "have", "here", "hope", "into", "just",
    "keep", "know", "last", "late", "life", "like", "long", "look", "make",
    "more", "most", "much", "need", "next", "none", "nothing", "once", "only",
    "over", "part", "plan", "plus", "push", "rain", "ready", "real", "right",
    "said", "same", "seem", "seen", "send", "show", "side", "size", "some",
    "soon", "stay", "such", "sure", "take", "talk", "tell", "than", "that",
    "them", "then", "they", "this", "time", "toda", "today", "very", "want",
    "week", "well", "what", "when", "will", "with", "work", "year", "your",
})


def _meaningful_interests(raw: List[str], limit: int = 5) -> List[str]:
    """Return only human-readable, non-noise interest topics."""
    out: List[str] = []
    seen: set = set()
    for topic in raw:
        t = str(topic or "").strip().lower()
        if not t or len(t) < 5:
            continue
        if ":" in t:          # drop intent strings like "weather:current"
            continue
        if t in _INTEREST_NOISE:
            continue
        if t not in seen:
            out.append(t)
            seen.add(t)
        if len(out) >= limit:
            break
    return out


def personality_to_system_instructions(personality: Dict[str, Any] | None) -> str:
    shaped = _as_personality(personality)
    curiosity = float(shaped["curiosity_weight"])
    directness = float(shaped["directness"])
    humor_threshold = float(shaped["humor_threshold"])
    concern = float(shaped["concern_sensitivity"])
    interests = _meaningful_interests(list(shaped.get("interests") or []))

    lines = [
        "Current ALICE personality drift:",
        f"- Follow-up behavior: {_describe_weight(curiosity, 'ask follow-up questions rarely; prefer completing the current answer', 'ask one useful follow-up only when it moves the task forward', 'show active curiosity and ask relevant follow-ups when useful')}.",
        f"- Directness: {_describe_weight(directness, 'use a warmer, softer tone before corrections', 'be clear and balanced', 'be direct, concise, and practical')}.",
        f"- Humor: {_describe_weight(humor_threshold, 'dry wit and humor are welcome; lean into it when the moment calls for it', 'light humor is fine when the tone is right', 'keep responses focused; humor only when clearly invited')}.",
        f"- Concern sensitivity: {_describe_weight(concern, 'do not over-index on mild stress signals', 'acknowledge stress briefly when relevant', 'notice stress quickly and respond with calm, practical support')}.",
    ]
    if interests:
        lines.append("- Topics this user cares about: " + ", ".join(interests) + ".")
    return "\n".join(lines)


def build_personality_system_instructions(
    world_model: WorldModel | None = None,
) -> str:
    return PersonalityLayer(world_model=world_model).build_system_instructions()


def apply_personality_to_system_prompt(
    system_prompt: str,
    world_model: WorldModel | None = None,
) -> str:
    base = str(system_prompt or "").strip()
    instructions = build_personality_system_instructions(world_model=world_model)
    if not instructions:
        return base
    return f"{base}\n\n{instructions}" if base else instructions
