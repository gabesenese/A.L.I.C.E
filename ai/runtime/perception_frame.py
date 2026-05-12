from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import re
from typing import Any, Dict


_BOUNDARY_PHRASES = (
    "i want to",
    "i need to",
    "i was wondering if",
    "i was wondering whether",
    "can you",
    "could you",
    "do i have",
    "are there",
    "let's",
    "lets",
    "now i",
    "but can you",
    "also can you",
)


@dataclass
class PerceptionFrame:
    raw_input: str
    social_context: str = ""
    actual_request: str = ""
    explicit_goal: str = ""
    topic: str = ""
    entities: Dict[str, Any] = field(default_factory=dict)
    user_energy_signal: str = ""
    user_mood_signal: str = ""
    time_reference: str = ""
    is_greeting: bool = False
    is_continuation: bool = False
    is_memory_rights_request: bool = False
    is_action_request: bool = False
    is_question: bool = False
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _extract_actual_request(text: str) -> str:
    low = text.lower()
    best_idx = None
    best_phrase = ""
    for phrase in _BOUNDARY_PHRASES:
        idx = low.find(phrase)
        if idx >= 0 and (best_idx is None or idx < best_idx):
            best_idx = idx
            best_phrase = phrase
    if best_idx is None:
        return ""
    out = text[best_idx:].strip(" ,;:.!?")
    if best_phrase in {"i was wondering if", "i was wondering whether"}:
        out = re.sub(
            r"^\s*i was wondering (?:if|whether)\s+",
            "",
            out,
            flags=re.IGNORECASE,
        )
    return out.strip()


def _clean_social_context(raw: str, actual_request: str) -> str:
    if not raw.strip():
        return ""
    if not actual_request:
        return ""
    low_raw = raw.lower()
    low_req = actual_request.lower()
    idx = low_raw.find(low_req)
    if idx < 0:
        return ""
    prefix = raw[:idx].strip(" ,;:.!?")
    if not prefix:
        return ""
    chunks = [part.strip(" ,;:.!?") for part in re.split(r",| and | but ", prefix) if part.strip()]
    return "; ".join(chunks[:3])


def build_perception_frame(
    user_input: str,
    *,
    local_time: datetime | None = None,
    operator_state: Dict[str, Any] | None = None,
    project_memory: Dict[str, Any] | None = None,
) -> PerceptionFrame:
    raw = str(user_input or "").strip()
    low = raw.lower()
    actual_request = _extract_actual_request(raw)
    social_context = _clean_social_context(raw, actual_request)
    request_low = actual_request.lower()

    is_greeting = low in {"hi", "hello", "hey", "hi alice", "hello alice", "hey alice"}
    is_memory_rights = bool(
        re.search(r"\b(delete|erase|remove|forget)\b.{0,40}\b(memory|memories|data|topic)\b", low)
    )
    is_action_request = bool(actual_request) or is_memory_rights
    is_question = "?" in raw or low.startswith(("what", "how", "can", "could", "do", "are"))
    is_continuation = bool(
        actual_request
        and any(token in request_low for token in ("work on", "continue", "move on", "delete", "open notes", "notes"))
    )

    mood = "unknown"
    if "positive" in low:
        mood = "positive"
    elif "tired" in low:
        mood = "tired"
    elif "focused" in low:
        mood = "focused"
    elif social_context:
        mood = "neutral"

    energy = "unknown"
    if "long day" in low or "tired" in low:
        energy = "low"
    elif "focused" in low:
        energy = "high"
    elif mood == "positive":
        energy = "medium"

    if "notes" in request_low:
        topic = "notes"
    elif "alice" in request_low or "alice" in low:
        topic = "Alice"
    elif "memory" in low or "memories" in low:
        topic = "memory"
    else:
        topic = ""

    if not actual_request and is_memory_rights:
        actual_request = "delete memories from your data"
    if is_greeting and not actual_request:
        social_context = ""

    time_ref = "unknown"
    if local_time is not None:
        h = int(local_time.hour)
        if 5 <= h <= 11:
            time_ref = "morning"
        elif 12 <= h <= 16:
            time_ref = "afternoon"
        elif 17 <= h <= 21:
            time_ref = "evening"
        else:
            time_ref = "night"

    return PerceptionFrame(
        raw_input=raw,
        social_context=social_context,
        actual_request=actual_request,
        explicit_goal=actual_request,
        topic=topic,
        entities={},
        user_energy_signal=energy,
        user_mood_signal=mood,
        time_reference=time_ref,
        is_greeting=is_greeting,
        is_continuation=is_continuation,
        is_memory_rights_request=is_memory_rights,
        is_action_request=is_action_request,
        is_question=is_question,
        confidence=0.8 if actual_request or is_memory_rights or is_greeting else 0.5,
    )
