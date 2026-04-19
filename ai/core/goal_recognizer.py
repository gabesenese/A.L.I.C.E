"""Shared goal/vision declaration recognition for routing and orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional


@dataclass
class GoalSignal:
    goal: str
    project_direction: str
    markers: List[str]
    confidence: float


class GoalRecognizer:
    """Detect strategic declarative project direction statements."""

    _LEAD_PATTERNS = (
        r"\bi\s+(?:want|need|am\s+trying|am\s+aiming|intend|plan)\s+to\b",
        r"\bi'?m\s+(?:building|working\s+on|trying\s+to)\b",
        r"\bmy\s+(?:goal|objective)\s+is\s+to\b",
        r"\bi\s+do\s+not\s+want\b",
        r"\bi\s+don't\s+want\b",
    )

    _DIRECTION_MARKERS = (
        "agent",
        "autonomous",
        "autonomy",
        "chatbot",
        "architecture",
        "orchestration",
        "planning",
        "plan-first",
        "think in steps",
        "step by step",
        "task persistence",
        "stateful",
        "initiative",
        "project vision",
        "project direction",
        "more like",
        "reliable",
        "deterministic",
        "verification",
        "execution path",
        "queue",
    )

    _TOOL_TARGET_MARKERS = (
        "email",
        "calendar",
        "note",
        "notes",
        "file",
        "files",
        "weather",
        "time",
        "reminder",
        "music",
        "song",
        "search",
    )

    _TOOL_VERB_RE = re.compile(
        r"\b(open|launch|send|delete|remove|list|search|read|write|schedule|set|remind|play|run)\b",
        re.IGNORECASE,
    )

    _GOAL_EXTRACT_RE = re.compile(
        r"\b(?:"
        r"i\s+(?:want|need|am\s+trying|am\s+aiming|intend|plan)\s+to"
        r"|my\s+(?:goal|objective)\s+is\s+to"
        r"|i'?m\s+(?:building|working\s+on|trying\s+to)"
        r")\s+(.+)",
        re.IGNORECASE,
    )

    _INFORMATIONAL_INTRO_RE = re.compile(
        r"\b(?:i\s+(?:want|need|would\s+like)\s+to\s+know|tell\s+me)\b",
        re.IGNORECASE,
    )
    _INFORMATIONAL_CUE_RE = re.compile(
        r"\b(?:what|how|why|difference\s+between|compare|comparison|define|explain)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _is_direct_informational_request(cls, text: str) -> bool:
        low = str(text or "").strip().lower()
        if not low:
            return False
        has_intro = bool(cls._INFORMATIONAL_INTRO_RE.search(low))
        has_cue = bool(cls._INFORMATIONAL_CUE_RE.search(low) or "?" in low)
        return bool(has_intro and has_cue)

    def detect(self, text: str) -> Optional[GoalSignal]:
        raw = str(text or "").strip()
        if not raw:
            return None

        if self._is_direct_informational_request(raw):
            return None

        low = raw.lower()
        has_goal_lead = any(re.search(pat, low) for pat in self._LEAD_PATTERNS)
        markers = [m for m in self._DIRECTION_MARKERS if m in low]

        if not has_goal_lead:
            return None
        if not markers:
            return None

        has_tool_target = any(token in low for token in self._TOOL_TARGET_MARKERS)
        has_tool_verb = bool(self._TOOL_VERB_RE.search(low))
        if has_tool_target and has_tool_verb:
            return None

        goal_text = raw
        match = self._GOAL_EXTRACT_RE.search(raw)
        if match:
            goal_text = match.group(1).strip(" .!?;:") or raw

        project_direction = "agentic_autonomy"
        if "chatbot" in low and "agent" not in low and "autonomy" not in low:
            project_direction = "non_chatbot_behavior"

        confidence = 0.82 + min(0.12, 0.02 * len(markers))
        confidence = max(0.0, min(0.96, confidence))

        return GoalSignal(
            goal=goal_text[:180],
            project_direction=project_direction,
            markers=markers[:10],
            confidence=confidence,
        )


_goal_recognizer: GoalRecognizer | None = None


def get_goal_recognizer() -> GoalRecognizer:
    global _goal_recognizer
    if _goal_recognizer is None:
        _goal_recognizer = GoalRecognizer()
    return _goal_recognizer
