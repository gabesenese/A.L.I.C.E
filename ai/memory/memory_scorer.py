from __future__ import annotations

import math
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

_EMOTIONAL_KEYWORDS = re.compile(
    r"\b(love|hate|fear|excited|anxious|proud|ashamed|angry|happy|sad|scared|"
    r"delighted|devastated|overwhelmed|grateful|frustrated|hopeful|worried|"
    r"thrilled|depressed|joyful|disgusted|surprised|confused|confident)\b",
    re.IGNORECASE,
)

_HIGH_VALUE_TAGS = {
    "goal",
    "decision",
    "milestone",
    "fact",
    "preference",
    "health",
    "relationship",
}

_SOURCE_SCORES = {
    "system_verified": 0.95,
    "document_ingest": 0.90,
    "plugin_result": 0.85,
    "user_explicit": 0.80,
    "conversation": 0.55,
    "ambient": 0.40,
}


class MemoryScorer:
    """
    Extended composite scorer for memory entries.

    Dimensions (weights sum to 1.0):
      recency    0.25 — exponential decay from creation; half-life = decay_hours
      access     0.20 — log-normalised access frequency
      base       0.25 — stored importance at creation
      emotional  0.10 — presence of emotional-salience keywords
      tag_value  0.10 — bonus for high-value tag categories
      source     0.10 — reliability by source type

    Output is 0.0–1.0 and is intended to be written back to memories.importance
    by the maintenance scheduler.
    """

    _WEIGHTS = {
        "recency": 0.25,
        "access": 0.20,
        "base": 0.25,
        "emotional": 0.10,
        "tag_value": 0.10,
        "source": 0.10,
    }

    def __init__(self, decay_hours: float = 168.0) -> None:
        # 168 h = 7-day half-life
        self._decay_lambda = math.log(2) / decay_hours
        self._lock = threading.Lock()
        self._max_access: int = 1

    def score(self, entry: Any) -> float:
        w = self._WEIGHTS
        raw = (
            w["recency"] * self._recency(entry)
            + w["access"] * self._access(entry)
            + w["base"] * self._base(entry)
            + w["emotional"] * self._emotional(entry)
            + w["tag_value"] * self._tag_value(entry)
            + w["source"] * self._source(entry)
        )
        return min(1.0, max(0.0, raw))

    def batch_score(self, entries: List[Any]) -> Dict[str, float]:
        if entries:
            max_ac = max(
                (int(getattr(e, "access_count", 0)) for e in entries), default=1
            )
            with self._lock:
                self._max_access = max(self._max_access, max_ac, 1)
        return {getattr(e, "id", str(i)): self.score(e) for i, e in enumerate(entries)}

    # ------------------------------------------------------------------
    # Sub-scorers
    # ------------------------------------------------------------------

    def _recency(self, entry: Any) -> float:
        ts = getattr(entry, "timestamp", None) or getattr(entry, "created_at", None)
        if not ts:
            return 0.5
        try:
            dt = datetime.fromisoformat(str(ts).rstrip("Z"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_h = max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)
            return math.exp(-self._decay_lambda * age_h)
        except (ValueError, TypeError):
            return 0.5

    def _access(self, entry: Any) -> float:
        count = int(getattr(entry, "access_count", 0))
        with self._lock:
            max_ac = max(self._max_access, count, 1)
        return math.log1p(count) / math.log1p(max_ac)

    def _base(self, entry: Any) -> float:
        return min(1.0, max(0.0, float(getattr(entry, "importance", 0.5))))

    def _emotional(self, entry: Any) -> float:
        content = str(getattr(entry, "content", "") or "")
        hits = len(_EMOTIONAL_KEYWORDS.findall(content))
        return min(1.0, hits * 0.25)

    def _tag_value(self, entry: Any) -> float:
        tags = getattr(entry, "tags", None) or []
        if not tags:
            return 0.0
        tag_set = {t.lower() for t in tags}
        matches = len(tag_set & _HIGH_VALUE_TAGS)
        return min(1.0, matches * 0.35)

    def _source(self, entry: Any) -> float:
        ctx = getattr(entry, "context", None)
        if not isinstance(ctx, dict):
            return 0.6
        source = str(ctx.get("source") or ctx.get("origin") or "").lower().strip()
        return _SOURCE_SCORES.get(source, 0.60)


_scorer: Optional[MemoryScorer] = None


def get_memory_scorer() -> MemoryScorer:
    global _scorer
    if _scorer is None:
        _scorer = MemoryScorer()
    return _scorer
