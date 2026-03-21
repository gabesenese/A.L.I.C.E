"""Consolidates episodic traces into compact semantic summaries."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List


class MemoryConsolidator:
    def consolidate(self, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        intents = Counter(str(ep.get("intent", "unknown")) for ep in episodes)
        top = intents.most_common(5)
        return {
            "episode_count": len(episodes),
            "top_intents": [{"intent": k, "count": v} for k, v in top],
            "summary": ", ".join(f"{k}({v})" for k, v in top) if top else "",
        }
