"""Tiny semantic index using token overlap for in-process retrieval."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set


class SemanticMemoryIndex:
    def __init__(self) -> None:
        self._docs: Dict[str, str] = {}
        self._inv: Dict[str, Set[str]] = defaultdict(set)

    def add(self, doc_id: str, text: str) -> None:
        doc_id = str(doc_id or "").strip()
        if not doc_id:
            return
        text = str(text or "")
        self._docs[doc_id] = text
        for token in set(text.lower().split()):
            if len(token) >= 3:
                self._inv[token].add(doc_id)

    def search(self, query: str, limit: int = 3) -> List[Dict[str, str]]:
        tokens = [t for t in str(query or "").lower().split() if len(t) >= 3]
        if not tokens:
            return []

        scores: Dict[str, int] = defaultdict(int)
        for token in tokens:
            for doc_id in self._inv.get(token, set()):
                scores[doc_id] += 1

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [
            {
                "doc_id": doc_id,
                "score": str(score),
                "text": self._docs.get(doc_id, "")[:280],
            }
            for doc_id, score in ranked[: max(1, int(limit or 3))]
        ]
