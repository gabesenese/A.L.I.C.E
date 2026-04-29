"""
A.L.I.C.E. Utterance Fingerprint Store
========================================
Caches successful parse outcomes so that repeated or near-identical
utterances skip the full NLP pipeline and reuse the known-good result.

How it works
------------
1. Normalize utterance (lowercase, collapse whitespace, strip punctuation)
2. SHA-256 the normalized text → fingerprint key
3. On lookup: return CachedParse if fingerprint exists
4. On store: write entry to JSON store, keep max MAX_ENTRIES in memory

File
----
data/analytics/utterance_fingerprints.json
  {fingerprint: {text, intent, frame_name, slots, confidence, hits, last_used}}

Usage
-----
>>> from ai.core.utterance_fingerprint import get_fingerprint_store
>>> fp = get_fingerprint_store()
>>> hit = fp.lookup("find my meeting notes")
>>> if hit and hit.confidence >= 0.80:
...     use(hit.intent, hit.slots)
>>> else:
...     result = full_nlp(text)
...     fp.store("find my meeting notes", result.intent, "SEARCH_NOTE", slots, 0.89)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

MAX_ENTRIES = 500  # max cached fingerprints
MIN_CONFIDENCE = 0.72  # minimum confidence to store as fingerprint


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CachedParse:
    fingerprint: str
    normalized_text: str
    intent: str
    frame_name: Optional[str]
    slots: Dict[str, Any]
    confidence: float
    hits: int = 1
    last_used: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fingerprint": self.fingerprint,
            "normalized_text": self.normalized_text,
            "intent": self.intent,
            "frame_name": self.frame_name,
            "slots": self.slots,
            "confidence": round(self.confidence, 4),
            "hits": self.hits,
            "last_used": self.last_used,
        }


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def _fingerprint(normalized: str) -> str:
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class FingerprintStore:
    """
    Thread-safe fingerprint cache backed by a local JSON file.
    """

    def __init__(self, store_path: Optional[str] = None):
        if store_path is None:
            store_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "data",
                "analytics",
                "utterance_fingerprints.json",
            )
        self._path = Path(store_path).resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: Dict[str, CachedParse] = {}
        self._dirty = False
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, text: str) -> Optional[CachedParse]:
        """Return a CachedParse for *text* if one exists, else None."""
        norm = _normalize(text)
        fp = _fingerprint(norm)
        with self._lock:
            entry = self._cache.get(fp)
            if entry:
                entry.hits += 1
                entry.last_used = datetime.utcnow().isoformat()
                self._dirty = True
                logger.debug(
                    "[FINGERPRINT] HIT %s → %s (%.2f)",
                    fp[:8],
                    entry.intent,
                    entry.confidence,
                )
                return entry
        return None

    def store(
        self,
        text: str,
        intent: str,
        frame_name: Optional[str],
        slots: Dict[str, Any],
        confidence: float,
    ) -> None:
        """Cache a successful parse result (only if confidence >= MIN_CONFIDENCE)."""
        if confidence < MIN_CONFIDENCE:
            return
        norm = _normalize(text)
        fp = _fingerprint(norm)
        with self._lock:
            if fp in self._cache:
                # Refresh confidence if new parse is more confident
                existing = self._cache[fp]
                if confidence > existing.confidence:
                    existing.confidence = confidence
                    existing.intent = intent
                    existing.frame_name = frame_name
                    existing.slots = slots
                existing.hits += 1
                existing.last_used = datetime.utcnow().isoformat()
            else:
                self._cache[fp] = CachedParse(
                    fingerprint=fp,
                    normalized_text=norm,
                    intent=intent,
                    frame_name=frame_name,
                    slots=slots,
                    confidence=confidence,
                )
            self._dirty = True
            self._evict_if_needed()
        logger.debug("[FINGERPRINT] STORED %s → %s (%.2f)", fp[:8], intent, confidence)

    def invalidate(self, text: str) -> None:
        """Remove a single entry (e.g. after the user corrected Alice)."""
        norm = _normalize(text)
        fp = _fingerprint(norm)
        with self._lock:
            if fp in self._cache:
                del self._cache[fp]
                self._dirty = True

    def flush(self) -> None:
        """Persist dirty cache to disk."""
        with self._lock:
            if not self._dirty:
                return
            try:
                data = {k: v.as_dict() for k, v in self._cache.items()}
                self._path.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                self._dirty = False
                logger.debug("[FINGERPRINT] Flushed %d entries to disk", len(data))
            except Exception as exc:
                logger.warning("[FINGERPRINT] Could not persist: %s", exc)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._cache),
                "total_hits": sum(e.hits for e in self._cache.values()),
                "max_entries": MAX_ENTRIES,
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for fp, d in raw.items():
                self._cache[fp] = CachedParse(
                    fingerprint=d.get("fingerprint", fp),
                    normalized_text=d.get("normalized_text", ""),
                    intent=d.get("intent", ""),
                    frame_name=d.get("frame_name"),
                    slots=d.get("slots", {}),
                    confidence=float(d.get("confidence", 0.0)),
                    hits=int(d.get("hits", 1)),
                    last_used=d.get("last_used", ""),
                )
            logger.info("[FINGERPRINT] Loaded %d cached parses", len(self._cache))
        except Exception as exc:
            logger.warning("[FINGERPRINT] Could not load store: %s", exc)

    def _evict_if_needed(self) -> None:
        """LRU eviction: drop oldest-used entries when over MAX_ENTRIES."""
        if len(self._cache) <= MAX_ENTRIES:
            return
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda item: item[1].last_used,
        )
        to_remove = len(self._cache) - MAX_ENTRIES
        for fp, _ in sorted_entries[:to_remove]:
            del self._cache[fp]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[FingerprintStore] = None
_inst_lock = threading.Lock()


def get_fingerprint_store() -> FingerprintStore:
    global _instance
    if _instance is None:
        with _inst_lock:
            if _instance is None:
                _instance = FingerprintStore()
    return _instance
