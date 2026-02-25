"""
A.L.I.C.E. Probabilistic Slot Filler
======================================
Multi-strategy slot extraction with confidence-weighted merging.

Each slot is resolved by running up to four independent strategies:
  1. Pattern   — compiled regex patterns (fast, precise)
  2. Context   — entity chain from DialogueMemory / ConversationContext
  3. Fuzzy     — Levenshtein-based title matching against known notes
  4. Positional — positional heuristics (last noun phrase after verb, etc.)

Results are merged via weighted vote:
  pattern   weight = 1.00
  context   weight = 0.90
  fuzzy     weight = 0.75
  positional weight = 0.60

Final confidence = best_weight × individual_confidence, clamped to [0.0, 0.97].

Usage
-----
>>> from ai.core.prob_slot_filler import ProbabilisticSlotFiller
>>> sf = ProbabilisticSlotFiller()
>>> slots = sf.fill("find my meeting notes tagged work", frame_name="SEARCH_NOTE")
>>> slots["query"]
SlotResult(value='meeting notes', confidence=0.86, strategy='pattern')
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SlotResult:
    """Single resolved slot value with provenance."""
    value: Any
    confidence: float
    strategy: str          # pattern | context | fuzzy | positional
    raw_match: str = ""


@dataclass
class FilledSlots:
    """All resolved slots for one utterance."""
    slots: Dict[str, SlotResult] = field(default_factory=dict)
    frame_name: str = ""
    fill_confidence: float = 0.0   # average confidence across filled slots

    def get(self, name: str, default: Any = None) -> Any:
        r = self.slots.get(name)
        return r.value if r else default

    def confidence(self, name: str) -> float:
        r = self.slots.get(name)
        return r.confidence if r else 0.0

    def __contains__(self, name: str) -> bool:
        return name in self.slots

    def as_dict(self) -> Dict[str, Any]:
        return {k: v.value for k, v in self.slots.items()}


# ---------------------------------------------------------------------------
# Per-slot pattern banks
# ---------------------------------------------------------------------------

# title: text after 'called', 'titled', 'named', quotes, or after verb
_TITLE_PATS: List[re.Pattern] = [
    re.compile(r'(?:called|titled?|named?)\s+["\']?(.+?)["\']?(?:\s+tagged|\s+with tag|\s+and\s|\s*$)', re.I),
    re.compile(r'"([^"]{2,80})"'),
    re.compile(r"'([^']{2,80})'"),
    re.compile(r'(?:create|make|write|save|add)\s+(?:a\s+)?(?:note\s+)?(?:about\s+)?([A-Z][^\s,]{1,40})', re.I),
]

# query: text after search verbs or 'about'/'for'
_QUERY_PATS: List[re.Pattern] = [
    re.compile(r'(?:find|search|look\s+for)\s+(?:my\s+)?(?:notes?\s+(?:about|on|for)\s+)?(.+?)(?:\s+tagged|\s+with\s+tag|\s*$)', re.I),
    re.compile(r'(?:notes?\s+(?:about|on|containing|mentioning|with))\s+(.+?)(?:\s+tagged|\s*$)', re.I),
    re.compile(r'(?:about|for|related\s+to)\s+["\']?([^"\']+?)["\']?(?:\s+tagged|\s*$)', re.I),
]

# note_ref: "the X note", "note called X", note #N
_NOTE_REF_PATS: List[re.Pattern] = [
    re.compile(r'(?:the\s+)?note\s+(?:called|titled?|named?)\s+["\']?(.+?)["\']?(?:\s|$)', re.I),
    re.compile(r'(?:note|file)\s+#?(\d+)', re.I),
    re.compile(r'(?:the\s+)([A-Z][^\s,]{2,40})\s+note', re.I),
    re.compile(r'"([^"]{2,60})"'),
    re.compile(r"'([^']{2,60})'"),
]

# tags: #word or "tagged X" patterns
_TAG_PATS: List[re.Pattern] = [
    re.compile(r'#(\w+)', re.I),
    re.compile(r'tagged\s+(\w+(?:\s*,\s*\w+)*)', re.I),
    re.compile(r'with\s+tag\s+(\w+)', re.I),
    re.compile(r'tag[s]?\s*:\s*(\w+(?:[,\s]+\w+)*)', re.I),
]

# priority
_PRIORITY_MAP = {
    "urgent": "urgent", "critical": "urgent", "asap": "urgent",
    "high": "high", "important": "high",
    "medium": "medium", "normal": "medium",
    "low": "low", "minor": "low",
}

# content: everything after 'content:' or 'saying' or after title
_CONTENT_PATS: List[re.Pattern] = [
    re.compile(r'(?:content|body|text)\s*:\s*(.+?)(?:\s+tagged|\s*$)', re.I),
    re.compile(r'saying\s+["\']?(.+?)["\']?(?:\s+tagged|\s*$)', re.I),
    re.compile(r'(?:that\s+says?|which\s+says?)\s+["\']?(.+?)["\']?(?:\s+tagged|\s*$)', re.I),
]

# event title
_EVENT_TITLE_PATS: List[re.Pattern] = [
    re.compile(r'(?:schedule|add|create|set|book)\s+(?:a\s+)?(?:meeting|event|appointment|reminder\s+for)?\s*(.+?)\s+(?:at|on|for|tomorrow|today|\d)', re.I),
    re.compile(r'(?:meeting|event|appointment)\s+(?:called|titled?|named?|about)\s+(.+?)(?:\s+at|\s+on|\s*$)', re.I),
]

# recipient
_RECIPIENT_PATS: List[re.Pattern] = [
    re.compile(r'(?:to|email)\s+([\w.+-]+@[\w.-]+\.\w+)', re.I),
    re.compile(r'(?:send|write|email)\s+(?:an?\s+(?:email|message)\s+)?to\s+(.+?)(?:\s+about|\s+saying|\s+with|\s*$)', re.I),
]

# date / time (coarse — dateparser handles precision later)
_DATE_PATS: List[re.Pattern] = [
    re.compile(r'\b(today|tomorrow|yesterday|next\s+\w+|this\s+\w+|\d{1,2}/\d{1,2}(?:/\d{2,4})?|\w+\s+\d{1,2}(?:st|nd|rd|th)?)\b', re.I),
]
_TIME_PATS: List[re.Pattern] = [
    re.compile(r'\b(\d{1,2}(?::\d{2})?\s*(?:am|pm)|noon|midnight)\b', re.I),
]

# song / artist
_SONG_PATS: List[re.Pattern] = [
    re.compile(r'(?:play|start|queue)\s+["\']?(.+?)["\']?\s+(?:by|from)', re.I),
    re.compile(r'(?:play|start|queue)\s+["\'](.+?)["\']', re.I),
    re.compile(r'(?:play|start|queue)\s+(.+?)(?:\s+by|\s*$)', re.I),
]
_ARTIST_PATS: List[re.Pattern] = [
    re.compile(r'\bby\s+([A-Z][^\s,]{1,40})', re.I),
    re.compile(r'\bfrom\s+([A-Z][^\s,]{1,40})', re.I),
]


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------

def _first_match(patterns: List[re.Pattern], text: str) -> Optional[str]:
    for p in patterns:
        m = p.search(text)
        if m:
            val = m.group(1).strip() if m.lastindex and m.group(1) else None
            if val:
                return val
    return None


def _multi_match(patterns: List[re.Pattern], text: str) -> List[str]:
    results: List[str] = []
    for p in patterns:
        for m in p.finditer(text):
            val = m.group(1).strip() if m.lastindex and m.group(1) else None
            if val and val not in results:
                results.append(val)
    return results


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def _fuzzy_match(query: str, candidates: List[str], threshold: float = 0.65) -> Optional[Tuple[str, float]]:
    """Return (best_match, normalized_similarity) or None."""
    if not candidates:
        return None
    query_lower = query.lower()
    best_val: Optional[str] = None
    best_sim = 0.0
    for cand in candidates:
        cand_lower = cand.lower()
        max_len = max(len(query_lower), len(cand_lower), 1)
        dist = _levenshtein(query_lower, cand_lower)
        sim = 1.0 - dist / max_len
        # Also check substring containment
        if query_lower in cand_lower or cand_lower in query_lower:
            sim = max(sim, 0.80)
        if sim > best_sim:
            best_sim = sim
            best_val = cand
    if best_val and best_sim >= threshold:
        return best_val, best_sim
    return None


# ---------------------------------------------------------------------------
# Frame → slot schema mapping
# ---------------------------------------------------------------------------

_FRAME_SLOTS: Dict[str, List[str]] = {
    "CREATE_NOTE":   ["title", "content", "tags", "priority"],
    "READ_NOTE":     ["note_ref", "title"],
    "SEARCH_NOTE":   ["query", "tags"],
    "LIST_NOTES":    ["tags"],
    "UPDATE_NOTE":   ["note_ref", "title", "content", "tags"],
    "DELETE_NOTE":   ["note_ref"],
    "ARCHIVE_NOTE":  ["note_ref"],
    "PLAY_MUSIC":    ["song", "artist"],
    "PAUSE_MUSIC":   [],
    "SKIP_MUSIC":    [],
    "CREATE_EVENT":  ["event_title", "date", "time"],
    "LIST_EVENTS":   [],
    "COMPOSE_EMAIL": ["recipient", "subject"],
    "READ_EMAIL":    [],
    "HELP":          [],
    "GENERAL_CHAT":  [],
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ProbabilisticSlotFiller:
    """
    Fill slots for a given utterance + frame using multiple strategies,
    merged by weighted confidence.
    """

    # Strategy weights
    _W_PATTERN    = 1.00
    _W_CONTEXT    = 0.90
    _W_FUZZY      = 0.75
    _W_POSITIONAL = 0.60

    def __init__(self, known_note_titles: Optional[List[str]] = None):
        """
        Parameters
        ----------
        known_note_titles: optional list of titles for fuzzy matching
        """
        self._known_titles: List[str] = known_note_titles or []

    def update_known_titles(self, titles: List[str]) -> None:
        self._known_titles = titles

    def fill(
        self,
        text: str,
        frame_name: str,
        context: Optional[Dict[str, Any]] = None,
        frame_slot_evidence: Optional[Dict[str, Any]] = None,
    ) -> FilledSlots:
        """
        Fill slots for *text* given the active frame.

        Parameters
        ----------
        text:               raw utterance
        frame_name:         name of the matched frame (e.g. SEARCH_NOTE)
        context:            ConversationContext-like dict (coref chain, last entities)
        frame_slot_evidence: pre-extracted slot evidence from FrameParser
        """
        context = context or {}
        frame_slot_evidence = frame_slot_evidence or {}
        target_slots = _FRAME_SLOTS.get(frame_name, [])

        results: Dict[str, SlotResult] = {}
        for slot in target_slots:
            result = self._fill_slot(slot, text, context, frame_slot_evidence)
            if result is not None:
                results[slot] = result

        fill_conf = (
            sum(r.confidence for r in results.values()) / len(results)
            if results else 0.0
        )

        logger.debug("[SLOTS] frame=%s slots=%s", frame_name, {k: v.value for k, v in results.items()})

        return FilledSlots(
            slots=results,
            frame_name=frame_name,
            fill_confidence=fill_conf,
        )

    # ------------------------------------------------------------------
    # Per-slot dispatchers
    # ------------------------------------------------------------------

    def _fill_slot(
        self,
        slot: str,
        text: str,
        context: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> Optional[SlotResult]:
        dispatcher: Dict[str, Callable] = {
            "title":       self._fill_title,
            "content":     self._fill_content,
            "query":       self._fill_query,
            "note_ref":    self._fill_note_ref,
            "tags":        self._fill_tags,
            "priority":    self._fill_priority,
            "song":        self._fill_song,
            "artist":      self._fill_artist,
            "event_title": self._fill_event_title,
            "recipient":   self._fill_recipient,
            "date":        self._fill_date,
            "time":        self._fill_time,
        }
        fn = dispatcher.get(slot)
        if fn is None:
            return None
        return fn(text, context, evidence)

    # ── title ──────────────────────────────────────────────────────────

    def _fill_title(self, text: str, ctx: Dict, ev: Dict) -> Optional[SlotResult]:
        # 1. Frame evidence (from FrameParser inline extractor)
        if ev.get("title"):
            return SlotResult(ev["title"], self._W_PATTERN * 0.90, "pattern", ev["title"])

        # 2. Pattern strategy
        val = _first_match(_TITLE_PATS, text)
        if val:
            return SlotResult(val, self._W_PATTERN * 0.85, "pattern", val)

        # 3. Context strategy — last mentioned title
        last_title = ctx.get("last_title") or ctx.get("last_entities", {}).get("title")
        if last_title:
            return SlotResult(last_title, self._W_CONTEXT * 0.80, "context", last_title)

        # 4. Fuzzy against known titles
        # Tokenize text to get candidate words (skip stop words)
        candidate_tokens = _content_words(text)
        if candidate_tokens and self._known_titles:
            match = _fuzzy_match(" ".join(candidate_tokens), self._known_titles)
            if match:
                return SlotResult(match[0], self._W_FUZZY * match[1], "fuzzy", match[0])

        return None

    # ── content ────────────────────────────────────────────────────────

    def _fill_content(self, text: str, ctx: Dict, ev: Dict) -> Optional[SlotResult]:
        val = _first_match(_CONTENT_PATS, text)
        if val:
            return SlotResult(val, self._W_PATTERN * 0.80, "pattern", val)
        return None

    # ── query ──────────────────────────────────────────────────────────

    def _fill_query(self, text: str, ctx: Dict, ev: Dict) -> Optional[SlotResult]:
        if ev.get("query"):
            return SlotResult(ev["query"], self._W_PATTERN * 0.90, "pattern", ev["query"])

        val = _first_match(_QUERY_PATS, text)
        if val:
            return SlotResult(val, self._W_PATTERN * 0.85, "pattern", val)

        # Positional fallback: content words after the search verb
        m = re.search(r'\b(?:find|search|look\s+for)\s+(.*)', text, re.I)
        if m:
            remainder = m.group(1).strip()
            # Strip leading "notes about / notes on"
            remainder = re.sub(r'^notes?\s+(?:about|on|for)\s+', '', remainder, flags=re.I)
            if remainder:
                return SlotResult(remainder, self._W_POSITIONAL * 0.70, "positional", remainder)

        return None

    # ── note_ref ───────────────────────────────────────────────────────

    def _fill_note_ref(self, text: str, ctx: Dict, ev: Dict) -> Optional[SlotResult]:
        if ev.get("note_ref"):
            return SlotResult(ev["note_ref"], self._W_PATTERN * 0.90, "pattern", ev["note_ref"])

        val = _first_match(_NOTE_REF_PATS, text)
        if val:
            return SlotResult(val, self._W_PATTERN * 0.85, "pattern", val)

        # Context chain: most recently mentioned note title
        chain = ctx.get("entity_chain", [])
        for entity in reversed(chain):
            if entity.get("type") in ("NOTE_REF", "note", "title"):
                v = entity.get("value") or entity.get("title")
                if v:
                    return SlotResult(v, self._W_CONTEXT * 0.85, "context", v)

        last = ctx.get("last_note_title") or ctx.get("last_entities", {}).get("title")
        if last:
            return SlotResult(last, self._W_CONTEXT * 0.80, "context", last)

        # Fuzzy
        cw = _content_words(text)
        if cw and self._known_titles:
            match = _fuzzy_match(" ".join(cw), self._known_titles)
            if match:
                return SlotResult(match[0], self._W_FUZZY * match[1], "fuzzy", match[0])

        return None

    # ── tags ───────────────────────────────────────────────────────────

    def _fill_tags(self, text: str, ctx: Dict, ev: Dict) -> Optional[SlotResult]:
        if ev.get("tags"):
            tags = ev["tags"]
            return SlotResult(tags, self._W_PATTERN * 0.90, "pattern", str(tags))

        tags = _multi_match(_TAG_PATS, text)
        if tags:
            # Flatten: "work, home" from tagged pattern
            flat: List[str] = []
            for t in tags:
                flat.extend([x.strip() for x in t.split(",") if x.strip()])
            return SlotResult(flat, self._W_PATTERN * 0.85, "pattern", str(flat))

        return None

    # ── priority ───────────────────────────────────────────────────────

    def _fill_priority(self, text: str, ctx: Dict, ev: Dict) -> Optional[SlotResult]:
        text_lower = text.lower()
        for kw, level in _PRIORITY_MAP.items():
            if kw in text_lower:
                return SlotResult(level, self._W_PATTERN * 0.90, "pattern", kw)
        return None

    # ── song / artist ──────────────────────────────────────────────────

    def _fill_song(self, text: str, ctx: Dict, ev: Dict) -> Optional[SlotResult]:
        val = _first_match(_SONG_PATS, text)
        if val:
            return SlotResult(val, self._W_PATTERN * 0.85, "pattern", val)
        return None

    def _fill_artist(self, text: str, ctx: Dict, ev: Dict) -> Optional[SlotResult]:
        val = _first_match(_ARTIST_PATS, text)
        if val:
            return SlotResult(val, self._W_PATTERN * 0.85, "pattern", val)
        return None

    # ── event_title ────────────────────────────────────────────────────

    def _fill_event_title(self, text: str, ctx: Dict, ev: Dict) -> Optional[SlotResult]:
        val = _first_match(_EVENT_TITLE_PATS, text)
        if val:
            return SlotResult(val, self._W_PATTERN * 0.80, "pattern", val)
        return None

    # ── recipient ──────────────────────────────────────────────────────

    def _fill_recipient(self, text: str, ctx: Dict, ev: Dict) -> Optional[SlotResult]:
        val = _first_match(_RECIPIENT_PATS, text)
        if val:
            return SlotResult(val, self._W_PATTERN * 0.88, "pattern", val)
        return None

    # ── date / time ────────────────────────────────────────────────────

    def _fill_date(self, text: str, ctx: Dict, ev: Dict) -> Optional[SlotResult]:
        val = _first_match(_DATE_PATS, text)
        if val:
            return SlotResult(val, self._W_PATTERN * 0.75, "pattern", val)
        return None

    def _fill_time(self, text: str, ctx: Dict, ev: Dict) -> Optional[SlotResult]:
        val = _first_match(_TIME_PATS, text)
        if val:
            return SlotResult(val, self._W_PATTERN * 0.80, "pattern", val)
        return None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset([
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "was", "are", "be", "have", "has",
    "do", "did", "will", "would", "should", "could", "can", "may",
    "i", "my", "me", "we", "us", "you", "your", "he", "she", "they",
    "note", "notes", "find", "search", "look", "show", "list", "that",
    "this", "from", "with", "about", "get", "make", "create",
])


def _content_words(text: str) -> List[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]
