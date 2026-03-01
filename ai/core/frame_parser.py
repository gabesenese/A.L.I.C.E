"""
A.L.I.C.E. Semantic Frame Parser
=================================
Frame-based NLU: instead of a regex branch forest,
every utterance is matched against a typed FrameDefinition that
carries plugin, action, slot schema, and calibrated confidence.

Architecture
------------
1. FrameDefinition   — declarative description of an intent frame
2. FrameMatchResult  — scored match with extracted slot evidence
3. FrameParser       — main engine: match + rank + return best frame

Usage (standalone)
------------------
>>> from ai.core.frame_parser import FrameParser
>>> result = FrameParser().parse("find my meeting notes", context={})
>>> result.frame_name, result.action, result.confidence
('SEARCH_NOTE', 'search_notes', 0.87)
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FrameDefinition:
    """Declarative description of one intent frame."""

    name: str  # e.g. READ_NOTE
    plugin: str  # e.g. notes
    action: str  # e.g. get_note_content
    trigger_keywords: List[str]  # weighted +0.15 each (capped at 0.45)
    trigger_patterns: List[str]  # regex, each grants +0.20 (capped at 0.40)
    anti_patterns: List[str]  # regex that *reduce* score -0.20 each
    required_slots: List[str]  # slots that must be present for high conf
    optional_slots: List[str]
    base_confidence: float = 0.55
    priority: int = 5  # lower = higher priority when tied


@dataclass
class FrameMatchResult:
    """Scored result from frame matching."""

    frame_name: str
    plugin: str
    action: str
    confidence: float
    matched_keywords: List[str] = field(default_factory=list)
    matched_patterns: List[str] = field(default_factory=list)
    slot_evidence: Dict[str, Any] = field(default_factory=dict)
    missing_required_slots: List[str] = field(default_factory=list)
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Frame registry
# ---------------------------------------------------------------------------

_FRAMES: List[FrameDefinition] = [
    # ── Notes ──────────────────────────────────────────────────────────────
    FrameDefinition(
        name="CREATE_NOTE",
        plugin="notes",
        action="create_note",
        trigger_keywords=[
            "create",
            "write",
            "make",
            "add",
            "new",
            "jot",
            "record",
            "draft",
            "save",
            "compose",
            "note down",
            "take note",
        ],
        trigger_patterns=[
            r"\b(create|write|make|add|new|save|jot(?:\s+down)?|record|draft|compose)\b.{0,25}\bnote\b",
            r"\bnote\b.{0,20}\b(about|on|regarding|for|titled?)\b",
            r"\b(take|jot).{0,10}\bnote\b",
            r"\b(remind(?:er)?|memo)\b.{0,20}\babout\b",
        ],
        anti_patterns=[
            r"\b(find|search|look\s+for|show|read|open|get|list|delete|remove)\b",
        ],
        required_slots=[],
        optional_slots=["title", "content", "tags", "priority"],
        base_confidence=0.62,
        priority=4,
    ),
    FrameDefinition(
        name="READ_NOTE",
        plugin="notes",
        action="get_note_content",
        trigger_keywords=[
            "read",
            "open",
            "show",
            "display",
            "view",
            "get",
            "fetch",
            "what does",
            "what's in",
            "content of",
        ],
        trigger_patterns=[
            r"\b(read|open|show|display|view|get|fetch)\b.{0,20}\bnote\b",
            r"\b(what(?:'s|\s+is)\s+(in|inside|on))\b.{0,25}\bnote\b",
            r"\bnote\b.{0,15}\b(content|text|body|inside)\b",
            r"\b(show|display)\s+me\b",
        ],
        anti_patterns=[
            r"\b(find|search|look\s+for|list\s+all|create|write|make)\b",
        ],
        required_slots=["note_ref"],
        optional_slots=[],
        base_confidence=0.60,
        priority=3,
    ),
    FrameDefinition(
        name="SEARCH_NOTE",
        plugin="notes",
        action="search_notes",
        trigger_keywords=[
            "find",
            "search",
            "look for",
            "locate",
            "filter",
            "where is",
            "which note",
            "notes about",
            "notes with",
        ],
        trigger_patterns=[
            r"\b(find|search|look\s+for|locate|filter)\b.{0,20}\bnotes?\b",
            r"\bnotes?\b.{0,15}\b(about|tagged|with tag|containing|mentioning)\b",
            r"\b(which|what)\s+notes?\b",
            r"\bdo i have.{0,10}\bnotes?\b",
        ],
        anti_patterns=[
            r"\b(create|write|make|add|open|read|show content|delete|remove)\b",
        ],
        required_slots=["query"],
        optional_slots=["tags", "date_range"],
        base_confidence=0.65,
        priority=3,
    ),
    FrameDefinition(
        name="LIST_NOTES",
        plugin="notes",
        action="list_notes",
        trigger_keywords=[
            "list",
            "show all",
            "all notes",
            "my notes",
            "notes i have",
            "see all",
            "display all",
        ],
        trigger_patterns=[
            r"\b(list|show|display)\s+(all|my)\s+notes?\b",
            r"\ball\s+(my\s+)?notes?\b",
            r"\bwhat notes\b",
            r"\bnotes?\s+i\s+have\b",
        ],
        anti_patterns=[
            r"\b(find|search|look\s+for|about|tagged)\b",
        ],
        required_slots=[],
        optional_slots=["tags", "date_range"],
        base_confidence=0.68,
        priority=2,
    ),
    FrameDefinition(
        name="UPDATE_NOTE",
        plugin="notes",
        action="update_note",
        trigger_keywords=[
            "update",
            "edit",
            "change",
            "modify",
            "revise",
            "rename",
            "append",
            "add to",
            "rewrite",
        ],
        trigger_patterns=[
            r"\b(update|edit|change|modify|revise|rename|rewrite)\b.{0,20}\bnote\b",
            r"\b(append|add)\b.{0,15}\b(to|into)\b.{0,20}\bnote\b",
            r"\bnote\b.{0,15}\b(update|edit|change|rename)\b",
        ],
        anti_patterns=[],
        required_slots=["note_ref"],
        optional_slots=["title", "content", "tags"],
        base_confidence=0.65,
        priority=4,
    ),
    FrameDefinition(
        name="DELETE_NOTE",
        plugin="notes",
        action="delete_note",
        trigger_keywords=[
            "delete",
            "remove",
            "trash",
            "erase",
            "discard",
            "get rid of",
        ],
        trigger_patterns=[
            r"\b(delete|remove|trash|erase|discard)\b.{0,20}\bnote\b",
            r"\bnote\b.{0,15}\b(delete|remove|trash|erase)\b",
            r"\bget\s+rid\s+of\b.{0,20}\bnote\b",
        ],
        anti_patterns=[],
        required_slots=["note_ref"],
        optional_slots=[],
        base_confidence=0.70,
        priority=3,
    ),
    FrameDefinition(
        name="ARCHIVE_NOTE",
        plugin="notes",
        action="archive_note",
        trigger_keywords=["archive", "store away", "put away", "stash"],
        trigger_patterns=[
            r"\b(archive|store\s+away|put\s+away|stash)\b.{0,20}\bnote\b",
        ],
        anti_patterns=[],
        required_slots=["note_ref"],
        optional_slots=[],
        base_confidence=0.72,
        priority=4,
    ),
    # ── Music ──────────────────────────────────────────────────────────────
    FrameDefinition(
        name="PLAY_MUSIC",
        plugin="music",
        action="play",
        trigger_keywords=["play", "start music", "queue", "put on"],
        trigger_patterns=[
            r"\b(play|start|queue|put\s+on)\b.{0,30}\b(song|music|track|album|playlist|by)\b",
            r"\bplay\b.{0,20}\bby\b",
            r"\blisten\s+to\b",
        ],
        anti_patterns=[r"\b(pause|stop|skip|next|volume)\b"],
        required_slots=["song"],
        optional_slots=["artist", "album", "playlist"],
        base_confidence=0.70,
        priority=3,
    ),
    FrameDefinition(
        name="PAUSE_MUSIC",
        plugin="music",
        action="pause",
        trigger_keywords=["pause", "stop music", "halt"],
        trigger_patterns=[
            r"\b(pause|stop)\b.{0,15}\b(music|song|track|playing|it)\b",
            r"\b(pause|stop)\b\s*$",
        ],
        anti_patterns=[r"\b(play|skip|next|volume)\b"],
        required_slots=[],
        optional_slots=[],
        base_confidence=0.75,
        priority=2,
    ),
    FrameDefinition(
        name="SKIP_MUSIC",
        plugin="music",
        action="skip",
        trigger_keywords=["skip", "next song", "next track", "forward"],
        trigger_patterns=[
            r"\b(skip|next|forward)\b.{0,15}\b(song|track|music)?\b",
        ],
        anti_patterns=[],
        required_slots=[],
        optional_slots=[],
        base_confidence=0.73,
        priority=2,
    ),
    # ── Calendar ───────────────────────────────────────────────────────────
    FrameDefinition(
        name="CREATE_EVENT",
        plugin="calendar",
        action="create_event",
        trigger_keywords=[
            "schedule",
            "add event",
            "create event",
            "set reminder",
            "book",
        ],
        trigger_patterns=[
            r"\b(schedule|add|create|set|book)\b.{0,20}\b(event|meeting|appointment|reminder)\b",
            r"\b(meeting|appointment)\b.{0,20}\b(at|on|for)\b",
        ],
        anti_patterns=[r"\b(cancel|delete|remove|list|what|show)\b"],
        required_slots=["event_title"],
        optional_slots=["date", "time", "location"],
        base_confidence=0.68,
        priority=4,
    ),
    FrameDefinition(
        name="LIST_EVENTS",
        plugin="calendar",
        action="list_events",
        trigger_keywords=["calendar", "schedule", "what's on", "upcoming events"],
        trigger_patterns=[
            r"\b(what(?:'s|\s+is)\s+on\s+my\s+calendar)\b",
            r"\b(upcoming|scheduled)\s+(events?|meetings?)\b",
            r"\b(show|list)\s+(my\s+)?(calendar|events?|appointments?)\b",
        ],
        anti_patterns=[r"\b(create|add|schedule|book)\b"],
        required_slots=[],
        optional_slots=["date_range"],
        base_confidence=0.65,
        priority=3,
    ),
    # ── Email ──────────────────────────────────────────────────────────────
    FrameDefinition(
        name="COMPOSE_EMAIL",
        plugin="email",
        action="compose",
        trigger_keywords=["send email", "write email", "compose", "email to"],
        trigger_patterns=[
            r"\b(send|write|compose|draft)\b.{0,20}\b(email|mail|message)\b",
            r"\bemail\b.{0,15}\bto\b",
        ],
        anti_patterns=[r"\b(read|check|find|search|open)\b"],
        required_slots=["recipient"],
        optional_slots=["subject", "body"],
        base_confidence=0.68,
        priority=4,
    ),
    FrameDefinition(
        name="READ_EMAIL",
        plugin="email",
        action="read_emails",
        trigger_keywords=["read email", "check email", "inbox", "new emails"],
        trigger_patterns=[
            r"\b(read|check|open|show)\b.{0,15}\b(email|mail|inbox|messages?)\b",
            r"\bany\s+(new\s+)?(emails?|messages?)\b",
        ],
        anti_patterns=[r"\b(send|write|compose|reply)\b"],
        required_slots=[],
        optional_slots=[],
        base_confidence=0.65,
        priority=3,
    ),
    # ── System / conversation ──────────────────────────────────────────────
    FrameDefinition(
        name="HELP",
        plugin="conversation",
        action="help",
        trigger_keywords=["help", "what can you do", "commands", "capabilities"],
        trigger_patterns=[
            r"\b(help|assist|support)\b",
            r"\bwhat\s+can\s+you\s+do\b",
            r"\bhow\s+do\s+i\b",
        ],
        anti_patterns=[],
        required_slots=[],
        optional_slots=[],
        base_confidence=0.70,
        priority=6,
    ),
    FrameDefinition(
        name="GENERAL_CHAT",
        plugin="conversation",
        action="general",
        trigger_keywords=["hi", "hello", "hey", "thanks", "goodbye", "bye"],
        trigger_patterns=[
            r"^(hi|hello|hey|sup|yo|howdy)\b",
            r"\b(thank you|thanks|cheers|bye|goodbye|see you)\b",
        ],
        anti_patterns=[],
        required_slots=[],
        optional_slots=[],
        base_confidence=0.50,
        priority=8,
    ),
]

# Index frames by name for fast lookup
_FRAME_INDEX: Dict[str, FrameDefinition] = {f.name: f for f in _FRAMES}


# ---------------------------------------------------------------------------
# Inline slot evidence extractor (lightweight, fast)
# ---------------------------------------------------------------------------

_TITLE_PATTERNS = [
    re.compile(
        r'(?:called|titled?|named?|about)\s+["\']?(.+?)["\']?(?:\s+(?:tagged|with|and|$)|$)',
        re.IGNORECASE,
    ),
    re.compile(r'"([^"]{2,60})"'),
    re.compile(r"'([^']{2,60})'"),
]
_QUERY_PATTERNS = [
    re.compile(
        r'(?:about|for|related\s+to|containing|with)\s+["\']?(.+?)["\']?(?:\s+(?:tagged|in|$)|$)',
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:find|search|look\s+for)\s+(?:notes?\s+(?:about|on)\s+)?(.+?)(?:\s+(?:tagged|$)|$)",
        re.IGNORECASE,
    ),
]
_NOTE_REF_PATTERNS = [
    re.compile(
        r'(?:note|the)\s+(?:called|titled?|named?)\s+["\']?(.+?)["\']?(?:\s|$)',
        re.IGNORECASE,
    ),
    re.compile(r"(?:note|file)\s+#?(\w[\w\s-]{1,40})", re.IGNORECASE),
]
_TAG_PATTERN = re.compile(r"#(\w+)", re.IGNORECASE)


def _extract_inline_slots(text: str, frame: FrameDefinition) -> Dict[str, Any]:
    slots: Dict[str, Any] = {}
    if "title" in frame.required_slots + frame.optional_slots:
        for p in _TITLE_PATTERNS:
            m = p.search(text)
            if m:
                slots["title"] = m.group(1).strip()
                break
    if "query" in frame.required_slots + frame.optional_slots:
        for p in _QUERY_PATTERNS:
            m = p.search(text)
            if m:
                slots["query"] = m.group(1).strip()
                break
    if "note_ref" in frame.required_slots + frame.optional_slots:
        for p in _NOTE_REF_PATTERNS:
            m = p.search(text)
            if m:
                slots["note_ref"] = m.group(1).strip()
                break
    tags = _TAG_PATTERN.findall(text)
    if tags and "tags" in frame.optional_slots:
        slots["tags"] = tags
    return slots


# ---------------------------------------------------------------------------
# FrameParser
# ---------------------------------------------------------------------------


class FrameParser:
    """
    Match an utterance against the frame registry and return the best
    FrameMatchResult.

    Scoring
    -------
    score = base_confidence
          + sum(+0.15 per matched keyword, capped at +0.45)
          + sum(+0.20 per matched pattern, capped at +0.40)
          - sum(+0.20 per matched anti-pattern, capped at -0.40)
          - 0.10 per missing required slot
          ∈ [0.0, 0.97]
    """

    def __init__(self):
        self._frames = _FRAMES
        self._compiled: Dict[str, Tuple[List[re.Pattern], List[re.Pattern]]] = {}
        for frame in self._frames:
            pos = [re.compile(p, re.IGNORECASE) for p in frame.trigger_patterns]
            neg = [re.compile(p, re.IGNORECASE) for p in frame.anti_patterns]
            self._compiled[frame.name] = (pos, neg)

    def parse(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        top_n: int = 3,
    ) -> Optional[FrameMatchResult]:
        """
        Match *text* against all frames. Returns the best match or None
        if no frame scores above 0.35.

        Parameters
        ----------
        text:    raw / already-cleaned user utterance
        context: optional dict with keys like 'last_plugin', 'last_action'
                 used for context-boosting
        top_n:   how many candidates to evaluate (internal use)
        """
        if not text:
            return None

        text_norm = text.lower().strip()
        candidates: List[FrameMatchResult] = []

        for frame in self._frames:
            result = self._score_frame(text_norm, text, frame, context or {})
            if result.confidence > 0.30:
                candidates.append(result)

        if not candidates:
            return None

        # Sort: confidence desc, then priority asc (lower = more specific)
        candidates.sort(
            key=lambda r: (-r.confidence, _FRAME_INDEX[r.frame_name].priority)
        )

        best = candidates[0]
        logger.debug(
            "[FRAME] Best=%s (%.2f) | top candidates: %s",
            best.frame_name,
            best.confidence,
            [(c.frame_name, round(c.confidence, 2)) for c in candidates[:3]],
        )
        return best

    def parse_ranked(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[FrameMatchResult]:
        """Return all matches sorted by confidence descending."""
        if not text:
            return []
        text_norm = text.lower().strip()
        results = [
            self._score_frame(text_norm, text, frame, context or {})
            for frame in self._frames
        ]
        results = [r for r in results if r.confidence > 0.25]
        results.sort(key=lambda r: (-r.confidence, _FRAME_INDEX[r.frame_name].priority))
        return results

    # ------------------------------------------------------------------
    # Internal scoring
    # ------------------------------------------------------------------

    def _score_frame(
        self,
        text_lower: str,
        text_original: str,
        frame: FrameDefinition,
        context: Dict[str, Any],
    ) -> FrameMatchResult:
        score = frame.base_confidence
        matched_keywords: List[str] = []
        matched_patterns: List[str] = []

        # Keyword hits
        keyword_bonus = 0.0
        for kw in frame.trigger_keywords:
            if kw in text_lower:
                keyword_bonus += 0.15
                matched_keywords.append(kw)
        score += min(0.45, keyword_bonus)

        # Pattern hits
        pos_patterns, neg_patterns = self._compiled[frame.name]
        pattern_bonus = 0.0
        for pat in pos_patterns:
            if pat.search(text_lower):
                pattern_bonus += 0.20
                matched_patterns.append(pat.pattern)
        score += min(0.40, pattern_bonus)

        # Anti-pattern penalty
        anti_penalty = 0.0
        for pat in neg_patterns:
            if pat.search(text_lower):
                anti_penalty += 0.20
        score -= min(0.40, anti_penalty)

        # Context boost: same plugin as last action
        if context.get("last_plugin") == frame.plugin and matched_keywords:
            score += 0.05

        # Slot evidence extraction
        slot_evidence = _extract_inline_slots(text_original, frame)

        # Required slot penalty if slots missing
        missing_required: List[str] = []
        for rs in frame.required_slots:
            if rs not in slot_evidence:
                missing_required.append(rs)
                score -= 0.08  # penalty but not a killer — slot_filler may fill it

        score = max(0.0, min(0.97, score))

        return FrameMatchResult(
            frame_name=frame.name,
            plugin=frame.plugin,
            action=frame.action,
            confidence=score,
            matched_keywords=matched_keywords,
            matched_patterns=matched_patterns,
            slot_evidence=slot_evidence,
            missing_required_slots=missing_required,
            raw_text=text_original,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def get_frame(name: str) -> Optional[FrameDefinition]:
        return _FRAME_INDEX.get(name)

    @staticmethod
    def frame_names() -> List[str]:
        return [f.name for f in _FRAMES]
