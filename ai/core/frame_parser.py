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

import json
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Negation detection helpers  (#3 — negation-aware anti-patterns)
# ---------------------------------------------------------------------------

_NEGATION_WORDS: frozenset = frozenset(
    {
        "not",
        "no",
        "never",
        "don't",
        "dont",
        "won't",
        "wont",
        "can't",
        "cant",
        "isn't",
        "isnt",
        "aren't",
        "arent",
        "didn't",
        "didnt",
        "wouldn't",
        "wouldnt",
        "shouldn't",
        "shouldnt",
        "do not",
        "does not",
        "please don't",
        "please dont",
    }
)
_NEG_WINDOW = 32  # characters to look back from a keyword for a negation word
_CREATE_NOTE_WEAK_VERB_RE = re.compile(r"\b(make|add|create|write)\b", re.IGNORECASE)
_CREATE_NOTE_EXPLICIT_EVIDENCE_PATTERNS: Tuple[re.Pattern, ...] = (
    re.compile(r"\bnotes?\b", re.IGNORECASE),
    re.compile(r"\bmemo\b", re.IGNORECASE),
    re.compile(r"\bjot\s+down\b", re.IGNORECASE),
    re.compile(r"\bwrite\s+this\s+down\b", re.IGNORECASE),
    re.compile(r"\bwrite\b.{0,40}\bdown\b", re.IGNORECASE),
    re.compile(r"\bsave\s+this\b", re.IGNORECASE),
    re.compile(r"\bsave\b.{0,40}\bto\s+my\s+notes?\b", re.IGNORECASE),
    re.compile(r"\bremember\s+this\b", re.IGNORECASE),
    re.compile(r"\btake\s+note\b", re.IGNORECASE),
    re.compile(r"\badd\s+this\s+to\s+my\s+notes\b", re.IGNORECASE),
)


def _has_negation_before(
    text_lower: str, keyword: str, window: int = _NEG_WINDOW
) -> bool:
    """Return True if a negation word appears in the *window* chars before *keyword*."""
    idx = text_lower.find(keyword)
    if idx < 0:
        return False
    snippet = text_lower[max(0, idx - window) : idx]
    return any(neg in snippet for neg in _NEGATION_WORDS)


def _has_create_note_explicit_evidence(text_lower: str) -> bool:
    return any(
        pattern.search(text_lower) for pattern in _CREATE_NOTE_EXPLICIT_EVIDENCE_PATTERNS
    )


def _keyword_matches_with_boundaries(text_lower: str, keyword: str) -> bool:
    token = str(keyword or "").strip().lower()
    if not token:
        return False
    if " " in token:
        parts = [re.escape(part) for part in token.split() if part]
        if not parts:
            return False
        pattern = r"\b" + r"\s+".join(parts) + r"\b"
    else:
        pattern = r"\b" + re.escape(token) + r"\b"
    return bool(re.search(pattern, text_lower))


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
    allow_zero_evidence: bool = False


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
        action="create",
        trigger_keywords=[
            "create",
            "write",
            "make",
            "add",
            "new",
            "jot",
            "jot down",
            "record",
            "draft",
            "save",
            "save this",
            "remember this",
            "write this down",
            "add this to my notes",
            "compose",
            "note down",
            "take note",
        ],
        trigger_patterns=[
            r"\b(create|write|make|add|new|save|jot(?:\s+down)?|record|draft|compose)\b.{0,25}\bnote\b",
            r"\bnote\b.{0,20}\b(about|on|regarding|for|titled?)\b",
            r"\b(take|jot).{0,10}\bnote\b",
            r"\b(remind(?:er)?|memo)\b.{0,20}\babout\b",
            r"\bwrite\s+this\s+down\b",
            r"\bsave\s+this(?:\s+to\s+my\s+notes?)?\b",
            r"\bremember\s+this\b",
            r"\badd\s+this\s+to\s+my\s+notes?\b",
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
        action="read",
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
        action="search",
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
        ],
        anti_patterns=[
            r"\b(create|write|make|add|open|read|show content|delete|remove|do i have|how many)\b",
        ],
        required_slots=["query"],
        optional_slots=["tags", "date_range"],
        base_confidence=0.65,
        priority=3,
    ),
    FrameDefinition(
        name="LIST_NOTES",
        plugin="notes",
        action="list",
        trigger_keywords=[
            "list notes",
            "list all",
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
            r"\bwhat\b.{0,10}\bnotes?\b",
            r"\bnotes?\b.{0,15}\b(i\s+have|do\s+i\s+have)\b",
        ],
        anti_patterns=[
            r"\b(find|search|look\s+for|about|tagged|in it|inside|in the|how many|save this|write this down|remember this)\b",
        ],
        required_slots=[],
        optional_slots=["tags", "date_range"],
        base_confidence=0.68,
        priority=2,
    ),
    FrameDefinition(
        name="UPDATE_NOTE",
        plugin="notes",
        action="update",
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
        action="delete",
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
        action="archive",
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
        allow_zero_evidence=True,
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
          + sum(learned_weight per matched keyword, capped at +0.45)   [#2]
          - sum(0.10 per negated keyword, capped at -0.20)             [#3]
          + sum(+0.20 per matched pattern, capped at +0.40)
          - sum(+0.20 per matched anti-pattern, capped at -0.40)
          - 0.10 per missing required slot
          ∈ [0.0, 0.97]

    Extensions
    ----------
    * learned keyword weights (#2) — per-frame per-keyword floats stored in
      data/frame_keyword_weights.json, updated via record_outcome().
    * negation-aware scoring (#3) — a keyword preceded by a negation word
      yields a -0.10 deduction instead of its normal bonus.
    * compound detection (#4) — parse_compound() splits "X and then Y"
      utterances and returns a list of FrameMatchResults.
    * YAML registry (#5) — extra frames loaded from data/frames/*.yaml at
      startup; they overlay / extend the built-in _FRAMES list.
    """

    _WEIGHTS_PATH = "data/frame_keyword_weights.json"
    _YAML_FRAMES_DIR = "data/frames"

    def __init__(self):
        self._frames: List[FrameDefinition] = list(_FRAMES)
        self._compiled: Dict[str, Tuple[List[re.Pattern], List[re.Pattern]]] = {}
        for frame in self._frames:
            pos = [re.compile(p, re.IGNORECASE) for p in frame.trigger_patterns]
            neg = [re.compile(p, re.IGNORECASE) for p in frame.anti_patterns]
            self._compiled[frame.name] = (pos, neg)

        # (#2) Learned keyword weights: {frame_name: {keyword: weight}}
        self._keyword_weights: Dict[str, Dict[str, float]] = {}
        self._load_keyword_weights()

        # (#5) YAML-driven extra frames
        self._load_yaml_frames()

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

        # (#2 + #3) Keyword hits with learned weights and negation awareness
        frame_weights = self._keyword_weights.get(frame.name, {})
        keyword_bonus = 0.0
        negation_deduction = 0.0
        for kw in frame.trigger_keywords:
            if _keyword_matches_with_boundaries(text_lower, kw):
                if _has_negation_before(text_lower, kw):
                    # Negated keyword → small deduction instead of bonus
                    negation_deduction += 0.10
                else:
                    weight = frame_weights.get(kw, 0.15)
                    keyword_bonus += weight
                    matched_keywords.append(kw)
        score += min(0.45, keyword_bonus)
        score -= min(0.20, negation_deduction)

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

        has_evidence = bool(matched_keywords or matched_patterns)
        if not has_evidence and not frame.allow_zero_evidence:
            score -= 0.50

        if frame.name == "CREATE_NOTE":
            has_weak_create_verb = bool(_CREATE_NOTE_WEAK_VERB_RE.search(text_lower))
            has_explicit_notes_evidence = _has_create_note_explicit_evidence(text_lower)
            if has_weak_create_verb and not has_explicit_notes_evidence:
                score -= 0.55

        # ── Context-aware boosts ─────────────────────────────────────────────
        last_plugin = context.get("last_plugin")
        last_intent = context.get("last_intent", "")
        last_domain = (
            last_intent.split(":")[0] if ":" in last_intent else last_plugin or ""
        )
        word_count = len(text_lower.split())

        if last_plugin == frame.plugin:
            if matched_keywords or matched_patterns:
                # Strong same-domain continuation boost (was +0.05, now tiered)
                score += 0.18
            elif word_count <= 5:
                # Short utterance in same domain with no explicit keywords —
                # still likely a follow-up (e.g. "and that one?", "delete it")
                score += 0.10

        # Inherited-slot credit: if prior context already filled a required slot,
        # don't penalise for it being absent in the current short utterance.
        prior_entities: Dict[str, Any] = context.get("last_entities") or {}
        if prior_entities and frame.required_slots:
            inherited: List[str] = [
                rs for rs in frame.required_slots if rs in prior_entities
            ]
            # Each inherited required slot cancels out would-be penalty (+0.08 each)
            score += len(inherited) * 0.08

        # Short follow-up bonus: utterances ≤ 5 tokens in same domain get a
        # small lift to compete with conversational catch-alls.
        if word_count <= 5 and last_domain == frame.plugin:
            score += 0.05

        # Slot evidence extraction
        slot_evidence = _extract_inline_slots(text_original, frame)
        # Merge inherited slots from context so slot consumers see full picture
        for key, val in prior_entities.items():
            if key not in slot_evidence:
                slot_evidence[key] = val

        # Required slot penalty if slots missing (after inheritance)
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

    # ------------------------------------------------------------------
    # (#2) Learned keyword weights
    # ------------------------------------------------------------------

    def _load_keyword_weights(self) -> None:
        path = Path(self._WEIGHTS_PATH)
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    self._keyword_weights = json.load(fh)
                logger.debug(
                    "[FrameParser] Loaded keyword weights for %d frames",
                    len(self._keyword_weights),
                )
            except Exception as exc:
                logger.debug("[FrameParser] Could not load keyword weights: %s", exc)

    def _save_keyword_weights(self) -> None:
        try:
            path = Path(self._WEIGHTS_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self._keyword_weights, fh, indent=2)
        except Exception as exc:
            logger.debug("[FrameParser] Could not save keyword weights: %s", exc)

    def record_outcome(
        self,
        frame_name: str,
        matched_keywords: List[str],
        was_correct: bool,
        alpha: float = 0.05,
    ) -> None:
        """
        EMA-update keyword weights for *frame_name* based on outcome.

        Correct match → nudge weights toward 0.22 (reward).
        Wrong match   → nudge weights toward 0.08 (soften).
        """
        target = 0.22 if was_correct else 0.08
        frame_w = self._keyword_weights.setdefault(frame_name, {})
        for kw in matched_keywords:
            current = frame_w.get(kw, 0.15)
            frame_w[kw] = round(current * (1.0 - alpha) + target * alpha, 4)
        if matched_keywords:
            self._save_keyword_weights()

    # ------------------------------------------------------------------
    # (#4) Multi-frame compound detection
    # ------------------------------------------------------------------

    _COMPOUND_SEP = re.compile(
        r"\s+(?:and(?:\s+(?:also|then))?|then|also|,\s*(?:and)?)\s+",
        re.IGNORECASE,
    )

    def parse_compound(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        min_confidence: float = 0.52,
    ) -> List[FrameMatchResult]:
        """
        Split a potentially compound utterance into parts, match each part,
        and return a list of FrameMatchResults.

        If only one confident frame is found (or the utterance is not compound),
        returns a single-element list from the normal parse.  Returns [] if
        nothing scores above the threshold.

        Example
        -------
        "create a note about the meeting and then send an email to bob"
        → [FrameMatchResult(CREATE_NOTE, …), FrameMatchResult(COMPOSE_EMAIL, …)]
        """
        parts = [
            p.strip() for p in self._COMPOUND_SEP.split(text) if len(p.strip()) > 7
        ]

        if len(parts) < 2:
            result = self.parse(text, context)
            return [result] if result else []

        sub_results: List[FrameMatchResult] = []
        for part in parts:
            r = self.parse(part, context)
            if r and r.confidence >= min_confidence:
                sub_results.append(r)

        if len(sub_results) >= 2:
            logger.debug(
                "[COMPOUND] Detected %d sub-frames: %s",
                len(sub_results),
                [r.frame_name for r in sub_results],
            )
            return sub_results

        # Could not reliably split — fall back to full-text parse
        full = self.parse(text, context)
        return [full] if full else []

    # ------------------------------------------------------------------
    # (#5) YAML-driven frame registry
    # ------------------------------------------------------------------

    def _load_yaml_frames(self) -> None:
        """
        Load extra / override FrameDefinitions from data/frames/*.yaml.

        Each YAML file must be a list of frame dicts with keys matching
        the FrameDefinition fields.  Frames whose *name* already exists in
        the built-in registry are replaced; new names are appended.

        Example YAML entry
        ------------------
        - name: CUSTOM_FRAME
          plugin: my_plugin
          action: do_thing
          trigger_keywords: [thing, do, execute]
          trigger_patterns: ["\\bdo\\s+thing\\b"]
          anti_patterns: []
          required_slots: []
          optional_slots: [target]
          base_confidence: 0.60
          priority: 5
        """
        frames_dir = Path(self._YAML_FRAMES_DIR)
        if not frames_dir.exists():
            return
        try:
            import yaml  # type: ignore[import]
        except ImportError:
            logger.debug(
                "[FrameParser] PyYAML not installed; YAML frame loading skipped"
            )
            return

        for yaml_path in sorted(frames_dir.glob("*.yaml")):
            try:
                with open(yaml_path, "r", encoding="utf-8") as fh:
                    entries = yaml.safe_load(fh) or []
                for entry in entries:
                    frame = FrameDefinition(
                        name=entry["name"],
                        plugin=entry["plugin"],
                        action=entry["action"],
                        trigger_keywords=entry.get("trigger_keywords", []),
                        trigger_patterns=entry.get("trigger_patterns", []),
                        anti_patterns=entry.get("anti_patterns", []),
                        required_slots=entry.get("required_slots", []),
                        optional_slots=entry.get("optional_slots", []),
                        base_confidence=float(entry.get("base_confidence", 0.55)),
                        priority=int(entry.get("priority", 5)),
                    )
                    # Replace existing frame or append new one
                    self._frames = [f for f in self._frames if f.name != frame.name]
                    self._frames.append(frame)
                    # Compile patterns
                    pos = [re.compile(p, re.IGNORECASE) for p in frame.trigger_patterns]
                    neg = [re.compile(p, re.IGNORECASE) for p in frame.anti_patterns]
                    self._compiled[frame.name] = (pos, neg)
                    # Keep global index current so get_frame() works
                    _FRAME_INDEX[frame.name] = frame
                    logger.info(
                        "[FrameParser] YAML frame loaded: %s from %s",
                        frame.name,
                        yaml_path.name,
                    )
            except Exception as exc:
                logger.warning("[FrameParser] Failed to load %s: %s", yaml_path, exc)
