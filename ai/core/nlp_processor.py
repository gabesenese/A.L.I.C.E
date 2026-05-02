"""
A.L.I.C.E. Advanced NLP Processor
Natural Language Understanding

Advanced approach: Semantic understanding over regex patterns.

Features:
- Semantic intent classification (no more manual patterns)
- Advanced slot filling with structured data extraction
- Temporal expression normalization (tomorrow -> actual dates)
- Custom NER for domain entities (tags, priorities, categories)
- Coreference resolution (track "it", "this", "that")
- Multi-label emotion & urgency detection
- Performance optimization with caching
- Context-aware conversation tracking
"""

import re
import logging
import importlib
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
from collections import OrderedDict, defaultdict, deque, Counter
from pathlib import Path
import threading

# Core NLP libraries
try:
    nltk_mod = importlib.import_module("nltk")
    sentiment_mod = importlib.import_module("nltk.sentiment")
    word_tokenize = nltk_mod.word_tokenize
    SentimentIntensityAnalyzer = sentiment_mod.SentimentIntensityAnalyzer
except ImportError:  # pragma: no cover

    def word_tokenize(text: str):
        return text.split()

    SentimentIntensityAnalyzer = None
    logging.warning(
        "[WARN] NLTK not available. Using basic tokenization and neutral sentiment."
    )

try:
    sklearn_text = importlib.import_module("sklearn.feature_extraction.text")
    sklearn_pairwise = importlib.import_module("sklearn.metrics.pairwise")
    TfidfVectorizer = sklearn_text.TfidfVectorizer
    cosine_similarity = sklearn_pairwise.cosine_similarity
except ImportError:  # pragma: no cover
    TfidfVectorizer = None
    cosine_similarity = None

# Advanced temporal parsing
try:
    dateparser = importlib.import_module("dateparser")
except ImportError:  # pragma: no cover
    dateparser = None
    logging.warning(
        "[WARN] dateparser not available. Temporal parsing will be limited."
    )

try:
    parsedatetime_mod = importlib.import_module("parsedatetime")
    Calendar = parsedatetime_mod.Calendar
except ImportError:  # pragma: no cover
    Calendar = None
    logging.warning(
        "[WARN] parsedatetime not available. Temporal parsing will be limited."
    )

# Semantic intent classification
try:
    from ai.core.intent_classifier import get_intent_classifier

    SEMANTIC_CLASSIFIER_AVAILABLE = True
except ImportError:
    SEMANTIC_CLASSIFIER_AVAILABLE = False
    logging.warning(
        "[WARN] Semantic intent classifier not available. Using fallback patterns."
    )

try:
    from ai.core.llm_intent_classifier import get_llm_intent_classifier

    LLM_INTENT_CLASSIFIER_AVAILABLE = True
except ImportError:
    LLM_INTENT_CLASSIFIER_AVAILABLE = False
    logging.warning(
        "[WARN] LLM intent classifier not available. Semantic-only routing will be used."
    )

# Entity normalizer (P0 Improvement)
try:
    from ai.core.entity_normalizer import get_normalizer as _get_entity_normalizer

    ENTITY_NORMALIZER_AVAILABLE = True
except ImportError:
    ENTITY_NORMALIZER_AVAILABLE = False
    logging.warning("[WARN] Entity normalizer not available. Using raw entities.")

# Feature flags for A/B testing
try:
    from ai.core.feature_flags import get_feature_flags as _get_feature_flags

    FEATURE_FLAGS_AVAILABLE = True
except ImportError:
    FEATURE_FLAGS_AVAILABLE = False
    logging.warning("[WARN] Feature flags not available. All features enabled.")

# Advanced NLP stack — Frame Parser, Probabilistic Slot Filler, Advanced Coreference
try:
    from ai.core.frame_parser import (
        FrameParser as _FrameParser,
        FrameMatchResult as _FrameMatchResult,
    )
    from ai.core.prob_slot_filler import (
        ProbabilisticSlotFiller as _ProbSlotFiller,
        FilledSlots as _FilledSlots,
    )
    from ai.core.coreference import (
        LegacyCoreferenceResolverCompat as _AdvancedCoref,
        DialogueMemory as _DialogueMemory,
    )
    from ai.core.utterance_fingerprint import (
        get_fingerprint_store as _get_fingerprint_store,
    )

    ADVANCED_NLP_AVAILABLE = True
except ImportError:
    ADVANCED_NLP_AVAILABLE = False
    _FrameParser = None
    _ProbSlotFiller = None
    _AdvancedCoref = None
    _DialogueMemory = None
    _get_fingerprint_store = None
    logging.warning("[WARN] Advanced NLP modules not available. Using legacy routing.")

try:
    from ai.core.conversation_memory import ConversationMemory
    from ai.core.implicit_intent_detector import ImplicitIntentDetector
    from ai.core.dialogue_state_machine import DialogueStateMachine
    INTELLIGENCE_FOUNDATIONS_AVAILABLE = True
except ImportError:
    INTELLIGENCE_FOUNDATIONS_AVAILABLE = False
    ConversationMemory = None
    ImplicitIntentDetector = None
    DialogueStateMachine = None
    logging.warning("[WARN] Tier foundation modules unavailable. Running base NLP stack.")

from ai.core.perception import Perception, PerceptionResult
from ai.core.followup_resolver import FollowUpResolver, FollowUpResult
from ai.core.route_coordinator import RouteCoordinator, RouteCoordinatorConfig
from ai.core.goal_recognizer import get_goal_recognizer
from ai.core.foundation_layers import FoundationLayers
from ai.plugins.registry import PluginRegistry, discover_plugins

logger = logging.getLogger(__name__)

# ============================================================================
# MODULE-LEVEL PATTERN CONSTANTS  (avoids per-call list allocation)
# ============================================================================

# Negation detection (shared by EmotionDetector and _detect_intent_semantic)
_NEGATION_WORDS: frozenset = frozenset({
    "not", "no", "never", "don't", "dont", "won't", "wont",
    "can't", "cant", "stop", "isn't", "isnt", "aren't", "arent",
    "didn't", "didnt", "wouldn't", "wouldnt", "shouldn't", "shouldnt",
})

_TOKEN_PATTERN = re.compile(r'"[^"]+"|#\w+|\d+(?:st|nd|rd|th)?|[A-Za-z]+|[^\w\s]')
_ORDINAL_TOKEN_RE = re.compile(r"\d+(?:st|nd|rd|th)")
_ALPHA_TOKEN_RE = re.compile(r"[A-Za-z]+")

_NOTE_TERMS: frozenset = frozenset({"note", "notes", "list", "lists", "todo", "task", "tasks"})
_EMAIL_TERMS: frozenset = frozenset({"email", "emails", "mail", "inbox", "sender", "subject"})
_CALENDAR_TERMS: frozenset = frozenset({"calendar", "event", "events", "meeting", "schedule"})
_WEATHER_TERMS: frozenset = frozenset({
    "weather", "forecast", "temperature", "rain", "snow", "sunny", "cloudy", "overcast",
    "wind", "windy", "storm", "humidity", "humid", "outside", "precipitation",
    "precip", "drizzle", "thunder", "thunderstorm",
})
_WEATHER_FORECAST_TERMS: frozenset = frozenset({
    "tomorrow", "tonight", "week", "weekend", "forecast", "monday", "tuesday",
    "wednesday", "thursday", "friday", "saturday", "sunday", "chance", "expect", "later",
})
_WEATHER_EVENT_TERMS: frozenset = frozenset({"snow", "rain", "storm", "drizzle", "thunder"})
_WEATHER_FUTURE_TERMS: frozenset = frozenset({"will", "gonna", "going", "chance", "expect", "is"})
_SYSTEM_TERMS: frozenset = frozenset({"system", "cpu", "memory", "disk", "battery", "status"})
_NON_WEATHER_TARGET_TERMS: frozenset = frozenset({
    "note", "notes", "memo", "email", "emails", "mail", "calendar", "event", "events", "meeting",
})
_WAKE_WORD_PREFIX_RE = re.compile(
    r"^\s*(?:hey|ok|okay)?\s*(?:assistant|alice)\b[\s,:\-]*",
    re.IGNORECASE,
)


def _has_negation_before(text_lower: str, keyword: str, window_chars: int = 40) -> bool:
    """Return True if a negation word appears within *window_chars* before *keyword*."""
    idx = text_lower.find(keyword)
    if idx == -1:
        return False
    preceding = text_lower[max(0, idx - window_chars):idx]
    return bool(_NEGATION_WORDS & set(preceding.split()))


def _is_negated_command(text_lower: str) -> bool:
    """Return True if any of the first 4 tokens is a negation word."""
    return bool(_NEGATION_WORDS & set(text_lower.split()[:4]))


# ── PHASE 1 pattern sets ──────────────────────────────────────────────────────
# System
_P1_SYSTEM_STATUS: frozenset = frozenset({"status", "doing", "health", "how"})
_P1_SYSTEM_RESOURCES: frozenset = frozenset({"cpu", "memory", "disk", "battery", "gpu"})
_P1_SYSTEM_RESOURCE_VERBS: frozenset = frozenset({"usage", "available", "how much", "low", "check", "is"})

# File operations
_P1_FILE_CREATE: frozenset = frozenset({"create", "make", "new"})
_P1_FILE_READ: frozenset = frozenset({"read", "open", "show", "display", "view"})
_P1_FILE_DELETE: frozenset = frozenset({"delete", "remove", "trash"})
_P1_FILE_MOVE: frozenset = frozenset({"move", "rename", "relocate"})
_P1_FILE_LIST: frozenset = frozenset({"list", "show"})
_P1_FILE_MARKERS: frozenset = frozenset({
    "file", "document", ".txt", ".pdf", ".csv", ".json", ".yaml", ".md",
    "folder", "directory",
})

# Memory
_P1_MEMORY_STORE: frozenset = frozenset({"remember", "keep in mind", "save this"})
_P1_MEMORY_STORE_OBJ: frozenset = frozenset({"that", "this", "i", "my", "prefer"})
_P1_MEMORY_RECALL: tuple = (
    "what do you remember",
    "do you remember",
    "what do you know",
    "can you remember",
    "could you remember",
    "would you remember",
)
_P1_MEMORY_SEARCH_SUBJ: tuple = ("what did we", "what have we", "what did i")
_P1_MEMORY_SEARCH_VERB: frozenset = frozenset({"talk", "discuss", "say", "tell"})

# Email
_P1_EMAIL_COMPOSE_VERBS: frozenset = frozenset({"compose", "draft", "write", "send"})
_P1_EMAIL_NOUNS: frozenset = frozenset({"email", "mail", "message", "to"})
_P1_EMAIL_DELETE_VERBS: frozenset = frozenset({"delete", "remove", "trash"})
_P1_EMAIL_DELETE_NOUNS: frozenset = frozenset({"email", "mail", "message"})
_P1_EMAIL_REPLY_VERBS: frozenset = frozenset({"reply", "respond"})
_P1_EMAIL_SEARCH_VERBS: frozenset = frozenset({"search", "find", "look for"})
_P1_EMAIL_SEARCH_NOUNS: frozenset = frozenset({"email", "mail", "inbox", "message", "from"})
_P1_EMAIL_LIST_VERBS: frozenset = frozenset({"show", "list", "recent", "latest"})
_P1_EMAIL_LIST_NOUNS: frozenset = frozenset({"email", "emails", "mail", "mails", "inbox"})
_P1_EMAIL_READ_VERBS: frozenset = frozenset({"read", "open", "display", "view"})
_P1_EMAIL_READ_NOUNS: frozenset = frozenset({"email", "emails", "mail", "message"})

# Notes
_P1_NOTES_APPEND_VERBS: frozenset = frozenset({"add", "put", "append", "include"})
# Regex for append: "add X to [my/the] [any words] [note/list/notes]"
# Uses regex at detection time — see _detect_intent_phase1
_P1_NOTES_CREATE_VERBS: frozenset = frozenset({"create", "new", "make", "write"})
_P1_NOTES_KEYWORDS: frozenset = frozenset({"note", "notes", "memo"})
_P1_NOTES_LIST_VERBS: frozenset = frozenset({"show", "list", "display", "see"})
_P1_NOTES_LIST_NOUNS: frozenset = frozenset({"note", "notes", "all notes"})
_P1_NOTES_SEARCH_VERBS: frozenset = frozenset({"find", "search"})
_P1_NOTES_DELETE_VERBS: frozenset = frozenset({"delete", "remove"})
# Read content patterns: "what is in the X [note]", "what's inside X", "what is in it?"
_P1_NOTES_READ_CONTENT_RE = re.compile(
    r"what(?:'s|\s+is)\s+in\s+(?:the\s+|my\s+)?(?P<name>[a-z][\w\s]+?)\s*(?:note|list|notes)?[?!.]*\s*$"
    r"|what(?:'s|\s+is)\s+inside\s+(?:the\s+|my\s+)?(?P<name2>[a-z][\w\s]+?)\s*(?:note|list|notes)?[?!.]*\s*$"
    r"|(?:show|read|open)\s+(?:me\s+)?(?:the\s+)?(?P<name3>[a-z][\w\s]+?)\s+(?:note|list)\s+content"
    r"|what(?:'s|\s+is)\s+in\s+(?:it|this|that)\b[?!.]*\s*$",
    re.IGNORECASE,
)
# Regex for append: matches "add/put X to [my/the/a] [any words] [note/list]"
_P1_NOTES_APPEND_RE = re.compile(
    r"\b(?:add|put|append|include)\b.+?\bto\s+(?:my|the|a|an)?\s*\w+(?:\s+\w+)*?\s+(?:note|list|notes)\b",
    re.IGNORECASE,
)

# Reminders
_P1_REMINDER_SET: tuple = (
    "remind me", "set a reminder", "add a reminder", "create a reminder",
    "alert me", "notify me when", "don't let me forget",
)
_P1_REMINDER_LIST: tuple = (
    "my reminders", "what reminders", "show reminders", "list reminders",
    "any reminders", "upcoming reminders", "pending reminders",
)
_P1_REMINDER_CANCEL_VERBS: frozenset = frozenset({"cancel", "delete", "remove"})

# Short conversational acknowledgments — must be caught before semantic classifier
# so phrases like "will do", "got it" can't be mis-routed to music:play
_P1_CONV_ACK: frozenset = frozenset({
    "will do", "got it", "noted", "understood", "sure", "sure thing",
    "sounds good", "sounds great", "alright", "okay", "ok", "okie",
    "roger", "roger that", "copy that", "aye", "yep", "yup", "yeah",
    "sure will", "on it", "done", "all good", "no problem", "np",
    "perfect", "great", "awesome", "nice",
    # affirmative agreement phrases
    "good idea", "great idea", "nice idea", "good point", "fair point",
    "fair enough", "makes sense", "that makes sense", "good call", "nice one",
    "true", "exactly", "absolutely", "definitely", "of course",
    "i see", "i know", "i know right", "right", "for sure",
    "you're welcome", "youre welcome", "no worries", "anytime",
})

# Greetings / thanks / status
_P1_THANKS: tuple = ("thanks", "thank you", "thx", "thank", "thanks for")
_P1_STATUS_INQUIRY: tuple = (
    "how are you", "how are you doing", "how is it going", "how have you been"
)
_P1_GREETING_WORDS: frozenset = frozenset({"hi", "hey", "hello", "yo", "sup", "hiya"})

# Vague / clarification
_P1_VAGUE_PRONOUNS: tuple = (
    "who is he", "who is she", "who is that", "what is that", "who is that person"
)
_P1_VAGUE_TEMPORAL: frozenset = frozenset({
    "yesterday", "tomorrow", "last week", "last month", "next week", "next month"
})
_P1_VAGUE_REQUESTS: tuple = (
    "add this to", "put this in", "do that thing",
    "can you do that", "that thing", "this thing",
)

# Weather
_P1_WEATHER_KEYWORDS: frozenset = frozenset({
    "weather", "forecast", "temperature", "rain", "snow",
    "sunny", "cloudy", "overcast", "humid", "cold", "hot",
    "wind", "windy", "storm",
})
_P1_FORECAST_WORDS: frozenset = frozenset({
    "tomorrow", "tonight", "weekend", "next week", "forecast",
    "this week", "7 day", "7-day",
})

# PHASE 3 fallback
_P3_FORECAST_PHRASES: tuple = (
    "forecast", "this week", "next week", "weekend", "tomorrow",
    "7 day", "7-day", "next few days", "next 7 days",
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
)
_P3_WEATHER_WORDS: frozenset = frozenset({"weather", "temperature", "outside"})
_P3_VAGUE_PATTERNS: tuple = (
    "who is he", "who is she", "who is that",
    "what is that", "what about that",
    "add this to", "put this in",
    "what happened", "what about",
    "how do i", "how can i",
    "tell me about the", "tell me about a",
)


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class Entity:
    """Represents an extracted entity with metadata"""

    type: str
    value: str
    confidence: float
    start_pos: int = -1
    end_pos: int = -1
    normalized_value: Any = None


@dataclass
class Slot:
    """Represents a filled slot in structured extraction"""

    name: str
    value: Any
    confidence: float
    raw_text: str = ""


@dataclass
class ProcessedQuery:
    """Complete NLP processing result"""

    original_text: str
    clean_text: str
    tokens: List[str]
    intent: str
    intent_confidence: float
    entities: Dict[str, List[Entity]]
    slots: Dict[str, Slot]
    sentiment: Dict[str, float]
    emotions: List[str]
    urgency_level: str
    is_question: bool
    keywords: List[str]
    context_entities: Dict[str, str] = field(default_factory=dict)
    parsed_command: Dict[str, Any] = field(default_factory=dict)
    plugin_scores: Dict[str, float] = field(default_factory=dict)
    token_debug: List[Dict[str, Any]] = field(default_factory=list)
    intent_candidates: List[Dict[str, Any]] = field(default_factory=list)
    intent_plausibility: float = 1.0
    plausibility_issues: List[str] = field(default_factory=list)
    validation_score: float = 1.0  # Intent-entity cross-validation (0.0-1.0)
    validation_issues: List[str] = field(default_factory=list)  # Detected mismatches


@dataclass
class ConversationContext:
    """Tracks conversation state for coreference resolution"""

    last_intent: Optional[str] = None
    last_entities: Dict[str, Any] = field(default_factory=dict)
    mentioned_notes: deque = field(default_factory=lambda: deque(maxlen=5))
    mentioned_events: deque = field(default_factory=lambda: deque(maxlen=5))
    mentioned_songs: deque = field(default_factory=lambda: deque(maxlen=3))
    query_history: deque = field(default_factory=lambda: deque(maxlen=10))
    dialogue_state: str = "idle"
    last_plugin: Optional[str] = None
    pending_clarification: Dict[str, Any] = field(default_factory=dict)
    turn_index: int = 0
    # Semantic frame of the most-recently completed turn.
    # Stored as a plain dict (a copy of parsed_command.modifiers["frame"])
    # so it survives across turns without carrying live object references.
    last_frame: Optional[Dict[str, Any]] = None


@dataclass
class TokenSegment:
    """Surface-level segment prior to lexical tokenization."""

    text: str
    kind: str  # utterance | meta_command | plugin_phrase
    start_pos: int
    end_pos: int


@dataclass
class RichToken:
    """Token with lexical, semantic, and positional metadata."""

    text: str
    normalized: str
    kind: str  # word | number | ordinal | symbol | hashtag | quoted_span | pronoun | date_like | command
    role: str  # action | object | modifier | meta | reference | value | unknown
    start_pos: int
    end_pos: int
    flags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedCommand:
    """Structured interpretation produced from rich tokens."""

    action: str = "unknown"
    object_type: str = "unknown"
    title_hint: Optional[str] = None
    sentence_type: str = "declarative"  # question | imperative | declarative
    modifiers: Dict[str, Any] = field(default_factory=dict)
    references: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RouteDecision:
    """Calibrated routing output used by final intent selection."""

    intent: str
    confidence: float
    plugin: str
    action: str
    trace: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# SLOT FILLING SYSTEM
# ============================================================================


class SlotFiller:
    """
    Extracts structured data from queries

    Example:
        "Create note about meeting tomorrow at 2pm tagged work high priority"
        -> {
            'title': 'meeting',
            'date': '2025-01-26',
            'time': '14:00',
            'tags': ['work'],
            'priority': 'high'
        }
    """

    # Slot templates per intent category
    SLOT_TEMPLATES = {
        "note_create": [
            "title",
            "content",
            "tags",
            "priority",
            "category",
            "date",
            "time",
            "note_type",
        ],
        "note_search": ["query", "tags", "date_range", "priority", "category"],
        "note_update": ["note_id", "title", "content", "tags", "priority"],
        "note_delete": ["note_id", "query"],
        "music_play": [
            "song",
            "artist",
            "album",
            "playlist",
            "genre",
            "mood",
            "service",
        ],
        "music_control": ["action", "volume"],
        "calendar_create": [
            "event",
            "date",
            "time",
            "duration",
            "location",
            "attendees",
            "recurring",
        ],
        "calendar_search": ["query", "date_range", "event_type"],
        "email_compose": ["recipient", "subject", "body", "cc", "bcc", "attachments"],
        "email_search": ["sender", "subject", "date_range", "has_attachment", "status"],
    }

    def __init__(self, temporal_parser):
        self.temporal_parser = temporal_parser
        # Wrap in the stable TemporalUnderstanding abstraction.  TemporalUnderstanding
        # is defined later in this module but is always available at call time because
        # Python evaluates function bodies lazily — the module is fully loaded
        # before any instance of SlotFiller is created.
        self.temporal: "TemporalUnderstanding" = TemporalUnderstanding(temporal_parser)  # type: ignore[name-defined]

        # Priority keywords
        self.priority_map = {
            "urgent": "urgent",
            "critical": "urgent",
            "asap": "urgent",
            "immediately": "urgent",
            "high": "high",
            "important": "high",
            "medium": "medium",
            "normal": "medium",
            "low": "low",
            "minor": "low",
        }

        # Note type keywords
        self.note_type_map = {
            "todo": "todo",
            "task": "todo",
            "checklist": "todo",
            "idea": "idea",
            "thought": "idea",
            "brainstorm": "idea",
            "meeting": "meeting",
            "notes": "meeting",
            "reminder": "reminder",
            "alert": "reminder",
        }

        # Category keywords
        self.category_map = {
            "work": "work",
            "business": "work",
            "office": "work",
            "personal": "personal",
            "home": "personal",
            "family": "personal",
            "project": "project",
            "dev": "project",
            "development": "project",
            "health": "health",
            "fitness": "health",
            "study": "study",
            "learning": "study",
            "education": "study",
        }

    def extract_slots(
        self, text: str, intent: str, entities: Dict[str, List[Entity]]
    ) -> Dict[str, Slot]:
        """Extract slots based on intent and entities"""
        slots = {}
        text_lower = text.lower()

        # Determine template
        template_key = self._get_template_key(intent)
        if template_key not in self.SLOT_TEMPLATES:
            return slots

        slot_names = self.SLOT_TEMPLATES[template_key]

        # Extract each slot
        for slot_name in slot_names:
            extractor = getattr(self, f"_extract_{slot_name}", None)
            if extractor:
                value, confidence, raw = extractor(text, text_lower, entities)
                if value is not None:
                    slots[slot_name] = Slot(slot_name, value, confidence, raw)

        return slots

    def _get_template_key(self, intent: str) -> str:
        """Map intent to slot template"""
        # Note intents
        if "create" in intent or "add" in intent:
            if "note" in intent:
                return "note_create"
            elif "event" in intent or "calendar" in intent:
                return "calendar_create"
        elif "search" in intent or "find" in intent or "list" in intent:
            if "note" in intent:
                return "note_search"
            elif "calendar" in intent:
                return "calendar_search"
            elif "email" in intent:
                return "email_search"
        elif "play" in intent or "music" in intent:
            return "music_play"
        elif "email" in intent and ("compose" in intent or "send" in intent):
            return "email_compose"

        return intent

    # ==================== NOTE SLOT EXTRACTORS ====================

    def _extract_title(
        self, text: str, text_lower: str, entities: Dict
    ) -> Tuple[Optional[str], float, str]:
        """Extract note title"""
        # Patterns like "note about X", "create X note", "X task"
        patterns = [
            r"(?:note|task|reminder)\s+(?:about|for|titled?)\s+([^,\n]+)",
            r"(?:create|add|make)\s+(?:a\s+)?(?:note\s+)?(?:about\s+)?([^,\n]+)",
            r"(?:called|named|titled)\s+([^,\n]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                title = match.group(1).strip()
                # Clean up common trailing words
                title = re.sub(
                    r"\s+(tagged?|with|priority|at|on|tomorrow|today).*$", "", title
                )
                if len(title) > 2:
                    return title, 0.85, match.group(0)

        # Fallback: extract first noun phrase (simple heuristic)
        words = text_lower.split()
        if len(words) >= 3:
            # Look for pattern after action words
            action_words = {"create", "add", "make", "new", "note", "task", "reminder"}
            for i, word in enumerate(words):
                if word in action_words and i + 1 < len(words):
                    # Take next 2-4 words as title
                    title_words = words[i + 1 : min(i + 5, len(words))]
                    title = " ".join(title_words)
                    title = re.sub(r"\s+(tagged?|with|priority|at|on).*$", "", title)
                    if len(title) > 2:
                        return title, 0.6, " ".join(words[i : i + 5])

        return None, 0.0, ""

    def _extract_content(
        self, text: str, text_lower: str, entities: Dict
    ) -> Tuple[Optional[str], float, str]:
        """Extract note content"""
        # Content is usually after title or in quotes
        patterns = [
            r'content[:\s]+"?([^"\n]+)"?',
            r'body[:\s]+"?([^"\n]+)"?',
            r'text[:\s]+"?([^"\n]+)"?',
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1).strip(), 0.9, match.group(0)

        return None, 0.0, ""

    def _extract_tags(
        self, text: str, text_lower: str, entities: Dict
    ) -> Tuple[Optional[List[str]], float, str]:
        """Extract tags from #tagname or 'tagged X'"""
        tags = []

        # Hashtag style: #work #urgent
        hashtag_pattern = r"#(\w+)"
        hashtags = re.findall(hashtag_pattern, text_lower)
        tags.extend(hashtags)

        # Explicit tagging: "tagged work urgent" or "tag it as work"
        tag_patterns = [
            r"tagged?\s+(?:as\s+)?(?:with\s+)?([a-z,\s]+?)(?:\s+priority|\s+category|$)",
            r"tags?\s+(?:are\s+)?(?:with\s+)?([a-z,\s]+?)(?:\s+priority|\s+category|$)",
        ]

        for pattern in tag_patterns:
            match = re.search(pattern, text_lower)
            if match:
                tag_text = match.group(1).strip()
                # Split by comma or space
                new_tags = [
                    t.strip() for t in re.split(r"[,\s]+", tag_text) if t.strip()
                ]
                tags.extend(new_tags)

        if tags:
            # Remove duplicates, keep order
            seen = set()
            unique_tags = []
            for tag in tags:
                if tag not in seen:
                    seen.add(tag)
                    unique_tags.append(tag)
            return unique_tags, 0.9, " ".join(f"#{t}" for t in unique_tags)

        return None, 0.0, ""

    def _extract_priority(
        self, text: str, text_lower: str, entities: Dict
    ) -> Tuple[Optional[str], float, str]:
        """Extract priority level"""
        for keyword, priority in self.priority_map.items():
            if re.search(rf"\b{keyword}\b", text_lower):
                return priority, 0.95, keyword

        # Check for explicit priority syntax
        priority_pattern = r"priority[:\s]+(urgent|high|medium|low)"
        match = re.search(priority_pattern, text_lower)
        if match:
            return match.group(1), 0.98, match.group(0)

        return None, 0.0, ""

    def _extract_category(
        self, text: str, text_lower: str, entities: Dict
    ) -> Tuple[Optional[str], float, str]:
        """Extract category"""
        for keyword, category in self.category_map.items():
            if re.search(rf"\b{keyword}\b", text_lower):
                return category, 0.8, keyword

        # Explicit category syntax
        cat_pattern = r"category[:\s]+(\w+)"
        match = re.search(cat_pattern, text_lower)
        if match:
            return match.group(1), 0.95, match.group(0)

        return None, 0.0, ""

    def _extract_note_type(
        self, text: str, text_lower: str, entities: Dict
    ) -> Tuple[Optional[str], float, str]:
        """Extract note type"""
        for keyword, note_type in self.note_type_map.items():
            if re.search(rf"\b{keyword}\b", text_lower):
                return note_type, 0.85, keyword

        return None, 0.0, ""

    def _extract_date(
        self, text: str, text_lower: str, entities: Dict
    ) -> Tuple[Optional[str], float, str]:
        """Extract and normalize date"""
        _tr = self.temporal.parse(text)
        result = _tr.as_dict() if _tr else None
        if result and result.get("date"):
            return (
                result["date"],
                result.get("confidence", 0.8),
                result.get("raw_text", ""),
            )
        return None, 0.0, ""

    def _extract_time(
        self, text: str, text_lower: str, entities: Dict
    ) -> Tuple[Optional[str], float, str]:
        """Extract and normalize time"""
        _tr = self.temporal.parse(text)
        result = _tr.as_dict() if _tr else None
        if result and result.get("time"):
            return (
                result["time"],
                result.get("confidence", 0.8),
                result.get("raw_text", ""),
            )
        return None, 0.0, ""

        # ==================== OTHER SLOT EXTRACTORS ====================

    def _extract_query(
        self, text: str, text_lower: str, entities: Dict
    ) -> Tuple[Optional[str], float, str]:
        """Extract search query"""
        # For search intents, the main query is usually the text minus action words
        query = re.sub(
            r"\b(search|find|show|list|get|fetch)\s+", "", text_lower, count=1
        )
        query = re.sub(r"\b(notes?|emails?|events?|tasks?)\b", "", query).strip()

        if len(query) > 2:
            return query, 0.7, query

        return None, 0.0, ""

    def _extract_album(
        self, text: str, text_lower: str, entities: Dict
    ) -> Tuple[Optional[str], float, str]:
        """Extract album name"""
        return None, 0.0, ""  # Implement if needed

    def _extract_playlist(
        self, text: str, text_lower: str, entities: Dict
    ) -> Tuple[Optional[str], float, str]:
        """Extract playlist name"""
        return None, 0.0, ""  # Implement if needed

    def _extract_service(
        self, text: str, text_lower: str, entities: Dict
    ) -> Tuple[Optional[str], float, str]:
        """Extract music service"""
        services = ["spotify", "apple music", "youtube music", "pandora"]
        for service in services:
            if service in text_lower:
                return service, 0.98, service
        return None, 0.0, ""


# ============================================================================
# TEMPORAL EXPRESSION NORMALIZATION
# ============================================================================


class TemporalParser:
    """
    Parse and normalize temporal expressions

    Examples:
        "tomorrow at 2pm" -> {'date': '2025-01-26', 'time': '14:00'}
        "next week" -> {'date': '2025-02-02'}
        "in 3 days" -> {'date': '2025-01-28'}
        "morning" -> {'time': '09:00'}
    """

    def __init__(self):
        if Calendar is not None:
            version_context_style = getattr(
                parsedatetime_mod, "VERSION_CONTEXT_STYLE", None
            )
            if version_context_style is not None:
                self.cal = Calendar(version=version_context_style)
            else:
                self.cal = Calendar()
        else:
            self.cal = None

        # Time of day mappings
        self.time_of_day = {
            "morning": "09:00",
            "noon": "12:00",
            "afternoon": "14:00",
            "evening": "18:00",
            "night": "20:00",
            "midnight": "00:00",
        }

    def parse_temporal_expression(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse temporal expressions from text"""
        result = {}
        confidence = 0.0
        raw_text = ""

        # Try dateparser first (most comprehensive)
        parsed_date = None
        if dateparser is not None:
            parsed_date = dateparser.parse(
                text,
                settings={
                    "PREFER_DATES_FROM": "future",
                    "RETURN_AS_TIMEZONE_AWARE": False,
                    "RELATIVE_BASE": datetime.now(),
                },
            )

        if parsed_date:
            result["date"] = parsed_date.strftime("%Y-%m-%d")
            result["time"] = parsed_date.strftime("%H:%M")
            confidence = 0.9
            raw_text = text
        else:
            # Try parsedatetime for relative expressions
            if self.cal is not None:
                time_struct, parse_context = self.cal.parse(text)
                # In parsedatetime 2.x+, parse returns (time_struct, pdtContext)
                # pdtContext.hasDateOrTime returns True if parsing was successful
                parse_success = (
                    parse_context.hasDateOrTime
                    if hasattr(parse_context, "hasDateOrTime")
                    else parse_context > 0
                )
                if parse_success:
                    parsed_dt = datetime(*time_struct[:6])
                    result["date"] = parsed_dt.strftime("%Y-%m-%d")
                    result["time"] = parsed_dt.strftime("%H:%M")
                    confidence = 0.8
                    raw_text = text

        # Check for time of day keywords
        text_lower = text.lower()
        for keyword, time_str in self.time_of_day.items():
            if keyword in text_lower:
                result["time"] = time_str
                confidence = max(confidence, 0.85)
                if not raw_text:
                    raw_text = keyword

        if result:
            result["confidence"] = confidence
            result["raw_text"] = raw_text
            return result

        return None

    def normalize_duration(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse duration expressions"""
        patterns = [
            (r"(\d+)\s*(?:minute|min)s?", "minutes"),
            (r"(\d+)\s*(?:hour|hr)s?", "hours"),
            (r"(\d+)\s*(?:day)s?", "days"),
            (r"(\d+)\s*(?:week)s?", "weeks"),
            (r"(\d+)\s*(?:month)s?", "months"),
        ]

        for pattern, unit in patterns:
            match = re.search(pattern, text.lower())
            if match:
                value = int(match.group(1))
                return {"value": value, "unit": unit, "raw_text": match.group(0)}

        return None


# ============================================================================
# CUSTOM NER FOR DOMAIN ENTITIES
# ============================================================================


class DomainEntityExtractor:
    """
    Extract domain-specific entities

    Entities:
    - NOTE_TAG: #work, #personal
    - PRIORITY: urgent, high, low
    - NOTE_TYPE: todo, idea, meeting
    - CATEGORY: work, personal, project
    - MUSIC_GENRE: rock, pop, jazz
    - MUSIC_MOOD: upbeat, chill, relaxing
    """

    ENTITY_PATTERNS = {
        "NOTE_TAG": r"#(\w+)",
        "PRIORITY": r"\b(urgent|critical|high|important|medium|normal|low|minor)\b",
        "NOTE_TYPE": r"\b(todo|task|idea|thought|meeting|reminder)\b",
        "CATEGORY": r"\b(work|personal|project|health|study)\b",
    }

    def extract(self, text: str) -> Dict[str, List[Entity]]:
        """Extract all domain entities"""
        entities = defaultdict(list)
        text_lower = text.lower()

        for entity_type, pattern in self.ENTITY_PATTERNS.items():
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                entity = Entity(
                    type=entity_type,
                    value=match.group(1) if match.groups() else match.group(0),
                    confidence=0.95,
                    start_pos=match.start(),
                    end_pos=match.end(),
                )
                entities[entity_type].append(entity)

        return dict(entities)


# ============================================================================
# COREFERENCE RESOLUTION
# ============================================================================


class CoreferenceResolver:
    """
    Resolve pronouns to entities mentioned in conversation.

    Uses intent-type matching and salience ordering: most-recently
    mentioned entity of the appropriate type wins.
    """

    # Pronouns that can refer to a note / file / generic object
    _OBJECT_PRONOUNS = {"it", "this", "that", "the note", "the task", "the file",
                         "the event", "the song", "the email", "the reminder"}
    # Pronouns that can refer to a person
    _PERSON_PRONOUNS = {"them", "he", "she", "they"}

    def resolve(self, text: str, context: "ConversationContext") -> str:
        """Resolve coreferences in text using conversation context."""
        resolved_text = text
        text_lower = text.lower()

        protected_it_phrases = (
            "what time is it",
            "how is it going",
            "how's it going",
            "hows it going",
        )

        all_pronouns = self._OBJECT_PRONOUNS | self._PERSON_PRONOUNS
        for pronoun in sorted(all_pronouns, key=len, reverse=True):  # longest first
            if pronoun == "it" and any(phrase in text_lower for phrase in protected_it_phrases):
                continue
            if re.search(rf"\b{re.escape(pronoun)}\b", text_lower):
                replacement = self._find_referent(pronoun, context)
                if replacement:
                    resolved_text = re.sub(
                        rf"\b{re.escape(pronoun)}\b",
                        replacement,
                        resolved_text,
                        count=1,
                        flags=re.IGNORECASE,
                    )
                    logger.info(f"[COREF] Resolved '{pronoun}' -> '{replacement}'")
                    text_lower = resolved_text.lower()  # re-scan after each substitution

        return resolved_text

    def _find_referent(
        self, pronoun: str, context: "ConversationContext"
    ) -> Optional[str]:
        """Find what the pronoun refers to, using intent-type matching."""
        last_intent = context.last_intent or ""

        # Person pronouns → last person entity
        if pronoun in self._PERSON_PRONOUNS:
            if "sender" in context.last_entities:
                return str(context.last_entities["sender"])
            if "recipient" in context.last_entities:
                return str(context.last_entities["recipient"])
            return None

        # Object pronouns: pick the best candidate by domain
        if "note" in last_intent or "task" in last_intent:
            if context.mentioned_notes:
                return str(context.mentioned_notes[-1])
            if "title" in context.last_entities:
                return str(context.last_entities["title"])
            if "note_id" in context.last_entities:
                return str(context.last_entities["note_id"])

        if "calendar" in last_intent or "event" in last_intent:
            if context.mentioned_events:
                return str(context.mentioned_events[-1])
            if "event" in context.last_entities:
                return str(context.last_entities["event"])

        if "music" in last_intent or "song" in last_intent:
            if context.mentioned_songs:
                return str(context.mentioned_songs[-1])
            if "song" in context.last_entities:
                return str(context.last_entities["song"])

        if "email" in last_intent:
            for key in ("subject", "sender", "email_id"):
                if key in context.last_entities:
                    return str(context.last_entities[key])

        # Generic fallback: most-recently filled slot (salience: title > query > any)
        for key in ("title", "query", "note_id", "song", "event"):
            if key in context.last_entities:
                return str(context.last_entities[key])

        # Last resort: first value in last_entities
        if context.last_entities:
            return str(next(iter(context.last_entities.values())))

        return None


# ============================================================================
# ENHANCED EMOTION & URGENCY DETECTION
# ============================================================================


class EmotionDetector:
    """
    Multi-label emotion detection + urgency analysis

    Emotions: angry, excited, worried, confused, satisfied, frustrated
    Urgency: none, low, medium, high, critical
    """

    EMOTION_KEYWORDS = {
        "angry": ["angry", "mad", "pissed", "furious", "annoyed", "irritated"],
        "excited": ["excited", "awesome", "amazing", "great", "fantastic", "wonderful"],
        "worried": ["worried", "concerned", "anxious", "nervous", "stressed"],
        "confused": ["confused", "lost", "unclear", "dont understand"],
        "satisfied": ["thanks", "thank you", "perfect", "exactly", "good"],
        "frustrated": [
            "not working",
            "broken",
            "frustrated",
            "cant",
            "wont",
            "doesnt work",
        ],
    }

    URGENCY_KEYWORDS = {
        "critical": [
            "emergency",
            "critical",
            "asap",
            "immediately",
            "right now",
            "urgent",
        ],
        "high": ["urgent", "important", "soon", "quickly", "hurry"],
        "medium": ["when you can", "sometime", "later"],
        "low": ["no rush", "whenever", "eventually"],
    }

    def detect_emotions(self, text: str, sentiment_scores: Dict) -> List[str]:
        """Detect emotions from text"""
        emotions = []
        text_lower = text.lower()

        for emotion, keywords in self.EMOTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower and not _has_negation_before(text_lower, keyword):
                    emotions.append(emotion)
                    break

        # Infer from sentiment if no explicit emotions
        if not emotions:
            compound = sentiment_scores.get("compound", 0)
            if compound > 0.5:
                emotions.append("satisfied")
            elif compound < -0.5:
                emotions.append("frustrated")

        return emotions

    def detect_urgency(self, text: str) -> str:
        """Detect urgency level"""
        text_lower = text.lower()

        # Check from highest to lowest
        for level in ["critical", "high", "medium", "low"]:
            for keyword in self.URGENCY_KEYWORDS[level]:
                if keyword in text_lower:
                    return level

        return "none"


# ============================================================================
# MAIN NLP PROCESSOR V2
# ============================================================================


class NLPProcessor:
    """
    NLP processing for A.L.I.C.E.

    No more regex hell. Pure intelligence.
    """

    _instance = None
    _lock = threading.Lock()

    # Foundation 2 routing policy thresholds.
    UNKNOWN_FALLBACK_CONF_HARD = 0.35
    UNKNOWN_FALLBACK_CONF_SOFT = 0.45
    UNKNOWN_FALLBACK_PLAUS_SOFT = 0.60
    UNKNOWN_FALLBACK_PLAUS_HARD = 0.45
    ROUTE_UNCERTAINTY_THRESHOLD = 0.55
    CLARIFICATION_INTENT_CONFIDENCE_THRESHOLD = 0.45
    CLARIFICATION_CONFIDENCE_MIN = 0.42
    CLARIFICATION_CONFIDENCE_MAX = 0.62
    CONVERSATION_CATEGORY_GATE_THRESHOLD = 0.88

    def __new__(cls, *args: Any, **kwargs: Any):
        """Singleton pattern for performance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, plugin_registry: Optional[PluginRegistry] = None):
        if hasattr(self, "_initialized"):
            if plugin_registry is not None:
                self.plugin_registry = plugin_registry
            return

        self._initialized = True

        # Core components
        self.sentiment_analyzer = (
            SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
        )
        self.vectorizer = (
            TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            if TfidfVectorizer
            else None
        )

        # Advanced components
        self.temporal_parser = TemporalParser()
        self.slot_filler = SlotFiller(self.temporal_parser)
        self.domain_ner = DomainEntityExtractor()
        self.emotion_detector = EmotionDetector()

        # Stable temporal abstraction — consumers should use this instead of
        # calling temporal_parser directly so parser API changes stay local.
        self.temporal = TemporalUnderstanding(self.temporal_parser)

        # Advanced NLP stack
        if ADVANCED_NLP_AVAILABLE:
            self.coref_resolver = _AdvancedCoref()
            self.frame_parser = _FrameParser()
            self.prob_slot_filler = _ProbSlotFiller()
            self.dialogue_memory = self.coref_resolver.memory
            self._fp_store = _get_fingerprint_store()
        else:
            self.coref_resolver = CoreferenceResolver()
            self.frame_parser = None
            self.prob_slot_filler = None
            self.dialogue_memory = None

        # Entity normalizer (P0 Improvement)
        if ENTITY_NORMALIZER_AVAILABLE:
            self.entity_normalizer = _get_entity_normalizer()
        else:
            self.entity_normalizer = None

        # Feature flags for A/B testing
        if FEATURE_FLAGS_AVAILABLE:
            self.feature_flags = _get_feature_flags()
        else:
            self.feature_flags = None

        # Conversation context
        self.context = ConversationContext()
        self.followup_resolver = FollowUpResolver()
        self.goal_recognizer = get_goal_recognizer()
        if plugin_registry is not None:
            self.plugin_registry = plugin_registry
        else:
            try:
                self.plugin_registry = discover_plugins()
            except Exception:
                self.plugin_registry = PluginRegistry()

        try:
            from ai.optimization.runtime_thresholds import get_thresholds

            thresholds = get_thresholds()
            self.UNKNOWN_FALLBACK_CONF_HARD = float(
                thresholds.get(
                    "unknown_fallback_conf_hard", self.UNKNOWN_FALLBACK_CONF_HARD
                )
            )
            self.UNKNOWN_FALLBACK_CONF_SOFT = float(
                thresholds.get(
                    "unknown_fallback_conf_soft", self.UNKNOWN_FALLBACK_CONF_SOFT
                )
            )
            self.UNKNOWN_FALLBACK_PLAUS_SOFT = float(
                thresholds.get(
                    "unknown_fallback_plaus_soft", self.UNKNOWN_FALLBACK_PLAUS_SOFT
                )
            )
            self.UNKNOWN_FALLBACK_PLAUS_HARD = float(
                thresholds.get(
                    "unknown_fallback_plaus_hard", self.UNKNOWN_FALLBACK_PLAUS_HARD
                )
            )
            self.ROUTE_UNCERTAINTY_THRESHOLD = float(
                thresholds.get(
                    "route_uncertainty_threshold", self.ROUTE_UNCERTAINTY_THRESHOLD
                )
            )
            self.CLARIFICATION_INTENT_CONFIDENCE_THRESHOLD = float(
                thresholds.get(
                    "clarification_intent_confidence_threshold",
                    self.CLARIFICATION_INTENT_CONFIDENCE_THRESHOLD,
                )
            )
            self.CLARIFICATION_CONFIDENCE_MIN = float(
                thresholds.get(
                    "clarification_confidence_min", self.CLARIFICATION_CONFIDENCE_MIN
                )
            )
            self.CLARIFICATION_CONFIDENCE_MAX = float(
                thresholds.get(
                    "clarification_confidence_max", self.CLARIFICATION_CONFIDENCE_MAX
                )
            )
            self.CONVERSATION_CATEGORY_GATE_THRESHOLD = float(
                thresholds.get(
                    "conversation_category_gate_threshold",
                    self.CONVERSATION_CATEGORY_GATE_THRESHOLD,
                )
            )
        except Exception as e:
            logger.debug(f"Runtime thresholds unavailable for NLP processor: {e}")

        self.route_coordinator = RouteCoordinator(
            RouteCoordinatorConfig(
                unknown_fallback_conf_hard=self.UNKNOWN_FALLBACK_CONF_HARD,
                unknown_fallback_conf_soft=self.UNKNOWN_FALLBACK_CONF_SOFT,
                unknown_fallback_plaus_soft=self.UNKNOWN_FALLBACK_PLAUS_SOFT,
                unknown_fallback_plaus_hard=self.UNKNOWN_FALLBACK_PLAUS_HARD,
                route_uncertainty_threshold=self.ROUTE_UNCERTAINTY_THRESHOLD,
                clarification_intent_confidence_threshold=self.CLARIFICATION_INTENT_CONFIDENCE_THRESHOLD,
                clarification_confidence_min=self.CLARIFICATION_CONFIDENCE_MIN,
                clarification_confidence_max=self.CLARIFICATION_CONFIDENCE_MAX,
                conversation_category_gate_threshold=self.CONVERSATION_CATEGORY_GATE_THRESHOLD,
            )
        )
        self.foundation_layers = FoundationLayers(budget_ms=120.0)

        # Tier foundations: explicit short-horizon memory + implicit intent + dialogue state machine
        if INTELLIGENCE_FOUNDATIONS_AVAILABLE:
            self.conversation_memory = ConversationMemory(max_turns=20)
            self.implicit_intent_detector = ImplicitIntentDetector()
            self.dialogue_state_machine = DialogueStateMachine(max_clarifying_turns=5)
        else:
            self.conversation_memory = None
            self.implicit_intent_detector = None
            self.dialogue_state_machine = None

        # Semantic intent classifier
        self.semantic_classifier = None
        self._semantic_classifier_init_attempted = False
        self.llm_gateway = None
        self.llm_intent_classifier = None

        # Load learned corrections into pattern matching
        self.learned_corrections = self._load_learned_corrections()

        # Intent-Entity Cross-Validation Matrix (P0 Improvement)
        # Maps intent prefixes to required/expected entity types
        # Note: 'query' can be empty for list-all operations
        self._validation_matrix = {
            "notes:create": {"required": [], "expected": ["title", "content", "tags"]},
            "notes:search": {
                "required": [],
                "expected": ["query", "tags", "date_range"],
            },  # Changed: query not required for list-all
            "notes:update": {
                "required": ["note_id", "title"],
                "expected": ["content", "tags"],
            },
            "notes:delete": {
                "required": [],
                "expected": ["note_id", "title", "query"],
            },  # Plugin handles "note not found" gracefully; don't block at gate
            "calendar:create": {
                "required": ["event", "date"],
                "expected": ["time", "location"],
            },
            "calendar:search": {
                "required": [],
                "expected": ["query", "date_range"],
            },  # List-all allowed
            "email:compose": {
                "required": ["recipient", "subject"],
                "expected": ["body"],
            },
            "email:search": {
                "required": [],
                "expected": ["sender", "subject", "date_range"],
            },  # List-all allowed
        }
        self.tokenizer_profile = "default"
        self.command_vocabulary = self._load_command_vocabulary()
        self._plugin_actions = {
            "notes": {
                "create",
                "append",
                "read",
                "list",
                "search",
                "delete",
                "query_exist",
            },
            "email": {"compose", "read", "list", "search", "delete", "reply"},
            "calendar": {"create", "list", "search", "update", "delete"},
            "weather": {"current", "forecast"},
            "system": {"status", "debug_tokens"},
            "conversation": {
                "general",
                "question",
                "meta_question",
                "clarification_needed",
                "goal_statement",
            },
        }
        self._intent_action_defaults = {
            "notes": "list",
            "email": "list",
            "calendar": "list",
            "weather": "current",
            "system": "status",
            "conversation": "general",
        }
        self._grammar_action_weights = {
            "create": 1.25,
            "append": 1.20,
            "read": 1.15,
            "list": 1.05,
            "search": 1.05,
            "delete": 1.05,
            "query_exist": 1.10,
        }
        self._typo_replacements = {
            "raed": "read",
            "nots": "notes",
            "notse": "notes",
            "emial": "email",
            "calender": "calendar",
        }
        self._noisy_channel_lexicon = {
            *(self.command_vocabulary.get("verbs", set())),
            *(self.command_vocabulary.get("objects", set())),
            "calendar",
            "email",
            "notes",
            "note",
            "todo",
            "task",
            "tasks",
            "weather",
            "forecast",
            "temperature",
            "rain",
            "snow",
            "sunny",
            "cloudy",
            "overcast",
            "wind",
            "windy",
            "storm",
            "read",
            "open",
            "show",
            "create",
            "append",
            "list",
            "search",
            "delete",
        }

        # Cache for performance (OrderedDict enables true LRU eviction)
        self._entity_cache: OrderedDict = OrderedDict()
        self._cache_lock = threading.Lock()
        # Lazy-built embedding index for semantic learned-corrections lookup
        self._correction_embeddings = None
        self._correction_keys: Optional[List[str]] = None

    def _ensure_semantic_classifier(self):
        """Lazily initialize semantic classifier only when needed."""
        if self.semantic_classifier is not None:
            return self.semantic_classifier

        if self._semantic_classifier_init_attempted:
            return None

        self._semantic_classifier_init_attempted = True
        if not SEMANTIC_CLASSIFIER_AVAILABLE:
            return None

        try:
            self.semantic_classifier = get_intent_classifier()
            logger.info("[OK] Semantic intent classifier loaded (lazy)")
            return self.semantic_classifier
        except Exception as e:
            logger.warning(f"[WARN] Failed to load semantic classifier: {e}")
            return None

    def attach_llm_gateway(self, llm_gateway: Any) -> None:
        """Attach LLM gateway for optional low-confidence intent arbitration."""
        self.llm_gateway = llm_gateway
        if self.llm_intent_classifier is not None:
            try:
                self.llm_intent_classifier.llm_gateway = llm_gateway
            except Exception:
                pass

    def _ensure_llm_intent_classifier(self):
        """Lazily initialize LLM intent classifier when an LLM gateway is available."""
        if self.llm_intent_classifier is not None:
            return self.llm_intent_classifier
        if not LLM_INTENT_CLASSIFIER_AVAILABLE:
            return None
        if self.llm_gateway is None:
            return None
        try:
            self.llm_intent_classifier = get_llm_intent_classifier(self.llm_gateway)
            return self.llm_intent_classifier
        except Exception as e:
            logger.debug(f"[NLP] Could not initialize LLM intent classifier: {e}")
            return None

    def _recent_context_for_llm_intent(self, limit: int = 3) -> List[str]:
        """Collect compact recent user-turn context for hybrid intent arbitration."""
        if self.conversation_memory is None:
            return []
        try:
            turns = self.conversation_memory.recent(limit=limit)
            return [str(turn.user_input or "").strip() for turn in turns if str(turn.user_input or "").strip()]
        except Exception:
            return []

    def _normalize_hybrid_intent(self, intent_name: str) -> str:
        """Normalize hybrid classifier output into runtime intent label format."""
        intent = str(intent_name or "").strip().lower()
        if not intent:
            return ""
        if ":" in intent:
            return intent

        if intent in self._intent_action_defaults:
            return f"{intent}:{self._intent_action_defaults[intent]}"

        aliases = {
            "question": "conversation:question",
            "conversation": "conversation:general",
            "meta_question": "conversation:meta_question",
            "clarification_needed": "conversation:clarification_needed",
            "status_inquiry": "status_inquiry",
            "thanks": "thanks",
            "greeting": "greeting",
        }
        return aliases.get(intent, "")

    def _maybe_apply_llm_intent_fallback(
        self,
        *,
        normalized_text: str,
        route: RouteDecision,
        weighted_candidates: List[Tuple[str, float]],
        parsed_command: ParsedCommand,
    ) -> RouteDecision:
        """Use LLM hybrid intent arbitration for low-confidence/ambiguous routes."""
        llm_classifier = self._ensure_llm_intent_classifier()
        if llm_classifier is None:
            return route

        trace = dict(getattr(route, "trace", {}) or {})
        calibration = dict(trace.get("calibration", {}) or {})
        margin = float(calibration.get("margin", 1.0) or 1.0)
        route_conf = float(route.confidence or 0.0)
        low_confidence = route_conf < 0.68
        ambiguous_margin = margin < 0.08

        if not (low_confidence or ambiguous_margin):
            return route

        # Keep explicit high-impact commands deterministic when confidence is already acceptable.
        explicit_action = str(getattr(parsed_command, "action", "") or "").strip().lower()
        if explicit_action in {"delete", "remove", "update", "append", "send", "compose"} and route_conf >= 0.60:
            return route

        llm_context = self._recent_context_for_llm_intent(limit=3)
        try:
            llm_intent_raw, llm_confidence, llm_source = llm_classifier.classify_hybrid(
                query=normalized_text,
                semantic_confidence=route_conf,
                semantic_intent=route.intent,
                context=llm_context,
            )
        except Exception as e:
            logger.debug(f"[NLP] LLM intent fallback failed: {e}")
            return route

        llm_intent = self._normalize_hybrid_intent(str(llm_intent_raw or ""))
        llm_confidence = float(llm_confidence or 0.0)
        if not llm_intent:
            return route

        adopt_llm = bool(llm_confidence >= max(route_conf + 0.05, 0.45))
        if not adopt_llm:
            return route

        # If the route already points to a concrete non-conversation action, keep it.
        if (
            route.intent.startswith(("notes:", "email:", "calendar:", "file_operations:", "memory:", "reminder:", "system:", "weather:", "time:"))
            and llm_intent.startswith("conversation:")
            and route_conf >= 0.62
        ):
            return route

        if ":" in llm_intent:
            plugin, action = llm_intent.split(":", 1)
        else:
            plugin, action = "conversation", "general"

        trace.update(
            {
                "llm_intent_fallback": {
                    "applied": True,
                    "previous_intent": route.intent,
                    "previous_confidence": route_conf,
                    "candidate_intent": llm_intent,
                    "candidate_confidence": llm_confidence,
                    "source": str(llm_source or "llm_hybrid"),
                    "weighted_candidates": weighted_candidates[:3],
                }
            }
        )

        return RouteDecision(
            intent=llm_intent,
            confidence=max(0.2, min(0.97, llm_confidence)),
            plugin=plugin,
            action=action,
            trace=trace,
        )

    def set_tokenizer_profile(self, profile: str) -> None:
        """Set tokenizer behavior profile: strict, default, llm-assisted."""
        profile_value = (profile or "default").strip().lower()
        if profile_value not in {"strict", "default", "llm-assisted"}:
            profile_value = "default"
        self.tokenizer_profile = profile_value

    def _load_command_vocabulary(self) -> Dict[str, Set[str]]:
        """Load shared command vocabulary from command directory with safe defaults."""
        verbs = {
            "create",
            "add",
            "make",
            "write",
            "append",
            "show",
            "list",
            "read",
            "open",
            "find",
            "search",
            "delete",
            "remove",
            "edit",
            "update",
            "summarize",
            "archive",
            "unarchive",
            "pin",
            "unpin",
            "play",
            "pause",
            "send",
            "check",
        }
        objects = {
            "note",
            "notes",
            "list",
            "lists",
            "task",
            "tasks",
            "email",
            "emails",
            "calendar",
            "event",
            "events",
            "reminder",
            "reminders",
        }
        meta = {"/help", "/plugins", "/debug", "/debug tokens", "/profile"}

        try:
            command_path = Path("command_directory.md")
            if command_path.exists():
                content = command_path.read_text(encoding="utf-8", errors="ignore")
                phrases = re.findall(r'"([^"]+)"', content)
                for phrase in phrases:
                    parts = [p for p in re.findall(r"[a-zA-Z]+", phrase.lower()) if p]
                    if not parts:
                        continue
                    verbs.add(parts[0])
                    if len(parts) > 1:
                        objects.add(parts[-1])
        except Exception as e:
            logger.debug(
                f"Could not load command vocabulary from command_directory.md: {e}"
            )

        return {
            "verbs": verbs,
            "objects": objects,
            "meta": meta,
        }

    def _edit_distance(self, source: str, target: str) -> int:
        """Small Levenshtein implementation for typo correction."""
        if source == target:
            return 0
        if not source:
            return len(target)
        if not target:
            return len(source)

        previous = list(range(len(target) + 1))
        for i, source_char in enumerate(source, start=1):
            current = [i]
            for j, target_char in enumerate(target, start=1):
                substitution = previous[j - 1] + (
                    0 if source_char == target_char else 1
                )
                insertion = current[j - 1] + 1
                deletion = previous[j] + 1
                current.append(min(substitution, insertion, deletion))
            previous = current
        return previous[-1]

    def _closest_lexicon_term(self, token: str, max_distance: int = 1) -> Optional[str]:
        """Find likely command lexicon correction for noisy input tokens."""
        if not token or not token.isalpha() or len(token) < 4:
            return None
        if token in self._noisy_channel_lexicon:
            return token

        def _is_adjacent_transposition(source: str, target: str) -> bool:
            if len(source) != len(target):
                return False
            mismatches = [
                index
                for index, pair in enumerate(zip(source, target))
                if pair[0] != pair[1]
            ]
            if len(mismatches) != 2:
                return False
            first, second = mismatches
            return (
                second == first + 1
                and source[first] == target[second]
                and source[second] == target[first]
            )

        candidate = None
        best_distance = max_distance + 1
        for term in self._noisy_channel_lexicon:
            if abs(len(term) - len(token)) > max_distance:
                continue
            if token[0] != term[0] or token[-1] != term[-1]:
                continue

            if _is_adjacent_transposition(token, term):
                return term

            distance = self._edit_distance(token, term)
            if distance < best_distance:
                best_distance = distance
                candidate = term
                if distance == 1:
                    break

        if best_distance <= max_distance:
            return candidate
        return None

    def _apply_noisy_channel_normalization(self, text: str) -> str:
        """Apply typo-aware correction for command words while preserving structure."""
        if not text:
            return text

        parts = re.findall(r"[A-Za-z]+|[^A-Za-z]+", text)
        adjusted: List[str] = []
        for part in parts:
            if part.isalpha():
                lower = part.lower()
                replacement = self._typo_replacements.get(
                    lower
                ) or self._closest_lexicon_term(lower)
                adjusted.append(replacement if replacement else part)
            else:
                adjusted.append(part)

        return "".join(adjusted)

    def _build_weighted_parse(
        self, parsed: ParsedCommand, plugin_scores: Dict[str, float], text: str
    ) -> List[Tuple[str, float]]:
        """Produce weighted intent candidates from parsed command and plugin scores."""
        candidates: List[Tuple[str, float]] = []
        if (
            parsed.object_type == "note"
            and parsed.action in self._plugin_actions["notes"]
        ):
            weight = self._grammar_action_weights.get(parsed.action, 1.0)
            confidence = min(0.96, 0.68 * weight)
            candidates.append((f"notes:{parsed.action}", confidence))

        if (
            parsed.object_type == "weather"
            and parsed.action in self._plugin_actions["weather"]
        ):
            candidates.append((f"weather:{parsed.action}", 0.84))

        if parsed.object_type == "unknown":
            lower = text.lower()
            if re.search(r"\b(email|mail|inbox)\b", lower):
                action = (
                    "compose"
                    if re.search(r"\b(send|compose|draft|write)\b", lower)
                    else "list"
                )
                candidates.append((f"email:{action}", 0.66))
            if re.search(r"\b(calendar|meeting|event|schedule)\b", lower):
                action = (
                    "create"
                    if re.search(r"\b(create|add|schedule|book)\b", lower)
                    else "list"
                )
                candidates.append((f"calendar:{action}", 0.66))

        if plugin_scores:
            ranked = sorted(
                plugin_scores.items(), key=lambda item: item[1], reverse=True
            )
            if ranked:
                plugin, score = ranked[0]
                if plugin in self._intent_action_defaults and score > 1.0:
                    action = self._intent_action_defaults[plugin]
                    if (
                        plugin == "notes"
                        and parsed.action in self._plugin_actions["notes"]
                    ):
                        action = parsed.action
                    candidates.append(
                        (f"{plugin}:{action}", min(0.85, 0.5 + (score / 12.0)))
                    )

        deduped: Dict[str, float] = {}
        for intent, score in candidates:
            deduped[intent] = max(score, deduped.get(intent, 0.0))
        return sorted(deduped.items(), key=lambda item: item[1], reverse=True)

    def _retrieval_first_parse(
        self, tokens: List[RichToken]
    ) -> Optional[Tuple[str, float]]:
        """Resolve explicit object-reference follow-ups only.

        Foundation 2 policy: broad topic inheritance now belongs to
        FollowUpResolver, so this stage only handles high-precision commands
        like "open it", "delete that", "what's in it?".
        """
        content_tokens = [token for token in tokens if token.kind != "symbol"]
        if len(content_tokens) > 8:
            return None

        token_words = {token.normalized for token in content_tokens}
        last_intent = self.context.last_intent or ""
        if not last_intent:
            return None

        reference_present = any(token.flags.get("is_reference") for token in content_tokens)
        explicit_reference_tokens = {
            "it",
            "this",
            "that",
            "one",
            "first",
            "second",
            "third",
            "those",
            "these",
        }
        has_reference = reference_present or bool(token_words & explicit_reference_tokens)
        if not has_reference:
            return None

        _delete_words = {"delete", "remove", "trash", "erase", "discard"}
        _content_cues = {"in", "inside", "contents", "content", "contains"}

        if last_intent.startswith("notes:") or self.context.mentioned_notes:
            if not token_words.isdisjoint(_delete_words):
                return "notes:delete", 0.85
            if not token_words.isdisjoint(_content_cues):
                return "notes:read_content", 0.88
            if "list" in token_words:
                return "notes:list", 0.82
            if not token_words.isdisjoint({"open", "read", "show"}):
                return "notes:read", 0.82

        if last_intent.startswith("email:") and not token_words.isdisjoint(
            {"read", "open", "show", "reply", "forward"}
        ):
            return "email:read", 0.78

        if last_intent.startswith("calendar:") and not token_words.isdisjoint(
            {"list", "show", "open", "move", "reschedule", "cancel"}
        ):
            return "calendar:list", 0.76

        if last_intent.startswith("reminder:") and not token_words.isdisjoint(
            {"show", "list", "cancel", "remove", "snooze"}
        ):
            return "reminder:list", 0.76

        return None

    def _calibrate_route_decision(
        self,
        weighted_candidates: List[Tuple[str, float]],
        plugin_scores: Dict[str, float],
        semantic_intent: Optional[Tuple[str, float]],
        parsed: ParsedCommand,
    ) -> RouteDecision:
        """Two-stage calibrated router: combine grammar, plugin, and semantic signals."""
        combined: Dict[str, float] = {}
        trace: Dict[str, Any] = {
            "weighted_candidates": weighted_candidates,
            "semantic": semantic_intent,
            "plugin_scores": plugin_scores,
        }

        for intent, score in weighted_candidates:
            combined[intent] = max(combined.get(intent, 0.0), score)

        if semantic_intent:
            semantic_name, semantic_score = semantic_intent
            # Phase 1 high-confidence results (≥0.88) are authoritative — return
            # immediately without letting plugin keyword scoring override them.
            if semantic_score >= 0.88:
                if ":" in semantic_name:
                    s_plugin, s_action = semantic_name.split(":", 1)
                else:
                    s_plugin, s_action = "conversation", "general"
                return RouteDecision(
                    intent=semantic_name,
                    confidence=semantic_score,
                    plugin=s_plugin,
                    action=s_action,
                    trace={**trace, "source": "phase1_authoritative"},
                )
            combined[semantic_name] = max(
                combined.get(semantic_name, 0.0), semantic_score * 0.92
            )

        if plugin_scores:
            plugin_name, plugin_value = max(
                plugin_scores.items(), key=lambda item: item[1]
            )
            if plugin_name in self._intent_action_defaults and plugin_value > 1.4:
                plugin_action = self._intent_action_defaults[plugin_name]
                plugin_intent = f"{plugin_name}:{plugin_action}"
                combined[plugin_intent] = max(
                    combined.get(plugin_intent, 0.0),
                    min(0.82, 0.42 + (plugin_value / 10.0)),
                )

        if not combined:
            return RouteDecision(
                intent="conversation:general",
                confidence=0.3,
                plugin="conversation",
                action="general",
                trace=trace,
            )

        best_intent, best_score = max(combined.items(), key=lambda item: item[1])
        sorted_scores = sorted(combined.values(), reverse=True)
        margin = (
            sorted_scores[0] - sorted_scores[1]
            if len(sorted_scores) > 1
            else sorted_scores[0]
        )
        calibrated = max(0.2, min(0.97, best_score * (0.88 + min(0.2, margin))))

        if ":" in best_intent:
            plugin, action = best_intent.split(":", 1)
        else:
            plugin, action = "conversation", "general"

        trace["calibration"] = {
            "margin": margin,
            "combined": sorted(
                combined.items(), key=lambda item: item[1], reverse=True
            )[:5],
        }

        return RouteDecision(
            intent=best_intent,
            confidence=calibrated,
            plugin=plugin,
            action=action,
            trace=trace,
        )

    def _build_top_intent_candidates(
        self,
        *,
        route: Optional[RouteDecision],
        weighted_candidates: List[Tuple[str, float]],
        semantic_intent: Optional[Tuple[str, float]],
        plugin_scores: Dict[str, float],
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Build top-N intent candidates with normalized scores and traceable sources."""
        aggregate: Dict[str, Dict[str, Any]] = {}

        def _upsert(intent_name: str, score: float, source: str) -> None:
            if not intent_name:
                return
            bounded = max(0.0, min(1.0, float(score or 0.0)))
            existing = aggregate.get(intent_name)
            if existing is None or bounded > float(existing.get("score", 0.0)):
                aggregate[intent_name] = {
                    "intent": intent_name,
                    "score": bounded,
                    "source": source,
                }

        if route is not None:
            _upsert(route.intent, route.confidence, "route")
            calibration = (route.trace or {}).get("calibration", {}) or {}
            combined = calibration.get("combined") or []
            for item in combined:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    _upsert(str(item[0]), float(item[1]), "calibration")

        for candidate_intent, candidate_score in weighted_candidates or []:
            _upsert(candidate_intent, candidate_score, "weighted_parse")

        if semantic_intent and len(semantic_intent) == 2:
            _upsert(str(semantic_intent[0]), float(semantic_intent[1]), "semantic")

        if plugin_scores:
            plugin_ranked = sorted(
                plugin_scores.items(), key=lambda item: item[1], reverse=True
            )[:2]
            for plugin_name, plugin_score in plugin_ranked:
                default_action = self._intent_action_defaults.get(plugin_name)
                if default_action:
                    implied_intent = f"{plugin_name}:{default_action}"
                    implied_score = min(0.9, 0.45 + (float(plugin_score) / 12.0))
                    _upsert(implied_intent, implied_score, "plugin_prior")

        ranked = sorted(
            aggregate.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True
        )[: max(1, int(limit))]
        for idx, item in enumerate(ranked, start=1):
            item["rank"] = idx
            item["score"] = round(float(item["score"]), 4)
        return ranked

    def _validate_intent_plausibility(
        self,
        text: str,
        intent: str,
        parsed_command: ParsedCommand,
        plugin_scores: Dict[str, float],
    ) -> Tuple[float, List[str]]:
        """Estimate whether the chosen intent is plausible for the utterance."""
        text_lower = (text or "").lower()
        chosen_intent = (intent or "conversation:general").lower()
        issues: List[str] = []
        score = 0.62
        rich_conceptual = self._is_rich_conceptual_request(text)

        def _contains_cue(cue: str) -> bool:
            cue_text = (cue or "").strip().lower()
            if not cue_text:
                return False
            if " " in cue_text:
                return cue_text in text_lower
            return bool(re.search(rf"\b{re.escape(cue_text)}\b", text_lower))

        weather_cues = _P1_WEATHER_KEYWORDS | _P1_FORECAST_WORDS | {"wind", "storm"}
        conversation_cues = {
            "brainstorm",
            "ideas",
            "idea",
            "explore",
            "plan",
            "strategy",
            "think",
            "could we",
            "how might",
            "let us",
            "let's",
            "discussion",
        }
        cue_map = {
            "notes": {"note", "notes", "memo", "todo", "task"},
            "email": {"email", "mail", "inbox", "subject", "reply"},
            "calendar": {"calendar", "event", "meeting", "schedule"},
            "reminder": {
                "remind",
                "reminder",
                "reminders",
                "alert",
                "notify",
                "forget",
                "don't let me forget",
            },
            "weather": weather_cues,
            "time": {"time", "clock", "hour", "timezone", "date"},
            "system": {"cpu", "memory", "disk", "battery", "status", "system"},
        }
        negative_evidence_map = {
            "weather": {
                "brainstorm",
                "architecture",
                "design",
                "software design",
                "system design",
                "api",
                "database",
                "refactor",
                "codebase",
            },
            "notes": {
                "weather",
                "forecast",
                "temperature",
                "rain",
                "snow",
                "internal code",
                "codebase",
                "source code",
                "python file",
                "python files",
            },
        }

        intent_plugin = chosen_intent.split(":", 1)[0]
        has_conversation_cue = any(_contains_cue(cue) for cue in conversation_cues)

        if intent_plugin in cue_map:
            matched = any(_contains_cue(cue) for cue in cue_map[intent_plugin])
            if matched:
                score += 0.22
            else:
                score -= 0.14
                issues.append(f"missing_{intent_plugin}_cues")

        if intent_plugin == "weather" and not any(_contains_cue(cue) for cue in weather_cues):
            score -= 0.28
            issues.append("weather_without_weather_signals")

        if intent_plugin == "weather":
            has_location_signal = bool(
                re.search(r"\b(?:in|at|for)\s+[a-z][a-z\s'\-]{2,30}\b", text_lower)
            )
            has_temperature_signal = any(
                _contains_cue(cue)
                for cue in {
                    "temperature",
                    "temp",
                    "degrees",
                    "celsius",
                    "fahrenheit",
                    "hot",
                    "cold",
                    "humid",
                }
            )
            has_forecast_signal = any(
                _contains_cue(cue)
                for cue in {
                    "forecast",
                    "tomorrow",
                    "tonight",
                    "weekend",
                    "next week",
                    "this week",
                    "7 day",
                    "rain",
                    "snow",
                    "sunny",
                    "cloudy",
                    "storm",
                    "wind",
                }
            )
            if not (has_location_signal or has_temperature_signal or has_forecast_signal):
                score -= 0.24
                issues.append("missing_weather_required_entities")

        contradiction_cues = negative_evidence_map.get(intent_plugin, set())
        contradiction_hits = [cue for cue in contradiction_cues if _contains_cue(cue)]
        if contradiction_hits:
            score -= 0.24
            issues.append(f"negative_evidence_{intent_plugin}_contradiction")
            if intent_plugin == "weather" and not any(_contains_cue(cue) for cue in weather_cues):
                score -= 0.10
                issues.append("weather_contradiction_without_weather_support")

        if has_conversation_cue and intent_plugin not in {"conversation", "learning"}:
            score -= 0.16
            issues.append("conversational_prompt_vs_action_intent")

        if rich_conceptual:
            if intent_plugin in {"conversation", "learning"}:
                score += 0.20
            else:
                score -= 0.12
                issues.append("rich_conceptual_vs_action_intent")

        if chosen_intent.startswith("conversation:"):
            score += 0.10

        if (
            parsed_command.object_type == "unknown"
            and not parsed_command.references
            and intent_plugin in {
                "notes",
                "email",
                "calendar",
            }
        ):
            score -= 0.08
            issues.append("no_object_for_domain_intent")

        if intent_plugin in {"notes", "email", "calendar"} and parsed_command.references:
            score += 0.18
            if parsed_command.action in {"read", "show", "open", "list"}:
                score += 0.08

        if plugin_scores:
            best_plugin, best_value = max(
                plugin_scores.items(), key=lambda item: float(item[1])
            )
            if best_plugin != intent_plugin and float(best_value) > 1.25:
                score -= 0.09
                issues.append("plugin_score_disagrees_with_intent")

        return max(0.0, min(1.0, score)), issues

    def _classify_intent_category(
        self,
        text: str,
        parsed_command: ParsedCommand,
        plugin_scores: Dict[str, float],
    ) -> str:
        """Coarse intent category gate used to suppress tool execution for conversational turns."""
        text_lower = (text or "").lower()
        words = set(re.findall(r"\b[a-z']+\b", text_lower))

        conversation_cues = {
            "brainstorm",
            "architecture",
            "design",
            "ideas",
            "idea",
            "strategy",
            "discuss",
            "talk",
            "chat",
            "think",
            "plan",
            "why",
            "how",
        }
        tool_cues = {
            "email",
            "calendar",
            "meeting",
            "note",
            "notes",
            "file",
            "files",
            "weather",
            "forecast",
            "temperature",
            "reminder",
            "reminders",
            "time",
            "date",
            "create",
            "delete",
            "read",
            "write",
            "open",
            "search",
            "list",
            "show",
            "set",
            "cancel",
            "remove",
            "add",
        }

        has_conversation_cue = bool(words & conversation_cues) or "?" in text_lower
        explicit_tool_cues = bool(words & tool_cues)
        has_tool_cue = explicit_tool_cues

        action_hint = str(parsed_command.action or "").lower().strip()
        object_hint = str(parsed_command.object_type or "").lower().strip()
        tool_objects = {
            "note",
            "notes",
            "email",
            "calendar",
            "event",
            "reminder",
            "file",
            "files",
            "weather",
            "time",
            "date",
        }
        if action_hint not in {"unknown", "general"}:
            # Avoid false tool classification for conceptual conversation turns.
            if explicit_tool_cues or object_hint in tool_objects:
                has_tool_cue = True
        if plugin_scores:
            best_plugin, best_score = max(plugin_scores.items(), key=lambda item: float(item[1]))
            tool_plugins = set(self.plugin_registry.all_actions.keys()) | {
                "notes",
                "email",
                "calendar",
                "file_operations",
                "memory",
                "reminder",
                "system",
                "weather",
                "time",
            }
            if best_plugin in tool_plugins and float(best_score) >= 1.35:
                has_tool_cue = True

        design_conversation_cues = {
            "brainstorm",
            "architecture",
            "design",
            "ideas",
            "idea",
            "strategy",
            "plan",
        }
        if has_conversation_cue and bool(words & design_conversation_cues) and not explicit_tool_cues:
            return "conversation"

        if has_conversation_cue and not has_tool_cue:
            return "conversation"
        if has_tool_cue and not has_conversation_cue:
            return "tool"
        return "mixed"

    def _is_rich_conceptual_request(self, text: str) -> bool:
        """Detect high-signal conceptual prompts that should route to direct explanation."""
        low = str(text or "").lower().strip()
        if not low:
            return False

        tokens = re.findall(r"\b[a-z0-9']+\b", low)
        if len(tokens) < 6:
            return False

        task_patterns = (
            r"\blet'?s\s+imagine\b",
            r"\bhow\s+would\b",
            r"\bhow\s+can\s+i\s+create\b",
            r"\bhow\s+can\s+i\s+build\b",
            r"\bhow\s+would\s+i\s+build\b",
            r"\bhow\s+would\s+someone\s+build\b",
            r"\bwould\s+.+\s+be\s+built\b",
            r"\bhow\s+.+\s+be\s+created\b",
            r"\barchitecture\b",
        )
        has_task = any(re.search(pattern, low) for pattern in task_patterns)
        if not has_task:
            return False

        has_build_verb = bool(re.search(r"\b(create|created|build|built|make|made)\b", low))

        constraint_cues = {
            "no fiction",
            "non fiction",
            "non-fiction",
            "real world",
            "real-world",
            "today's technology",
            "todays technology",
            "with today's technology",
        }
        has_constraint = any(cue in low for cue in constraint_cues)
        if not has_constraint:
            return False

        subject_cues = {
            "assistant",
            "fictional inventor",
            "assistant",
            "ai",
            "technology",
            "system",
            "architecture",
            "foundations",
        }
        has_subject = any(cue in low for cue in subject_cues)

        if has_subject and has_constraint and has_build_verb:
            return True

        stop = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "for",
            "with",
            "in",
            "on",
            "how",
            "would",
            "today",
            "no",
            "fiction",
            "real",
            "world",
        }
        strong_terms = [t for t in tokens if len(t) >= 5 and t not in stop]

        return has_subject or len(set(strong_terms)) >= 3

    def _is_conceptual_build_architecture_prompt(self, text: str) -> bool:
        """Detect conceptual build/design questions that should use a dedicated system-design route."""
        low = str(text or "").lower().strip()
        if not low:
            return False
        if not self._is_rich_conceptual_request(low):
            return False

        has_build = bool(re.search(r"\b(create|created|build|built|make|made)\b", low))
        has_subject = any(
            cue in low
            for cue in (
                "ai",
                "assistant",
                "assistant",
                "system",
                "architecture",
            )
        )
        has_question_frame = bool(
            re.search(
                r"\b(how\s+can\s+i|how\s+would\s+i|how\s+would\s+someone|how\s+would|let'?s\s+imagine)\b",
                low,
            )
        )
        return has_build and has_subject and has_question_frame

    def _is_answerability_direct_question(self, text: str) -> bool:
        """Rule gate: direct answer for specific domain questions instead of clarification."""
        low = str(text or "").lower().strip()
        if not low:
            return False

        tokens = set(re.findall(r"\b[a-z0-9']+\b", low))
        question_terms = {"what", "which", "how", "why"}
        domain_terms = {"optimizer", "optimizers", "nlp", "model", "models", "training"}
        ambiguity_terms = {"something", "anything", "stuff", "idk"}

        has_question_term = bool(tokens & question_terms)
        is_question = ("?" in low) or bool(re.match(r"^\s*(what|which|how|why)\b", low))
        has_domain_content = bool(tokens & domain_terms)
        has_ambiguity_markers = bool(tokens & ambiguity_terms)

        return bool(
            has_question_term
            and is_question
            and has_domain_content
            and not has_ambiguity_markers
        )

    def _should_force_unknown_fallback(
        self,
        *,
        intent: str,
        confidence: float,
        plausibility: float,
        uncertainty: Optional[Dict[str, Any]],
        text: str = "",
    ) -> bool:
        """Decide whether to force a safe clarification fallback for unstable intent."""
        normalized_intent = (intent or "").lower()
        if normalized_intent.startswith("conversation:"):
            return False
        if normalized_intent == "learning:explanation_request":
            return False
        if self._is_rich_conceptual_request(text):
            return False
        if uncertainty and uncertainty.get("needs_clarification"):
            return True
        if confidence < self.UNKNOWN_FALLBACK_CONF_HARD:
            return True
        if (
            confidence < self.UNKNOWN_FALLBACK_CONF_SOFT
            and plausibility < self.UNKNOWN_FALLBACK_PLAUS_SOFT
        ):
            return True
        if plausibility < self.UNKNOWN_FALLBACK_PLAUS_HARD:
            return True
        return False

    # -------------------------------------------------------------------------
    # Semantic frame editing for follow-ups (Improvement 2)
    # -------------------------------------------------------------------------

    def _merge_followup_frame(
        self,
        text: str,
        parsed_command: ParsedCommand,
    ) -> Optional[Dict[str, Any]]:
        """
        When a short follow-up mutates only part of the previous frame (e.g.
        "and tomorrow?", "for my notes instead", "make it 5pm"), produce a
        merged frame dict that inherits unchanged slots from ``context.last_frame``
        and overwrites only the slot(s) that appear in *text*.

        Returns the merged frame dict, or None if no merge is applicable.

        Merge rules
        -----------
        * **Temporal slot** — if a time/date expression is detected in text,
          update the ``slots.date`` and/or ``slots.time`` entries.
        * **Target slot** — if text contains a domain-pivot (e.g. "in my notes",
          "for the calendar") update the ``target`` slot and (optionally) the
          frame name.
        * All other slots are inherited from ``last_frame["slots"]``.

        The merged frame is tagged ``"merged": true`` so downstream consumers
        know it is a synthesised result.
        """
        if self.context.last_frame is None:
            return None

        text_lower = text.lower().strip()
        last_frame = self.context.last_frame

        # Only apply merging for genuinely short follow-ups
        word_count = len(text_lower.split())
        if word_count > 10:
            return None

        # Clone the last frame, stripping stale result fields
        merged: Dict[str, Any] = {
            "name": last_frame.get("name"),
            "confidence": max(0.60, last_frame.get("confidence", 0.60) * 0.90),
            "keywords": list(last_frame.get("keywords", [])),
            "slots": dict(last_frame.get("slots", {})),
            "fill_confidence": last_frame.get("fill_confidence", 0.5),
            "merged": True,
        }
        mutated = False

        # ── Temporal mutation ─────────────────────────────────────────────────
        _temporal_tr = self.temporal.parse(text)
        temporal = _temporal_tr.as_dict() if _temporal_tr else None
        if temporal:
            if temporal.get("date"):
                merged["slots"]["date"] = temporal["date"]
                mutated = True
            if temporal.get("time"):
                merged["slots"]["time"] = temporal["time"]
                mutated = True

        # ── Target / domain mutation ──────────────────────────────────────────
        _target_pivots: Dict[str, str] = {
            "note": "notes",
            "notes": "notes",
            "memo": "notes",
            "email": "email",
            "calendar": "calendar",
            "reminder": "reminder",
        }
        for kw, domain in _target_pivots.items():
            if re.search(rf"\b{kw}\b", text_lower):
                if merged["slots"].get("target") != domain:
                    merged["slots"]["target"] = domain
                    mutated = True
                break

        if not mutated:
            return None

        logger.debug(
            "[FRAME_MERGE] Follow-up '%s' merged into frame '%s' with updated slots: %s",
            text[:60],
            merged["name"],
            {k: v for k, v in merged["slots"].items() if k in ("date", "time", "target")},
        )
        return merged

    def _build_uncertainty_prompt(
        self,
        route: RouteDecision,
        parsed: ParsedCommand,
        plugin_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """Build disambiguation metadata when intent confidence is weak."""
        if route.confidence >= self.ROUTE_UNCERTAINTY_THRESHOLD:
            return {}

        ranked_plugins = sorted(
            plugin_scores.items(), key=lambda item: item[1], reverse=True
        )
        options = [name for name, score in ranked_plugins[:3] if score > 0.5]
        if route.plugin not in options:
            options.insert(0, route.plugin)
        options = [option for option in options if option]

        question = "Could you clarify what you want me to do?"
        if options:
            question = f"Did you mean {', '.join(options[:2])}{' or ' + options[2] if len(options) > 2 else ''}?"

        return {
            "needs_clarification": True,
            "question": question,
            "candidate_plugins": options,
            "route_confidence": route.confidence,
            "parsed_action": parsed.action,
        }

    def _surface_segment(self, text: str) -> List[TokenSegment]:
        """Surface layer: split raw text into analyzable segments."""
        if not text:
            return []

        segments: List[TokenSegment] = []
        offset = 0
        for piece in re.split(r"[;\n]+", text):
            segment_text = piece.strip()
            if not segment_text:
                offset += len(piece) + 1
                continue

            start = text.find(segment_text, offset)
            end = start + len(segment_text)
            offset = end

            kind = "meta_command" if segment_text.startswith("/") else "utterance"
            if re.search(
                r"\b(?:plugin|note|email|calendar|music|task|reminder)s?\b",
                segment_text,
                re.IGNORECASE,
            ):
                kind = "plugin_phrase" if kind == "utterance" else kind

            segments.append(
                TokenSegment(text=segment_text, kind=kind, start_pos=start, end_pos=end)
            )

        return segments

    def _normalize_for_tokenizer(self, text: str) -> str:
        """Normalize user input before lexical tokenization."""
        normalized = (text or "").strip()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = _WAKE_WORD_PREFIX_RE.sub("", normalized, count=1).strip()

        contractions = {
            "what's": "what is",
            "i'm": "i am",
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "it's": "it is",
        }
        for source, target in contractions.items():
            normalized = re.sub(
                rf"\b{re.escape(source)}\b", target, normalized, flags=re.IGNORECASE
            )

        typo_map = {
            "notets": "notes",
            "nots": "notes",
            "emial": "email",
            "calender": "calendar",
        }
        for source, target in typo_map.items():
            normalized = re.sub(
                rf"\b{re.escape(source)}\b", target, normalized, flags=re.IGNORECASE
            )

        normalized = re.sub(r"\binside of\b", "in", normalized, flags=re.IGNORECASE)
        normalized = re.sub(
            r"\bi wanna\b", "i want to", normalized, flags=re.IGNORECASE
        )
        normalized = self._apply_noisy_channel_normalization(normalized)

        if self.tokenizer_profile == "strict":
            normalized = re.sub(
                r"\b(?:please|kindly)\b", "", normalized, flags=re.IGNORECASE
            )
            normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    def _lexical_tokenize(self, segments: List[TokenSegment]) -> List[RichToken]:
        """Lexical layer: tokenize segments into rich tokens with type/role/span metadata."""
        tokens: List[RichToken] = []

        action_words = self.command_vocabulary.get("verbs", set())
        object_words = self.command_vocabulary.get("objects", set())
        pronouns = {"this", "that", "it", "them", "those", "these", "one", "last"}
        ordinal_words = {
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
        }
        meta_words = self.command_vocabulary.get("meta", set())

        for segment in segments:
            for match in _TOKEN_PATTERN.finditer(segment.text):
                raw = match.group(0)
                norm = raw.lower()
                start = segment.start_pos + match.start()
                end = segment.start_pos + match.end()

                if raw.startswith('"') and raw.endswith('"'):
                    kind = "quoted_span"
                    role = "value"
                elif raw.startswith("#"):
                    kind = "hashtag"
                    role = "modifier"
                elif _ORDINAL_TOKEN_RE.fullmatch(norm):
                    kind = "ordinal"
                    role = "reference"
                elif raw.isdigit():
                    kind = "number"
                    role = "value"
                elif _ALPHA_TOKEN_RE.fullmatch(raw):
                    kind = "word"
                    if norm in action_words:
                        role = "action"
                    elif norm in ordinal_words:
                        role = "reference"
                        kind = "ordinal"
                    elif norm in object_words:
                        role = "object"
                    elif norm in pronouns:
                        role = "reference"
                        kind = "pronoun"
                    elif norm in {"today", "tomorrow", "yesterday", "week", "month"}:
                        role = "modifier"
                        kind = "date_like"
                    else:
                        role = "unknown"
                else:
                    kind = "symbol"
                    role = "meta" if raw in meta_words else "unknown"

                token = RichToken(
                    text=raw,
                    normalized=norm,
                    kind=kind,
                    role=role,
                    start_pos=start,
                    end_pos=end,
                    flags={
                        "is_ordinal": kind == "ordinal",
                        "is_reference": role == "reference",
                        "is_meta_command": segment.kind == "meta_command"
                        or norm in meta_words,
                    },
                )
                tokens.append(token)

        return tokens

    def _extract_semantic_hints(
        self, text: str, tokens: List[RichToken]
    ) -> ParsedCommand:
        """Semantic hint layer: derive structured command from token sequence."""
        token_text = [token.normalized for token in tokens if token.kind != "symbol"]
        lower = text.lower()
        parsed = ParsedCommand()

        if text.endswith("?") or (
            token_text
            and token_text[0]
            in {"what", "when", "where", "who", "why", "how", "do", "is", "are", "can"}
        ):
            parsed.sentence_type = "question"
        elif token_text and token_text[0] in self.command_vocabulary.get(
            "verbs", set()
        ):
            parsed.sentence_type = "imperative"

        if re.search(r"\b(do i have|is there)\b", lower) and re.search(
            r"\bnote|notes|list|lists\b", lower
        ):
            parsed.action = "query_exist"
            parsed.object_type = "note"
        elif re.search(r"\b(list|show)\b", lower) and re.search(
            r"\bnotes?|lists?\b", lower
        ):
            parsed.action = "list"
            parsed.object_type = "note"
        elif re.search(r"\b(read|open|show)\b", lower) and re.search(
            r"\bnotes?|lists?|it|that|this\b", lower
        ):
            parsed.action = "read"
            parsed.object_type = "note"
        elif re.search(r"\b(create|make|new|add)\b", lower) and re.search(
            r"\bnotes?|memo\b", lower
        ):
            parsed.action = "create"
            parsed.object_type = "note"
        elif (
            re.search(r"\b(add|append|put|include)\b", lower)
            and re.search(r"\bto\b", lower)
            and re.search(r"\bnotes?|lists?\b", lower)
        ):
            parsed.action = "append"
            parsed.object_type = "note"
        elif re.search(r"\b(send|compose|draft|reply)\b", lower) and re.search(
            r"\bemail|mail|inbox\b", lower
        ):
            parsed.action = "compose" if not re.search(r"\breply\b", lower) else "reply"
            parsed.object_type = "email"
        elif re.search(r"\b(read|open|show|list)\b", lower) and re.search(
            r"\bemail|mail|inbox\b", lower
        ):
            parsed.action = "read" if re.search(r"\b(read|open)\b", lower) else "list"
            parsed.object_type = "email"
        elif re.search(r"\b(schedule|create|add|book)\b", lower) and re.search(
            r"\bcalendar|event|meeting\b", lower
        ):
            parsed.action = "create"
            parsed.object_type = "calendar"
        elif re.search(r"\b(show|list|find|search)\b", lower) and re.search(
            r"\bcalendar|event|meeting|schedule\b", lower
        ):
            parsed.action = "list" if re.search(r"\b(show|list)\b", lower) else "search"
            parsed.object_type = "calendar"
        elif any(word in lower for word in _P1_WEATHER_KEYWORDS):
            parsed.action = "forecast" if any(word in lower for word in _P1_FORECAST_WORDS) else "current"
            parsed.object_type = "weather"

        title_match = re.search(
            r"(?:called|named|titled|about)\s+([a-z0-9\s'\-]+)$", lower
        )
        if not title_match:
            title_match = re.search(
                r"(?:read|open|show)\s+(?:the\s+)?([a-z0-9\s'\-]+?)\s+notes?\b", lower
            )
        if title_match:
            parsed.title_hint = title_match.group(1).strip(" .,!?")

        for token in tokens:
            if token.flags.get("is_reference"):
                parsed.references.append(
                    {
                        "text": token.normalized,
                        "start": token.start_pos,
                        "end": token.end_pos,
                    }
                )

        if re.search(r"\b(?:my|mine)\b", lower):
            parsed.modifiers["scope"] = "my"
        if re.search(r"\b(today|tomorrow|yesterday|this week|next week)\b", lower):
            parsed.modifiers["time"] = re.search(
                r"\b(today|tomorrow|yesterday|this week|next week)\b", lower
            ).group(1)
        if re.search(r"\b(low|medium|high|urgent)\b", lower):
            parsed.modifiers["priority"] = re.search(
                r"\b(low|medium|high|urgent)\b", lower
            ).group(1)

        return parsed

    def _compute_plugin_scores(
        self, tokens: List[RichToken], parsed: ParsedCommand
    ) -> Dict[str, float]:
        """Compute plugin routing scores from token features and parsed command."""
        scores = {
            "notes": 0.0,
            "email": 0.0,
            "calendar": 0.0,
            "weather": 0.0,
            "system": 0.0,
            "conversation": 0.2,
        }

        normalized = [token.normalized for token in tokens]
        token_counts = Counter(normalized)
        normalized_set = set(token_counts.keys())
        bigrams = {
            f"{normalized[i]} {normalized[i + 1]}" for i in range(len(normalized) - 1)
        }
        scores["notes"] += 1.2 * sum(token_counts[word] for word in _NOTE_TERMS)
        scores["email"] += 1.2 * sum(token_counts[word] for word in _EMAIL_TERMS)
        scores["calendar"] += 1.2 * sum(token_counts[word] for word in _CALENDAR_TERMS)
        scores["weather"] += 1.35 * sum(token_counts[word] for word in _WEATHER_TERMS)
        scores["system"] += 1.2 * sum(token_counts[word] for word in _SYSTEM_TERMS)
        if _WEATHER_FORECAST_TERMS & normalized_set:
            scores["weather"] += 0.8
        if (
            _WEATHER_EVENT_TERMS & normalized_set
            and _WEATHER_FUTURE_TERMS & normalized_set
        ):
            scores["weather"] += 0.9

        if parsed.object_type == "note":
            scores["notes"] += 1.5
        if parsed.object_type == "email":
            scores["email"] += 1.5
        if parsed.object_type == "calendar":
            scores["calendar"] += 1.5
        if parsed.object_type == "weather":
            scores["weather"] += 1.6
        if (
            parsed.action in {"read", "append", "create", "list", "query_exist"}
            and parsed.object_type == "note"
        ):
            scores["notes"] += 1.0
        if (
            parsed.action in {"compose", "read", "list", "search", "reply"}
            and parsed.object_type == "email"
        ):
            scores["email"] += 1.0
        if (
            parsed.action in {"create", "list", "search"}
            and parsed.object_type == "calendar"
        ):
            scores["calendar"] += 1.0
        if (
            parsed.action in {"current", "forecast"}
            and parsed.object_type == "weather"
        ):
            scores["weather"] += 1.0
        if parsed.references:
            scores["notes"] += 0.6

        if "read it" in bigrams or "show it" in bigrams:
            scores["notes"] += 0.5
            if self.context.last_intent and self.context.last_intent.startswith(
                "email:"
            ):
                scores["email"] += 0.5
        if "system status" in bigrams:
            scores["system"] += 0.8

        if parsed.sentence_type == "question":
            scores["conversation"] += 0.15
        if self.context.last_plugin and self.context.last_plugin in scores:
            scores[self.context.last_plugin] += 0.35

        action_bias = {
            "query_exist": "notes",
            "append": "notes",
            "compose": "email",
            "reply": "email",
            "forecast": "weather",
            "current": "weather",
            "status": "system",
        }
        preferred = action_bias.get(parsed.action)
        if preferred:
            scores[preferred] += 0.45

        # Strong weather language should outrank weak lexical collisions in notes/email/calendar.
        weather_signal_strength = sum(token_counts[word] for word in _WEATHER_TERMS)
        has_explicit_non_weather_target = bool(normalized_set & _NON_WEATHER_TARGET_TERMS)
        if weather_signal_strength >= 1 and not has_explicit_non_weather_target:
            scores["notes"] *= 0.35
            scores["email"] *= 0.45
            scores["calendar"] *= 0.45
        if weather_signal_strength >= 2:
            scores["notes"] *= 0.20
            scores["email"] *= 0.30
            scores["calendar"] *= 0.30

        if self.tokenizer_profile == "strict":
            for key in ("email", "calendar", "weather", "system"):
                if scores[key] < 1.0:
                    scores[key] *= 0.7

        if self.tokenizer_profile == "llm-assisted":
            for key in ("notes", "email", "calendar", "weather", "system"):
                scores[key] *= 1.05

        dynamic_scores = self.plugin_registry.score_all(
            text=" ".join(normalized),
            tokens=normalized,
        )
        for plugin_name, plugin_score in dynamic_scores.items():
            scores[plugin_name] = float(scores.get(plugin_name, 0.0) + plugin_score)

        return scores

    async def aprocess(self, text: str, use_context: bool = True) -> ProcessedQuery:
        return await asyncio.to_thread(self.process, text, use_context)

    async def aclassify(self, text: str, use_context: bool = True) -> str:
        result = await self.aprocess(text=text, use_context=use_context)
        return str(result.intent)

    async def aextract_slots(
        self,
        text: str,
        intent: str,
        entities: Dict[str, List[Entity]],
    ) -> Dict[str, Slot]:
        return await asyncio.to_thread(self.slot_filler.extract_slots, text, intent, entities)

    def debug_tokenizer(self, text: str) -> Dict[str, Any]:
        """Development HUD for tokenizer introspection."""
        normalized = self._normalize_for_tokenizer(self._clean_text(text))
        segments = self._surface_segment(normalized)
        tokens = self._lexical_tokenize(segments)
        parsed = self._extract_semantic_hints(normalized, tokens)
        scores = self._compute_plugin_scores(tokens, parsed)
        return {
            "raw_text": text,
            "normalized_text": normalized,
            "segments": [segment.__dict__ for segment in segments],
            "tokens": [token.__dict__ for token in tokens],
            "parsed_command": parsed.__dict__,
            "plugin_scores": scores,
        }

    def _load_learned_corrections(self) -> Dict[str, str]:
        """Load learned corrections to override pattern matching"""
        try:
            corrections_file = Path("memory/curated_patterns.json")
            if corrections_file.exists():
                with open(corrections_file, "r", encoding="utf-8") as f:
                    patterns = json.load(f)

                if isinstance(patterns, dict) and "corrections" in patterns:
                    learned = {}
                    for correction in patterns["corrections"]:
                        if (
                            "user_input" in correction
                            and "expected_intent" in correction
                        ):
                            # Map user input to learned intent for fast override
                            learned[correction["user_input"].lower()] = correction[
                                "expected_intent"
                            ]

                    if learned:
                        logger.info(
                            f"Loaded {len(learned)} learned corrections into NLP processor"
                        )
                    return learned
        except Exception as e:
            logger.debug(f"Could not load learned corrections: {e}")

        return {}

        logger.info(
            "[OK] NLPProcessor initialized with advanced semantic understanding"
        )

    def _ensure_correction_embeddings(self) -> None:
        """Lazily build a normalised embedding matrix over all learned-correction keys."""
        if self._correction_embeddings is not None or not self.learned_corrections:
            return
        classifier = self._ensure_semantic_classifier()
        if classifier is None or not hasattr(classifier, "_model") or classifier._model is None:
            return
        try:
            import numpy as np
            keys = list(self.learned_corrections.keys())
            embs = classifier._model.encode(keys, normalize_embeddings=True)
            self._correction_keys = keys
            self._correction_embeddings = embs
            logger.info(
                "[LEARNED-SEM] Built correction embedding index (%d entries)", len(keys)
            )
        except Exception as exc:
            logger.debug("[LEARNED-SEM] Could not build correction embeddings: %s", exc)

    def _try_semantic_correction(self, text: str) -> Optional[Tuple[str, float]]:
        """Return (intent, confidence) if a learned correction matches semantically.

        Uses cosine similarity over sentence-transformer embeddings.  Only fires
        when similarity > 0.90 to avoid spurious overrides.
        """
        if not self.learned_corrections:
            return None
        self._ensure_correction_embeddings()
        if self._correction_embeddings is None or self._correction_keys is None:
            return None
        classifier = self._ensure_semantic_classifier()
        if classifier is None or not hasattr(classifier, "_model") or classifier._model is None:
            return None
        try:
            import numpy as np
            query_emb = classifier._model.encode(
                [text.lower()], normalize_embeddings=True
            )
            sims = (self._correction_embeddings @ query_emb.T).flatten()
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim > 0.90:
                matched_key = self._correction_keys[best_idx]
                learned_intent = self.learned_corrections[matched_key]
                logger.info(
                    "[LEARNED-SEM] '%s' ~ '%s' (%.2f) -> %s",
                    text, matched_key, best_sim, learned_intent,
                )
                return learned_intent, min(0.93, 0.80 + best_sim * 0.13)
        except Exception as exc:
            logger.debug("[LEARNED-SEM] Semantic lookup failed: %s", exc)
        return None

    def process(self, text: str, use_context: bool = True) -> ProcessedQuery:
        """
        Complete NLP processing pipeline

        Steps:
        1. Check learned corrections (HIGHEST PRIORITY)
        2. Resolve coreferences (if context enabled)
        3. Detect intent (semantic-first, regex fallback)
        4. Extract entities (domain-specific + general)
        5. Fill slots (structured data extraction)
        6. Analyze sentiment & emotions
        7. Detect urgency
        8. Extract keywords
        9. Update conversation context
        """
        turn_budget = self.foundation_layers.new_budget()

        # Step 1: Coreference resolution
        if use_context:
            # Check if using advanced coreference with ambiguity detection
            if hasattr(self.coref_resolver, "_engine"):
                # Using AdvancedCoref through compat wrapper
                resolved_result = self.coref_resolver._engine.resolve(
                    text,
                    self.context.__dict__ if hasattr(self.context, "__dict__") else {},
                )
                resolved_text = resolved_result.text

                # Track ambiguity detection (P0-2)
                if (
                    self.feature_flags
                    and self.feature_flags.is_enabled("nlp_ambiguity_resolver")
                    and resolved_result.needs_clarification
                ):
                    from ai.infrastructure.metrics_collector import MetricsCollector

                    metrics = MetricsCollector()
                    metrics.track_ambiguity_detection(
                        resolved_result.entity_type or "unknown",
                        len(resolved_result.candidates),
                    )
                    # Store ambiguity info for clarification prompts
                    if hasattr(self.context, "pending_clarification"):
                        self.context.pending_clarification = {
                            "type": "ambiguity",
                            "candidates": resolved_result.candidates,
                            "entity_type": resolved_result.entity_type,
                            "confidence": resolved_result.confidence,
                        }
            else:
                # Fallback to simple resolve
                resolved_text = self.coref_resolver.resolve(text, self.context)
        else:
            resolved_text = text

        # Step 1.5: Fingerprint lookup — use cached parse as a strong prior
        _fp_prior_intent: Optional[str] = None
        _fp_prior_conf: float = 0.0
        if hasattr(self, "_fp_store") and self._fp_store is not None:
            _fp_hit = self._fp_store.lookup(resolved_text)
            if _fp_hit and _fp_hit.confidence >= 0.80:
                _fp_prior_intent = _fp_hit.intent
                _fp_prior_conf = _fp_hit.confidence
                logger.debug(
                    "[FINGERPRINT] Prior: %s (%.2f)", _fp_prior_intent, _fp_prior_conf
                )

        # Step 2: Clean + normalize for tokenizer layers
        clean_text = self._clean_text(resolved_text)
        clean_text = self.foundation_layers.normalize_input(clean_text)
        normalized_text = self._normalize_for_tokenizer(clean_text)
        contextual_text = normalized_text
        if self.conversation_memory is not None:
            contextual_text = self.conversation_memory.build_contextual_input(normalized_text)

        # Step 3: Surface + lexical + semantic hint layers
        segments = self._surface_segment(normalized_text)
        rich_tokens = self._lexical_tokenize(segments)
        parsed_command = self._extract_semantic_hints(normalized_text, rich_tokens)
        plugin_scores = self._compute_plugin_scores(rich_tokens, parsed_command)
        retrieval_route = self._retrieval_first_parse(rich_tokens)
        intent_category = self._classify_intent_category(
            normalized_text,
            parsed_command,
            plugin_scores,
        )
        parsed_command.modifiers["intent_category"] = intent_category
        tokens = [
            token.normalized
            for token in rich_tokens
            if token.kind
            in {
                "word",
                "number",
                "ordinal",
                "pronoun",
                "hashtag",
                "quoted_span",
                "date_like",
            }
        ]

        # Step 4: Check learned corrections FIRST (highest priority)
        # Also check the original (pre-coref-resolution) text as a fallback, so coref
        # changes (e.g. "it" in "what time is it?" resolved to a note title) don't prevent
        # a known correction from firing.
        route: Optional[RouteDecision] = None  # may be set in the else branch below
        semantic_intent: Optional[Tuple[str, float]] = None
        weighted_candidates: List[Tuple[str, float]] = []
        _correction_key = normalized_text.lower()
        _orig_key = text.strip().lower()
        _correction_intent = (
            self.learned_corrections.get(_correction_key)
            if self.learned_corrections else None
        ) or (
            self.learned_corrections.get(_orig_key)
            if self.learned_corrections and _orig_key != _correction_key else None
        )
        _send_followup_correction_guard = bool(
            re.search(r"\bsend\b", _correction_key)
            and re.search(r"\b(her|him|them)\b", _correction_key)
        )
        if _send_followup_correction_guard:
            _correction_intent = None
        if _correction_intent:
            learned_intent = _correction_intent
            logger.info(
                f"[LEARNED] Using correction for '{normalized_text}' -> {learned_intent}"
            )
            intent = learned_intent
            intent_confidence = 0.95  # High confidence for learned patterns
        else:
            # ── Reminder early override: runs BEFORE retrieval_first_parse so
            # no context-based short-query logic can shadow reminder intents. ──
            _tl = normalized_text.lower()
            _reminder_intent = None
            if any(
                phrase in _tl
                for phrase in [
                    "remind me",
                    "set a reminder",
                    "add a reminder",
                    "create a reminder",
                    "alert me",
                    "notify me when",
                    "don't let me forget",
                    "do not let me forget",
                    "dont let me forget",
                ]
            ):
                _reminder_intent = "reminder:set"
            elif any(
                phrase in _tl
                for phrase in [
                    "my reminders",
                    "what reminders",
                    "show reminders",
                    "list reminders",
                    "any reminders",
                    "upcoming reminders",
                    "pending reminders",
                ]
            ) or ("reminder" in _tl and "do i have" in _tl):
                _reminder_intent = "reminder:list"
            elif (
                any(w in _tl for w in ["cancel", "delete", "remove"])
                and "reminder" in _tl
            ):
                _reminder_intent = "reminder:cancel"

            _send_followup_intent = None
            _tl_words = set(_tl.split())
            if (
                "send" in _tl_words
                and any(p in _tl_words for p in {"her", "him", "them"})
            ):
                _send_followup_intent = "email:compose"

            if _reminder_intent:
                route = RouteDecision(
                    intent=_reminder_intent,
                    confidence=0.95,
                    plugin="reminder",
                    action=_reminder_intent.split(":", 1)[1],
                    trace={"source": "reminder_early_override"},
                )
                intent = _reminder_intent
                intent_confidence = 0.95
            elif _send_followup_intent:
                route = RouteDecision(
                    intent=_send_followup_intent,
                    confidence=0.90,
                    plugin="email",
                    action="compose",
                    trace={"source": "send_followup_override"},
                )
                intent = _send_followup_intent
                intent_confidence = 0.90
            # Step 5: Intent detection (retrieval-first + semantic + calibrated routing)
            elif retrieval_route:
                intent, intent_confidence = retrieval_route
                route = RouteDecision(
                    intent=intent,
                    confidence=intent_confidence,
                    plugin=intent.split(":", 1)[0],
                    action=intent.split(":", 1)[1] if ":" in intent else "general",
                    trace={"source": "retrieval_first"},
                )
            else:
                weighted_candidates = self._build_weighted_parse(
                    parsed_command, plugin_scores, normalized_text
                )
                shallow_confidence = max(plugin_scores.values()) if plugin_scores else 0.0
                run_deep_stage = self.foundation_layers.should_run_deep_stage(
                    turn_budget, shallow_confidence=shallow_confidence
                )
                if parsed_command.sentence_type == "question":
                    run_deep_stage = True
                parsed_command.modifiers["staged_inference"] = {
                    "deep_stage_enabled": run_deep_stage,
                    "elapsed_ms": round(turn_budget.elapsed_ms(), 3),
                    "shallow_confidence": round(float(shallow_confidence), 3),
                }
                if run_deep_stage:
                    semantic_intent = self._detect_intent_semantic(
                        contextual_text,
                        parsed_command=parsed_command,
                        plugin_scores=plugin_scores,
                        return_structured=False,
                    )
                else:
                    semantic_intent = None

                if self.implicit_intent_detector is not None:
                    _implicit_matches = self.implicit_intent_detector.detect(
                        normalized_text,
                        recent_topic=(
                            self.conversation_memory.latest_topic()
                            if self.conversation_memory is not None
                            else ""
                        ),
                    )
                    if _implicit_matches:
                        parsed_command.modifiers["implicit_intents"] = [
                            m.as_dict() for m in _implicit_matches[:3]
                        ]
                        for m in _implicit_matches[:2]:
                            weighted_candidates.append((m.intent, float(m.confidence)))

                route = self._calibrate_route_decision(
                    weighted_candidates, plugin_scores, semantic_intent, parsed_command
                )
                route = self._maybe_apply_llm_intent_fallback(
                    normalized_text=normalized_text,
                    route=route,
                    weighted_candidates=weighted_candidates,
                    parsed_command=parsed_command,
                )
                intent = route.intent
                intent_confidence = route.confidence

            intent, intent_confidence = self.route_coordinator.apply_initial_routing_policy(
                intent=intent,
                intent_confidence=float(intent_confidence or 0.0),
                route=route,
                parsed_command=parsed_command,
                plugin_scores=plugin_scores,
                semantic_intent=semantic_intent,
                weighted_candidates=weighted_candidates,
                build_uncertainty_prompt=self._build_uncertainty_prompt,
                build_top_intent_candidates=self._build_top_intent_candidates,
                validate_intent_plausibility=self._validate_intent_plausibility,
                should_force_unknown_fallback=self._should_force_unknown_fallback,
                normalized_text=normalized_text,
            )

        intent, intent_confidence = self.route_coordinator.apply_category_gate(
            intent=intent,
            intent_confidence=float(intent_confidence or 0.0),
            intent_category=intent_category,
            parsed_command=parsed_command,
            normalized_text=normalized_text,
            previous_intent=str(self.context.last_intent or ""),
        )

        self.route_coordinator.ensure_metadata(
            parsed_command=parsed_command,
            route=route,
            weighted_candidates=weighted_candidates,
            semantic_intent=semantic_intent,
            plugin_scores=plugin_scores,
            build_top_intent_candidates=self._build_top_intent_candidates,
            validate_intent_plausibility=self._validate_intent_plausibility,
            normalized_text=normalized_text,
            intent=intent,
        )

        clarification_state = self.foundation_layers.clarification_policy(
            plugin_scores=plugin_scores,
            confidence=float(intent_confidence or 0.0),
        )
        parsed_command.modifiers["clarification_policy"] = clarification_state

        authorization = self.foundation_layers.authorization_policy(
            intent=intent,
            text=normalized_text,
        )
        parsed_command.modifiers["authorization"] = authorization
        if not authorization.get("allowed", True):
            intent = "conversation:clarification_needed"
            intent_confidence = min(float(intent_confidence or 0.0), 0.45)
            parsed_command.modifiers["tool_execution_disabled"] = True

        # Fingerprint prior: if we have a cached high-confidence parse, use it as a boost
        if (
            _fp_prior_intent
            and _fp_prior_intent == intent
            and _fp_prior_conf > intent_confidence
        ):
            intent_confidence = min(0.97, _fp_prior_conf)
            logger.debug(
                "[FINGERPRINT] Boosted confidence %.2f → %.2f",
                intent_confidence,
                _fp_prior_conf,
            )
        elif _fp_prior_intent and _fp_prior_conf >= 0.85 and intent_confidence < 0.70:
            # Prior disagrees but is very confident — override
            logger.info(
                "[FINGERPRINT] Prior override: %s (%.2f) → %s (%.2f)",
                intent,
                intent_confidence,
                _fp_prior_intent,
                _fp_prior_conf,
            )
            intent = _fp_prior_intent
            intent_confidence = _fp_prior_conf

        # ── Advanced NLP: Semantic Frame Parser + Probabilistic Slot Filler ──
        frame_result = None
        filled_slots = None
        if self.frame_parser is not None:
            _ctx_dict = {
                "last_plugin": self.context.last_plugin,
                "last_intent": self.context.last_intent or "",
                "last_note_title": (
                    str(self.context.mentioned_notes[-1])
                    if self.context.mentioned_notes
                    else None
                ),
                "last_entities": self.context.last_entities,
                "turn_index": self.context.turn_index,
                "entity_chain": (
                    self.dialogue_memory.entity_chain_dict()
                    if self.dialogue_memory
                    else []
                ),
            }
            frame_result = self.frame_parser.parse(normalized_text, context=_ctx_dict)
            if frame_result is not None and frame_result.confidence > 0.50:
                filled_slots = self.prob_slot_filler.fill(
                    text=normalized_text,
                    frame_name=frame_result.frame_name,
                    context=_ctx_dict,
                    frame_slot_evidence=frame_result.slot_evidence,
                )
                # Override routing if frame is more confident.
                # Phase 1 high-confidence results are authoritative — do NOT override.
                _phase1_locked = bool(
                    route is not None
                    and hasattr(route, "trace")
                    and route.trace.get("source") == "phase1_authoritative"
                )
                if not _phase1_locked and frame_result.confidence > intent_confidence + 0.07:
                    logger.info(
                        "[FRAME] Override route %s (%.2f) → frame %s (%.2f)",
                        intent,
                        intent_confidence,
                        frame_result.frame_name,
                        frame_result.confidence,
                    )
                    intent = f"{frame_result.plugin}:{frame_result.action}"
                    intent_confidence = min(0.97, frame_result.confidence)
                parsed_command.modifiers["frame"] = {
                    "name": frame_result.frame_name,
                    "confidence": frame_result.confidence,
                    "keywords": frame_result.matched_keywords,
                    "slots": filled_slots.as_dict() if filled_slots else {},
                    "fill_confidence": (
                        filled_slots.fill_confidence if filled_slots else 0.0
                    ),
                }
            elif retrieval_route is not None:
                # ── Semantic frame editing for short follow-ups ───────────────
                # Even when the frame parser finds no strong match for a terse
                # follow-up ("and tomorrow?", "for my notes", "make it 5pm"),
                # we can synthesize a merged frame by mutating only the changed
                # slot(s) from context.last_frame.
                _merged = self._merge_followup_frame(normalized_text, parsed_command)
                if _merged:
                    parsed_command.modifiers["frame"] = _merged

        # ── Foundation 2: single follow-up authority path ───────────────────
        # Apply follow-up inheritance in one place after baseline routing and
        # frame adjustments, so later stages don't duplicate topic carry-over.
        _followup_result = self.followup_resolver.resolve(
            user_input=normalized_text,
            nlp_intent=intent,
            nlp_confidence=float(intent_confidence or 0.0),
            last_intent=self.context.last_intent,
            conversation_topics=None,
            perception_followup_domain=None,
            turn_distance=0,
        )
        parsed_command.modifiers["followup_resolution"] = {
            "was_followup": _followup_result.was_followup,
            "domain": _followup_result.domain,
            "reason": _followup_result.reason,
            "resolved_intent": _followup_result.resolved_intent,
            "confidence": float(_followup_result.confidence),
        }
        if _followup_result.was_followup:
            intent = _followup_result.resolved_intent
            intent_confidence = max(
                float(intent_confidence or 0.0),
                float(_followup_result.confidence),
            )
            if intent.startswith(("notes:", "email:", "calendar:", "reminder:", "weather:")):
                parsed_command.modifiers["unknown_intent_fallback"] = False
                parsed_command.modifiers.pop("disambiguation", None)
            route = RouteDecision(
                intent=intent,
                confidence=float(intent_confidence),
                plugin=intent.split(":", 1)[0] if ":" in intent else "conversation",
                action=intent.split(":", 1)[1] if ":" in intent else "general",
                trace={
                    "source": "followup_resolver",
                    "reason": _followup_result.reason,
                    "domain": _followup_result.domain,
                },
            )

        _pending_slot = self.context.pending_clarification if hasattr(self.context, "pending_clarification") else {}
        _pending_slot_type = str(
            (_pending_slot or {}).get("slot_type")
            or (_pending_slot or {}).get("type")
            or ""
        ).strip().lower()
        if isinstance(_pending_slot, dict) and _pending_slot_type in {"help_narrowing", "narrowing"}:
            _slot_text = normalized_text.strip().strip(".!,;:")
            _slot_tokens = re.findall(r"\b[a-z0-9']+\b", _slot_text.lower())
            _has_action_word = bool(
                re.search(
                    r"\b(open|launch|play|send|create|make|delete|remove|search|find|show|list|read|write|edit|update|schedule|set|remind)\b",
                    _slot_text.lower(),
                )
            )
            _social_only = set(_slot_tokens).issubset(
                {"hi", "hello", "hey", "thanks", "thank", "ok", "okay", "sure", "bye", "goodbye"}
            )
            _looks_like_help_opener = bool(
                re.search(
                    r"\b(i\s+need\s+help|can\s+you\s+help|help\s+me|help\s+with\s+this|i\s+am\s+(?:a\s+)?beginner|i'?m\s+(?:a\s+)?beginner|want\s+an\s+explanation)\b",
                    _slot_text.lower(),
                )
            )
            _expected_shape = str(_pending_slot.get("expected_answer_shape") or "").strip().lower()
            _ordinal_selected = ""
            if re.search(r"\b(second|2nd|two)\b", _slot_text.lower()):
                _ordinal_selected = "second"
            elif re.search(r"\b(third|3rd|three)\b", _slot_text.lower()):
                _ordinal_selected = "third"
            elif re.search(r"\b(first|1st)\b", _slot_text.lower()):
                _ordinal_selected = "first"
            elif len(_slot_tokens) == 1 and _slot_tokens[0] == "one":
                _ordinal_selected = "first"

            _shape_allows_fill = True
            if _expected_shape == "single_token":
                _shape_allows_fill = len(_slot_tokens) == 1
            elif _expected_shape in {"short_topic_or_subdomain", "short_disambiguation_or_selection", "ordinal_or_short_phrase"}:
                _shape_allows_fill = (1 <= len(_slot_tokens) <= 5)

            if (
                _slot_text
                and "?" not in _slot_text
                and 1 <= len(_slot_tokens) <= 5
                and not _has_action_word
                and not _social_only
                and not _looks_like_help_opener
                and _shape_allows_fill
            ):
                intent = "conversation:clarification_needed"
                intent_confidence = max(float(intent_confidence or 0.0), 0.84)
                _slot_name = str(_pending_slot.get("slot") or "project_subdomain")
                parsed_command.modifiers["pending_slot_followup"] = {
                    "filled": True,
                    "slot": _slot_name,
                    "value": _slot_text,
                    "parent_topic": str(_pending_slot.get("parent_topic") or ""),
                    "reason": "pending_help_narrowing_slot",
                    "expected_answer_shape": _expected_shape,
                }
                if _ordinal_selected:
                    parsed_command.modifiers["pending_slot_followup"]["selected_reference"] = _ordinal_selected
                    parsed_command.modifiers["selected_object_reference"] = _ordinal_selected
                parsed_command.modifiers["followup_resolution"] = {
                    "was_followup": True,
                    "domain": "conversation",
                    "reason": "pending_help_narrowing_slot",
                    "resolved_intent": intent,
                    "confidence": float(intent_confidence),
                }
                route = RouteDecision(
                    intent=intent,
                    confidence=float(intent_confidence),
                    plugin="conversation",
                    action="clarification_needed",
                    trace={"source": "pending_help_slot"},
                )

        if isinstance(_pending_slot, dict) and _pending_slot_type in {"route_choice", "help_route_choice"}:
            _choice_text = normalized_text.strip().lower()
            _choice_tokens = re.findall(r"\b[a-z0-9']+\b", _choice_text)
            _compact_choice = len(_choice_tokens) <= 4
            _choice = ""
            if re.search(r"\b(explanation|explain|walk\s*through|teach|how|why)\b", _choice_text):
                _choice = "explanation"
            elif re.search(r"\b(direct\s+action|action|do\s+it|execute|perform|run\s+it)\b", _choice_text):
                _choice = "direct_action"
            elif re.search(r"\b(quick\s+search|search|look\s*up|find\s+info|web|online)\b", _choice_text):
                _choice = "quick_search"

            if _choice and _compact_choice and not self._is_rich_conceptual_request(_choice_text):
                intent = "conversation:clarification_needed"
                intent_confidence = max(float(intent_confidence or 0.0), 0.88)
                parsed_command.modifiers["pending_slot_followup"] = {
                    "filled": True,
                    "slot": "route_choice",
                    "value": _choice,
                    "parent_topic": str(_pending_slot.get("parent_topic") or "conversation"),
                    "parent_request": str(_pending_slot.get("parent_request") or ""),
                    "parent_intent": str(_pending_slot.get("parent_intent") or "conversation:help"),
                    "reason": "pending_route_choice_slot",
                    "expected_answer_shape": "single_token",
                }
                parsed_command.modifiers["route_choice"] = _choice
                parsed_command.modifiers["followup_resolution"] = {
                    "was_followup": True,
                    "domain": "conversation",
                    "reason": "pending_route_choice_slot",
                    "resolved_intent": intent,
                    "confidence": float(intent_confidence),
                }
                route = RouteDecision(
                    intent=intent,
                    confidence=float(intent_confidence),
                    plugin="conversation",
                    action="clarification_needed",
                    trace={"source": "pending_route_choice_slot"},
                )

        if self._is_rich_conceptual_request(normalized_text) and intent == "conversation:clarification_needed":
            intent = (
                "learning:system_design"
                if self._is_conceptual_build_architecture_prompt(normalized_text)
                else "conversation:question"
            )
            intent_confidence = max(float(intent_confidence or 0.0), 0.78)
            parsed_command.modifiers["pending_unknown_fallback"] = False
            parsed_command.modifiers["unknown_intent_fallback"] = False
            parsed_command.modifiers.pop("disambiguation", None)
            parsed_command.modifiers.pop("pending_slot_followup", None)
            if isinstance(route, RouteDecision):
                route.intent = intent
                route.confidence = float(intent_confidence)
                if intent.startswith("learning:"):
                    route.plugin = "learning"
                    route.action = intent.split(":", 1)[1]
                else:
                    route.plugin = "conversation"
                    route.action = "question"
                route.trace = {
                    **(route.trace or {}),
                    "source": "rich_conceptual_override",
                }

        if self._is_answerability_direct_question(normalized_text):
            parsed_command.modifiers["allow_direct_answer"] = True
            parsed_command.modifiers["block_clarification"] = True
            parsed_command.modifiers["answerability_gate"] = {
                "matched": True,
                "reason": "specific_domain_question",
            }
            if intent in {
                "conversation:clarification_needed",
                "conversation:question",
                "conversation:general",
                "vague_question",
                "vague_request",
                "vague_temporal_question",
            }:
                intent = "learning:explanation_request"
                intent_confidence = max(float(intent_confidence or 0.0), 0.90)
                parsed_command.modifiers["pending_unknown_fallback"] = False
                parsed_command.modifiers["unknown_intent_fallback"] = False
                parsed_command.modifiers.pop("disambiguation", None)
                if isinstance(route, RouteDecision):
                    route.intent = intent
                    route.confidence = float(intent_confidence)
                    route.plugin = "learning"
                    route.action = "explanation_request"
                    route.trace = {
                        **(route.trace or {}),
                        "source": "answerability_gate_override",
                    }

        # ── Dialogue-state gate ────────────────────────────────────────────────
        # If Alice is waiting for the user to disambiguate (dialogue_state ==
        # "clarifying") and the current response looks like a clarification
        # answer (an ordinal, yes/no, pronoun, etc.) rather than a new command,
        # map it to the pending candidate plugin instead of a raw NLP intent.
        _clarif_resolution_cues = (
            "that one", "the first", "first one", "second one", "third one",
            "yes", "no", "neither", "that", "option",
        )
        if (
            self.context.dialogue_state == "clarifying"
            and self.context.pending_clarification
            and intent_confidence < 0.70
            and any(cue in normalized_text.lower() for cue in _clarif_resolution_cues)
        ):
            _pending_plugins = self.context.pending_clarification.get("candidate_plugins", [])
            if _pending_plugins:
                _tl_low = normalized_text.lower()
                _idx = 0
                if any(w in _tl_low for w in ["second", " 2 ", "two"]):
                    _idx = 1
                elif any(w in _tl_low for w in ["third", " 3 ", "three"]):
                    _idx = 2
                _sel = _pending_plugins[min(_idx, len(_pending_plugins) - 1)]
                intent = f"{_sel}:general"
                intent_confidence = 0.78
                parsed_command.modifiers["selected_object_reference"] = f"candidate_plugin:{_sel}"
                route = RouteDecision(
                    intent=intent,
                    confidence=intent_confidence,
                    plugin=_sel,
                    action="general",
                    trace={"source": "dialogue_state_gate"},
                )
                logger.debug("[DIALOGUE_GATE] Clarification resolved -> %s", intent)

        # ── Semantic learned-correction fallback ──────────────────────────────
        # If main intent confidence is low, attempt a nearest-neighbour lookup
        # over the learned_corrections embedding index.  Only overrides when
        # cosine similarity > 0.90 (see _try_semantic_correction).
        _followup_locked = bool(
            route is not None
            and isinstance(getattr(route, "trace", None), dict)
            and route.trace.get("source") == "followup_resolver"
        )
        if intent_confidence < 0.70 and self.learned_corrections and not _followup_locked:
            _sem = self._try_semantic_correction(normalized_text)
            if _sem:
                _sem_intent, _sem_conf = _sem
                logger.info(
                    "[LEARNED-SEM] Override %s (%.2f) -> %s (%.2f)",
                    intent, intent_confidence, _sem_intent, _sem_conf,
                )
                intent = _sem_intent
                intent_confidence = _sem_conf

        # ── Compound intent detection ─────────────────────────────────────────
        # "create a note AND remind me tomorrow" -> primary intent + secondary_intents
        # Uses FrameParser.parse_compound() when available for structured splitting;
        # falls back to the lightweight semantic splitter.
        _compound_sep = re.compile(r"\s+(?:and|then|also)\s+", re.IGNORECASE)
        _compound_parts = _compound_sep.split(normalized_text.strip())
        if len(_compound_parts) > 1:
            # Prefer structured compound parsing via the frame parser
            if self.frame_parser is not None and hasattr(self.frame_parser, 'parse_compound'):
                _frame_ctx = {
                    "last_plugin": parsed_command.modifiers.get("last_plugin"),
                    "last_intent": parsed_command.modifiers.get("last_intent", ""),
                }
                _compound_frames = self.frame_parser.parse_compound(
                    normalized_text.strip(), context=_frame_ctx
                )
                if len(_compound_frames) >= 2:
                    parsed_command.modifiers["compound_frames"] = [
                        {
                            "frame_name": r.frame_name,
                            "plugin": r.plugin,
                            "action": r.action,
                            "confidence": r.confidence,
                            "slot_evidence": r.slot_evidence,
                        }
                        for r in _compound_frames
                    ]
                    logger.debug(
                        "[COMPOUND] Frame-split: %s",
                        [r.frame_name for r in _compound_frames],
                    )
            else:
                # Lightweight semantic fallback
                _secondary_intents = []
                for _part in _compound_parts[1:]:
                    _part = _part.strip()
                    if len(_part) > 8:
                        try:
                            _si, _sc = self._detect_intent_semantic(_part)
                            if (
                                _sc > 0.65
                                and _si != "conversation:general"
                                and _si != intent
                            ):
                                _secondary_intents.append(
                                    {"intent": _si, "confidence": _sc, "text": _part}
                                )
                        except Exception:
                            pass
                if _secondary_intents:
                    parsed_command.modifiers["secondary_intents"] = _secondary_intents
                    logger.debug(
                        "[COMPOUND] Secondary intents: %s", _secondary_intents
                    )

        # Final safety pass: late frame/semantic overrides can change intent after
        # early fallback checks, so enforce clarification fallback one more time.
        _final_plausibility, _final_issues = self._validate_intent_plausibility(
            normalized_text,
            intent,
            parsed_command,
            plugin_scores,
        )
        parsed_command.modifiers["intent_plausibility"] = {
            "score": _final_plausibility,
            "issues": _final_issues,
        }
        _strong_action_frame = bool(
            frame_result is not None
            and frame_result.confidence >= 0.82
            and str(getattr(frame_result, "plugin", "")).lower()
            in {"notes", "email", "calendar", "reminder", "file_operations"}
        )
        _followup_locked_final = bool(
            route is not None
            and isinstance(getattr(route, "trace", None), dict)
            and route.trace.get("source") == "followup_resolver"
            and intent.startswith(("notes:", "email:", "calendar:", "reminder:", "weather:"))
        )

        intent, intent_confidence = self.route_coordinator.apply_final_fallback(
            intent=intent,
            intent_confidence=float(intent_confidence or 0.0),
            parsed_command=parsed_command,
            final_plausibility=float(_final_plausibility),
            strong_action_frame=_strong_action_frame,
            followup_locked_final=_followup_locked_final,
            should_force_unknown_fallback=self._should_force_unknown_fallback,
            normalized_text=normalized_text,
        )

        # Continue with rest of processing

        # Step 3: Entity extraction
        entities = self._extract_all_entities(normalized_text)

        # Step 4: Slot filling
        slots = self.slot_filler.extract_slots(normalized_text, intent, entities)

        # Step 4.5: Entity Normalization (P0 Improvement)
        if (
            self.entity_normalizer
            and self.feature_flags
            and self.feature_flags.is_enabled("nlp_entity_normalizer")
        ):
            from ai.infrastructure.metrics_collector import MetricsCollector

            metrics = MetricsCollector()

            for slot_name, slot in slots.items():
                if not slot.value:
                    continue
                # Normalize based on slot type
                if slot_name == "tags" and isinstance(slot.value, list):
                    normalized_tags = self.entity_normalizer.normalize_batch(
                        slot.value, "tag"
                    )
                    for nt in normalized_tags:
                        if nt.normalized != nt.original:
                            metrics.track_entity_normalization(
                                "tag", nt.rule_applied or "default"
                            )
                    slot.value = [nt.normalized for nt in normalized_tags]
                elif slot_name in ("title", "query", "note_id"):
                    normalized = self.entity_normalizer.normalize(slot.value, "title")
                    if normalized.normalized != normalized.original:
                        metrics.track_entity_normalization(
                            "title", normalized.rule_applied or "default"
                        )
                    slot.value = normalized.normalized
                    slot.confidence = min(slot.confidence, normalized.confidence)
                elif slot_name in ("date", "time", "date_range"):
                    normalized = self.entity_normalizer.normalize(
                        str(slot.value), "datetime"
                    )
                    if normalized.normalized != str(slot.value):
                        metrics.track_entity_normalization(
                            "datetime", normalized.rule_applied or "default"
                        )
                        slot.value = normalized.normalized
                        slot.confidence = min(slot.confidence, normalized.confidence)
        elif not self.feature_flags or not self.feature_flags.is_enabled(
            "nlp_entity_normalizer"
        ):
            # Feature disabled for A/B testing
            pass

        # Final authorization check after all route/frame overrides
        authorization = self.foundation_layers.authorization_policy(
            intent=intent,
            text=normalized_text,
        )
        parsed_command.modifiers["authorization"] = authorization
        if not authorization.get("allowed", True):
            intent = "conversation:clarification_needed"
            intent_confidence = min(float(intent_confidence or 0.0), 0.45)
            parsed_command.modifiers["tool_execution_disabled"] = True

        # Step 5: Sentiment analysis
        if self.sentiment_analyzer is not None:
            sentiment = self.sentiment_analyzer.polarity_scores(normalized_text)
            compound = sentiment.get("compound", 0)
            sentiment["category"] = (
                "positive"
                if compound >= 0.05
                else ("negative" if compound <= -0.05 else "neutral")
            )
        else:
            sentiment = {
                "compound": 0.0,
                "pos": 0.0,
                "neu": 1.0,
                "neg": 0.0,
                "category": "neutral",
            }

        # Step 6: Emotion detection
        emotions = self.emotion_detector.detect_emotions(normalized_text, sentiment)

        # Step 7: Urgency detection
        urgency = self.emotion_detector.detect_urgency(normalized_text)

        # Step 8: Keyword extraction
        keywords = self._extract_keywords(normalized_text)

        # Step 9: Check if question
        is_question = self._is_question(normalized_text)

        # Step 10: Intent-Entity Cross-Validation (P0 Improvement)
        validation_score = 1.0
        validation_issues = []
        if self.feature_flags and self.feature_flags.is_enabled(
            "nlp_intent_entity_validation"
        ):
            validation_score, validation_issues = self._validate_intent_entity_match(
                intent, slots
            )

            # Track validation metrics
            from ai.infrastructure.metrics_collector import MetricsCollector

            metrics = MetricsCollector()
            metrics.track_intent_entity_validation(
                intent, validation_score, validation_issues
            )
        elif not self.feature_flags or not self.feature_flags.is_enabled(
            "nlp_intent_entity_validation"
        ):
            # Feature disabled for A/B testing
            pass

        plan_snapshot = self.foundation_layers.update_plan_memory(
            intent=intent,
            parsed_command=parsed_command.__dict__,
            text=normalized_text,
        )
        parsed_command.modifiers["plan_memory"] = plan_snapshot
        self.foundation_layers.apply_grounding(
            parsed_command.__dict__,
            world_state={
                "location": (self.context.last_entities or {}).get("location"),
                "timezone": "UTC",
            },
        )
        eval_snapshot = self.foundation_layers.record_turn(
            confidence=float(intent_confidence or 0.0),
            clarification=bool(clarification_state.get("needs_clarification")),
            safety_blocked=not authorization.get("allowed", True),
        )
        parsed_command.modifiers["evaluation_harness"] = eval_snapshot
        parsed_command.modifiers["latency_budget"] = {
            "elapsed_ms": round(turn_budget.elapsed_ms(), 3),
            "remaining_ms": round(turn_budget.remaining_ms(), 3),
        }

        # Build result
        result = ProcessedQuery(
            original_text=text,
            clean_text=normalized_text,
            tokens=tokens,
            intent=intent,
            intent_confidence=intent_confidence,
            entities=entities,
            slots=slots,
            sentiment=sentiment,
            emotions=emotions,
            urgency_level=urgency,
            is_question=is_question,
            keywords=keywords,
            parsed_command=parsed_command.__dict__,
            plugin_scores=plugin_scores,
            token_debug=[token.__dict__ for token in rich_tokens],
            intent_candidates=parsed_command.modifiers.get("intent_candidates", []),
            intent_plausibility=float(
                (parsed_command.modifiers.get("intent_plausibility") or {}).get(
                    "score", 1.0
                )
            ),
            plausibility_issues=list(
                (parsed_command.modifiers.get("intent_plausibility") or {}).get(
                    "issues", []
                )
            ),
            validation_score=validation_score,
            validation_issues=validation_issues,
        )

        # Step 11: Update conversation context
        if use_context:
            self._update_context(result)
            # Update dialogue memory
            if self.dialogue_memory is not None:
                _filled = filled_slots.as_dict() if filled_slots else {}
                self.dialogue_memory.update_from_nlp_result(intent, _filled)
            # Store successful high-confidence parses in the fingerprint cache
            if (
                hasattr(self, "_fp_store")
                and self._fp_store is not None
                and intent_confidence >= 0.80
            ):
                _frame_name = frame_result.frame_name if frame_result else None
                _fp_slots = filled_slots.as_dict() if filled_slots else {}
                self._fp_store.store(
                    resolved_text, intent, _frame_name, _fp_slots, intent_confidence
                )

        logger.info(
            "[NLP] Intent: %s (%.2f) | Frame: %s | Slots: %d | Emotions: %s | Urgency: %s",
            intent,
            intent_confidence,
            frame_result.frame_name if frame_result else "none",
            len(slots),
            emotions,
            urgency,
        )

        # Emit thought trace (dev diagnostic — controlled by 'thought_trace' flag)
        if self.feature_flags and self.feature_flags.is_enabled("thought_trace"):
            _correction_source = "unknown"
            _routing_trace = (
                parsed_command.modifiers.get("routing_trace") or {}
                if hasattr(parsed_command, "modifiers")
                else {}
            )
            _src = _routing_trace.get("source", "")
            if "learned" in _src:
                _correction_source = "learned"
            elif "fingerprint" in _src:
                _correction_source = "fingerprint"
            elif "retrieval" in _src:
                _correction_source = "retrieval"
            elif "semantic" in _src or "calibrat" in _src:
                _correction_source = "semantic"
            elif "phase1" in _src or "pattern" in _src:
                _correction_source = "phase1"
            elif "reminder_early_override" in _src or "dialogue_state_gate" in _src:
                _correction_source = _src

            _frame_merged = bool(
                parsed_command.modifiers.get("frame", {}).get("merged")
                if hasattr(parsed_command, "modifiers")
                else False
            )
            _followup = _frame_merged or (
                hasattr(self.context, "turn_count")
                and getattr(self.context, "turn_count", 0) > 0
                and len(normalized_text.split()) <= 6
            )

            self._emit_thought_trace(
                turn_index=getattr(self.context, "turn_count", 0),
                raw_text=text,
                normalized_text=normalized_text,
                intent=intent,
                intent_conf=intent_confidence,
                frame_result=frame_result,
                followup_detected=_followup,
                frame_merged=_frame_merged,
                correction_source=_correction_source,
                emotions=emotions,
                urgency=urgency,
            )

        return result

    def _emit_thought_trace(
        self,
        *,
        turn_index: int,
        raw_text: str,
        normalized_text: str,
        intent: str,
        intent_conf: float,
        frame_result,
        followup_detected: bool,
        frame_merged: bool,
        correction_source: str,
        emotions: List[str],
        urgency: str,
    ) -> None:
        """
        Build and log a ``ThoughtTrace`` for the current turn.

        Written as a JSONL line to ``data/analytics/thought_trace.jsonl``
        and emitted at DEBUG level.
        """
        try:
            from ai.core.failure_taxonomy import ThoughtTrace
        except ImportError:
            return

        _policy_src = "hand_tuned"
        if self.feature_flags:
            if self.feature_flags.is_enabled("policy_learned_model"):
                _policy_src = "learned_model"

        trace = ThoughtTrace(
            turn_index=turn_index,
            raw_text=raw_text,
            normalized_text=normalized_text,
            intent=intent,
            intent_conf=intent_conf,
            frame_name=frame_result.frame_name if frame_result else None,
            frame_conf=frame_result.confidence if frame_result else 0.0,
            followup_detected=followup_detected,
            frame_merged=frame_merged,
            correction_source=correction_source,
            dialogue_state=getattr(self.context, "dialogue_state", "active"),
            policy_source=_policy_src,
            emotions=list(emotions) if emotions else [],
            urgency=urgency,
        )

        logger.debug("[THOUGHT_TRACE] %s", trace.compact())

        import pathlib, json as _json
        _log_dir = pathlib.Path("data") / "analytics"
        try:
            _log_dir.mkdir(parents=True, exist_ok=True)
            with open(_log_dir / "thought_trace.jsonl", "a", encoding="utf-8") as _fh:
                _fh.write(_json.dumps(trace.to_dict()) + "\n")
        except Exception as _exc:
            logger.debug("[THOUGHT_TRACE] Could not write log: %s", _exc)

    def _detect_intent_semantic(
        self,
        text: str,
        parsed_command: Optional[ParsedCommand] = None,
        plugin_scores: Optional[Dict[str, float]] = None,
        return_structured: bool = False,
    ) -> Tuple[str, float]:
        """Detect intent using explicit patterns (primary) then semantic classifier (fallback)"""

        text_lower = text.lower()

        # Negation guard: "don't create a note" should NOT fire notes:create
        _negated = _is_negated_command(text_lower)

        if text_lower.startswith("/debug tokens"):
            return "system:debug_tokens", 0.98

        # PHASE 0: Meta-conversational questions directed *at* Alice.
        # These ask about Alice's previous output or behaviour and must be
        # routed to conversation before any domain classifier can misfire.
        # Examples: "why are you saying still?", "what do you mean by that?",
        #           "why did you say mainly clear?", "what are you talking about?"
        _META_QUESTION_PHRASES = (
            "why are you",
            "why did you",
            "why do you",
            "what do you mean",
            "what did you mean",
            "why would you",
            "how come you",
            "what are you talking",
            "what are you saying",
            "who told you",
        )
        if any(phrase in text_lower for phrase in _META_QUESTION_PHRASES):
            return "conversation:meta_question", 0.93

        # PHASE 0b: Identity / self-inquiry questions about Alice herself.
        # "who made you", "who created you", "who are you", "what are you", etc.
        # Must catch these before the semantic classifier mistakes them for note operations.
        _IDENTITY_PHRASES = (
            "who made you",
            "who created you",
            "who built you",
            "who designed you",
            "who are you",
            "what are you",
            "tell me about yourself",
            "who developed you",
            "who programmed you",
            "who wrote you",
            "who invented you",
            "are you an ai",
            "are you a robot",
            "are you human",
        )
        if any(phrase in text_lower for phrase in _IDENTITY_PHRASES):
            return "conversation:meta_question", 0.95

        mapped = None  # Initialize to prevent UnboundLocalError
        if parsed_command and parsed_command.object_type == "note":
            # Check if this is actually a file operation (e.g., "read the file called notes.txt")
            has_file_context = (
                "file" in text_lower
                or ".txt" in text_lower
                or ".pdf" in text_lower
                or ".csv" in text_lower
                or ".json" in text_lower
                or ".yaml" in text_lower
                or ".md" in text_lower
                or "folder" in text_lower
                or "directory" in text_lower
            )

            # If file context exists, skip note mapping and let file patterns handle it
            if not has_file_context:
                action_map = {
                    "query_exist": "notes:query_exist",
                    "list": "notes:list",
                    "read": "notes:read",
                    "append": "notes:append",
                    "create": "notes:create",
                }
                mapped = action_map.get(parsed_command.action)
            if mapped:
                return mapped, 0.88

        # PHASE 1: HIGH-CONFIDENCE EXPLICIT PATTERNS (these override semantic classification)
        # System intents - check BEFORE time (system has "how's" pattern that could match time)
        if "system" in text_lower and any(word in text_lower for word in _P1_SYSTEM_STATUS):
            return "system:status", 0.9
        # Resource check: "how is/is my + cpu/memory/disk/battery"
        if any(word in text_lower for word in ["how is", "is my", "is the"]) and any(
            word in text_lower for word in _P1_SYSTEM_RESOURCES
        ):
            return "system:status", 0.9
        if any(word in text_lower for word in _P1_SYSTEM_RESOURCES) and any(
            word in text_lower for word in _P1_SYSTEM_RESOURCE_VERBS
        ):
            return "system:status", 0.9

        # File operations - MUST CHECK BEFORE notes (to prevent "notes.txt" matching notes plugin)
        # These patterns check for explicit file context markers like "file", "document", file extensions
        has_file_marker = any(w in text_lower for w in _P1_FILE_MARKERS)

        if has_file_marker:
            # Create: "create/make/new + file"
            if any(word in text_lower for word in _P1_FILE_CREATE) and not _negated:
                return "file_operations:create", 0.95
            # Read: "read/open/show + file/contents"
            if any(word in text_lower for word in _P1_FILE_READ):
                return "file_operations:read", 0.95
            # Delete: "delete/remove + file"
            if any(word in text_lower for word in _P1_FILE_DELETE) and not _negated:
                return "file_operations:delete", 0.95
            # Move: "move/rename + file"
            if any(word in text_lower for word in _P1_FILE_MOVE) and not _negated:
                return "file_operations:move", 0.95
            # List: "list/show + files"
            if any(word in text_lower for word in _P1_FILE_LIST):
                return "file_operations:list", 0.95

        # Memory operations - user preference/recall patterns
        # Recall: "what/do/can/could/would you remember" forms should win over store.
        if any(phrase in text_lower for phrase in _P1_MEMORY_RECALL):
            return "memory:recall", 0.95
        # Store: imperative "remember/save/keep + that/this" (exclude question-style prompts).
        is_memory_question = text_lower.endswith("?") and (
            "remember" in text_lower or "know" in text_lower
        )
        if not is_memory_question and any(
            word in text_lower for word in _P1_MEMORY_STORE
        ) and any(word in text_lower for word in _P1_MEMORY_STORE_OBJ):
            return "memory:store", 0.95
        # Search: "what did we talk about/discuss"
        if any(phrase in text_lower for phrase in _P1_MEMORY_SEARCH_SUBJ) and any(
            word in text_lower for word in _P1_MEMORY_SEARCH_VERB
        ):
            return "memory:search", 0.95

        # Email intents - VERY explicit: action word + email word (0.85-0.9 confidence)
        # Compose: "compose/draft/write + email/mail"
        if any(word in text_lower for word in _P1_EMAIL_COMPOSE_VERBS) and any(
            word in text_lower for word in _P1_EMAIL_NOUNS
        ) and not _negated:
            return "email:compose", 0.9
        # Delete: "delete/remove/trash + email/mail"
        if any(word in text_lower for word in _P1_EMAIL_DELETE_VERBS) and any(
            word in text_lower for word in _P1_EMAIL_DELETE_NOUNS
        ) and not _negated:
            return "email:delete", 0.9
        # Reply: "reply/respond to email/mail"
        if any(word in text_lower for word in _P1_EMAIL_REPLY_VERBS) and any(
            word in text_lower for word in _P1_EMAIL_DELETE_NOUNS
        ) and not _negated:
            return "email:reply", 0.85
        # Search: "find/search + email/mail + (from/subject/sender)"
        if any(word in text_lower for word in _P1_EMAIL_SEARCH_VERBS) and any(
            word in text_lower for word in _P1_EMAIL_SEARCH_NOUNS
        ):
            return "email:search", 0.85
        # List: "show/list/recent + email(s)/mail(s)"
        if any(word in text_lower for word in _P1_EMAIL_LIST_VERBS) and any(
            word in text_lower for word in _P1_EMAIL_LIST_NOUNS
        ):
            return "email:list", 0.85
        # Read: "read/open/display + email(s)/mail/message"
        if any(word in text_lower for word in _P1_EMAIL_READ_VERBS) and any(
            word in text_lower for word in _P1_EMAIL_READ_NOUNS
        ):
            return "email:read", 0.85

        # Goal-statement guard: avoid routing reflective project goals into tools.
        _goal_phrases = (
            "i want to",
            "i would like to",
            "i'd like to",
            "i'm trying to",
            "im trying to",
            "i am trying to",
            "i'm working on",
            "i am working on",
            "i'm learning",
            "i am learning",
            "trying to learn",
            "want to learn",
            "learn how to",
            "my goal is",
            "i plan to",
            "i'm aiming to",
            "i am aiming to",
        )
        _note_markers = (
            "note",
            "notes",
            "memo",
            "list",
            "lists",
            "jot down",
            "write down",
            "remember this",
            "save this",
        )
        if any(p in text_lower for p in _goal_phrases) or "ai system" in text_lower:
            if not any(m in text_lower for m in _note_markers):
                return "conversation:goal_statement", 0.9

        # Notes intents - distinguish VERY clearly
        # Explicit note-capture requests should route to notes:create even without "note" noun.
        if (
            re.search(
                r"\b(write\b.{0,40}\bdown|save\b.{0,40}(?:\bto\s+my\s+notes?\b)?|remember\b.{0,40}|add\b.{0,40}\bto\s+my\s+notes?\b)\b",
                text_lower,
            )
            and not _negated
        ):
            return "notes:create", 0.93
        # Read note content: "what is in the grocery list?", "what's inside my notes?"
        # Must come before list/append to avoid misclassification.
        if _P1_NOTES_READ_CONTENT_RE.search(text_lower) and not _negated:
            _mentions_note_word = bool(re.search(r"\b(note|notes|list|lists|memo)\b", text_lower))
            _pronoun_only = bool(re.search(r"\b(it|this|that)\b", text_lower))
            _notes_context = bool(
                getattr(self.context, "last_intent", "").startswith("notes:")
                or getattr(self.context, "mentioned_notes", None)
            )
            if _mentions_note_word or (_pronoun_only and _notes_context):
                return "notes:read_content", 0.92
        # Append/add to existing note: must come BEFORE create to prevent misclassification
        # Matches: "add X to my grocery list", "add X to the note", "put X on my shopping list"
        if (
            any(word in text_lower for word in _P1_NOTES_APPEND_VERBS)
            and _P1_NOTES_APPEND_RE.search(text_lower)
            and not _negated
        ):
            return "notes:append", 0.9
        # Create: "create/new/make + note" - requires explicit note/memo keyword
        if any(word in text_lower for word in _P1_NOTES_CREATE_VERBS) and any(
            word in text_lower for word in _P1_NOTES_KEYWORDS
        ) and not _negated:
            return "notes:create", 0.9
        # "add a note" - create, not append (no "to the" structure)
        if text_lower.startswith("add") and any(
            word in text_lower for word in _P1_NOTES_KEYWORDS
        ) and not _negated:
            return "notes:create", 0.9
        # Existence/count questions: "do i have notes?", "how many notes do i have?"
        if (
            re.search(r"\b(do i have|how many|i have)\b", text_lower)
            and re.search(r"\bnotes?\b", text_lower)
        ):
            return "notes:query_exist", 0.90
        # List: "show/list + note(s)"
        if any(word in text_lower for word in _P1_NOTES_LIST_VERBS) and any(
            word in text_lower for word in _P1_NOTES_LIST_NOUNS
        ):
            return "notes:list", 0.9
        # Search: "find/search + note(s)"
        if any(word in text_lower for word in _P1_NOTES_SEARCH_VERBS) and any(
            word in text_lower for word in _P1_NOTES_KEYWORDS
        ):
            return "notes:search", 0.9
        # Delete: "delete/remove + note(s)"
        if any(word in text_lower for word in _P1_NOTES_DELETE_VERBS) and any(
            word in text_lower for word in _P1_NOTES_KEYWORDS
        ) and not _negated:
            return "notes:delete", 0.85

        # ── Reminder intents (must come BEFORE thanks/greetings) ─────────────────
        # Set: "remind me to X", "set a reminder", "alert me", "notify me"
        if any(
            phrase in text_lower
            for phrase in [
                "remind me",
                "set a reminder",
                "add a reminder",
                "create a reminder",
                "alert me",
                "notify me when",
                "don't let me forget",
            ]
        ):
            return "reminder:set", 0.95
        # List: "what reminders", "show my reminders", "do I have any reminders"
        if any(
            phrase in text_lower
            for phrase in [
                "my reminders",
                "what reminders",
                "show reminders",
                "list reminders",
                "any reminders",
                "upcoming reminders",
                "pending reminders",
            ]
        ):
            return "reminder:list", 0.95
        # Cancel: "cancel reminder", "delete reminder", "remove reminder"
        if (
            any(word in text_lower for word in ["cancel", "delete", "remove"])
            and "reminder" in text_lower
        ):
            return "reminder:cancel", 0.95
        # ─────────────────────────────────────────────────────────────────────────

        # Time queries: "what time is it?", "what's the time?", "current time"
        # Must come before greetings/weather to prevent misclassification.
        if re.search(
            r"\b(what(?:'s|\s+is)\s+(?:the\s+)?(?:current\s+)?time"
            r"|what time is it"
            r"|tell me the time"
            r"|current time)\b",
            text_lower,
        ):
            return "time:current", 0.95
        # Date queries: "what's today's date?", "what is today?", "what day is it?"
        if re.search(
            r"\b(what(?:'s|\s+is)\s+(?:today(?:'s)?\s+date|the\s+date(?:\s+today)?)"
            r"|what day is it"
            r"|what(?:'s|\s+is)\s+the\s+day"
            r"|today(?:'s)?\s+date)\b",
            text_lower,
        ):
            return "time:current", 0.95

        # Short conversational acknowledgments — catch before semantic classifier
        # e.g. "will do", "got it", "sure", "sounds good", "noted"
        _text_stripped = text_lower.strip(".,!? ")
        if _text_stripped in _P1_CONV_ACK or len(text_lower.split()) <= 3 and _text_stripped in _P1_CONV_ACK:
            return "conversation:ack", 0.92

        # Thanks - check BEFORE greetings (to prevent "thanks" being matched by semantic classifier)
        if any(phrase in text_lower for phrase in _P1_THANKS):
            return "thanks", 0.9

        # Status inquiry: "how are you", "how is alice", "how are you doing"
        if any(phrase in text_lower for phrase in _P1_STATUS_INQUIRY):
            return "status_inquiry", 0.85

        # Understanding/self-review prompts should not collapse into generic help.
        if (
            re.search(
                r"\b(review|summarize|recap|go over|tell me)\b.{0,60}\b(understand(?:ing|stood)?|what you (?:understand|understood)|what we are doing|our plan)\b",
                text_lower,
            )
            or re.search(
                r"\bwhat did you understand\b",
                text_lower,
            )
        ):
            return "conversation:understanding_review", 0.92

        # Greetings (high confidence) - must be after Thanks to not interfere
        greeting_tokens = set(re.findall(r"\b[a-z']+\b", text_lower))
        if _P1_GREETING_WORDS & greeting_tokens and len(text_lower.split()) <= 4:
            return "greeting", 0.9

        if self._is_answerability_direct_question(text_lower):
            return "learning:explanation_request", 0.9

        # Knowledge/definition questions: "what is X?", "what are X?", "who is X?", "define X"
        # Must be before vague patterns to prevent misclassification
        if (
            text_lower.startswith("what is ")
            or text_lower.startswith("what are ")
            or text_lower.startswith("who is ")
            and len(text_lower.split()) > 2
            or text_lower.startswith("define ")
            or text_lower.startswith("explain ")
            or "can you explain" in text_lower
            or "can you tell me what" in text_lower
        ):
            # Exclude weather queries: "what is the weather in London?"
            if any(
                word in text_lower
                for word in [
                    "weather",
                    "temperature",
                    "forecast",
                    "rain",
                    "snow",
                    "sunny",
                ]
            ):
                pass  # Fall through to weather detection below
            # Exclude time queries: "what is the time?"
            elif (
                any(word in text_lower for word in ["time", "clock", "date"])
                and len(text_lower.split()) <= 6
            ):
                pass  # Fall through to time detection below
            # Exclude vague pronouns: "what is that", "who is he/she"
            elif not any(
                vague in text_lower
                for vague in [
                    "what is that",
                    "who is he",
                    "who is she",
                    "who is that person",
                ]
            ):
                return "conversation:question", 0.85

        # Goal/vision declarations: strategic project direction (not immediate tool action).
        goal_signal = self.goal_recognizer.detect(text_lower)
        if goal_signal is not None:
            return "conversation:goal_statement", float(goal_signal.confidence or 0.86)

        # Strong conceptual/build route: keep these prompts in direct architecture explanation mode.
        if self._is_conceptual_build_architecture_prompt(text_lower):
            return "learning:system_design", 0.93
        if self._is_rich_conceptual_request(text_lower):
            return "conversation:question", 0.92

        # CLARIFICATION INTENTS (must run before semantic to catch vague patterns)
        # Vague pronouns without context: "who is he/she", "what is that", "who is that"
        if any(pattern in text_lower for pattern in _P1_VAGUE_PRONOUNS):
            return "vague_question", 0.8

        # Vague temporal questions: "what about yesterday/tomorrow", "what happened last week"
        if any(word in text_lower for word in _P1_VAGUE_TEMPORAL) and (
            "what" in text_lower
            or "when" in text_lower
            or "what about" in text_lower
            or "what happened" in text_lower
        ):
            return "vague_temporal_question", 0.8

        # Vague requests without clear object: "add this to", "put this in", "do that thing"
        if any(pattern in text_lower for pattern in _P1_VAGUE_REQUESTS):
            return "vague_request", 0.8

        # Schedule action without specifics
        if (
            "schedule" in text_lower
            and any(word in text_lower for word in ["it", "that", "this"])
            and any(word in text_lower for word in ["for", "at"])
        ):
            return "schedule_action", 0.8

        # Tell me about - ambiguous topic
        if any(
            pattern in text_lower
            for pattern in [
                "tell me about the",
                "tell me about a",
                "about the sun",
                "about the moon",
                "about the",
            ]
        ):
            if len(text_lower.split()) <= 6:  # Short phrase = likely vague
                return "vague_question", 0.75

        # Weather intents: "what's the weather in X", "will it rain", "is it cold outside?"
        # Must be in PHASE 1 (before semantic classifier) since semantic may misclassify.
        if any(word in text_lower for word in _P1_WEATHER_KEYWORDS):
            if any(word in text_lower for word in _P1_FORECAST_WORDS):
                return "weather:forecast", 0.88
            return "weather:current", 0.88

        # PHASE 1.5: Conversational guard before semantic fallback.
        # Prevent force-fitting casual chat into tool intents.
        _words = set(re.findall(r"\b[a-z']+\b", text_lower))
        _conversation_cues = {
            "how", "why", "what", "hey", "hi", "hello", "day", "feeling",
            "doing", "going", "think", "chat", "talk",
        }
        _tool_cues = {
            "email", "calendar", "meeting", "note", "notes", "file", "files",
            "weather", "forecast", "temperature", "reminder", "time", "date",
            "create", "delete", "read", "write", "open", "search", "list", "show",
        }
        if (
            len(_words) <= 12
            and ("?" in text_lower or not _words.isdisjoint(_conversation_cues))
            and _words.isdisjoint(_tool_cues)
        ):
            return "conversation:general", 0.72

        # PHASE 2: Fallback to semantic classification
        # Try semantic classification for lower-confidence cases
        semantic_classifier = self._ensure_semantic_classifier()
        if semantic_classifier:
            try:
                # Raised threshold to avoid over-eager semantic force-fit.
                result = semantic_classifier.get_plugin_action(text, threshold=0.58)
                if result and result.get("confidence", 0) >= 0.68:
                    # Map plugin:action to intent
                    plugin = result.get("plugin", "")
                    action = result.get("action", "")
                    intent = f"{plugin}:{action}"
                    confidence = result.get("confidence", 0.0)

                    logger.debug(f"[NLP] Semantic intent: {intent} ({confidence:.3f})")
                    return intent, confidence
            except Exception as e:
                logger.warning(f"[WARN] Semantic classification failed: {e}")

        # PHASE 3: Additional fallback patterns

        # Clarification/meta-question cues
        if any(
            phrase in text_lower
            for phrase in [
                "can i ask you something about",
                "i have a question about",
                "let me ask you about",
                "can i ask you about",
            ]
        ):
            return "conversation:meta_question", 0.8

        # Thanks
        if any(phrase in text_lower for phrase in ["thanks", "thank you", "thx"]):
            return "thanks", 0.9

        # Status inquiry
        if any(
            phrase in text_lower
            for phrase in ["how are you", "how are you doing", "how is it going"]
        ):
            return "status_inquiry", 0.85

        # Weather intents - CHECK EARLY before vague patterns
        # This ensures "is that wednesday?" triggers weather:forecast, not vague_question
        if any(phrase in text_lower for phrase in _P3_FORECAST_PHRASES):
            return "weather:forecast", 0.8

        if any(word in text_lower for word in _P3_WEATHER_WORDS):
            return "weather:current", 0.75

        # Vague patterns requiring clarification
        if any(pattern in text_lower for pattern in _P3_VAGUE_PATTERNS):
            return "vague_question", 0.75
        if "?" in text:
            return "conversation:question", 0.5

        if plugin_scores:
            best_plugin = max(plugin_scores.items(), key=lambda item: item[1])
            plugin_name, score = best_plugin
            if score >= 2.2:
                plugin_intent_map = {
                    "notes": "notes:list",
                    "email": "email:list",
                    "calendar": "calendar:list",
                    "weather": "weather:current",
                    "system": "system:status",
                }
                if plugin_name in plugin_intent_map:
                    return plugin_intent_map[plugin_name], min(
                        0.85, 0.55 + (score / 10.0)
                    )

        return "conversation:general", 0.3

    def _extract_all_entities(self, text: str) -> Dict[str, List[Entity]]:
        """Extract all entities (domain-specific + general)"""

        # Check cache (promote to most-recently-used on hit)
        cache_key = hash(text)
        with self._cache_lock:
            if cache_key in self._entity_cache:
                self._entity_cache.move_to_end(cache_key)
                return self._entity_cache[cache_key]

        entities = {}
        text_lower = text.lower()

        # Domain-specific entities (custom NER)
        domain_entities = self.domain_ner.extract(text)
        entities.update(domain_entities)

        # General entities (regex patterns)
        general_patterns = {
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "PHONE": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "URL": r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "NUMBER": r"\b\d+\b",
            "PERCENTAGE": r"\b\d+(?:\.\d+)?%\b",
        }

        for entity_type, pattern in general_patterns.items():
            matches = re.finditer(pattern, text)
            entity_list = []
            for match in matches:
                entity = Entity(
                    type=entity_type,
                    value=match.group(0),
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end(),
                )
                entity_list.append(entity)

            if entity_list:
                entities[entity_type] = entity_list

        # Time range entities (for forecast queries)
        time_range_patterns = {
            "this week": "week",
            "next week": "next_week",
            "weekend": "weekend",
            "tomorrow": "tomorrow",
            "next few days": "next_few_days",
            "next 7 days": "7_days",
            "7 day": "7_days",
            "7-day": "7_days",
            "monday": "monday",
            "tuesday": "tuesday",
            "wednesday": "wednesday",
            "thursday": "thursday",
            "friday": "friday",
            "saturday": "saturday",
            "sunday": "sunday",
        }

        time_range_entities = []
        for phrase, normalized in time_range_patterns.items():
            if phrase in text_lower:
                time_range_entities.append(
                    Entity(
                        type="TIME_RANGE",
                        value=phrase,
                        confidence=0.85,
                        normalized_value=normalized,
                    )
                )

        if time_range_entities:
            entities["TIME_RANGE"] = time_range_entities

        # Cache result (OrderedDict LRU: move_to_end promotes recently used entries)
        with self._cache_lock:
            self._entity_cache[cache_key] = entities
            self._entity_cache.move_to_end(cache_key)
            # Evict least-recently-used entries when cache exceeds limit
            while len(self._entity_cache) > 1000:
                self._entity_cache.popitem(last=False)

        return entities

    def _update_context(self, result: ProcessedQuery):
        """Update conversation context"""
        _existing_pending = (
            dict(getattr(self.context, "pending_clarification", {}) or {})
            if hasattr(self, "context")
            else {}
        )
        self.context.last_intent = result.intent
        self.context.query_history.append(result.original_text)
        self.context.turn_index += 1
        self.context.last_plugin = (
            result.intent.split(":", 1)[0] if ":" in result.intent else "conversation"
        )
        if self.dialogue_state_machine is not None:
            _dialogue_state = self.dialogue_state_machine.observe_intent(result.intent)
            self.context.dialogue_state = _dialogue_state.value
        else:
            self.context.dialogue_state = (
                "clarifying"
                if result.intent == "conversation:clarification_needed"
                else "active"
            )

        disambiguation = {}
        _pending_slot_followup = {}
        if isinstance(result.parsed_command, dict):
            _modifiers = result.parsed_command.get("modifiers", {}) or {}
            disambiguation = _modifiers.get("disambiguation", {})
            _pending_slot_followup = _modifiers.get("pending_slot_followup", {})

        _existing_slot_type = str(
            (_existing_pending or {}).get("slot_type")
            or (_existing_pending or {}).get("type")
            or ""
        ).strip().lower()
        _preserve_existing_slot = bool(
            isinstance(_existing_pending, dict)
            and _existing_pending
            and _existing_slot_type in {"route_choice", "help_route_choice", "help_narrowing", "narrowing"}
            and bool(_existing_pending.get("active", True))
        )
        _slot_was_filled = bool(
            isinstance(_pending_slot_followup, dict)
            and _pending_slot_followup.get("filled")
        )

        if disambiguation:
            self.context.pending_clarification = disambiguation
        elif _slot_was_filled:
            self.context.pending_clarification = {}
        elif _preserve_existing_slot:
            self.context.pending_clarification = dict(_existing_pending)
        else:
            self.context.pending_clarification = {}

        # Persist the semantic frame for follow-up merging
        if isinstance(result.parsed_command, dict):
            _frame_data = result.parsed_command.get("modifiers", {}).get("frame")
            if _frame_data and isinstance(_frame_data, dict) and _frame_data.get("name"):
                self.context.last_frame = dict(_frame_data)

        # Extract entities to context
        self.context.last_entities = {}

        if self.conversation_memory is not None:
            self.conversation_memory.add_turn(
                user_input=result.original_text,
                intent=result.intent,
                response="",
                context_extracted={
                    "topic": (result.keywords[0] if result.keywords else ""),
                    "dialogue_state": self.context.dialogue_state,
                },
            )

        # From slots
        for slot_name, slot in result.slots.items():
            self.context.last_entities[slot_name] = slot.value

            # Track specific entity types
            if slot_name == "title" and "note" in result.intent:
                self.context.mentioned_notes.append(slot.value)
            elif slot_name == "song":
                self.context.mentioned_songs.append(slot.value)
            elif slot_name == "event":
                self.context.mentioned_events.append(slot.value)

    def _validate_intent_entity_match(
        self, intent: str, slots: Dict[str, Slot]
    ) -> Tuple[float, List[str]]:
        """
        Cross-validate intent with extracted entities (P0 Improvement).

        Returns:
            (validation_score, issues): Score 0.0-1.0 and list of detected issues

        Algorithm: Penalize missing required entities, boost matching expected entities
        Complexity: O(n) where n = number of slots
        """
        validation_score = 1.0
        issues = []

        # Find matching validation rule
        rules = self._validation_matrix.get(intent)
        if not rules:
            # Try prefix match (e.g., "notes:create_tagged" -> "notes:create")
            for pattern, rule in self._validation_matrix.items():
                if intent.startswith(pattern.split(":")[0] + ":"):
                    rules = rule
                    break

        if not rules:
            return validation_score, issues  # No validation rule for this intent

        # Extract present slot keys
        present_slots = {k for k, v in slots.items() if v.value}

        # Check required entities
        required = set(rules.get("required", []))
        missing_required = required - present_slots
        if missing_required:
            penalty = 0.25 * len(missing_required)  # -0.25 per missing required entity
            validation_score -= penalty
            issues.append(f"Missing required: {', '.join(missing_required)}")

        # Check expected entities (soft guidance, not critical)
        expected = set(rules.get("expected", []))
        if expected:
            present_expected = expected & present_slots
            match_ratio = len(present_expected) / len(expected)
            # Only penalize if NO expected entities present AND there's a clear expectation
            if match_ratio == 0.0 and len(expected) <= 2:
                validation_score -= (
                    0.05  # Small penalty for completely missing simple expectations
                )
                # Don't add to issues - this is just informational

        # Bonus for having unexpected but relevant entities (don't penalize creativity)
        validation_score = max(0.0, min(1.0, validation_score))

        return validation_score, issues

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"[^\w\s.,!?\'\-/#:;]", "", text)
        return text

    def _is_question(self, text: str) -> bool:
        """Check if text is a question"""
        if "?" in text:
            return True

        question_words = [
            "what",
            "when",
            "where",
            "who",
            "why",
            "how",
            "which",
            "can",
            "could",
            "would",
            "should",
            "is",
            "are",
            "do",
            "does",
        ]

        text_lower = text.lower()
        return any(text_lower.startswith(qw) for qw in question_words)

    def _extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Extract top keywords"""
        tokens = word_tokenize(text.lower())

        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "be",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
        }

        keywords = [
            token for token in tokens if token.isalnum() and token not in stopwords
        ]

        from collections import Counter

        keyword_freq = Counter(keywords)
        return [word for word, _ in keyword_freq.most_common(top_n)]

    def get_context(self) -> ConversationContext:
        """Get current conversation context"""
        return self.context

    def reset_context(self):
        """Reset conversation context"""
        self.context = ConversationContext()
        logger.info("[OK] Conversation context reset")


# ============================================================================
# FACTORY FUNCTION (for compatibility with existing code)
# ============================================================================


def get_nlp_processor() -> NLPProcessor:
    """Get singleton NLP processor instance"""
    return NLPProcessor()


# ============================================================================
# TEMPORAL UNDERSTANDING  (stable abstraction over TemporalParser)
# ============================================================================


@dataclass
class TemporalResult:
    """
    Normalised time/date extraction result.

    Fields
    ------
    start:
        ISO-8601 date string (YYYY-MM-DD) or None.
    end:
        ISO-8601 date string for range end, or None.
    time:
        Wall-clock time (HH:MM) or None.
    grain:
        Coarsest meaningful unit: 'minute' | 'hour' | 'day' | 'week' |
        'month' | 'unknown'.
    raw:
        The original text fragment that was parsed.
    confidence:
        Parser confidence, 0–1.
    """

    start: Optional[str]
    end: Optional[str]
    time: Optional[str]
    grain: str
    raw: str
    confidence: float

    def as_dict(self) -> Dict[str, Any]:
        """Backward-compatible dict for callers that still expect raw parser output."""
        return {
            "date": self.start,
            "end_date": self.end,
            "time": self.time,
            "grain": self.grain,
            "raw_text": self.raw,
            "confidence": self.confidence,
        }


class TemporalUnderstanding:
    """
    Thin stable wrapper around TemporalParser.

    Usage
    -----
        tu = TemporalUnderstanding(temporal_parser)
        result = tu.parse("tomorrow at 3pm")
        if result:
            print(result.start, result.time, result.grain)
    """

    def __init__(self, temporal_parser: Any) -> None:
        self._parser = temporal_parser

    def parse(self, text: str) -> Optional[TemporalResult]:
        """
        Parse a text fragment for temporal expressions.

        Returns a TemporalResult on success, None if no temporal
        expression was detected.
        """
        try:
            raw = self._parser.parse_temporal_expression(text)
        except Exception as exc:
            logger.debug("[TemporalUnderstanding] parse error for %r: %s", text, exc)
            return None

        if not raw:
            return None

        grain = self._infer_grain(raw, text)

        return TemporalResult(
            start=raw.get("date"),
            end=raw.get("end_date"),
            time=raw.get("time"),
            grain=grain,
            raw=raw.get("raw_text", text),
            confidence=raw.get("confidence", 0.7),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _WEEK_CUES = {
        "week",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "weekend",
    }
    _MONTH_CUES = {
        "month",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    }

    def _infer_grain(self, raw: Dict[str, Any], text: str) -> str:
        lower = text.lower()
        words = set(lower.split())

        if raw.get("end_date"):
            return "week" if words & self._WEEK_CUES else "day"
        if raw.get("time") and raw.get("date"):
            return "minute"
        if raw.get("time"):
            return "hour"
        if raw.get("date"):
            if words & self._WEEK_CUES:
                return "week"
            if words & self._MONTH_CUES:
                return "month"
            return "day"
        return "unknown"


def get_temporal_understanding(temporal_parser: Any) -> TemporalUnderstanding:
    """Factory helper — creates a TemporalUnderstanding bound to *temporal_parser*."""
    return TemporalUnderstanding(temporal_parser)


# Follow-up resolver and perception classes are extracted to dedicated modules.
