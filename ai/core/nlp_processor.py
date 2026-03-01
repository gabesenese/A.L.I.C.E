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
from collections import defaultdict, deque
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

logger = logging.getLogger(__name__)


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
        result = self.temporal_parser.parse_temporal_expression(text)
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
        result = self.temporal_parser.parse_temporal_expression(text)
        if result and result.get("time"):
            return (
                result["time"],
                result.get("confidence", 0.8),
                result.get("raw_text", ""),
            )
        return None, 0.0, ""

    # ==================== MUSIC SLOT EXTRACTORS ====================

    def _extract_song(
        self, text: str, text_lower: str, entities: Dict
    ) -> Tuple[Optional[str], float, str]:
        """Extract song name"""
        patterns = [
            r'"([^"]+)"',  # Quoted
            r"play\s+([^,\n]+?)(?:\s+by|\s+from|$)",
            r'song\s+"?([^"]+?)"?(?:\s+by|$)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                song = match.group(1).strip()
                if len(song) > 1:
                    return song, 0.85, match.group(0)

        return None, 0.0, ""

    def _extract_artist(
        self, text: str, text_lower: str, entities: Dict
    ) -> Tuple[Optional[str], float, str]:
        """Extract artist name"""
        patterns = [
            r"\bby\s+([A-Za-z\s&\']+?)(?:\s+from|\s+on|$)",
            r"\bartist\s+([A-Za-z\s&\']+?)(?:\s+from|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                artist = match.group(1).strip()
                if len(artist) > 1:
                    return artist, 0.9, match.group(0)

        return None, 0.0, ""

    def _extract_genre(
        self, text: str, text_lower: str, entities: Dict
    ) -> Tuple[Optional[str], float, str]:
        """Extract music genre"""
        genres = [
            "rock",
            "pop",
            "jazz",
            "classical",
            "hip hop",
            "rap",
            "country",
            "electronic",
            "blues",
            "reggae",
            "folk",
            "metal",
            "indie",
        ]

        for genre in genres:
            if re.search(rf"\b{genre}\b", text_lower):
                return genre, 0.95, genre

        return None, 0.0, ""

    def _extract_mood(
        self, text: str, text_lower: str, entities: Dict
    ) -> Tuple[Optional[str], float, str]:
        """Extract mood/vibe"""
        moods = [
            "upbeat",
            "relaxing",
            "chill",
            "energetic",
            "slow",
            "happy",
            "sad",
            "workout",
            "study",
            "sleep",
            "party",
            "romantic",
        ]

        for mood in moods:
            if re.search(rf"\b{mood}\b", text_lower):
                return mood, 0.85, mood

        return None, 0.0, ""

    def _extract_action(
        self, text: str, text_lower: str, entities: Dict
    ) -> Tuple[Optional[str], float, str]:
        """Extract music action"""
        actions = [
            "play",
            "pause",
            "stop",
            "skip",
            "next",
            "previous",
            "resume",
            "shuffle",
            "repeat",
        ]

        for action in actions:
            if re.search(rf"\b{action}\b", text_lower):
                return action, 0.95, action

        return None, 0.0, ""

    def _extract_volume(
        self, text: str, text_lower: str, entities: Dict
    ) -> Tuple[Optional[int], float, str]:
        """Extract volume level"""
        volume_match = re.search(
            r"\b(?:volume|sound)\s+(?:to\s+)?(\d{1,3})(?:%|percent)?\b", text_lower
        )
        if volume_match:
            volume = int(volume_match.group(1))
            return min(100, max(0, volume)), 0.98, volume_match.group(0)

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
                time_struct, parse_status = self.cal.parse(text)
                if parse_status > 0:
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
        "MUSIC_GENRE": r"\b(rock|pop|jazz|classical|hip hop|rap|country|electronic|blues|metal)\b",
        "MUSIC_MOOD": r"\b(upbeat|relaxing|chill|energetic|happy|sad|workout|study)\b",
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
    Resolve pronouns to entities mentioned in conversation

    Examples:
        User: "Create note shopping list"
        Then: "Add eggs to it" -> "it" resolves to "shopping list" note

        User: "Play Bohemian Rhapsody"
        Then: "pause it" -> "it" resolves to "Bohemian Rhapsody"
    """

    def __init__(self):
        self.pronouns = [
            "it",
            "this",
            "that",
            "the note",
            "the event",
            "the song",
            "the task",
            "the email",
            "the reminder",
        ]

    def resolve(self, text: str, context: ConversationContext) -> str:
        """Resolve coreferences in text using conversation context"""
        resolved_text = text
        text_lower = text.lower()

        # Check if text contains pronouns
        for pronoun in self.pronouns:
            if pronoun in text_lower:
                replacement = self._find_referent(pronoun, context)
                if replacement:
                    # Replace pronoun with actual entity
                    resolved_text = re.sub(
                        rf"\b{re.escape(pronoun)}\b",
                        replacement,
                        resolved_text,
                        count=1,
                        flags=re.IGNORECASE,
                    )
                    logger.info(f"[COREF] Resolved '{pronoun}' -> '{replacement}'")

        return resolved_text

    def _find_referent(
        self, pronoun: str, context: ConversationContext
    ) -> Optional[str]:
        """Find what the pronoun refers to"""
        # Check based on last intent
        if context.last_intent:
            if "note" in context.last_intent and context.mentioned_notes:
                return str(context.mentioned_notes[-1])
            elif "calendar" in context.last_intent and context.mentioned_events:
                return str(context.mentioned_events[-1])
            elif "music" in context.last_intent and context.mentioned_songs:
                return str(context.mentioned_songs[-1])

        # Check generic entities
        if context.last_entities:
            # Return most recent entity of appropriate type
            for entity_type in ["title", "song", "event", "note_id"]:
                if entity_type in context.last_entities:
                    return str(context.last_entities[entity_type])

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
                if keyword in text_lower:
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

    def __new__(cls):
        """Singleton pattern for performance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
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

        # Semantic intent classifier
        self.semantic_classifier = None
        self._semantic_classifier_init_attempted = False

        # Load learned corrections into pattern matching
        self.learned_corrections = self._load_learned_corrections()

        # Intent-Entity Cross-Validation Matrix (P0 Improvement)
        # Maps intent prefixes to required/expected entity types
        self._validation_matrix = {
            "notes:create": {"required": [], "expected": ["title", "content", "tags"]},
            "notes:search": {"required": ["query"], "expected": ["tags", "date_range"]},
            "notes:update": {"required": ["note_id", "title"], "expected": ["content", "tags"]},
            "notes:delete": {"required": ["note_id", "query"], "expected": []},
            "music:play": {"required": ["song", "artist"], "expected": ["album", "playlist"]},
            "calendar:create": {"required": ["event", "date"], "expected": ["time", "location"]},
            "calendar:search": {"required": ["query", "date_range"], "expected": []},
            "email:compose": {"required": ["recipient", "subject"], "expected": ["body"]},
            "email:search": {"required": ["sender", "subject"], "expected": ["date_range"]},
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
            "music": {"play", "pause", "next", "previous", "queue"},
            "system": {"status", "debug_tokens"},
            "conversation": {
                "general",
                "question",
                "meta_question",
                "clarification_needed",
            },
        }
        self._intent_action_defaults = {
            "notes": "list",
            "email": "list",
            "calendar": "list",
            "music": "play",
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
            "music",
            "notes",
            "note",
            "todo",
            "task",
            "tasks",
            "read",
            "open",
            "show",
            "create",
            "append",
            "list",
            "search",
            "delete",
        }

        # Cache for performance
        self._entity_cache = {}
        self._cache_lock = threading.Lock()

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
            "music",
            "song",
            "songs",
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
        """Resolve very short follow-up queries using context before broad intent matching."""
        content_tokens = [token for token in tokens if token.kind != "symbol"]
        if len(content_tokens) > 8:
            return None

        token_words = {token.normalized for token in content_tokens}
        last_intent = self.context.last_intent or ""
        recent_queries = [q.lower() for q in list(self.context.query_history)[-3:]]

        reference_present = any(
            token.flags.get("is_reference") for token in content_tokens
        )

        followup_connectors = {
            "what",
            "about",
            "and",
            "also",
            "that",
            "this",
            "it",
            "same",
        }
        time_range_cues = {
            "today",
            "tomorrow",
            "tonight",
            "week",
            "weekend",
            "month",
            "next",
        }
        weather_cues = {
            "weather",
            "forecast",
            "rain",
            "snow",
            "cold",
            "warm",
            "umbrella",
            "coat",
            "jacket",
            "wear",
            "outside",
            "temperature",
        }

        recent_weather_context = last_intent.startswith("weather:") or any(
            "weather" in query or "forecast" in query for query in recent_queries
        )

        if recent_weather_context:
            weather_followup = bool(token_words & weather_cues)
            temporal_followup = bool(token_words & time_range_cues)
            connector_followup = (
                reference_present or bool(token_words & followup_connectors)
            )

            if temporal_followup or (
                connector_followup and not token_words.isdisjoint(time_range_cues)
            ):
                return "weather:forecast", 0.86

            if weather_followup or connector_followup:
                if token_words & {"week", "weekend", "next", "tomorrow", "tonight"}:
                    return "weather:forecast", 0.84
                return "weather:current", 0.83

        # Generic short ambiguous follow-up: keep same topic intent family
        if last_intent and (reference_present or bool(token_words & followup_connectors)):
            if last_intent.startswith("notes:"):
                return "notes:read", 0.79
            if last_intent.startswith("email:"):
                return "email:read", 0.77
            if last_intent.startswith("calendar:"):
                return "calendar:list", 0.76
            if ":" in last_intent and not last_intent.startswith(
                ("conversation:", "system:")
            ):
                return last_intent, 0.75

        action_cues = {
            "read",
            "show",
            "open",
            "list",
            "play",
            "pause",
            "reply",
            "send",
            "schedule",
        }
        if token_words.isdisjoint(action_cues):
            return None

        if not reference_present and token_words.isdisjoint(
            {"read", "show", "open", "list", "reply", "send"}
        ):
            return None

        if last_intent.startswith("notes:") or self.context.mentioned_notes:
            if "list" in token_words:
                return "notes:list", 0.81
            return "notes:read", 0.82
        if last_intent.startswith("email:"):
            return "email:read", 0.78
        if last_intent.startswith("calendar:"):
            return "calendar:list", 0.76
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

    def _build_uncertainty_prompt(
        self,
        route: RouteDecision,
        parsed: ParsedCommand,
        plugin_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """Build disambiguation metadata when intent confidence is weak."""
        if route.confidence >= 0.55:
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
        token_pattern = re.compile(
            r'"[^"]+"|#\w+|\d+(?:st|nd|rd|th)?|[A-Za-z]+|[^\w\s]'
        )

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
            for match in token_pattern.finditer(segment.text):
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
                elif re.fullmatch(r"\d+(?:st|nd|rd|th)", norm):
                    kind = "ordinal"
                    role = "reference"
                elif raw.isdigit():
                    kind = "number"
                    role = "value"
                elif re.fullmatch(r"[A-Za-z]+", raw):
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
        elif re.search(r"\b(play|pause|skip|next)\b", lower) and re.search(
            r"\bmusic|song|songs|playlist|album\b", lower
        ):
            parsed.action = "play" if re.search(r"\bplay\b", lower) else "pause"
            parsed.object_type = "music"

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
            "music": 0.0,
            "system": 0.0,
            "conversation": 0.2,
        }

        normalized = [token.normalized for token in tokens]
        bigrams = {
            f"{normalized[i]} {normalized[i + 1]}" for i in range(len(normalized) - 1)
        }
        note_terms = {"note", "notes", "list", "lists", "todo", "task", "tasks"}
        email_terms = {"email", "emails", "mail", "inbox", "sender", "subject"}
        cal_terms = {"calendar", "event", "events", "meeting", "schedule"}
        music_terms = {"music", "song", "songs", "playlist", "album", "artist", "play"}
        system_terms = {"system", "cpu", "memory", "disk", "battery", "status"}

        scores["notes"] += sum(1.2 for word in normalized if word in note_terms)
        scores["email"] += sum(1.2 for word in normalized if word in email_terms)
        scores["calendar"] += sum(1.2 for word in normalized if word in cal_terms)
        scores["music"] += sum(1.2 for word in normalized if word in music_terms)
        scores["system"] += sum(1.2 for word in normalized if word in system_terms)

        if parsed.object_type == "note":
            scores["notes"] += 1.5
        if parsed.object_type == "email":
            scores["email"] += 1.5
        if parsed.object_type == "calendar":
            scores["calendar"] += 1.5
        if parsed.object_type == "music":
            scores["music"] += 1.5
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
        if parsed.action in {"play", "pause"} and parsed.object_type == "music":
            scores["music"] += 1.0
        if parsed.references:
            scores["notes"] += 0.6

        if "read it" in bigrams or "show it" in bigrams:
            scores["notes"] += 0.5
            if self.context.last_intent and self.context.last_intent.startswith(
                "email:"
            ):
                scores["email"] += 0.5
        if "next song" in bigrams:
            scores["music"] += 0.7
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
            "play": "music",
            "status": "system",
        }
        preferred = action_bias.get(parsed.action)
        if preferred:
            scores[preferred] += 0.45

        if self.tokenizer_profile == "strict":
            for key in ("email", "calendar", "music", "system"):
                if scores[key] < 1.0:
                    scores[key] *= 0.7

        if self.tokenizer_profile == "llm-assisted":
            for key in ("notes", "email", "calendar", "music", "system"):
                scores[key] *= 1.05

        return scores

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

        # Step 1: Coreference resolution
        if use_context:
            # Check if using advanced coreference with ambiguity detection
            if hasattr(self.coref_resolver, '_engine'):
                # Using AdvancedCoref through compat wrapper
                resolved_result = self.coref_resolver._engine.resolve(text, self.context.__dict__ if hasattr(self.context, '__dict__') else {})
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
                        len(resolved_result.candidates)
                    )
                    # Store ambiguity info for clarification prompts
                    if hasattr(self.context, 'pending_clarification'):
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
        normalized_text = self._normalize_for_tokenizer(clean_text)

        # Step 3: Surface + lexical + semantic hint layers
        segments = self._surface_segment(normalized_text)
        rich_tokens = self._lexical_tokenize(segments)
        parsed_command = self._extract_semantic_hints(normalized_text, rich_tokens)
        plugin_scores = self._compute_plugin_scores(rich_tokens, parsed_command)
        retrieval_route = self._retrieval_first_parse(rich_tokens)
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
        if (
            self.learned_corrections
            and normalized_text.lower() in self.learned_corrections
        ):
            learned_intent = self.learned_corrections[normalized_text.lower()]
            logger.info(
                f"[LEARNED] Using correction for '{normalized_text}' -> {learned_intent}"
            )
            intent = learned_intent
            intent_confidence = 0.95  # High confidence for learned patterns
        else:
            # Step 5: Intent detection (retrieval-first + semantic + calibrated routing)
            if retrieval_route:
                intent, intent_confidence = retrieval_route
                route = RouteDecision(
                    intent=intent,
                    confidence=intent_confidence,
                    plugin=intent.split(":", 1)[0],
                    action=intent.split(":", 1)[1] if ":" in intent else "general",
                    trace={"source": "retrieval_first"},
                )
            else:
                semantic_intent = self._detect_intent_semantic(
                    normalized_text,
                    parsed_command=parsed_command,
                    plugin_scores=plugin_scores,
                    return_structured=False,
                )
                weighted_candidates = self._build_weighted_parse(
                    parsed_command, plugin_scores, normalized_text
                )
                route = self._calibrate_route_decision(
                    weighted_candidates, plugin_scores, semantic_intent, parsed_command
                )
                intent = route.intent
                intent_confidence = route.confidence

            uncertainty = self._build_uncertainty_prompt(
                route, parsed_command, plugin_scores
            )
            if intent.startswith("vague_") and not uncertainty:
                uncertainty = {
                    "needs_clarification": True,
                    "question": "Can you clarify what action and target you mean?",
                    "candidate_plugins": ["notes", "email", "calendar", "music"],
                    "route_confidence": intent_confidence,
                    "parsed_action": parsed_command.action,
                }
            if not uncertainty and re.search(
                r"\b(do that thing|this thing|that thing|what about that|who is that)\b",
                normalized_text.lower(),
            ):
                uncertainty = {
                    "needs_clarification": True,
                    "question": "Could you clarify what you want me to do?",
                    "candidate_plugins": ["notes", "email", "calendar", "music"],
                    "route_confidence": intent_confidence,
                    "parsed_action": parsed_command.action,
                }
            if uncertainty:
                parsed_command.modifiers["disambiguation"] = uncertainty
                if intent_confidence < 0.45 and not intent.startswith(
                    ("notes:", "email:", "calendar:", "music:", "system:")
                ):
                    intent = "conversation:clarification_needed"
                    intent_confidence = max(intent_confidence, 0.41)

            parsed_command.modifiers["routing_trace"] = route.trace

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
                "last_note_title": (
                    str(self.context.mentioned_notes[-1])
                    if self.context.mentioned_notes
                    else None
                ),
                "last_entities": self.context.last_entities,
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
                # Override routing if frame is more confident
                if frame_result.confidence > intent_confidence + 0.07:
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
                    normalized_tags = self.entity_normalizer.normalize_batch(slot.value, "tag")
                    for nt in normalized_tags:
                        if nt.normalized != nt.original:
                            metrics.track_entity_normalization("tag", nt.rule_applied or "default")
                    slot.value = [nt.normalized for nt in normalized_tags]
                elif slot_name in ("title", "query", "note_id"):
                    normalized = self.entity_normalizer.normalize(slot.value, "title")
                    if normalized.normalized != normalized.original:
                        metrics.track_entity_normalization("title", normalized.rule_applied or "default")
                    slot.value = normalized.normalized
                    slot.confidence = min(slot.confidence, normalized.confidence)
                elif slot_name in ("date", "time", "date_range"):
                    normalized = self.entity_normalizer.normalize(str(slot.value), "datetime")
                    if normalized.normalized != str(slot.value):
                        metrics.track_entity_normalization("datetime", normalized.rule_applied or "default")
                        slot.value = normalized.normalized
                        slot.confidence = min(slot.confidence, normalized.confidence)
        elif not self.feature_flags or not self.feature_flags.is_enabled("nlp_entity_normalizer"):
            # Feature disabled for A/B testing
            pass

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
        if self.feature_flags and self.feature_flags.is_enabled("nlp_intent_entity_validation"):
            validation_score, validation_issues = self._validate_intent_entity_match(intent, slots)
            
            # Track validation metrics
            from ai.infrastructure.metrics_collector import MetricsCollector
            metrics = MetricsCollector()
            metrics.track_intent_entity_validation(intent, validation_score, validation_issues)
        elif not self.feature_flags or not self.feature_flags.is_enabled("nlp_intent_entity_validation"):
            # Feature disabled for A/B testing
            pass

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

        return result

    def _detect_intent_semantic(
        self,
        text: str,
        parsed_command: Optional[ParsedCommand] = None,
        plugin_scores: Optional[Dict[str, float]] = None,
        return_structured: bool = False,
    ) -> Tuple[str, float]:
        """Detect intent using explicit patterns (primary) then semantic classifier (fallback)"""

        text_lower = text.lower()

        if text_lower.startswith("/debug tokens"):
            return "system:debug_tokens", 0.98

        if parsed_command and parsed_command.object_type == "note":
            action_map = {
                "query_exist": "notes:list",
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
        if "system" in text_lower and any(
            word in text_lower for word in ["status", "doing", "health", "how"]
        ):
            return "system:status", 0.9
        # Resource check: "how is/is my + cpu/memory/disk/battery"
        if any(word in text_lower for word in ["how is", "is my", "is the"]) and any(
            word in text_lower for word in ["cpu", "memory", "disk", "battery", "gpu"]
        ):
            return "system:status", 0.9
        if any(
            word in text_lower for word in ["cpu", "memory", "disk", "battery"]
        ) and any(
            word in text_lower
            for word in ["usage", "available", "how much", "low", "check", "is"]
        ):
            return "system:status", 0.9

        # Email intents - VERY explicit: action word + email word (0.85-0.9 confidence)
        # Compose: "compose/draft/write + email/mail"
        if any(
            word in text_lower for word in ["compose", "draft", "write", "send"]
        ) and any(word in text_lower for word in ["email", "mail", "message", "to"]):
            return "email:compose", 0.9
        # Delete: "delete/remove/trash + email/mail"
        if any(word in text_lower for word in ["delete", "remove", "trash"]) and any(
            word in text_lower for word in ["email", "mail", "message"]
        ):
            return "email:delete", 0.9
        # Reply: "reply/respond to email/mail"
        if any(word in text_lower for word in ["reply", "respond"]) and any(
            word in text_lower for word in ["email", "mail", "message"]
        ):
            return "email:reply", 0.85
        # Search: "find/search + email/mail + (from/subject/sender)"
        if any(word in text_lower for word in ["search", "find", "look for"]) and any(
            word in text_lower for word in ["email", "mail", "inbox", "message", "from"]
        ):
            return "email:search", 0.85
        # List: "show/list/recent + email(s)/mail(s)"
        if any(
            word in text_lower for word in ["show", "list", "recent", "latest"]
        ) and any(
            word in text_lower for word in ["email", "emails", "mail", "mails", "inbox"]
        ):
            return "email:list", 0.85
        # Read: "read/open/display + email(s)/mail/message + optional (first/last/latest/number)"
        if any(
            word in text_lower for word in ["read", "open", "display", "view"]
        ) and any(
            word in text_lower for word in ["email", "emails", "mail", "message"]
        ):
            return "email:read", 0.85

        # Notes intents - distinguish VERY clearly
        # Append/add to existing note: must come BEFORE create to prevent misclassification
        # Patterns: "add X to the list", "add X to my note", "put X on the list"
        if any(
            word in text_lower for word in ["add", "put", "append", "include"]
        ) and any(
            phrase in text_lower
            for phrase in [
                "to the list",
                "to my list",
                "to the note",
                "to my note",
                "on the list",
                "on my list",
                "to grocery",
                "to shopping",
            ]
        ):
            return "notes:append", 0.9
        # Create: "create/new/make + note" - requires explicit note/memo keyword
        if any(
            word in text_lower for word in ["create", "new", "make", "write"]
        ) and any(word in text_lower for word in ["note", "notes", "memo"]):
            return "notes:create", 0.9
        # "add a note" - create, not append (no "to the" structure)
        if text_lower.startswith("add") and any(
            word in text_lower for word in ["note", "memo"]
        ):
            return "notes:create", 0.9
        # List: "show/list + note(s)"
        if any(
            word in text_lower for word in ["show", "list", "display", "see"]
        ) and any(word in text_lower for word in ["note", "notes", "all notes"]):
            return "notes:list", 0.9
        # Search: "find/search + note(s)"
        if any(word in text_lower for word in ["find", "search"]) and any(
            word in text_lower for word in ["note", "notes", "memo"]
        ):
            return "notes:search", 0.9
        # Delete: "delete/remove + note(s)"
        if any(word in text_lower for word in ["delete", "remove"]) and any(
            word in text_lower for word in ["note", "notes", "memo"]
        ):
            return "notes:delete", 0.85

        # Thanks - check BEFORE greetings (to prevent "thanks" being matched by semantic classifier)
        if any(
            phrase in text_lower
            for phrase in ["thanks", "thank you", "thx", "thank", "thanks for"]
        ):
            return "thanks", 0.9

        # Status inquiry: "how are you", "how is alice", "how are you doing"
        if any(
            phrase in text_lower
            for phrase in [
                "how are you",
                "how are you doing",
                "how is it going",
                "how have you been",
            ]
        ):
            return "status_inquiry", 0.85

        # Greetings (high confidence) - must be after Thanks to not interfere
        greeting_words = ["hi", "hey", "hello", "yo", "sup", "hiya"]
        greeting_tokens = set(re.findall(r"\b[a-z']+\b", text_lower))
        if (
            any(word in greeting_tokens for word in greeting_words)
            and len(text_lower.split()) <= 4
        ):
            return "greeting", 0.9

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
            # Exclude vague pronouns: "what is that", "who is he/she"
            if not any(
                vague in text_lower
                for vague in [
                    "what is that",
                    "who is he",
                    "who is she",
                    "who is that person",
                ]
            ):
                return "conversation:question", 0.85

        # CLARIFICATION INTENTS (must run before semantic to catch vague patterns)
        # Vague pronouns without context: "who is he/she", "what is that", "who is that"
        if any(
            pattern in text_lower
            for pattern in [
                "who is he",
                "who is she",
                "who is that",
                "what is that",
                "who is that person",
            ]
        ):
            return "vague_question", 0.8

        # Vague temporal questions: "what about yesterday/tomorrow", "what happened last week"
        if any(
            word in text_lower
            for word in [
                "yesterday",
                "tomorrow",
                "last week",
                "last month",
                "next week",
                "next month",
            ]
        ) and (
            "what" in text_lower
            or "when" in text_lower
            or "what about" in text_lower
            or "what happened" in text_lower
        ):
            return "vague_temporal_question", 0.8

        # Vague requests without clear object: "add this to", "put this in", "do that thing", "can you do that thing"
        if any(
            pattern in text_lower
            for pattern in [
                "add this to",
                "put this in",
                "do that thing",
                "can you do that",
                "that thing",
                "this thing",
            ]
        ):
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

        # PHASE 2: Fallback to semantic classification
        # Try semantic classification for lower-confidence cases
        semantic_classifier = self._ensure_semantic_classifier()
        if semantic_classifier:
            try:
                result = semantic_classifier.get_plugin_action(text, threshold=0.4)
                if result and result.get("confidence", 0) > 0.5:
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
        forecast_phrases = [
            "forecast",
            "this week",
            "next week",
            "weekend",
            "tomorrow",
            "7 day",
            "7-day",
            "next few days",
            "next 7 days",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
        if any(phrase in text_lower for phrase in forecast_phrases):
            return "weather:forecast", 0.8

        if any(word in text_lower for word in ["weather", "temperature", "outside"]):
            return "weather:current", 0.75

        # Vague patterns requiring clarification
        vague_patterns = [
            "who is he",
            "who is she",
            "who is that",  # Pronoun ref without context
            "what is that",
            "what about that",  # Ambiguous reference
            "add this to",
            "put this in",  # Action without object context
            "what happened",
            "what about",  # Vague temporal/domain
            "how do i",
            "how can i",  # Generic how questions
            "tell me about the",
            "tell me about a",  # Ambiguous topic
        ]
        if any(pattern in text_lower for pattern in vague_patterns):
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
                    "music": "music:play",
                    "system": "system:status",
                }
                if plugin_name in plugin_intent_map:
                    return plugin_intent_map[plugin_name], min(
                        0.85, 0.55 + (score / 10.0)
                    )

        return "conversation:general", 0.3

    def _extract_all_entities(self, text: str) -> Dict[str, List[Entity]]:
        """Extract all entities (domain-specific + general)"""

        # Check cache
        cache_key = hash(text)
        with self._cache_lock:
            if cache_key in self._entity_cache:
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

        # Cache result
        with self._cache_lock:
            self._entity_cache[cache_key] = entities
            # Limit cache size
            if len(self._entity_cache) > 1000:
                # Remove oldest 20%
                to_remove = list(self._entity_cache.keys())[:200]
                for key in to_remove:
                    del self._entity_cache[key]

        return entities

    def _update_context(self, result: ProcessedQuery):
        """Update conversation context"""
        self.context.last_intent = result.intent
        self.context.query_history.append(result.original_text)
        self.context.last_plugin = (
            result.intent.split(":", 1)[0] if ":" in result.intent else "conversation"
        )
        self.context.dialogue_state = (
            "clarifying"
            if result.intent == "conversation:clarification_needed"
            else "active"
        )

        disambiguation = {}
        if isinstance(result.parsed_command, dict):
            disambiguation = result.parsed_command.get("modifiers", {}).get(
                "disambiguation", {}
            )
        self.context.pending_clarification = disambiguation if disambiguation else {}

        # Extract entities to context
        self.context.last_entities = {}

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
        
        # Check expected entities (soft bonus/penalty)
        expected = set(rules.get("expected", []))
        if expected:
            present_expected = expected & present_slots
            match_ratio = len(present_expected) / len(expected)
            if match_ratio < 0.5:  # Less than half of expected entities
                validation_score -= 0.1
                issues.append(f"Few expected entities: {match_ratio:.0%}")
        
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
