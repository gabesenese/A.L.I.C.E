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
    logging.warning("[WARN] NLTK not available. Using basic tokenization and neutral sentiment.")

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
    logging.warning("[WARN] dateparser not available. Temporal parsing will be limited.")

try:
    parsedatetime_mod = importlib.import_module("parsedatetime")
    Calendar = parsedatetime_mod.Calendar
except ImportError:  # pragma: no cover
    Calendar = None
    logging.warning("[WARN] parsedatetime not available. Temporal parsing will be limited.")

# Semantic intent classification
try:
    from ai.intent_classifier import get_intent_classifier
    SEMANTIC_CLASSIFIER_AVAILABLE = True
except ImportError:
    SEMANTIC_CLASSIFIER_AVAILABLE = False
    logging.warning("[WARN] Semantic intent classifier not available. Using fallback patterns.")

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


@dataclass
class ConversationContext:
    """Tracks conversation state for coreference resolution"""
    last_intent: Optional[str] = None
    last_entities: Dict[str, Any] = field(default_factory=dict)
    mentioned_notes: deque = field(default_factory=lambda: deque(maxlen=5))
    mentioned_events: deque = field(default_factory=lambda: deque(maxlen=5))
    mentioned_songs: deque = field(default_factory=lambda: deque(maxlen=3))
    query_history: deque = field(default_factory=lambda: deque(maxlen=10))


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
        'note_create': ['title', 'content', 'tags', 'priority', 'category', 'date', 'time', 'note_type'],
        'note_search': ['query', 'tags', 'date_range', 'priority', 'category'],
        'note_update': ['note_id', 'title', 'content', 'tags', 'priority'],
        'note_delete': ['note_id', 'query'],
        
        'music_play': ['song', 'artist', 'album', 'playlist', 'genre', 'mood', 'service'],
        'music_control': ['action', 'volume'],
        
        'calendar_create': ['event', 'date', 'time', 'duration', 'location', 'attendees', 'recurring'],
        'calendar_search': ['query', 'date_range', 'event_type'],
        
        'email_compose': ['recipient', 'subject', 'body', 'cc', 'bcc', 'attachments'],
        'email_search': ['sender', 'subject', 'date_range', 'has_attachment', 'status'],
    }
    
    def __init__(self, temporal_parser):
        self.temporal_parser = temporal_parser
        
        # Priority keywords
        self.priority_map = {
            'urgent': 'urgent',
            'critical': 'urgent',
            'asap': 'urgent',
            'immediately': 'urgent',
            'high': 'high',
            'important': 'high',
            'medium': 'medium',
            'normal': 'medium',
            'low': 'low',
            'minor': 'low',
        }
        
        # Note type keywords
        self.note_type_map = {
            'todo': 'todo',
            'task': 'todo',
            'checklist': 'todo',
            'idea': 'idea',
            'thought': 'idea',
            'brainstorm': 'idea',
            'meeting': 'meeting',
            'notes': 'meeting',
            'reminder': 'reminder',
            'alert': 'reminder',
        }
        
        # Category keywords
        self.category_map = {
            'work': 'work',
            'business': 'work',
            'office': 'work',
            'personal': 'personal',
            'home': 'personal',
            'family': 'personal',
            'project': 'project',
            'dev': 'project',
            'development': 'project',
            'health': 'health',
            'fitness': 'health',
            'study': 'study',
            'learning': 'study',
            'education': 'study',
        }
    
    def extract_slots(self, text: str, intent: str, entities: Dict[str, List[Entity]]) -> Dict[str, Slot]:
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
            extractor = getattr(self, f'_extract_{slot_name}', None)
            if extractor:
                value, confidence, raw = extractor(text, text_lower, entities)
                if value is not None:
                    slots[slot_name] = Slot(slot_name, value, confidence, raw)
        
        return slots
    
    def _get_template_key(self, intent: str) -> str:
        """Map intent to slot template"""
        # Note intents
        if 'create' in intent or 'add' in intent:
            if 'note' in intent:
                return 'note_create'
            elif 'event' in intent or 'calendar' in intent:
                return 'calendar_create'
        elif 'search' in intent or 'find' in intent or 'list' in intent:
            if 'note' in intent:
                return 'note_search'
            elif 'calendar' in intent:
                return 'calendar_search'
            elif 'email' in intent:
                return 'email_search'
        elif 'play' in intent or 'music' in intent:
            return 'music_play'
        elif 'email' in intent and ('compose' in intent or 'send' in intent):
            return 'email_compose'
        
        return intent
    
    # ==================== NOTE SLOT EXTRACTORS ====================
    
    def _extract_title(self, text: str, text_lower: str, entities: Dict) -> Tuple[Optional[str], float, str]:
        """Extract note title"""
        # Patterns like "note about X", "create X note", "X task"
        patterns = [
            r'(?:note|task|reminder)\s+(?:about|for|titled?)\s+([^,\n]+)',
            r'(?:create|add|make)\s+(?:a\s+)?(?:note\s+)?(?:about\s+)?([^,\n]+)',
            r'(?:called|named|titled)\s+([^,\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                title = match.group(1).strip()
                # Clean up common trailing words
                title = re.sub(r'\s+(tagged?|with|priority|at|on|tomorrow|today).*$', '', title)
                if len(title) > 2:
                    return title, 0.85, match.group(0)
        
        # Fallback: extract first noun phrase (simple heuristic)
        words = text_lower.split()
        if len(words) >= 3:
            # Look for pattern after action words
            action_words = {'create', 'add', 'make', 'new', 'note', 'task', 'reminder'}
            for i, word in enumerate(words):
                if word in action_words and i + 1 < len(words):
                    # Take next 2-4 words as title
                    title_words = words[i+1:min(i+5, len(words))]
                    title = ' '.join(title_words)
                    title = re.sub(r'\s+(tagged?|with|priority|at|on).*$', '', title)
                    if len(title) > 2:
                        return title, 0.6, ' '.join(words[i:i+5])
        
        return None, 0.0, ""
    
    def _extract_content(self, text: str, text_lower: str, entities: Dict) -> Tuple[Optional[str], float, str]:
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
    
    def _extract_tags(self, text: str, text_lower: str, entities: Dict) -> Tuple[Optional[List[str]], float, str]:
        """Extract tags from #tagname or 'tagged X'"""
        tags = []
        
        # Hashtag style: #work #urgent
        hashtag_pattern = r'#(\w+)'
        hashtags = re.findall(hashtag_pattern, text_lower)
        tags.extend(hashtags)
        
        # Explicit tagging: "tagged work urgent" or "tag it as work"
        tag_patterns = [
            r'tagged?\s+(?:as\s+)?(?:with\s+)?([a-z,\s]+?)(?:\s+priority|\s+category|$)',
            r'tags?\s+(?:are\s+)?(?:with\s+)?([a-z,\s]+?)(?:\s+priority|\s+category|$)',
        ]
        
        for pattern in tag_patterns:
            match = re.search(pattern, text_lower)
            if match:
                tag_text = match.group(1).strip()
                # Split by comma or space
                new_tags = [t.strip() for t in re.split(r'[,\s]+', tag_text) if t.strip()]
                tags.extend(new_tags)
        
        if tags:
            # Remove duplicates, keep order
            seen = set()
            unique_tags = []
            for tag in tags:
                if tag not in seen:
                    seen.add(tag)
                    unique_tags.append(tag)
            return unique_tags, 0.9, ' '.join(f'#{t}' for t in unique_tags)
        
        return None, 0.0, ""
    
    def _extract_priority(self, text: str, text_lower: str, entities: Dict) -> Tuple[Optional[str], float, str]:
        """Extract priority level"""
        for keyword, priority in self.priority_map.items():
            if re.search(rf'\b{keyword}\b', text_lower):
                return priority, 0.95, keyword
        
        # Check for explicit priority syntax
        priority_pattern = r'priority[:\s]+(urgent|high|medium|low)'
        match = re.search(priority_pattern, text_lower)
        if match:
            return match.group(1), 0.98, match.group(0)
        
        return None, 0.0, ""
    
    def _extract_category(self, text: str, text_lower: str, entities: Dict) -> Tuple[Optional[str], float, str]:
        """Extract category"""
        for keyword, category in self.category_map.items():
            if re.search(rf'\b{keyword}\b', text_lower):
                return category, 0.8, keyword
        
        # Explicit category syntax
        cat_pattern = r'category[:\s]+(\w+)'
        match = re.search(cat_pattern, text_lower)
        if match:
            return match.group(1), 0.95, match.group(0)
        
        return None, 0.0, ""
    
    def _extract_note_type(self, text: str, text_lower: str, entities: Dict) -> Tuple[Optional[str], float, str]:
        """Extract note type"""
        for keyword, note_type in self.note_type_map.items():
            if re.search(rf'\b{keyword}\b', text_lower):
                return note_type, 0.85, keyword
        
        return None, 0.0, ""
    
    def _extract_date(self, text: str, text_lower: str, entities: Dict) -> Tuple[Optional[str], float, str]:
        """Extract and normalize date"""
        result = self.temporal_parser.parse_temporal_expression(text)
        if result and result.get('date'):
            return result['date'], result.get('confidence', 0.8), result.get('raw_text', '')
        return None, 0.0, ""
    
    def _extract_time(self, text: str, text_lower: str, entities: Dict) -> Tuple[Optional[str], float, str]:
        """Extract and normalize time"""
        result = self.temporal_parser.parse_temporal_expression(text)
        if result and result.get('time'):
            return result['time'], result.get('confidence', 0.8), result.get('raw_text', '')
        return None, 0.0, ""
    
    # ==================== MUSIC SLOT EXTRACTORS ====================
    
    def _extract_song(self, text: str, text_lower: str, entities: Dict) -> Tuple[Optional[str], float, str]:
        """Extract song name"""
        patterns = [
            r'"([^"]+)"',  # Quoted
            r'play\s+([^,\n]+?)(?:\s+by|\s+from|$)',
            r'song\s+"?([^"]+?)"?(?:\s+by|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                song = match.group(1).strip()
                if len(song) > 1:
                    return song, 0.85, match.group(0)
        
        return None, 0.0, ""
    
    def _extract_artist(self, text: str, text_lower: str, entities: Dict) -> Tuple[Optional[str], float, str]:
        """Extract artist name"""
        patterns = [
            r'\bby\s+([A-Za-z\s&\']+?)(?:\s+from|\s+on|$)',
            r'\bartist\s+([A-Za-z\s&\']+?)(?:\s+from|$)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                artist = match.group(1).strip()
                if len(artist) > 1:
                    return artist, 0.9, match.group(0)
        
        return None, 0.0, ""
    
    def _extract_genre(self, text: str, text_lower: str, entities: Dict) -> Tuple[Optional[str], float, str]:
        """Extract music genre"""
        genres = ['rock', 'pop', 'jazz', 'classical', 'hip hop', 'rap', 'country', 
                 'electronic', 'blues', 'reggae', 'folk', 'metal', 'indie']
        
        for genre in genres:
            if re.search(rf'\b{genre}\b', text_lower):
                return genre, 0.95, genre
        
        return None, 0.0, ""
    
    def _extract_mood(self, text: str, text_lower: str, entities: Dict) -> Tuple[Optional[str], float, str]:
        """Extract mood/vibe"""
        moods = ['upbeat', 'relaxing', 'chill', 'energetic', 'slow', 'happy', 'sad', 
                'workout', 'study', 'sleep', 'party', 'romantic']
        
        for mood in moods:
            if re.search(rf'\b{mood}\b', text_lower):
                return mood, 0.85, mood
        
        return None, 0.0, ""
    
    def _extract_action(self, text: str, text_lower: str, entities: Dict) -> Tuple[Optional[str], float, str]:
        """Extract music action"""
        actions = ['play', 'pause', 'stop', 'skip', 'next', 'previous', 'resume', 'shuffle', 'repeat']
        
        for action in actions:
            if re.search(rf'\b{action}\b', text_lower):
                return action, 0.95, action
        
        return None, 0.0, ""
    
    def _extract_volume(self, text: str, text_lower: str, entities: Dict) -> Tuple[Optional[int], float, str]:
        """Extract volume level"""
        volume_match = re.search(r'\b(?:volume|sound)\s+(?:to\s+)?(\d{1,3})(?:%|percent)?\b', text_lower)
        if volume_match:
            volume = int(volume_match.group(1))
            return min(100, max(0, volume)), 0.98, volume_match.group(0)
        
        return None, 0.0, ""
    
    # ==================== OTHER SLOT EXTRACTORS ====================
    
    def _extract_query(self, text: str, text_lower: str, entities: Dict) -> Tuple[Optional[str], float, str]:
        """Extract search query"""
        # For search intents, the main query is usually the text minus action words
        query = re.sub(r'\b(search|find|show|list|get|fetch)\s+', '', text_lower, count=1)
        query = re.sub(r'\b(notes?|emails?|events?|tasks?)\b', '', query).strip()
        
        if len(query) > 2:
            return query, 0.7, query
        
        return None, 0.0, ""
    
    def _extract_album(self, text: str, text_lower: str, entities: Dict) -> Tuple[Optional[str], float, str]:
        """Extract album name"""
        return None, 0.0, ""  # Implement if needed
    
    def _extract_playlist(self, text: str, text_lower: str, entities: Dict) -> Tuple[Optional[str], float, str]:
        """Extract playlist name"""
        return None, 0.0, ""  # Implement if needed
    
    def _extract_service(self, text: str, text_lower: str, entities: Dict) -> Tuple[Optional[str], float, str]:
        """Extract music service"""
        services = ['spotify', 'apple music', 'youtube music', 'pandora']
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
            version_context_style = getattr(parsedatetime_mod, "VERSION_CONTEXT_STYLE", None)
            if version_context_style is not None:
                self.cal = Calendar(version=version_context_style)
            else:
                self.cal = Calendar()
        else:
            self.cal = None
        
        # Time of day mappings
        self.time_of_day = {
            'morning': '09:00',
            'noon': '12:00',
            'afternoon': '14:00',
            'evening': '18:00',
            'night': '20:00',
            'midnight': '00:00',
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
                    'PREFER_DATES_FROM': 'future',
                    'RETURN_AS_TIMEZONE_AWARE': False,
                    'RELATIVE_BASE': datetime.now()
                }
            )
        
        if parsed_date:
            result['date'] = parsed_date.strftime('%Y-%m-%d')
            result['time'] = parsed_date.strftime('%H:%M')
            confidence = 0.9
            raw_text = text
        else:
            # Try parsedatetime for relative expressions
            if self.cal is not None:
                time_struct, parse_status = self.cal.parse(text)
                if parse_status > 0:
                    parsed_dt = datetime(*time_struct[:6])
                    result['date'] = parsed_dt.strftime('%Y-%m-%d')
                    result['time'] = parsed_dt.strftime('%H:%M')
                    confidence = 0.8
                    raw_text = text
        
        # Check for time of day keywords
        text_lower = text.lower()
        for keyword, time_str in self.time_of_day.items():
            if keyword in text_lower:
                result['time'] = time_str
                confidence = max(confidence, 0.85)
                if not raw_text:
                    raw_text = keyword
        
        if result:
            result['confidence'] = confidence
            result['raw_text'] = raw_text
            return result
        
        return None
    
    def normalize_duration(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse duration expressions"""
        patterns = [
            (r'(\d+)\s*(?:minute|min)s?', 'minutes'),
            (r'(\d+)\s*(?:hour|hr)s?', 'hours'),
            (r'(\d+)\s*(?:day)s?', 'days'),
            (r'(\d+)\s*(?:week)s?', 'weeks'),
            (r'(\d+)\s*(?:month)s?', 'months'),
        ]
        
        for pattern, unit in patterns:
            match = re.search(pattern, text.lower())
            if match:
                value = int(match.group(1))
                return {'value': value, 'unit': unit, 'raw_text': match.group(0)}
        
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
        'NOTE_TAG': r'#(\w+)',
        'PRIORITY': r'\b(urgent|critical|high|important|medium|normal|low|minor)\b',
        'NOTE_TYPE': r'\b(todo|task|idea|thought|meeting|reminder)\b',
        'CATEGORY': r'\b(work|personal|project|health|study)\b',
        'MUSIC_GENRE': r'\b(rock|pop|jazz|classical|hip hop|rap|country|electronic|blues|metal)\b',
        'MUSIC_MOOD': r'\b(upbeat|relaxing|chill|energetic|happy|sad|workout|study)\b',
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
                    end_pos=match.end()
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
        self.pronouns = ['it', 'this', 'that', 'the note', 'the event', 'the song', 
                        'the task', 'the email', 'the reminder']
    
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
                        rf'\b{re.escape(pronoun)}\b',
                        replacement,
                        resolved_text,
                        count=1,
                        flags=re.IGNORECASE
                    )
                    logger.info(f"[COREF] Resolved '{pronoun}' -> '{replacement}'")
        
        return resolved_text
    
    def _find_referent(self, pronoun: str, context: ConversationContext) -> Optional[str]:
        """Find what the pronoun refers to"""
        # Check based on last intent
        if context.last_intent:
            if 'note' in context.last_intent and context.mentioned_notes:
                return str(context.mentioned_notes[-1])
            elif 'calendar' in context.last_intent and context.mentioned_events:
                return str(context.mentioned_events[-1])
            elif 'music' in context.last_intent and context.mentioned_songs:
                return str(context.mentioned_songs[-1])
        
        # Check generic entities
        if context.last_entities:
            # Return most recent entity of appropriate type
            for entity_type in ['title', 'song', 'event', 'note_id']:
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
        'angry': ['angry', 'mad', 'pissed', 'furious', 'annoyed', 'irritated'],
        'excited': ['excited', 'awesome', 'amazing', 'great', 'fantastic', 'wonderful'],
        'worried': ['worried', 'concerned', 'anxious', 'nervous', 'stressed'],
        'confused': ['confused', 'lost', 'unclear', 'dont understand'],
        'satisfied': ['thanks', 'thank you', 'perfect', 'exactly', 'good'],
        'frustrated': ['not working', 'broken', 'frustrated', 'cant', 'wont', 'doesnt work'],
    }
    
    URGENCY_KEYWORDS = {
        'critical': ['emergency', 'critical', 'asap', 'immediately', 'right now', 'urgent'],
        'high': ['urgent', 'important', 'soon', 'quickly', 'hurry'],
        'medium': ['when you can', 'sometime', 'later'],
        'low': ['no rush', 'whenever', 'eventually'],
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
            compound = sentiment_scores.get('compound', 0)
            if compound > 0.5:
                emotions.append('satisfied')
            elif compound < -0.5:
                emotions.append('frustrated')
        
        return emotions
    
    def detect_urgency(self, text: str) -> str:
        """Detect urgency level"""
        text_lower = text.lower()
        
        # Check from highest to lowest
        for level in ['critical', 'high', 'medium', 'low']:
            for keyword in self.URGENCY_KEYWORDS[level]:
                if keyword in text_lower:
                    return level
        
        return 'none'


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
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        
        # Core components
        self.sentiment_analyzer = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2)) if TfidfVectorizer else None
        
        # Advanced components
        self.temporal_parser = TemporalParser()
        self.slot_filler = SlotFiller(self.temporal_parser)
        self.domain_ner = DomainEntityExtractor()
        self.coref_resolver = CoreferenceResolver()
        self.emotion_detector = EmotionDetector()
        
        # Conversation context
        self.context = ConversationContext()
        
        # Semantic intent classifier
        self.semantic_classifier = None
        if SEMANTIC_CLASSIFIER_AVAILABLE:
            try:
                self.semantic_classifier = get_intent_classifier()
                logger.info("[OK] Semantic intent classifier loaded")
            except Exception as e:
                logger.warning(f"[WARN] Failed to load semantic classifier: {e}")
        
        # Load learned corrections into pattern matching
        self.learned_corrections = self._load_learned_corrections()
        self.tokenizer_profile = "default"
        self.command_vocabulary = self._load_command_vocabulary()
        
        # Cache for performance
        self._entity_cache = {}
        self._cache_lock = threading.Lock()

    def set_tokenizer_profile(self, profile: str) -> None:
        """Set tokenizer behavior profile: strict, default, llm-assisted."""
        profile_value = (profile or "default").strip().lower()
        if profile_value not in {"strict", "default", "llm-assisted"}:
            profile_value = "default"
        self.tokenizer_profile = profile_value

    def _load_command_vocabulary(self) -> Dict[str, Set[str]]:
        """Load shared command vocabulary from command directory with safe defaults."""
        verbs = {
            "create", "add", "make", "write", "append", "show", "list", "read", "open",
            "find", "search", "delete", "remove", "edit", "update", "summarize", "archive",
            "unarchive", "pin", "unpin", "play", "pause", "send", "check",
        }
        objects = {
            "note", "notes", "list", "lists", "task", "tasks", "email", "emails", "calendar",
            "event", "events", "music", "song", "songs", "reminder", "reminders",
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
            logger.debug(f"Could not load command vocabulary from command_directory.md: {e}")

        return {
            "verbs": verbs,
            "objects": objects,
            "meta": meta,
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
            if re.search(r"\b(?:plugin|note|email|calendar|music|task|reminder)s?\b", segment_text, re.IGNORECASE):
                kind = "plugin_phrase" if kind == "utterance" else kind

            segments.append(TokenSegment(text=segment_text, kind=kind, start_pos=start, end_pos=end))

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
            normalized = re.sub(rf"\b{re.escape(source)}\b", target, normalized, flags=re.IGNORECASE)

        typo_map = {
            "notets": "notes",
            "nots": "notes",
            "emial": "email",
            "calender": "calendar",
        }
        for source, target in typo_map.items():
            normalized = re.sub(rf"\b{re.escape(source)}\b", target, normalized, flags=re.IGNORECASE)

        normalized = re.sub(r"\binside of\b", "in", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bi wanna\b", "i want to", normalized, flags=re.IGNORECASE)

        if self.tokenizer_profile == "strict":
            normalized = re.sub(r"\b(?:please|kindly)\b", "", normalized, flags=re.IGNORECASE)
            normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    def _lexical_tokenize(self, segments: List[TokenSegment]) -> List[RichToken]:
        """Lexical layer: tokenize segments into rich tokens with type/role/span metadata."""
        tokens: List[RichToken] = []
        token_pattern = re.compile(r'"[^"]+"|#\w+|\d+(?:st|nd|rd|th)?|[A-Za-z]+|[^\w\s]')

        action_words = self.command_vocabulary.get("verbs", set())
        object_words = self.command_vocabulary.get("objects", set())
        pronouns = {"this", "that", "it", "them", "those", "these", "one", "last"}
        ordinal_words = {
            "first", "second", "third", "fourth", "fifth", "sixth",
            "seventh", "eighth", "ninth", "tenth",
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
                        "is_meta_command": segment.kind == "meta_command" or norm in meta_words,
                    },
                )
                tokens.append(token)

        return tokens

    def _extract_semantic_hints(self, text: str, tokens: List[RichToken]) -> ParsedCommand:
        """Semantic hint layer: derive structured command from token sequence."""
        token_text = [token.normalized for token in tokens if token.kind != "symbol"]
        lower = text.lower()
        parsed = ParsedCommand()

        if text.endswith("?") or (token_text and token_text[0] in {"what", "when", "where", "who", "why", "how", "do", "is", "are", "can"}):
            parsed.sentence_type = "question"
        elif token_text and token_text[0] in self.command_vocabulary.get("verbs", set()):
            parsed.sentence_type = "imperative"

        if re.search(r"\b(do i have|is there)\b", lower) and re.search(r"\bnote|notes|list|lists\b", lower):
            parsed.action = "query_exist"
            parsed.object_type = "note"
        elif re.search(r"\b(list|show)\b", lower) and re.search(r"\bnotes?|lists?\b", lower):
            parsed.action = "list"
            parsed.object_type = "note"
        elif re.search(r"\b(read|open|show)\b", lower) and re.search(r"\bnotes?|lists?|it|that|this\b", lower):
            parsed.action = "read"
            parsed.object_type = "note"
        elif re.search(r"\b(create|make|new|add)\b", lower) and re.search(r"\bnotes?|memo\b", lower):
            parsed.action = "create"
            parsed.object_type = "note"
        elif re.search(r"\b(add|append|put|include)\b", lower) and re.search(r"\bto\b", lower) and re.search(r"\bnotes?|lists?\b", lower):
            parsed.action = "append"
            parsed.object_type = "note"

        title_match = re.search(r"(?:called|named|titled|about)\s+([a-z0-9\s'\-]+)$", lower)
        if not title_match:
            title_match = re.search(r"(?:read|open|show)\s+(?:the\s+)?([a-z0-9\s'\-]+?)\s+notes?\b", lower)
        if title_match:
            parsed.title_hint = title_match.group(1).strip(" .,!?")

        for token in tokens:
            if token.flags.get("is_reference"):
                parsed.references.append({
                    "text": token.normalized,
                    "start": token.start_pos,
                    "end": token.end_pos,
                })

        if re.search(r"\b(?:my|mine)\b", lower):
            parsed.modifiers["scope"] = "my"
        if re.search(r"\b(today|tomorrow|yesterday|this week|next week)\b", lower):
            parsed.modifiers["time"] = re.search(r"\b(today|tomorrow|yesterday|this week|next week)\b", lower).group(1)
        if re.search(r"\b(low|medium|high|urgent)\b", lower):
            parsed.modifiers["priority"] = re.search(r"\b(low|medium|high|urgent)\b", lower).group(1)

        return parsed

    def _compute_plugin_scores(self, tokens: List[RichToken], parsed: ParsedCommand) -> Dict[str, float]:
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
        if parsed.action in {"read", "append", "create", "list", "query_exist"} and parsed.object_type == "note":
            scores["notes"] += 1.0
        if parsed.references:
            scores["notes"] += 0.6

        if self.tokenizer_profile == "strict":
            for key in ("email", "calendar", "music", "system"):
                if scores[key] < 1.0:
                    scores[key] *= 0.7

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
                with open(corrections_file, 'r', encoding='utf-8') as f:
                    patterns = json.load(f)
                
                if isinstance(patterns, dict) and 'corrections' in patterns:
                    learned = {}
                    for correction in patterns['corrections']:
                        if 'user_input' in correction and 'expected_intent' in correction:
                            # Map user input to learned intent for fast override
                            learned[correction['user_input'].lower()] = correction['expected_intent']
                    
                    if learned:
                        logger.info(f"Loaded {len(learned)} learned corrections into NLP processor")
                    return learned
        except Exception as e:
            logger.debug(f"Could not load learned corrections: {e}")
        
        return {}
        
        logger.info("[OK] NLPProcessor initialized with advanced semantic understanding")
    
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
            resolved_text = self.coref_resolver.resolve(text, self.context)
        else:
            resolved_text = text

        # Step 2: Clean + normalize for tokenizer layers
        clean_text = self._clean_text(resolved_text)
        normalized_text = self._normalize_for_tokenizer(clean_text)

        # Step 3: Surface + lexical + semantic hint layers
        segments = self._surface_segment(normalized_text)
        rich_tokens = self._lexical_tokenize(segments)
        parsed_command = self._extract_semantic_hints(normalized_text, rich_tokens)
        plugin_scores = self._compute_plugin_scores(rich_tokens, parsed_command)
        tokens = [
            token.normalized
            for token in rich_tokens
            if token.kind in {"word", "number", "ordinal", "pronoun", "hashtag", "quoted_span", "date_like"}
        ]

        # Step 4: Check learned corrections FIRST (highest priority)
        if self.learned_corrections and normalized_text.lower() in self.learned_corrections:
            learned_intent = self.learned_corrections[normalized_text.lower()]
            logger.info(f"[LEARNED] Using correction for '{normalized_text}' -> {learned_intent}")
            intent = learned_intent
            intent_confidence = 0.95  # High confidence for learned patterns
        else:
            # Step 5: Intent detection (semantic + structured hints)
            intent, intent_confidence = self._detect_intent_semantic(
                normalized_text,
                parsed_command=parsed_command,
                plugin_scores=plugin_scores,
            )
        
        # Continue with rest of processing

        # Step 3: Entity extraction
        entities = self._extract_all_entities(normalized_text)
        
        # Step 4: Slot filling
        slots = self.slot_filler.extract_slots(normalized_text, intent, entities)
        
        # Step 5: Sentiment analysis
        if self.sentiment_analyzer is not None:
            sentiment = self.sentiment_analyzer.polarity_scores(normalized_text)
            compound = sentiment.get('compound', 0)
            sentiment['category'] = 'positive' if compound >= 0.05 else ('negative' if compound <= -0.05 else 'neutral')
        else:
            sentiment = {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0, 'category': 'neutral'}
        
        # Step 6: Emotion detection
        emotions = self.emotion_detector.detect_emotions(normalized_text, sentiment)
        
        # Step 7: Urgency detection
        urgency = self.emotion_detector.detect_urgency(normalized_text)
        
        # Step 8: Keyword extraction
        keywords = self._extract_keywords(normalized_text)
        
        # Step 9: Check if question
        is_question = self._is_question(normalized_text)
        
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
        )
        
        # Step 10: Update conversation context
        if use_context:
            self._update_context(result)
        
        logger.info(f"[NLP] Intent: {intent} ({intent_confidence:.2f}) | Slots: {len(slots)} | Emotions: {emotions} | Urgency: {urgency}")
        
        return result
    
    def _detect_intent_semantic(
        self,
        text: str,
        parsed_command: Optional[ParsedCommand] = None,
        plugin_scores: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, float]:
        """Detect intent using explicit patterns (primary) then semantic classifier (fallback)"""
        
        text_lower = text.lower()

        if text_lower.startswith('/debug tokens'):
            return 'system:debug_tokens', 0.98

        if parsed_command and parsed_command.object_type == 'note':
            action_map = {
                'query_exist': 'notes:list',
                'list': 'notes:list',
                'read': 'notes:read',
                'append': 'notes:append',
                'create': 'notes:create',
            }
            mapped = action_map.get(parsed_command.action)
            if mapped:
                return mapped, 0.88
        
        # PHASE 1: HIGH-CONFIDENCE EXPLICIT PATTERNS (these override semantic classification)
        # System intents - check BEFORE time (system has "how's" pattern that could match time)
        if 'system' in text_lower and any(word in text_lower for word in ['status', 'doing', 'health', 'how']):
            return 'system:status', 0.9
        # Resource check: "how is/is my + cpu/memory/disk/battery"
        if any(word in text_lower for word in ['how is', 'is my', 'is the']) and any(word in text_lower for word in ['cpu', 'memory', 'disk', 'battery', 'gpu']):
            return 'system:status', 0.9
        if any(word in text_lower for word in ['cpu', 'memory', 'disk', 'battery']) and any(word in text_lower for word in ['usage', 'available', 'how much', 'low', 'check', 'is']):
            return 'system:status', 0.9

        # Email intents - VERY explicit: action word + email word (0.85-0.9 confidence)
        # Compose: "compose/draft/write + email/mail"
        if any(word in text_lower for word in ['compose', 'draft', 'write', 'send']) and any(word in text_lower for word in ['email', 'mail', 'message', 'to']):
            return 'email:compose', 0.9
        # Delete: "delete/remove/trash + email/mail"
        if any(word in text_lower for word in ['delete', 'remove', 'trash']) and any(word in text_lower for word in ['email', 'mail', 'message']):
            return 'email:delete', 0.9
        # Reply: "reply/respond to email/mail"
        if any(word in text_lower for word in ['reply', 'respond']) and any(word in text_lower for word in ['email', 'mail', 'message']):
            return 'email:reply', 0.85
        # Search: "find/search + email/mail + (from/subject/sender)"
        if any(word in text_lower for word in ['search', 'find', 'look for']) and any(word in text_lower for word in ['email', 'mail', 'inbox', 'message', 'from']):
            return 'email:search', 0.85
        # List: "show/list/recent + email(s)/mail(s)"
        if any(word in text_lower for word in ['show', 'list', 'recent', 'latest']) and any(word in text_lower for word in ['email', 'emails', 'mail', 'mails', 'inbox']):
            return 'email:list', 0.85
        # Read: "read/open/display + email(s)/mail/message + optional (first/last/latest/number)"
        if any(word in text_lower for word in ['read', 'open', 'display', 'view']) and any(word in text_lower for word in ['email', 'emails', 'mail', 'message']):
            return 'email:read', 0.85

        # Notes intents - distinguish VERY clearly
        # Append/add to existing note: must come BEFORE create to prevent misclassification
        # Patterns: "add X to the list", "add X to my note", "put X on the list"
        if any(word in text_lower for word in ['add', 'put', 'append', 'include']) and \
           any(phrase in text_lower for phrase in ['to the list', 'to my list', 'to the note', 'to my note', 'on the list', 'on my list', 'to grocery', 'to shopping']):
            return 'notes:append', 0.9
        # Create: "create/new/make + note" - requires explicit note/memo keyword
        if any(word in text_lower for word in ['create', 'new', 'make', 'write']) and any(word in text_lower for word in ['note', 'notes', 'memo']):
            return 'notes:create', 0.9
        # "add a note" - create, not append (no "to the" structure)
        if text_lower.startswith('add') and any(word in text_lower for word in ['note', 'memo']):
            return 'notes:create', 0.9
        # List: "show/list + note(s)"
        if any(word in text_lower for word in ['show', 'list', 'display', 'see']) and any(word in text_lower for word in ['note', 'notes', 'all notes']):
            return 'notes:list', 0.9
        # Search: "find/search + note(s)"
        if any(word in text_lower for word in ['find', 'search']) and any(word in text_lower for word in ['note', 'notes', 'memo']):
            return 'notes:search', 0.9
        # Delete: "delete/remove + note(s)"
        if any(word in text_lower for word in ['delete', 'remove']) and any(word in text_lower for word in ['note', 'notes', 'memo']):
            return 'notes:delete', 0.85

        # Thanks - check BEFORE greetings (to prevent "thanks" being matched by semantic classifier)
        if any(phrase in text_lower for phrase in ['thanks', 'thank you', 'thx', 'thank', 'thanks for']):
            return 'thanks', 0.9

        # Status inquiry: "how are you", "how is alice", "how are you doing"
        if any(phrase in text_lower for phrase in ['how are you', 'how are you doing', 'how is it going', 'how have you been']):
            return 'status_inquiry', 0.85

        # Greetings (high confidence) - must be after Thanks to not interfere
        greeting_words = ['hi', 'hey', 'hello', 'yo', 'sup', 'hiya']
        if any(word in text_lower for word in greeting_words) and len(text_lower.split()) <= 4:
            return 'greeting', 0.9

        # CLARIFICATION INTENTS (must run before semantic to catch vague patterns)
        # Vague pronouns without context: "who is he/she", "what is that", "who is that"
        if any(pattern in text_lower for pattern in ['who is he', 'who is she', 'who is that', 'what is that', 'who is that person']):
            return 'vague_question', 0.8
        
        # Vague temporal questions: "what about yesterday/tomorrow", "what happened last week"
        if any(word in text_lower for word in ['yesterday', 'tomorrow', 'last week', 'last month', 'next week', 'next month']) and ('what' in text_lower or 'when' in text_lower or 'what about' in text_lower or 'what happened' in text_lower):
            return 'vague_temporal_question', 0.8
        
        # Vague requests without clear object: "add this to", "put this in", "do that thing", "can you do that thing"
        if any(pattern in text_lower for pattern in ['add this to', 'put this in', 'do that thing', 'can you do that', 'that thing', 'this thing']):
            return 'vague_request', 0.8
        
        # Schedule action without specifics
        if 'schedule' in text_lower and any(word in text_lower for word in ['it', 'that', 'this']) and any(word in text_lower for word in ['for', 'at']):
            return 'schedule_action', 0.8
        
        # Tell me about - ambiguous topic
        if any(pattern in text_lower for pattern in ['tell me about the', 'tell me about a', 'about the sun', 'about the moon', 'about the']):
            if len(text_lower.split()) <= 6:  # Short phrase = likely vague
                return 'vague_question', 0.75

        # PHASE 2: Fallback to semantic classification
        # Try semantic classification for lower-confidence cases
        if self.semantic_classifier:
            try:
                result = self.semantic_classifier.get_plugin_action(text, threshold=0.4)
                if result and result.get('confidence', 0) > 0.5:
                    # Map plugin:action to intent
                    plugin = result.get('plugin', '')
                    action = result.get('action', '')
                    intent = f"{plugin}:{action}"
                    confidence = result.get('confidence', 0.0)
                    
                    logger.debug(f"[NLP] Semantic intent: {intent} ({confidence:.3f})")
                    return intent, confidence
            except Exception as e:
                logger.warning(f"[WARN] Semantic classification failed: {e}")
        
        # PHASE 3: Additional fallback patterns

        # Clarification/meta-question cues
        if any(phrase in text_lower for phrase in [
            'can i ask you something about',
            'i have a question about',
            'let me ask you about',
            'can i ask you about'
        ]):
            return 'conversation:meta_question', 0.8

        # Thanks
        if any(phrase in text_lower for phrase in ['thanks', 'thank you', 'thx']):
            return 'thanks', 0.9

        # Status inquiry
        if any(phrase in text_lower for phrase in ['how are you', 'how are you doing', 'how is it going']):
            return 'status_inquiry', 0.85

        # Weather intents - CHECK EARLY before vague patterns
        # This ensures "is that wednesday?" triggers weather:forecast, not vague_question
        forecast_phrases = [
            'forecast', 'this week', 'next week', 'weekend', 'tomorrow',
            '7 day', '7-day', 'next few days', 'next 7 days',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'
        ]
        if any(phrase in text_lower for phrase in forecast_phrases):
            return 'weather:forecast', 0.8

        if any(word in text_lower for word in ['weather', 'temperature', 'outside']):
            return 'weather:current', 0.75

        # Vague patterns requiring clarification
        vague_patterns = [
            'who is he', 'who is she', 'who is that',  # Pronoun ref without context
            'what is that', 'what about that',         # Ambiguous reference
            'add this to', 'put this in',              # Action without object context
            'what happened', 'what about',             # Vague temporal/domain
            'how do i', 'how can i',                   # Generic how questions
            'tell me about the', 'tell me about a',    # Ambiguous topic
        ]
        if any(pattern in text_lower for pattern in vague_patterns):
            return 'vague_question', 0.75
        if '?' in text:
            return 'conversation:question', 0.5

        if plugin_scores:
            best_plugin = max(plugin_scores.items(), key=lambda item: item[1])
            plugin_name, score = best_plugin
            if score >= 2.2:
                plugin_intent_map = {
                    'notes': 'notes:list',
                    'email': 'email:list',
                    'calendar': 'calendar:list',
                    'music': 'music:play',
                    'system': 'system:status',
                }
                if plugin_name in plugin_intent_map:
                    return plugin_intent_map[plugin_name], min(0.85, 0.55 + (score / 10.0))

        return 'conversation:general', 0.3
    
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
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'PHONE': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'URL': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'NUMBER': r'\b\d+\b',
            'PERCENTAGE': r'\b\d+(?:\.\d+)?%\b',
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
                    end_pos=match.end()
                )
                entity_list.append(entity)
            
            if entity_list:
                entities[entity_type] = entity_list

        # Time range entities (for forecast queries)
        time_range_patterns = {
            'this week': 'week',
            'next week': 'next_week',
            'weekend': 'weekend',
            'tomorrow': 'tomorrow',
            'next few days': 'next_few_days',
            'next 7 days': '7_days',
            '7 day': '7_days',
            '7-day': '7_days',
            'monday': 'monday',
            'tuesday': 'tuesday',
            'wednesday': 'wednesday',
            'thursday': 'thursday',
            'friday': 'friday',
            'saturday': 'saturday',
            'sunday': 'sunday'
        }

        time_range_entities = []
        for phrase, normalized in time_range_patterns.items():
            if phrase in text_lower:
                time_range_entities.append(Entity(
                    type='TIME_RANGE',
                    value=phrase,
                    confidence=0.85,
                    normalized_value=normalized
                ))

        if time_range_entities:
            entities['TIME_RANGE'] = time_range_entities
        
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
        
        # Extract entities to context
        self.context.last_entities = {}
        
        # From slots
        for slot_name, slot in result.slots.items():
            self.context.last_entities[slot_name] = slot.value
            
            # Track specific entity types
            if slot_name == 'title' and 'note' in result.intent:
                self.context.mentioned_notes.append(slot.value)
            elif slot_name == 'song':
                self.context.mentioned_songs.append(slot.value)
            elif slot_name == 'event':
                self.context.mentioned_events.append(slot.value)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s.,!?\'\-/#:;]', '', text)
        return text
    
    def _is_question(self, text: str) -> bool:
        """Check if text is a question"""
        if '?' in text:
            return True
        
        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which', 
                         'can', 'could', 'would', 'should', 'is', 'are', 'do', 'does']
        
        text_lower = text.lower()
        return any(text_lower.startswith(qw) for qw in question_words)
    
    def _extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Extract top keywords"""
        tokens = word_tokenize(text.lower())
        
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                    'should', 'may', 'might', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
        
        keywords = [token for token in tokens if token.isalnum() and token not in stopwords]
        
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
