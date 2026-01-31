"""
Unified Context Engine for A.L.I.C.E
Consolidates context_manager + advanced_context_handler
Manages conversation state, user preferences, entity tracking, and coreference resolution
"""

import json
import os
import logging
import re
import pickle
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UserPreferences:
    """User preferences and settings"""
    name: str = "User"
    preferred_voice: str = "default"
    temperature_preference: float = 0.7
    verbose_mode: bool = False
    proactive_suggestions: bool = True
    location: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    timezone: Optional[str] = None
    custom_settings: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.custom_settings is None:
            self.custom_settings = {}
        
        # Auto-detect location if not set
        if self.location is None or self.city is None:
            self._detect_location()
    
    def _detect_location(self):
        """Auto-detect user location using IP geolocation"""
        try:
            import requests
            logger.info("Attempting to detect location...")
            response = requests.get('http://ip-api.com/json/', timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    self.city = data.get('city')
                    self.country = data.get('country')
                    self.location = f"{self.city}, {self.country}"
                    self.timezone = data.get('timezone')
                    logger.info(f"Location detected: {self.location} (IP-based, may not be exact)")
                    return
            
            logger.warning("Location detection returned no data")
            self._set_unknown_location()
        except Exception as e:
            logger.warning(f"Could not auto-detect location: {e}")
            self._set_unknown_location()
    
    def _set_unknown_location(self):
        """Set location to unknown when detection fails"""
        self.location = None
        self.city = None
        self.country = None
    
    def set_location(self, city: str, country: str = None):
        """Manually set user location"""
        self.city = city
        self.country = country
        if country:
            self.location = f"{city}, {country}"
        else:
            self.location = city
        logger.info(f"Location manually set to: {self.location}")


@dataclass
class ConversationState:
    """Current conversation state"""
    active_topic: Optional[str] = None
    mentioned_entities: List[str] = None
    last_intent: Optional[str] = None
    pending_tasks: List[str] = None
    context_window: List[Dict] = None
    session_start: datetime = None
    last_interaction: datetime = None
    
    def __post_init__(self):
        if self.mentioned_entities is None:
            self.mentioned_entities = []
        if self.pending_tasks is None:
            self.pending_tasks = []
        if self.context_window is None:
            self.context_window = []
        if self.session_start is None:
            self.session_start = datetime.now()
        if self.last_interaction is None:
            self.last_interaction = datetime.now()


@dataclass
class ContextEntity:
    """Represents an entity that can be referenced in conversation"""
    entity_id: str
    entity_type: str  # email, file, person, topic, etc.
    data: Dict[str, Any]
    mentions: List[str] = field(default_factory=list)
    last_mentioned: datetime = field(default_factory=datetime.now)
    relevance_score: float = 1.0
    aliases: List[str] = field(default_factory=list)


@dataclass
class ConversationTurn:
    """Represents a single conversation turn with metadata"""
    turn_id: str
    user_input: str
    assistant_response: str
    intent: str
    entities_mentioned: List[str]
    entities_resolved: Dict[str, str]
    timestamp: datetime
    context_used: List[str]
    pragmatic_signals: Dict[str, Any] = field(default_factory=dict)


class ContextEngine:
    """
    Unified context management system for A.L.I.C.E
    Combines:
    - User preferences and personalization
    - Conversation state and history
    - Entity tracking and coreference resolution
    - Pragmatic understanding
    - Short-term and long-term memory
    """
    
    def __init__(self, data_dir: str = "data/context"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Core components
        self.user_prefs: UserPreferences = UserPreferences()
        self.conv_state: ConversationState = ConversationState()
        
        # Memory systems
        self.short_term_memory: List[Dict] = []
        self.working_memory: Dict[str, Any] = {}
        self.semantic_memory: Dict[str, Any] = {}
        
        # System state
        self.system_status: Dict[str, Any] = {
            "online": True,
            "capabilities": [],
            "active_plugins": [],
            "performance": {}
        }
        
        # Entity tracking
        self.entities: Dict[str, ContextEntity] = {}
        self.current_focus: List[str] = []
        
        # Coreference resolution
        self.pronoun_stack: List[str] = []
        self.demonstrative_stack: List[str] = []
        self.conversation_turns: List[ConversationTurn] = []
        
        # Entity patterns for resolution
        self.entity_patterns = {
            "email": {
                "indicators": ["email", "message", "mail"],
                "pronouns": ["it", "this", "that", "the email", "the message"],
                "properties": ["from", "to", "subject", "content", "date"]
            },
            "person": {
                "indicators": ["from", "by", "person", "user", "sender"],
                "pronouns": ["he", "she", "they", "him", "her", "them"],
                "properties": ["name", "email_address", "relationship"]
            },
            "file": {
                "indicators": ["file", "document", "report", "attachment"],
                "pronouns": ["it", "this", "that", "the file"],
                "properties": ["name", "path", "type", "size"]
            },
            "topic": {
                "indicators": ["topic", "subject", "matter", "issue"],
                "pronouns": ["it", "this", "that"],
                "properties": ["name", "keywords", "related_entities"]
            }
        }
        
        # Discourse markers for pragmatic understanding
        self.discourse_markers: Dict[str, str] = {
            "by the way": "topic_shift",
            "speaking of": "topic_continuation",
            "anyway": "topic_shift",
            "also": "topic_addition",
            "but": "contrast",
            "however": "contrast",
            "meanwhile": "parallel_context"
        }
        
        # Load saved context
        self._load_context()
        
        logger.info("[OK] Context Engine initialized")
    
    def _load_context(self):
        """Load saved context from disk"""
        try:
            prefs_path = os.path.join(self.data_dir, "user_prefs.json")
            if os.path.exists(prefs_path):
                with open(prefs_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.user_prefs = UserPreferences(**data)
                logger.info(f"📂 Loaded preferences for {self.user_prefs.name}")
            
            memory_path = os.path.join(self.data_dir, "semantic_memory.pkl")
            if os.path.exists(memory_path):
                with open(memory_path, 'rb') as f:
                    self.semantic_memory = pickle.load(f)
                logger.info(f"Loaded {len(self.semantic_memory)} memory entries")
                
        except Exception as e:
            logger.warning(f"[WARNING] Could not load context: {e}")
    
    def save_context(self):
        """Save context to disk"""
        try:
            # Save user preferences
            prefs_path = os.path.join(self.data_dir, "user_prefs.json")
            with open(prefs_path, 'w', encoding='utf-8') as f:
                prefs_dict = asdict(self.user_prefs)
                json.dump(prefs_dict, f, indent=2)
            
            # Save semantic memory
            memory_path = os.path.join(self.data_dir, "semantic_memory.pkl")
            with open(memory_path, 'wb') as f:
                pickle.dump(self.semantic_memory, f)
            
            logger.info("Context saved successfully")
            
        except Exception as e:
            logger.error(f"[ERROR] Error saving context: {e}")
    
    def process_turn(self, user_input: str, assistant_response: str = "",
                    intent: str = "", entities: Dict = None) -> ConversationTurn:
        """Process a conversation turn and update context"""
        
        turn_id = f"turn_{len(self.conversation_turns) + 1}_{int(datetime.now().timestamp())}"
        
        # Extract and resolve entities
        mentioned_entities = self._extract_entities(user_input)
        resolved_entities = self._resolve_coreferences(user_input, mentioned_entities)
        
        # Detect pragmatic signals
        pragmatic_signals = self._analyze_pragmatics(user_input)
        
        # Update entity focus
        self._update_focus(mentioned_entities, pragmatic_signals)
        
        # Create turn record
        turn = ConversationTurn(
            turn_id=turn_id,
            user_input=user_input,
            assistant_response=assistant_response,
            intent=intent,
            entities_mentioned=mentioned_entities,
            entities_resolved=resolved_entities,
            timestamp=datetime.now(),
            context_used=self.current_focus.copy(),
            pragmatic_signals=pragmatic_signals
        )
        
        # Store turn and update stacks
        self.conversation_turns.append(turn)
        self._update_reference_stacks(mentioned_entities)
        self._update_entity_relevance()
        
        # Also update conversation state
        self.update_conversation(user_input, assistant_response, intent, mentioned_entities)
        
        return turn
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entity mentions from text"""
        entities_found = []
        
        # Look for explicit entity references
        for entity_id, entity in self.entities.items():
            for mention in entity.mentions + [entity_id] + entity.aliases:
                if mention.lower() in text.lower():
                    entities_found.append(entity_id)
                    break
        
        return list(set(entities_found))
    
    def _resolve_coreferences(self, text: str, mentioned_entities: List[str]) -> Dict[str, str]:
        """Resolve pronouns and references to specific entities"""
        resolutions = {}
        text_lower = text.lower()
        
        # Pronoun resolution patterns
        pronoun_patterns = {
            r'\bit\b': self._resolve_neutral_pronoun,
            r'\bthis\b': self._resolve_demonstrative,
            r'\bthat\b': self._resolve_demonstrative,
            r'\bthe (email|message|file|document)\b': self._resolve_definite_description,
            r'\bthe (latest|last|recent|first)\b': self._resolve_temporal_reference,
        }
        
        for pattern, resolver in pronoun_patterns.items():
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                reference = match.group()
                resolved_entity = resolver(reference, text, mentioned_entities)
                if resolved_entity:
                    resolutions[reference] = resolved_entity
        
        return resolutions
    
    def _resolve_neutral_pronoun(self, pronoun: str, text: str, mentioned: List[str]) -> Optional[str]:
        """Resolve 'it' to most recently mentioned neutral entity"""
        for entity_id in reversed(self.pronoun_stack):
            entity = self.entities.get(entity_id)
            if entity and entity.entity_type != "person":
                return entity_id
        return None
    
    def _resolve_demonstrative(self, demonstrative: str, text: str, mentioned: List[str]) -> Optional[str]:
        """Resolve 'this'/'that' based on recency and context"""
        if not self.demonstrative_stack:
            return None
        
        if "this" in demonstrative:
            return self.demonstrative_stack[-1] if self.demonstrative_stack else None
        
        if "that" in demonstrative and len(self.demonstrative_stack) > 1:
            return self.demonstrative_stack[-2]
        
        return self.demonstrative_stack[-1] if self.demonstrative_stack else None
    
    def _resolve_definite_description(self, phrase: str, text: str, mentioned: List[str]) -> Optional[str]:
        """Resolve definite descriptions like 'the email', 'the file'"""
        entity_type = None
        for etype, patterns in self.entity_patterns.items():
            for indicator in patterns["indicators"]:
                if indicator in phrase:
                    entity_type = etype
                    break
        
        if not entity_type:
            return None
        
        for entity_id in reversed(self.pronoun_stack):
            entity = self.entities.get(entity_id)
            if entity and entity.entity_type == entity_type:
                return entity_id
        
        return None
    
    def _resolve_temporal_reference(self, phrase: str, text: str, mentioned: List[str]) -> Optional[str]:
        """Resolve temporal references like 'the latest', 'the first'"""
        if "latest" in phrase or "last" in phrase or "recent" in phrase:
            return self.pronoun_stack[-1] if self.pronoun_stack else None
        
        if "first" in phrase:
            return self.current_focus[0] if self.current_focus else None
        
        return None
    
    def _analyze_pragmatics(self, text: str) -> Dict[str, Any]:
        """Analyze pragmatic signals in text"""
        signals = {}
        text_lower = text.lower()
        
        # Discourse markers
        for marker, signal_type in self.discourse_markers.items():
            if marker in text_lower:
                signals["discourse_marker"] = signal_type
                break
        
        # Politeness markers
        if any(word in text_lower for word in ["please", "could you", "would you", "thank you"]):
            signals["politeness"] = "high"
        
        # Urgency markers
        if any(word in text_lower for word in ["urgent", "asap", "quickly", "immediately", "now"]):
            signals["urgency"] = "high"
        
        # Certainty markers
        if any(word in text_lower for word in ["definitely", "certainly", "absolutely", "sure"]):
            signals["certainty"] = "high"
        elif any(word in text_lower for word in ["maybe", "perhaps", "possibly", "might"]):
            signals["certainty"] = "low"
        
        # Question types
        if text.strip().endswith("?"):
            if any(word in text_lower for word in ["what", "how", "why", "when", "where", "who"]):
                signals["question_type"] = "wh_question"
            elif any(word in text_lower for word in ["can", "could", "would", "should", "do", "did", "is", "are"]):
                signals["question_type"] = "yes_no_question"
        
        return signals
    
    def _update_focus(self, mentioned_entities: List[str], pragmatic_signals: Dict[str, Any]):
        """Update current conversation focus"""
        
        # Handle topic shifts
        if pragmatic_signals.get("discourse_marker") == "topic_shift":
            self.current_focus.clear()
        
        # Add newly mentioned entities to focus
        for entity_id in mentioned_entities:
            if entity_id not in self.current_focus:
                self.current_focus.append(entity_id)
        
        # Limit focus size
        if len(self.current_focus) > 5:
            focus_scores = []
            for entity_id in self.current_focus:
                entity = self.entities.get(entity_id)
                if entity:
                    recency_score = 1.0 / max(1, (datetime.now() - entity.last_mentioned).total_seconds() / 3600)
                    total_score = entity.relevance_score * recency_score
                    focus_scores.append((entity_id, total_score))
            
            focus_scores.sort(key=lambda x: x[1], reverse=True)
            self.current_focus = [entity_id for entity_id, _ in focus_scores[:5]]
    
    def _update_reference_stacks(self, mentioned_entities: List[str]):
        """Update pronoun and demonstrative reference stacks"""
        
        # Update pronoun stack
        for entity_id in mentioned_entities:
            if entity_id in self.pronoun_stack:
                self.pronoun_stack.remove(entity_id)
            self.pronoun_stack.append(entity_id)
        
        # Keep stack manageable
        if len(self.pronoun_stack) > 10:
            self.pronoun_stack = self.pronoun_stack[-10:]
        
        # Update demonstrative stack
        self.demonstrative_stack = self.pronoun_stack.copy()
    
    def _update_entity_relevance(self):
        """Update relevance scores for entities based on recent activity"""
        now = datetime.now()
        
        for entity_id, entity in self.entities.items():
            # Decay relevance over time
            time_since_mention = (now - entity.last_mentioned).total_seconds() / 3600
            decay_factor = max(0.1, 1.0 - (time_since_mention * 0.1))
            entity.relevance_score *= decay_factor
            
            # Boost relevance if in current focus
            if entity_id in self.current_focus:
                entity.relevance_score = min(1.0, entity.relevance_score * 1.2)
    
    def add_entity(self, entity_type: str, data: Dict[str, Any],
                   entity_id: Optional[str] = None, aliases: List[str] = None) -> str:
        """Add a new entity to context"""
        
        if not entity_id:
            entity_id = f"{entity_type}_{int(datetime.now().timestamp())}"
        
        entity = ContextEntity(
            entity_id=entity_id,
            entity_type=entity_type,
            data=data,
            aliases=aliases or [],
            last_mentioned=datetime.now()
        )
        
        self.entities[entity_id] = entity
        logger.info(f"Added entity {entity_id} of type {entity_type}")
        return entity_id
    
    def get_contextual_entities(self, query: str, max_entities: int = 5) -> List[Tuple[str, float]]:
        """Get entities relevant to a query"""
        # Fallback to focus-based selection
        return [(eid, self.entities[eid].relevance_score) for eid in self.current_focus[:max_entities]]
    
    def resolve_reference(self, reference_text: str) -> Optional[str]:
        """Resolve a textual reference to an entity ID"""
        reference_lower = reference_text.lower().strip()
        
        for entity_id, entity in self.entities.items():
            all_refs = [entity_id.lower()] + [m.lower() for m in entity.mentions] + [a.lower() for a in entity.aliases]
            if reference_lower in all_refs or any(ref in reference_lower for ref in all_refs):
                return entity_id
        
        return None
    
    def get_context_for_llm(self, query: str) -> str:
        """Generate context string for LLM"""
        context_parts = []
        
        # Get relevant entities
        relevant_entities = self.get_contextual_entities(query)
        
        if relevant_entities:
            context_parts.append("Relevant context:")
            for entity_id, score in relevant_entities:
                entity = self.entities[entity_id]
                entity_context = f"- {entity.entity_type.title()}: "
                
                if entity.entity_type == "email":
                    subject = entity.data.get("subject", "No subject")
                    sender = entity.data.get("from", "Unknown sender")
                    entity_context += f"'{subject}' from {sender}"
                elif entity.entity_type == "person":
                    name = entity.data.get("name", entity_id)
                    entity_context += f"{name}"
                elif entity.entity_type == "file":
                    name = entity.data.get("name", entity_id)
                    entity_context += f"'{name}'"
                else:
                    entity_context += str(entity.data)
                
                context_parts.append(entity_context)
        
        # Add recent conversation context
        if self.conversation_turns:
            recent_turns = self.conversation_turns[-3:]
            context_parts.append("\nRecent conversation:")
            for turn in recent_turns:
                if turn.entities_resolved:
                    resolved_info = ", ".join([f"{ref}→{eid}" for ref, eid in turn.entities_resolved.items()])
                    context_parts.append(f"- User: {turn.user_input[:100]} (resolved: {resolved_info})")
                else:
                    context_parts.append(f"- User: {turn.user_input[:100]}")
        
        return "\n".join(context_parts)
    
    def update_conversation(self, user_input: str, assistant_response: str,
                          intent: Optional[str] = None, entities: Optional[List] = None):
        """Update conversation state with new interaction"""
        
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "assistant": assistant_response,
            "intent": intent,
            "entities": entities or [],
            "session_id": str(self.conv_state.session_start.timestamp())
        }
        
        # Add to short-term memory
        self.short_term_memory.append(interaction)
        if len(self.short_term_memory) > 100:
            self.short_term_memory.pop(0)
        
        # Update conversation state
        self.conv_state.last_interaction = datetime.now()
        if intent:
            self.conv_state.last_intent = intent
        if entities:
            self.conv_state.mentioned_entities.extend(entities)
            self.conv_state.mentioned_entities = list(set(self.conv_state.mentioned_entities[-20:]))
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of current context for LLM"""
        
        session_duration = datetime.now() - self.conv_state.session_start
        
        recent_exchanges = self.short_term_memory[-5:] if self.short_term_memory else []
        
        summary = {
            "user_name": self.user_prefs.name,
            "session_duration_minutes": int(session_duration.total_seconds() / 60),
            "recent_topics": self.conv_state.mentioned_entities[-5:] if self.conv_state.mentioned_entities else [],
            "active_topic": self.conv_state.active_topic,
            "pending_tasks": self.conv_state.pending_tasks,
            "last_intent": self.conv_state.last_intent,
            "location": self.user_prefs.location,
            "city": self.user_prefs.city,
            "country": self.user_prefs.country,
            "time_of_day": self._get_time_of_day(),
            "recent_exchanges": recent_exchanges,
            "total_interactions": len(self.short_term_memory)
        }
        
        return summary
    
    def _get_time_of_day(self) -> str:
        """Get current time of day for context"""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    
    def store_fact(self, key: str, value: Any, category: str = "general"):
        """Store a fact in semantic memory"""
        if category not in self.semantic_memory:
            self.semantic_memory[category] = {}
        
        self.semantic_memory[category][key] = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "access_count": 0
        }
        logger.info(f"Stored fact: {key} in {category}")
    
    def recall_fact(self, key: str, category: str = "general") -> Optional[Any]:
        """Recall a fact from semantic memory"""
        if category in self.semantic_memory and key in self.semantic_memory[category]:
            fact = self.semantic_memory[category][key]
            fact["access_count"] += 1
            fact["last_accessed"] = datetime.now().isoformat()
            return fact["value"]
        return None
    
    def add_task(self, task: str):
        """Add a pending task"""
        self.conv_state.pending_tasks.append({
            "task": task,
            "created": datetime.now().isoformat(),
            "completed": False
        })
        logger.info(f"[OK] Task added: {task}")
    
    def complete_task(self, task_index: int):
        """Mark a task as completed"""
        if 0 <= task_index < len(self.conv_state.pending_tasks):
            self.conv_state.pending_tasks[task_index]["completed"] = True
            self.conv_state.pending_tasks[task_index]["completed_at"] = datetime.now().isoformat()
            logger.info(f"[OK] Task completed: {self.conv_state.pending_tasks[task_index]['task']}")
    
    def get_pending_tasks(self) -> List[str]:
        """Get list of pending tasks"""
        return [
            t["task"] for t in self.conv_state.pending_tasks
            if not t.get("completed", False)
        ]
    
    def set_active_topic(self, topic: str):
        """Set the current conversation topic"""
        self.conv_state.active_topic = topic
        logger.info(f"Topic set to: {topic}")
    
    def clear_short_term_memory(self):
        """Clear short-term memory"""
        self.short_term_memory = []
        self.conv_state = ConversationState()
        logger.info("🧹 Short-term memory cleared")
    
    def get_personalization_context(self) -> str:
        """Get personalization info for LLM system prompt"""
        context_parts = []
        
        if self.user_prefs.name != "User":
            context_parts.append(f"You are speaking with {self.user_prefs.name}.")
        
        if self.conv_state.active_topic:
            context_parts.append(f"Current topic: {self.conv_state.active_topic}.")
        
        if self.conv_state.mentioned_entities:
            entities = ", ".join(self.conv_state.mentioned_entities[-3:])
            context_parts.append(f"Recently discussed: {entities}.")
        
        pending = self.get_pending_tasks()
        if pending:
            context_parts.append(f"Pending tasks: {', '.join(pending[:3])}.")
        
        time_of_day = self._get_time_of_day()
        context_parts.append(f"Current time: {time_of_day}.")
        
        return " ".join(context_parts) if context_parts else ""
    
    def update_system_status(self, key: str, value: Any):
        """Update system status"""
        self.system_status[key] = value
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return self.system_status.copy()
    
    def save_state(self, filepath: str):
        """Save context state to file"""
        state = {
            "entities": {eid: {
                "entity_id": e.entity_id,
                "entity_type": e.entity_type,
                "data": e.data,
                "mentions": e.mentions,
                "last_mentioned": e.last_mentioned.isoformat(),
                "relevance_score": e.relevance_score,
                "aliases": e.aliases
            } for eid, e in self.entities.items()},
            "current_focus": self.current_focus,
            "pronoun_stack": self.pronoun_stack,
            "conversation_turns": len(self.conversation_turns),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Context state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load context state from file"""
        try:
            with open(filepath, "rb") as f:
                state = pickle.load(f)
            
            # Restore entities
            for eid, entity_data in state.get("entities", {}).items():
                entity = ContextEntity(
                    entity_id=entity_data["entity_id"],
                    entity_type=entity_data["entity_type"],
                    data=entity_data["data"],
                    mentions=entity_data["mentions"],
                    last_mentioned=datetime.fromisoformat(entity_data["last_mentioned"]),
                    relevance_score=entity_data["relevance_score"],
                    aliases=entity_data["aliases"]
                )
                self.entities[eid] = entity
            
            # Restore other state
            self.current_focus = state.get("current_focus", [])
            self.pronoun_stack = state.get("pronoun_stack", [])
            
            logger.info(f"Context state loaded from {filepath}")
            
        except Exception as e:
            logger.warning(f"Could not load context state: {e}")
    
    def __enter__(self):
        """Context manager enter"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save on exit"""
        self.save_context()


# Singleton instance
_context_engine_instance: Optional[ContextEngine] = None


def get_context_engine(data_dir: str = "data/context") -> ContextEngine:
    """Get or create the singleton ContextEngine instance"""
    global _context_engine_instance
    if _context_engine_instance is None:
        _context_engine_instance = ContextEngine(data_dir)
    return _context_engine_instance


if __name__ == "__main__":
    print("Testing Context Engine...")
    
    with ContextEngine() as ctx:
        ctx.user_prefs.name = "Gabriel"
        ctx.store_fact("favorite_color", "red", "preferences")
        ctx.add_task("Review ALICE improvements")
        ctx.update_conversation(
            "What's the weather?",
            "Let me check...",
            intent="weather_query",
            entities=["weather"]
        )
        
        summary = ctx.get_context_summary()
        print(f"\n📊 Context Summary:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        print(f"\n🎭 Personalization: {ctx.get_personalization_context()}")
    
    print("\n[OK] Context Engine working correctly")
