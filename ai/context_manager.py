"""
Advanced Context Manager for A.L.I.C.E
Maintains conversation state, user preferences, and system context
Similar to Jarvis's contextual awareness
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pickle

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


class ContextManager:
    """
    Advanced context management system for A.L.I.C.E
    Handles:
    - User preferences and personalization
    - Conversation state and history
    - Short-term and long-term memory
    - System status and capabilities
    """
    
    def __init__(self, data_dir: str = "data/context"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Core components
        self.user_prefs: UserPreferences = UserPreferences()
        self.conv_state: ConversationState = ConversationState()
        
        # Memory systems
        self.short_term_memory: List[Dict] = []  # Last N interactions
        self.working_memory: Dict[str, Any] = {}  # Current task variables
        self.semantic_memory: Dict[str, Any] = {}  # Facts and knowledge
        
        # System state
        self.system_status: Dict[str, Any] = {
            "online": True,
            "capabilities": [],
            "active_plugins": [],
            "performance": {}
        }
        
        # Load saved context
        self._load_context()
        
        logger.info("[OK] Context Manager initialized")
    
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
                # Convert dataclass to dict, handling datetime
                prefs_dict = asdict(self.user_prefs)
                json.dump(prefs_dict, f, indent=2)
            
            # Save semantic memory
            memory_path = os.path.join(self.data_dir, "semantic_memory.pkl")
            with open(memory_path, 'wb') as f:
                pickle.dump(self.semantic_memory, f)
            
            logger.info("Context saved successfully")
            
        except Exception as e:
            logger.error(f"[ERROR] Error saving context: {e}")
    
    def update_conversation(self, user_input: str, assistant_response: str, 
                          intent: Optional[str] = None, entities: Optional[List] = None):
        """Update conversation state with new interaction"""
        
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user": user_input,
            "assistant": assistant_response,
            "intent": intent,
            "entities": entities or [],
            "session_id": str(self.conv_state.session_start.timestamp())  # Track session
        }
        
        # Add to short-term memory (keep last 100 for better context)
        self.short_term_memory.append(interaction)
        if len(self.short_term_memory) > 100:
            self.short_term_memory.pop(0)
        
        # Update conversation state
        self.conv_state.last_interaction = datetime.now()
        if intent:
            self.conv_state.last_intent = intent
        if entities:
            self.conv_state.mentioned_entities.extend(entities)
            # Keep unique entities only
            self.conv_state.mentioned_entities = list(set(
                self.conv_state.mentioned_entities[-20:]  # Keep last 20 unique
            ))
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of current context for LLM"""
        
        session_duration = datetime.now() - self.conv_state.session_start
        
        # Get recent conversation for context
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
    
    def __enter__(self):
        """Context manager enter"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - save on exit"""
        self.save_context()


# Example usage
if __name__ == "__main__":
    print("Testing Context Manager...")
    
    with ContextManager() as ctx:
        # Set user preferences
        ctx.user_prefs.name = "Tony Stark"
        ctx.user_prefs.location = "Malibu"
        
        # Store some facts
        ctx.store_fact("favorite_color", "red", "preferences")
        ctx.store_fact("python_version", "3.11", "system")
        
        # Add tasks
        ctx.add_task("Review project documentation")
        ctx.add_task("Test voice recognition")
        
        # Simulate conversation
        ctx.update_conversation(
            "What's the weather like?",
            "Let me check that for you.",
            intent="weather_query",
            entities=["weather"]
        )
        
        # Get context summary
        summary = ctx.get_context_summary()
        print(f"\n📊 Context Summary:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        # Get personalization context
        print(f"\n🎭 Personalization: {ctx.get_personalization_context()}")
        
        # Recall facts
        color = ctx.recall_fact("favorite_color", "preferences")
        print(f"\nRecalled: favorite_color = {color}")
    
    print("\n[OK] Context saved automatically on exit")
