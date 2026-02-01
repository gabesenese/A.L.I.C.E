"""   
A.L.I.C.E - Advanced Linguistic Intelligence Computer Entity
Main Orchestrator - Intelligent Personal Assistant

Integrates all components:
- Advanced NLP with intent detection
- LLM engine (Ollama with Llama 3.3 70B)
- Context management and personalization
- Memory system with RAG
- Plugin system for extensibility
- Voice interaction (speech-to-text, text-to-speech)
- Task execution and automation
"""

# Suppress warnings before importing other modules
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

import sys
import logging
import re
from typing import Optional, Dict, Any
from datetime import datetime
from collections import defaultdict

from ai.goal_from_llm import get_goal_from_llm, GoalJSON
from ai.policy import get_policy_decision, PolicyDecision
from ai.runtime_thresholds import get_tool_path_confidence, get_goal_path_confidence, get_ask_threshold

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import ALICE components
from ai.nlp_processor import NLPProcessor
from ai.llm_engine import LocalLLMEngine, LLMConfig
from ai.context_engine import get_context_engine
from ai.memory_system import MemorySystem
from ai.conversation_summarizer import ConversationSummarizer
from ai.entity_relationship_tracker import EntityRelationshipTracker
from ai.active_learning_manager import ActiveLearningManager, CorrectionType, FeedbackType
from ai.email_plugin import GmailPlugin
from ai.plugin_system import (
    PluginManager, WeatherPlugin, TimePlugin,
    FileOperationsPlugin, SystemControlPlugin, WebSearchPlugin
)
from ai.document_plugin import DocumentPlugin
from ai.calendar_plugin import CalendarPlugin
from ai.music_plugin import MusicPlugin
from ai.notes_plugin import NotesPlugin
from ai.maps_plugin import MapsPlugin
from ai.task_executor import TaskExecutor
from speech.speech_engine import SpeechEngine, SpeechConfig

# New anticipatory AI systems
from ai.event_bus import get_event_bus, EventType, EventPriority
from ai.system_state import get_state_tracker, SystemStatus
from ai.observers import get_observer_manager
from ai.pattern_learner import get_pattern_learner
from ai.system_monitor import get_system_monitor
from ai.task_planner import get_planner
from ai.plan_executor import initialize_executor, get_executor
from ai.reasoning_engine import get_reasoning_engine, WorldEntity, EntityKind
from ai.proactive_assistant import get_proactive_assistant
from ai.error_recovery import get_error_recovery
from ai.smart_context_cache import get_context_cache
from ai.adaptive_context_selector import get_context_selector
from ai.predictive_prefetcher import get_prefetcher
from ai.response_optimizer import get_response_optimizer
from ai.self_reflection import get_self_reflection
from ai.learning_engine import get_learning_engine
from ai.conversational_engine import get_conversational_engine, ConversationalContext
from ai.llm_gateway import get_llm_gateway, LLMGateway
from ai.llm_policy import LLMCallType

# Advanced learning, testing, and telemetry
from ai.pattern_miner import PatternMiner
from ai.synthetic_corpus_generator import SyntheticCorpusGenerator
from ai.multimodal_context import MultimodalContext
from ai.lab_simulator import LabSimulator
from ai.red_team_tester import RedTeamTester

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ALICE:
    """
    Main A.L.I.C.E system
    Intelligent personal assistant with advanced AI capabilities
    """
    
    def __init__(
        self,
        voice_enabled: bool = False,
        llm_model: str = "llama3.1:8b",
        user_name: str = "User",
        debug: bool = False,
        privacy_mode: bool = False,
        llm_policy: str = "default"
    ):
        self.voice_enabled = voice_enabled
        self.debug = debug
        self.privacy_mode = privacy_mode
        self.llm_policy = llm_policy
        self.running = False
        
        # Conversation state for context-aware operations
        self.last_email_list = []  # Store last displayed email list
        self.last_email_context = None  # Store context of last email operation
        self.last_code_file = None  # Store last code file for follow-up queries
        self.pending_action = None  # Track multi-step actions (e.g., composing email)
        self.pending_data = {}  # Store data for pending actions
        
        # Enhanced conversation tracking
        self.conversation_summary = []  # Summary of recent exchanges
        self.referenced_items = {}  # Track items user has referenced (emails, files, etc.)
        self.conversation_topics = []  # Track topics discussed in this session
        
        # Code introspection context tracking (smart follow-up handling)
        self.code_context = {
            'last_files_shown': [],  # List of file paths recently shown
            'last_action': None,     # 'list', 'analyze', 'read', 'summary'
            'timestamp': None,       # When context was created
            'file_count': 0          # Number of files in last result
        }
        
        # Active learning tracking
        self.last_user_input = ""
        self.last_assistant_response = ""
        self.last_intent = ""
        self.last_entities = {}
        self.last_nlp_result = {}
        
        # Session-level response cache for fast conversational responses
        # Caches (input_normalized, intent) -> response for quick lookup
        self._response_cache = {}
        self._cache_max_size = 50  # Keep last 50 responses per session
        
        # Conversation summarizer for intelligent context management
        self.summarizer = None  # Will be initialized after LLM engine
        
        # Advanced context handler
        self.advanced_context = None  # Will be initialized after NLP processor
        
        # Event-driven architecture (anticipatory AI)
        self.event_bus = None
        self.state_tracker = None
        self.observer_manager = None
        self.pattern_learner = None
        self.system_monitor = None
        self.planner = None
        self.plan_executor = None
        
        # Event-driven architecture
        self.event_bus = None
        self.state_tracker = None
        self.observer_manager = None
        self.pattern_learner = None
        self.system_monitor = None
        self.planner = None
        self.plan_executor = None
        
        logger.info("=" * 80)
        logger.info("Initializing A.L.I.C.E - Advanced Linguistic Intelligence Computer Entity")
        logger.info("=" * 80)
        
        # Initialize components
        try:
            # 1. NLP Processor
            logger.info("Loading NLP processor...")
            self.nlp = NLPProcessor()
            
            # 1.5. Unified Context Engine (combines context_manager + advanced_context_handler)
            logger.info("Loading unified context engine...")
            self.context = get_context_engine()
            self.context.user_prefs.name = user_name
            # Backward compatibility: older code paths reference advanced_context
            self.advanced_context = self.context
            
            # 2. Reasoning Engine (combines world_state + reference_resolver + goal_resolver + verifier)
            logger.info("Loading reasoning engine...")
            self.reasoning_engine = get_reasoning_engine(user_name)
            self.error_recovery = get_error_recovery()
            self.context_cache = get_context_cache()
            self.context_selector = get_context_selector()
            self.prefetcher = get_prefetcher(self.reasoning_engine)
            self.response_optimizer = get_response_optimizer(self.reasoning_engine)
            self.self_reflection = get_self_reflection()
            logger.info("[OK] Reasoning engine initialized - unified entity/goal/verification tracking")
            
            # 2.9. Unified Learning Engine - collect and learn from interactions
            self.learning_engine = get_learning_engine()
            logger.info("[OK] Learning engine initialized - A.L.I.C.E will learn from your interactions")
            
            # 3. Memory System (needs to be before conversational engine)
            logger.info("Loading memory system...")
            self.memory = MemorySystem()
            
            # 3.5. Conversational engine - A.L.I.C.E's own conversational logic (no Ollama)
            self.conversational_engine = get_conversational_engine(
                memory_system=self.memory,
                world_state=self.reasoning_engine
            )
            logger.info("[OK] Conversational engine initialized - A.L.I.C.E thinks independently")
            
            # 4. LLM Engine
            logger.info("🤖 Loading LLM engine...")
            llm_config = LLMConfig(model=llm_model)
            self.llm = LocalLLMEngine(llm_config)
            
            # 4.1. LLM Gateway - Single entry point for all LLM calls
            logger.info("🚪 Initializing LLM Gateway with policy enforcement...")
            self.llm_gateway = get_llm_gateway(
                llm_engine=self.llm,
                learning_engine=self.learning_engine
            )
            logger.info("[OK] LLM Gateway active - all calls now policy-gated")
            
            # 4.2. Configure LLM Policy based on startup flag
            if self.llm_policy != "default":
                logger.info(f"⚙️  Configuring LLM policy: {self.llm_policy}")
                from ai.llm_policy import configure_minimal_policy
                
                if self.llm_policy == "minimal":
                    configure_minimal_policy()
                    logger.info("[OK] Minimal policy active - patterns-first, LLM only for generation")
                elif self.llm_policy == "strict":
                    # Strict mode: no LLM at all (future implementation)
                    logger.warning("[WARNING] Strict policy not yet implemented - using minimal")
                    configure_minimal_policy()
            
            # 4.5. Conversation Summarizer (now uses gateway for policy enforcement)
            logger.info("Loading conversation summarizer...")
            self.summarizer = ConversationSummarizer(llm_engine=self.llm, llm_gateway=self.llm_gateway)
            
            # 4.6. Entity Relationship Tracker
            logger.info("Loading entity relationship tracker...")
            self.relationship_tracker = EntityRelationshipTracker()
            
            # 4.7. Active Learning Manager
            logger.info("Loading active learning system...")
            self.learning_manager = ActiveLearningManager()
            
            # 5. Plugin System
            logger.info("Loading plugins...")
            self.plugins = PluginManager()
            self._register_plugins()
            
            # 6. Task Executor
            logger.info(" Loading task executor...")
            self.executor = TaskExecutor(safe_mode=True)
            
            # 7. Speech Engine (optional)
            self.speech = None
            if voice_enabled: 
                logger.info("Loading speech engine...")
                speech_config = SpeechConfig(wake_words=["alice", "hey alice", "ok alice"])
                self.speech = SpeechEngine(speech_config)
            
            # 8. Gmail Plugin
            logger.info("Loading Gmail integration...")
            try:
                self.gmail = GmailPlugin()
                if self.gmail.service:
                    logger.info(f"[OK] Gmail connected: {self.gmail.user_email}")
                else:
                    logger.warning("[WARNING] Gmail not configured - run setup")
                    self.gmail = None
            except Exception as e:
                logger.warning(f"[WARNING] Gmail not available: {e}")
                self.gmail = None
            
            # 9. Advanced learning, testing, and telemetry systems
            logger.info("Initializing advanced learning systems...")
            try:
                # Pattern miner for learning from logged interactions
                self.pattern_miner = PatternMiner()
                logger.info("[OK] Pattern miner ready - will detect learnable patterns")
                
                # Synthetic corpus generator for pre-training
                self.synthetic_corpus_gen = SyntheticCorpusGenerator()
                logger.info("[OK] Synthetic corpus generator ready")
                
                # Multimodal context for system-aware interactions
                self.multimodal_context = MultimodalContext()
                logger.info("[OK] Multimodal context capture enabled")
                
                # Lab simulator for stress testing
                self.lab_simulator = LabSimulator()
                logger.info("[OK] Lab simulator ready for scenario generation")
                
                # Red team tester for security validation
                self.red_team_tester = RedTeamTester()
                logger.info("[OK] Red team tester ready")
                
            except Exception as e:
                logger.warning(f"[WARNING] Advanced learning systems initialization partial: {e}")
            
            # 9.5. Event-driven architecture
            logger.info("Initializing event-driven systems...")
            
            # Event bus
            self.event_bus = get_event_bus()
            logger.info("[OK] Event bus ready")
            
            # System state tracker
            self.state_tracker = get_state_tracker()
            self.state_tracker.start_monitoring(interval=30)  # Check every 30 seconds
            logger.info("[OK] System state tracker monitoring")
            
            # Pattern learner for anticipatory suggestions
            self.pattern_learner = get_pattern_learner()
            logger.info("[OK] Pattern learner ready")
            
            # System monitor for OS-level awareness
            self.system_monitor = get_system_monitor()
            self.system_monitor.start_monitoring()
            logger.info("[OK] System monitor tracking apps")
            
            # Task planner (separates planning from execution)
            self.planner = get_planner()
            logger.info("[OK] Task planner ready")
            
            # Plan executor
            self.plan_executor = initialize_executor(
                plugin_manager=self.plugins,
                llm_engine=self.llm,
                memory_system=self.memory
            )
            logger.info("[OK] Plan executor ready")
            
            # Observer manager for smart notifications
            self.observer_manager = get_observer_manager()
            self.observer_manager.set_notification_callback(self._handle_observer_notification)
            self.observer_manager.start_all()
            logger.info("[OK] Background observers watching")
            
            # Proactive assistant (reminders, follow-ups, proactive info)
            calendar_plugin = None
            notes_plugin = None
            for name, plugin in self.plugins.plugins.items():
                if 'Calendar' in name or isinstance(plugin, CalendarPlugin):
                    calendar_plugin = plugin
                if 'Notes' in name or isinstance(plugin, NotesPlugin):
                    notes_plugin = plugin
            self.proactive_assistant = get_proactive_assistant(
                world_state=self.reasoning_engine,
                calendar_plugin=calendar_plugin,
                notes_plugin=notes_plugin
            )
            self.proactive_assistant.set_notification_callback(self._handle_observer_notification)
            self.proactive_assistant.start()
            logger.info("[OK] Proactive assistant monitoring")
            
            logger.info("=" * 80)
            logger.info("[OK] A.L.I.C.E initialized successfully!")
            logger.info("=" * 80)
            
            # Store system capabilities in context
            self.context.update_system_status("capabilities", self.plugins.get_capabilities())
            self.context.update_system_status("voice_enabled", voice_enabled)
            self.context.update_system_status("llm_model", llm_model)
            
            # Load previous conversation state if available
            self._load_conversation_state()
            
        except Exception as e:
            logger.error(f"[ERROR] Initialization failed: {e}")
            raise
    
    def _register_plugins(self):
        """Register all available plugins"""
        # Register NotesPlugin early to ensure it handles note commands before calendar
        self.plugins.register_plugin(NotesPlugin())
        self.plugins.register_plugin(WeatherPlugin())
        self.plugins.register_plugin(MapsPlugin())
        self.plugins.register_plugin(TimePlugin())
        self.plugins.register_plugin(FileOperationsPlugin())
        self.plugins.register_plugin(SystemControlPlugin())
        self.plugins.register_plugin(WebSearchPlugin())
        self.plugins.register_plugin(DocumentPlugin())
        self.plugins.register_plugin(CalendarPlugin())
        self.plugins.register_plugin(MusicPlugin())
        
        logger.info(f"[OK] Registered {len(self.plugins.plugins)} plugins")
    
    def _handle_observer_notification(self, message: str, priority: EventPriority):
        """
        Handle notifications from background observers
        Display/speak them appropriately
        """
        priority_label = {
            EventPriority.LOW: "ℹ️",
            EventPriority.NORMAL: "",
            EventPriority.HIGH: "⚠️",
            EventPriority.CRITICAL: "🚨"
        }.get(priority, "•")
        
        notification = f"\n{priority_label} {message}\n"
        print(notification)
        
        # Speak if voice is enabled and priority is high enough
        if self.speech and priority.value >= EventPriority.NORMAL.value:
            self.speech.speak(message)
    
    def _log_action_for_learning(self, action: str, context: Dict[str, Any] = None):
        """
        Log user action to pattern learner
        
        Args:
            action: Action taken (e.g., "review_notes:finance")
            context: Current context (time, state, etc.)
        """
        if not self.pattern_learner:
            return
        
        # Build context
        full_context = context or {}
        full_context.update({
            'day': datetime.now().strftime("%A"),
            'hour': datetime.now().hour,
            'system_state': self.state_tracker.get_status().value if self.state_tracker else 'unknown'
        })
        
        # Log the action
        self.pattern_learner.observe_action(action, full_context)
    
    def _check_proactive_suggestions(self) -> Optional[str]:
        """
        Check if we should make proactive suggestions
        
        Returns:
            Suggestion text or None
        """
        if not self.pattern_learner:
            return None
        
        # Get context
        context = {
            'system_state': self.state_tracker.get_status().value if self.state_tracker else 'unknown',
            'running_apps': self.system_monitor.get_running_apps() if self.system_monitor else []
        }
        
        # Get suggestions
        suggestions = self.pattern_learner.get_suggestions(context)
        
        if suggestions:
            # Return top suggestion
            pattern, suggestion_text = suggestions[0]
            
            # Store pattern ID for tracking acceptance
            self._last_suggestion_pattern = pattern.pattern_id
            
            return f" {suggestion_text}"
        
        return None
    
    def _use_planner_executor(self, intent: str, entities: Dict[str, Any], query: str) -> Optional[str]:
        """
        Use task planner and executor for complex tasks
        
        Args:
            intent: Detected intent
            entities: Extracted entities
            query: User's query
        
        Returns:
            Response or None if not planned
        """
        # Only use planner for specific intents that benefit from structured execution
        plannable_intents = [
            'summarize_notes', 'check_calendar', 'send_email',
            'play_music', 'create_note', 'question'
        ]
        
        if intent not in plannable_intents:
            return None
        
        try:
            # Create execution plan
            context = {
                'user_prefs': vars(self.context.user_prefs),
                'system_state': self.state_tracker.get_status().value if self.state_tracker else 'unknown'
            }
            
            plan = self.planner.create_plan(intent, entities, context)
            
            # Validate plan
            if not self.planner.validate_plan(plan):
                logger.error(f"Invalid plan for intent {intent}")
                return None
            
            # Log plan explanation
            logger.info(f"Execution plan:\n{self.planner.explain_plan(plan)}")
            
            # Execute plan
            result = self.plan_executor.execute(plan)
            
            if result['success']:
                return result.get('result')
            else:
                logger.error(f"Plan execution failed: {result.get('error')}")
                return None
        
        except Exception as e:
            logger.error(f"Planner/executor error: {e}")
            return None
    
    def _build_llm_context(self, user_input: str, intent: str = "", entities: Dict = None, goal_res = None) -> str:
        """Build enhanced context for LLM with smart caching and adaptive selection"""
        # Check cache first
        cached = self.context_cache.get(user_input, intent or "", entities or {})
        if cached:
            self._think("Context cache hit → reusing context")
            return cached
        
        # Build all context parts
        context_parts = []
        context_types = []
        
        # 0. ACTIVE GOAL - Most important for understanding user intent
        if goal_res and goal_res.goal:
            goal = goal_res.goal
            goal_context = f"ACTIVE GOAL: The user is trying to {goal.description}. "
            goal_context += f"Intent: {goal.intent}. "
            if goal.entities:
                goal_ents = ", ".join(f"{k}={v}" for k, v in list(goal.entities.items())[:3])
                goal_context += f"Entities: {goal_ents}. "
            goal_context += "Use this goal to understand ambiguous inputs and guide your response to help complete this goal."
            context_parts.insert(0, goal_context)  # Put goal first - highest priority
            context_types.insert(0, "goal")
            self._think(f"Goal context → {goal.description[:50]}...")
        
        # 0.5. Self-Reflection Capability - ALWAYS include when user asks about code/access
        code_keywords = ['code', 'improve', 'analyze', 'read file', 'my code', 'your code', 'alice code', 
                        'show code', 'list files', 'access to', 'internal code', 'codebase', 'see code',
                        'have access', 'can you see', 'your files']
        if any(word in user_input.lower() for word in code_keywords):
            codebase_summary = self.self_reflection.get_codebase_summary()
            reflection_context = f"CRITICAL: You ARE A.L.I.C.E, an AI assistant with read-only access to your own codebase. "
            reflection_context += f"Your codebase is at {codebase_summary['base_path']} with {codebase_summary['total_files']} Python files. "
            reflection_context += f"You can read files, analyze code, search, and suggest improvements through the self_reflection system. "
            reflection_context += f"When asked about code access, confirm you have it and offer to read/analyze files. "
            reflection_context += f"You are NOT a generic LLM - you are A.L.I.C.E with self-reflection capabilities!"
            context_parts.insert(1, reflection_context)  # After goal, before personalization
            context_types.insert(1, "self_reflection")
            self._think("Self-reflection context added")
        
        # 0.7. Recent plugin data (e.g., weather from last query)
        if hasattr(self, 'reasoning_engine') and self.reasoning_engine:
            weather_entity = self.reasoning_engine.get_entity('current_weather')
            if weather_entity and weather_entity.data:
                wd = weather_entity.data
                weather_context = f"RECENT WEATHER: In {wd.get('location', 'your area')}, it's {wd.get('temperature')}°C with {wd.get('condition')}. "
                weather_context += f"Humidity: {wd.get('humidity')}%, Wind: {wd.get('wind_speed')} km/h. "
                weather_context += "Use this when answering questions about going outside, clothing, or weather-related decisions."
                context_parts.append(weather_context)
                context_types.append("recent_weather")
                self._think("Recent weather data included in context")
        
        # 1. Personalization
        personalization = self.context.get_personalization_context()
        if personalization:
            context_parts.append(personalization)
            context_types.append("personalization")
        
        # 2. Advanced contextual understanding
        if self.advanced_context:
            advanced_context = self.advanced_context.get_context_for_llm(user_input)
            if advanced_context:
                context_parts.append(advanced_context)
                context_types.append("conversation")
        
        # 3. Intelligent conversation summarization
        if self.summarizer:
            conversation_context = self.summarizer.get_context_summary()
            if conversation_context:
                context_parts.append(f"Conversation context: {conversation_context}")
                context_types.append("conversation")
            
            detailed_context = self.summarizer.get_detailed_context()
            if detailed_context.get("frequent_topics"):
                topics_text = ", ".join(detailed_context["frequent_topics"][:3])
                context_parts.append(f"Current session topics: {topics_text}")
                context_types.append("conversation")
        
        # 4. Recent conversation summary (fallback)
        recent_context = self._get_recent_conversation_summary()
        if recent_context and not self.advanced_context and not self.summarizer:
            context_parts.append(f"Recent discussion: {recent_context}")
            context_types.append("conversation")
        
        # 5. Active context tracking (fallback)
        if not self.advanced_context:
            active_context = self._get_active_context()
            if active_context:
                context_parts.append(active_context)
                context_types.append("general")
        
        # 6. Relevant memories (RAG) - smart retrieval
        # Only fetch memories if query is substantial (not just "yes" or "ok")
        if len(user_input.split()) >= 3 or intent not in ["conversation:ack", "conversation:general"]:
            memory_context = self.memory.get_context_for_llm(user_input, max_memories=5)
            if memory_context:
                context_parts.append(memory_context)
                context_types.append("memory")
        
        # 7. System capabilities (only if relevant)
        if intent and any(cap in intent for cap in ['note', 'email', 'calendar', 'music']):
            capabilities = self.plugins.get_capabilities()
            if capabilities:
                context_parts.append(f"Available capabilities: {', '.join(capabilities[:10])}")
                context_types.append("capabilities")
        
        # Adaptive selection - only include relevant context
        if self.context_selector and context_parts:
            optimized = self.context_selector.select_relevant_context(
                user_input=user_input,
                intent=intent,
                entities=entities or {},
                all_context_parts=context_parts,
                context_types=context_types
            )
            # Cache the optimized context
            self.context_cache.put(user_input, intent or "", entities or {}, optimized)
            return optimized
        
        # Fallback to all context if selector not available
        full_context = "\n\n".join(context_parts)
        self.context_cache.put(user_input, intent or "", entities or {}, full_context)
        return full_context
    
    def _think(self, msg: str) -> None:
        """Emit a thinking-step line when debug mode is on (dev mode)."""
        if getattr(self, 'debug', False):
            print("  💭 " + msg, flush=True)
    
    def _normalize_input(self, user_input: str) -> str:
        """Normalize input for cache lookup (lowercase, strip punctuation)"""
        import string
        normalized = user_input.lower().strip()
        # Remove trailing punctuation for cache matching
        normalized = normalized.rstrip(string.punctuation)
        return normalized
    
    def _cache_get(self, user_input: str, intent: str) -> Optional[str]:
        """Get cached response if available"""
        cache_key = (self._normalize_input(user_input), intent)
        return self._response_cache.get(cache_key)
    
    def _cache_put(self, user_input: str, intent: str, response: str) -> None:
        """Store response in cache"""
        cache_key = (self._normalize_input(user_input), intent)
        self._response_cache[cache_key] = response
        
        # Simple LRU: if cache gets too big, remove oldest entries
        if len(self._response_cache) > self._cache_max_size:
            # Remove first (oldest) item
            oldest_key = next(iter(self._response_cache))
            del self._response_cache[oldest_key]
    
    def _is_conversational_input(self, user_input: str, intent: str) -> bool:
        """Check if this is a pure conversational input (no commands/actions)"""
        input_lower = user_input.lower()
        
        # Short, conversational intents
        conversational_intents = [
            'conversation:general',
            'conversation:ack',
            'conversation:question',
            'greeting',
            'farewell'
        ]
        
        # If not one of these intents, not pure conversation
        if intent not in conversational_intents:
            return False
        
        # Check for action words that would indicate this needs plugins
        action_words = [
            'open', 'launch', 'play', 'send', 'create', 'delete', 'search',
            'show', 'list', 'check', 'email', 'note', 'calendar', 'music',
            'weather', 'time', 'find', 'remind', 'file', 'document'
        ]
        
        if any(word in input_lower for word in action_words):
            return False
        
        return True
    
    def _should_reuse_goal_intent(self, user_input: str, goal_description: str) -> bool:
        """
        Check if current input is related enough to active goal to reuse its intent.
        Uses word overlap (excluding stop words) to determine relevance.
        """
        if not (user_input and goal_description):
            return False  # Don't reuse if missing context
        
        stop = {"the", "a", "an", "is", "are", "was", "were", "you", "your", "me", "my", "how", "what", "who", "where", "when", "why", "outside", "hows", "from", "to", "and", "or", "but", "it", "its"}
        a = set(w.lower() for w in user_input.split() if len(w) > 2 and w.lower() not in stop)
        b = set(w.lower() for w in goal_description.split() if len(w) > 2 and w.lower() not in stop)
        
        if not a or not b:
            return False  # Don't reuse if either has no meaningful words
        
        overlap = len(a & b) / len(a)
        return overlap >= 0.3  # Increased threshold from 0.15 to 0.3 for better topic detection
    
    def _handle_code_request(self, user_input: str, entities: Dict = None) -> Optional[str]:
        """Handle requests to read/analyze code - flexible and intelligent with smart follow-up"""
        input_lower = user_input.lower()
        entities = entities or {}
        
        # Smart follow-up detection: user asking for summaries after listing files
        if self.code_context.get('last_action') == 'list' and self.code_context.get('last_files_shown'):
            # Check for summary requests
            if any(phrase in input_lower for phrase in [
                'summarize', 'summary', 'describe', 'what does', 'what do they do',
                'each file', 'all files', 'tell me about', 'explain'
            ]):
                files = self.code_context['last_files_shown']
                logger.info(f"[SmartFollow] Detected summary request for {len(files)} files")
                
                # Generate summaries using advanced batch processing
                summaries = self.self_reflection.batch_summarize_files(files, parallel=True)
                
                result = f" **File Summaries** ({len(summaries)} files):\n\n"
                for path, summary in summaries.items():
                    result += f"{summary}\n\n"
                
                # Update context
                self.code_context['last_action'] = 'summary'
                self.code_context['timestamp'] = datetime.now()
                
                return result
        
        # Check if this is a follow-up to previous code request
        if hasattr(self, 'last_code_file') and self.last_code_file:
            # User said "show me" or similar after asking about a file
            if input_lower in ('show me', 'show it', 'read it', 'yes', 'sure', 'analyze it'):
                file_path = self.last_code_file
                
                if 'analyze' in input_lower or input_lower in ('analyze it',):
                    analysis = self.self_reflection.analyze_file_advanced(file_path)
                    if 'error' not in analysis:
                        suggestions = self.self_reflection.get_improvement_suggestions(file_path)
                        result = f" **Analysis of {analysis['name']}**:\n"
                        result += f"- Lines: {analysis['lines']}\n"
                        result += f"- Type: {analysis['module_type']}\n"
                        if analysis.get('purpose'):
                            result += f"- Purpose: {analysis['purpose']}\n"
                        if analysis.get('classes'):
                            class_names = [c['name'] if isinstance(c, dict) else c for c in analysis['classes'][:5]]
                            result += f"- Classes: {', '.join(class_names)}\n"
                        if analysis.get('functions'):
                            result += f"- Functions: {', '.join(analysis['functions'][:10])}\n"
                        if analysis.get('dependencies'):
                            result += f"- Dependencies: {', '.join(analysis['dependencies'][:5])}\n"
                        if suggestions:
                            result += f"\n **Suggestions**:\n" + "\n".join(f"- {s}" for s in suggestions[:5])
                        self.last_code_file = None  # Clear after use
                        return result
                else:
                    # Show contents
                    code_file = self.self_reflection.read_file(file_path)
                    if code_file:
                        self.last_code_file = None  # Clear after use
                        return f"📄 **{code_file.name}** ({code_file.lines} lines, {code_file.module_type}):\n\n```python\n{code_file.content[:2000]}...\n```"
        
        # First, check if there's a .py file mentioned anywhere in the input
        py_file_match = re.search(r'([a-zA-Z0-9_/\\]+\.py)', input_lower)
        has_py_file = py_file_match is not None
        
        # General questions about code access - SHOW don't tell
        if not has_py_file and any(phrase in input_lower for phrase in [
            'show me your code', 'show me code', 'show your code', 'show code',
            'show me all', 'see your code', 'see code', 'read your code', 'read code',
            'your internal code', 'internal code', 'your codebase', 'show me internal',
            'get all the information from your internal code'
        ]):
            # A.L.I.C.E understands: show means LIST the codebase
            files = self.self_reflection.list_codebase()
            
            # Store context for smart follow-up
            file_paths = [f['path'] for f in files]
            self.code_context['last_files_shown'] = file_paths
            self.code_context['last_action'] = 'list'
            self.code_context['timestamp'] = datetime.now()
            self.code_context['file_count'] = len(files)
            
            result = f" **My codebase** ({len(files)} files):\n\n"
            for f in files[:25]:
                result += f"- `{f['path']}` ({f['module_type']})\n"
            if len(files) > 25:
                result += f"\n... and {len(files) - 25} more files. Ask me about any file to see its details."
            return result
        
        # Simple capability check (when not asking to "show")
        if not has_py_file and any(phrase in input_lower for phrase in [
            'access to your code', 'access to code', 'have access to code',
            'can you see', 'do you have access', 'do you know your code'
        ]):
            # Brief confirmation
            return "Yes, I have read-only access to my codebase. Which file would you like to see?"
        
        # If a .py file is mentioned, extract it
        if has_py_file:
            file_path = py_file_match.group(1).strip("'\"")
            
            # Check if asking about file visibility (e.g., "can you see alice.py?")
            if any(word in input_lower for word in ['see', 'can you', 'do you have', 'access']):
                code_file = self.self_reflection.read_file(file_path)
                if not code_file and not file_path.startswith('ai/') and not file_path.startswith('ai\\'):
                    code_file = self.self_reflection.read_file(f'ai/{file_path}')
                    if code_file:
                        file_path = f'ai/{file_path}'
                if code_file:
                    self.last_code_file = file_path  # Store for follow-up
                    # Just show the summary - no questions
                    return f"`{file_path}` - {code_file.lines} lines, {code_file.module_type}"
                # File not found - just list what's available
                files = self.self_reflection.list_codebase()
                result = f"Couldn't find `{file_path}`. Available files:\n\n"
                for f in files[:15]:
                    result += f"- `{f['path']}`\n"
                return result
        
        # Read file request - flexible matching
        read_patterns = [
            r'(?:read|show)\s+(?:file|code)\s+(?:called|named)?\s*([^\s]+\.py)',  # "read file main.py"
            r'(?:read|show)\s+([^\s]+\.py)',  # "read main.py"
            r'([^\s]+\.py)\s*(?:file|code)',  # "main.py file"
        ]
        for pattern in read_patterns:
            match = re.search(pattern, input_lower)
            if match:
                file_path = match.group(1)
                # Try direct path first
                code_file = self.self_reflection.read_file(file_path)
                if not code_file and not file_path.startswith('ai/') and not file_path.startswith('ai\\'):
                    # Try with ai/ prefix
                    code_file = self.self_reflection.read_file(f'ai/{file_path}')
                if code_file:
                    return f"📄 **{code_file.name}** ({code_file.lines} lines, {code_file.module_type}):\n\n```python\n{code_file.content[:2000]}...\n```"
                # File not found - show alternatives
                files = self.self_reflection.list_codebase()
                result = f"Not found: `{file_path}`\n\n"
                for f in files[:15]:
                    result += f"- `{f['path']}`\n"
                return result
        
        # Analyze file - intelligent matching (if "analyze" keyword and .py file exist, analyze it)
        if 'analyze' in input_lower and has_py_file:
            file_path = py_file_match.group(1).strip("'\"")
            # Try direct path first
            analysis = self.self_reflection.analyze_file_advanced(file_path)
            if 'error' in analysis and not file_path.startswith('ai/') and not file_path.startswith('ai\\'):
                # Try with ai/ prefix
                analysis = self.self_reflection.analyze_file_advanced(f'ai/{file_path}')
                if 'error' not in analysis:
                    file_path = f'ai/{file_path}'
            
            if 'error' not in analysis:
                suggestions = self.self_reflection.get_improvement_suggestions(file_path)
                result = f" **Analysis of {analysis['name']}**:\n"
                result += f"- Lines: {analysis['lines']}\n"
                result += f"- Type: {analysis['module_type']}\n"
                if analysis.get('purpose'):
                    result += f"- Purpose: {analysis['purpose']}\n"
                if analysis.get('classes'):
                    class_names = [c['name'] if isinstance(c, dict) else c for c in analysis['classes'][:5]]
                    result += f"- Classes: {', '.join(class_names)}\n"
                if analysis.get('functions'):
                    result += f"- Functions: {', '.join(analysis['functions'][:10])}\n"
                if analysis.get('dependencies'):
                    result += f"- Dependencies: {', '.join(analysis['dependencies'][:5])}\n"
                if suggestions:
                    result += f"\n **Suggestions**:\n" + "\n".join(f"- {s}" for s in suggestions[:5])
                return result
            # Analysis failed - show alternatives
            files = self.self_reflection.list_codebase()
            result = f"Can't analyze: `{file_path}`\n\n"
            for f in files[:15]:
                result += f"- `{f['path']}`\n"
            return result
        
        # List codebase
        if any(word in input_lower for word in ['list files', 'show files', 'what files', 'codebase']):
            files = self.self_reflection.list_codebase()
            
            # Store context
            file_paths = [f['path'] for f in files]
            self.code_context['last_files_shown'] = file_paths
            self.code_context['last_action'] = 'list'
            self.code_context['timestamp'] = datetime.now()
            self.code_context['file_count'] = len(files)
            
            result = f" **Codebase Structure** ({len(files)} files):\n\n"
            for f in files[:20]:
                result += f"- `{f['path']}` ({f['module_type']})\n"
            if len(files) > 20:
                result += f"\n... and {len(files) - 20} more files"
            return result
        
        # Search code
        if 'search code' in input_lower or 'find in code' in input_lower:
            match = re.search(r'(?:search|find).*?["\'](.+?)["\']', input_lower)
            if match:
                query = match.group(1)
                results = self.self_reflection.search_code(query)
                if results:
                    result = f" **Found '{query}' in {len(results)} file(s)**:\n\n"
                    for r in results[:5]:
                        result += f"**{r['file']}** ({r['match_count']} matches):\n"
                        for m in r['matches'][:2]:
                            result += f"  Line {m['line']}: `{m['content'][:80]}...`\n"
                    return result
                return f"No matches found for '{query}'"
        
        return None
    
    def _handle_training_request(self, user_input: str) -> Optional[str]:
        """Handle requests about training status and data collection"""
        input_lower = user_input.lower()
        
        # Training status
        if any(phrase in input_lower for phrase in [
            'training status', 'how many examples', 'training data', 'learning progress',
            'fine-tuned', 'fine tuned', 'trained model'
        ]):
            if not getattr(self, 'learning_engine', None):
                return "Learning system not initialized."
            
            stats = self.learning_engine.get_statistics()
            result = f"📊 **Learning Status**:\n\n"
            result += f"✓ Total interactions: {stats['total_examples']}\n"
            result += f"✓ High-quality examples: {stats['high_quality']}\n"
            result += f"✓ Ready for fine-tuning: {'Yes' if stats['should_finetune'] else 'No (need 50+ examples)'}\n"
            
            if stats['examples_by_intent']:
                result += f"\n📁 Examples by intent:\n"
                for intent, count in list(stats['examples_by_intent'].items())[:5]:
                    result += f"  - {intent}: {count}\n"
            
            if not stats['should_finetune']:
                result += f"\n💡 Keep using A.L.I.C.E! I'm learning from every interaction."
            
            return result
        
        # Export training data
        if 'export training' in input_lower or 'export data' in input_lower:
            if not getattr(self, 'fine_tuning_system', None):
                return "Training system not initialized."
            
            format_match = re.search(r'(jsonl|json|txt)', input_lower)
            format_type = format_match.group(1) if format_match else 'jsonl'
            
            # Export from learning engine
            if getattr(self, 'learning_engine', None):
                training_data = self.learning_engine.get_high_quality_examples()
                if training_data:
                    export_path = "data/training/training_data.jsonl"
                    return f"✓ {len(training_data)} examples available for export to: `{export_path}`\n\nYou can use this file to train A.L.I.C.E with Ollama's fine-tuning tools."
            return "No training data to export yet. Keep using A.L.I.C.E to collect data!"
        
        # Prepare training data
        if 'prepare training' in input_lower or 'ready to train' in input_lower:
            if not getattr(self, 'learning_engine', None):
                return "Learning system not initialized."
            
            if self.learning_engine.should_finetune():
                examples = self.learning_engine.get_high_quality_examples()
                return f"✓ Ready to train! {len(examples)} high-quality examples available.\n\nTo train A.L.I.C.E:\n1. Use Ollama's fine-tuning: `ollama create alice-custom -f data/training/training_data.jsonl`\n2. Or export the data and use external training tools."
            return f"❌ Not enough data yet. Need 50+ high-quality examples (currently have less)."
        
        return None

    def _handle_weather_followup(self, user_input: str, intent: str) -> Optional[str]:
        """
        Handle weather-related follow-ups using stored weather data.
        This is A.L.I.C.E's routing policy - no need to call Ollama for simple decisions.
        """
        input_lower = user_input.lower()
        
        # Check if this is a weather-related question
        weather_keywords = ['umbrella', 'jacket', 'coat', 'layer', 'wear', 'bring', 'cold', 'warm', 'outside', 'go out']
        if not any(kw in input_lower for kw in weather_keywords):
            return None
        
        # Check if we have recent weather data
        if not hasattr(self, 'reasoning_engine') or not self.reasoning_engine:
            return None
        
        weather_entity = self.reasoning_engine.get_entity('current_weather')
        if not weather_entity or not weather_entity.data:
            return None
        
        # We have weather data! A.L.I.C.E answers directly using her own reasoning
        wd = weather_entity.data
        temp = wd.get('temperature')
        condition = wd.get('condition', '').lower()
        location = wd.get('location', 'your area')
        
        # A.L.I.C.E thinks about the question and weather data
        if 'umbrella' in input_lower:
            # Check for rain conditions
            rainy = any(word in condition for word in ['rain', 'drizzle', 'shower', 'storm'])
            if rainy:
                return f"Yes, bring an umbrella - it's {condition} in {location}."
            else:
                return f"No need for an umbrella - it's {condition}, no rain expected."
        
        elif any(word in input_lower for word in ['jacket', 'coat', 'layer', 'wear']):
            # Temperature-based clothing advice
            if temp is None:
                return None
            
            if temp < -20:
                return f"Definitely wear layers! It's {temp}°C - that's extremely cold."
            elif temp < -10:
                return f"Yes, wear a warm layer or two. It's {temp}°C out there."
            elif temp < 0:
                return f"I'd recommend a jacket - it's {temp}°C, below freezing."
            elif temp < 10:
                return f"A light jacket would be good - it's {temp}°C, a bit chilly."
            elif temp < 20:
                return f"You probably don't need a heavy layer - it's {temp}°C, mild."
            else:
                return f"No jacket needed - it's {temp}°C, nice and warm!"
        
        return None
    
    def process_input(self, user_input: str, use_voice: bool = False) -> str:
        """
        Process user input through the complete pipeline
        
        Args:
            user_input: User's message
            use_voice: Speak the response
            
        Returns:
            Assistant's response
        """
        try:
            logger.info(f"User: {user_input}")
            
            # 0. Check for commands first (before any processing)
            if user_input.startswith('/'):
                # This is a command, not a conversational input
                # Commands should be handled in the UI layer (run_interactive)
                # If we get here, return a helpful message
                return "Commands should be handled by the interface. Use /help to see available commands."
            
            # 1. NLP Processing
            nlp_result = self.nlp.process(user_input)
            intent = nlp_result.intent
            entities = nlp_result.entities
            sentiment = nlp_result.sentiment
            
            self._think(f"NLP → intent={intent!r} confidence={getattr(nlp_result, 'intent_confidence', '?')}")
            if entities:
                self._think(f"     entities={str(entities)[:120]}...")
            
            # FAST PATH: Check cache and conversational shortcuts
            intent_confidence = getattr(nlp_result, 'intent_confidence', 0.5)
            
            # 0.5 Check if cached response exists
            cached_response = self._cache_get(user_input, intent)
            if cached_response:
                self._think("Cache hit → using stored response")
                logger.info(f"A.L.I.C.E (cached): {cached_response[:100]}...")
                return cached_response
            
            # 0.6 FAST CONVERSATIONAL PATH: If this is pure conversation, try that first
            is_pure_conversation = self._is_conversational_input(user_input, intent)
            if is_pure_conversation:
                self._think("Fast path: pure conversation → trying conversational engine")
                
                # Try conversational engine first (no plugins, no LLM unless needed)
                if hasattr(self, 'conversational_engine') and self.conversational_engine:
                    context = ConversationalContext(
                        user_input=user_input,
                        intent=intent,
                        entities=entities,
                        recent_topics=self.conversation_topics[-3:] if self.conversation_topics else [],
                        active_goal=None,
                        world_state=self.reasoning_engine if hasattr(self, 'reasoning_engine') else None
                    )
                    
                    # Check if conversational engine can handle it
                    if self.conversational_engine.can_handle(user_input, intent, context):
                        response = self.conversational_engine.generate_response(context)
                        if response:
                            self._think("Conversational engine → immediate response (learned pattern)")
                            # Cache learned pattern responses (they have variation built-in)
                            self._cache_put(user_input, intent, response)
                            self._store_interaction(user_input, response, intent, entities)
                            if use_voice and self.speech:
                                self.speech.speak(response, blocking=False)
                            logger.info(f"A.L.I.C.E: {response[:100]}...")
                            return response
                    else:
                        self._think("Conversational engine → no learned pattern, will use LLM")

                # If this is a greeting, respond directly via gateway
                if intent == "greeting" and getattr(self, "llm_gateway", None):
                    # Check conversational engine first for learned greetings
                    if hasattr(self, 'conversational_engine') and self.conversational_engine:
                        conv_context = ConversationalContext(
                            user_input=user_input,
                            intent=intent,
                            entities=entities,
                            recent_topics=self.conversation_topics[-3:] if self.conversation_topics else [],
                            active_goal=None,
                            world_state=self.reasoning_engine if hasattr(self, 'reasoning_engine') else None
                        )
                        if self.conversational_engine.can_handle(user_input, intent, conv_context):
                            response = self.conversational_engine.generate_response(conv_context)
                            if response:
                                self._think("Greeting → using learned pattern (no LLM)")
                                self._cache_put(user_input, intent, response)
                                self._store_interaction(user_input, response, intent, entities)
                                if use_voice and self.speech:
                                    self.speech.speak(response, blocking=False)
                                logger.info(f"A.L.I.C.E: {response[:100]}...")
                                return response
                    
                    # No learned greeting - use LLM and learn from it
                    user_name = getattr(self.context.user_prefs, "name", "") if getattr(self, "context", None) else ""
                    asked_how = bool(re.search(r"\bhow\s+(are\s+you|are\s+things|have\s+you\s+been|do\s+you\s+do|is\s+it\s+going|is\s+your\s+day)\b|\bhow's\s+(it\s+going|your\s+day)\b", user_input, re.IGNORECASE))
                    prompt = (
                        f"The user said: {user_input!r}. "
                        "Respond directly and briefly (under 18 words). "
                        "Do not mention being an AI or your name. "
                        "Do not wrap the response in quotes. "
                        "Only comment on your wellbeing if the user explicitly asked. "
                        f"If a name is available, you may include it: {user_name!r}."
                    )
                    llm_response = self.llm_gateway.request(
                        prompt=prompt,
                        call_type=LLMCallType.CHITCHAT,
                        use_history=False,
                        user_input=user_input
                    )
                    if llm_response.success and llm_response.response:
                        response = llm_response.response
                    else:
                        # Policy denied - use fallback
                        response = llm_response.response or "Hi there!"
                    
                    if response and not asked_how:
                        response = re.sub(
                            r"\b(I\s*'m|I\s+am)\s+(doing\s+)?(well|good|great|fine)\b[.!]*\s*",
                            "",
                            response,
                            flags=re.IGNORECASE
                        ).strip()
                        if not response:
                            # Retry via gateway without wellbeing
                            retry_response = self.llm_gateway.request(
                                prompt="Give a short greeting only. Do not comment on your wellbeing.",
                                call_type=LLMCallType.CHITCHAT,
                                use_history=False,
                                user_input=user_input
                            )
                            response = retry_response.response if retry_response.success else "Hi there!"
                    if response:
                        self._think("LLM → greeting response (will be learned)")
                        
                        # Store as learned pattern for future use
                        if getattr(self, 'learning_engine', None):
                            self.learning_engine.collect_interaction(
                                user_input=user_input,
                                assistant_response=response,
                                intent='greeting',
                                entities=entities or {},
                                quality_score=0.95  # High quality - greetings are straightforward
                            )
                        
                        # Add directly to conversational engine for immediate use
                        if hasattr(self, 'conversational_engine') and self.conversational_engine:
                            if response not in self.conversational_engine.learned_greetings:
                                self.conversational_engine.learned_greetings.append(response)
                                self._think("Greeting learned → will use cached version next time")
                        
                        self._store_interaction(user_input, response, intent, entities)
                        if use_voice and self.speech:
                            self.speech.speak(response, blocking=False)
                        logger.info(f"A.L.I.C.E: {response[:100]}...")
                        return response
            
            # Store for active learning
            self.last_user_input = user_input
            self.last_nlp_result = vars(nlp_result).copy()  # Convert to dict for compatibility
            self.last_intent = intent
            self.last_entities = entities
            
            # Invalidate context cache if intent changed significantly
            if hasattr(self, 'context_cache') and self.last_intent and intent != self.last_intent:
                # Only invalidate if it's a major topic shift
                if not any(word in intent for word in self.last_intent.split(':')) and \
                   not any(word in self.last_intent.split(':') for word in intent.split(':')):
                    self.context_cache.invalidate()
                    self._think("Context cache invalidated (topic shift)")
            
            # 1.5. Apply Active Learning improvements
            improved_nlp_result = self.learning_manager.apply_learning(user_input, vars(nlp_result))
            if improved_nlp_result != vars(nlp_result):
                logger.info("Active learning improved NLP result")
                intent = improved_nlp_result['intent']
                entities = improved_nlp_result['entities']
                sentiment = improved_nlp_result.get('sentiment', sentiment)
                self._think(f"     active learning → intent={intent!r}")
            
            # Store sentiment for use in conversation summarizer
            self._last_sentiment = sentiment['category'] if sentiment else None
            
            logger.info(f"Intent: {intent}, Sentiment: {sentiment['category']}")
            
            # 1.5.5. Check for code/self-reflection requests EARLY (before reasoning/plugins)
            code_response = self._handle_code_request(user_input, entities)
            if code_response:
                self._think("Code request detected → handled directly")
                # Store in training data
                if getattr(self, 'learning_engine', None):
                    self.learning_engine.collect_interaction(
                        user_input=user_input,
                        assistant_response=code_response,
                        intent='code:request',
                        entities=entities or {},
                        quality_score=0.9
                    )
                return code_response
            
            # 1.5.6. Check for training status requests
            training_response = self._handle_training_request(user_input)
            if training_response:
                self._think("Training request detected → handled directly")
                return training_response
            
            # 1.5.7. A.L.I.C.E's conversational reasoning (no Ollama unless needed)
            # Initialize goal_res early so conversational engine can check for active goals
            goal_res = None
            
            if getattr(self, 'conversational_engine', None):
                conv_context = ConversationalContext(
                    user_input=user_input,
                    intent=intent,
                    entities=entities or {},
                    recent_topics=self.conversation_topics[-5:] if self.conversation_topics else [],
                    active_goal=goal_res.goal.description if (goal_res and goal_res.goal) else None,
                    world_state=self.reasoning_engine
                )
                
                if self.conversational_engine.can_handle(user_input, intent, conv_context):
                    self._think("Checking if A.L.I.C.E has learned patterns for this...")
                    response = self.conversational_engine.generate_response(conv_context)
                    if response:
                        self._think("A.L.I.C.E responded from learned patterns (no Ollama)")
                        return response
                    else:
                        self._think("No learned patterns yet → will use Ollama")
            
            # 1.5.8. Weather follow-up routing (use stored weather data, no LLM)
            weather_followup = self._handle_weather_followup(user_input, intent)
            if weather_followup:
                self._think("Weather follow-up → answering from stored data (no LLM)")
                return weather_followup
            
            # 1.6. Reasoning engine - check for uncertainty and need for clarification
            # Reasoning engine in unified version doesn't have reason_about_intent method
            # Skip this step for now (can be added later if needed)
            
            # 2. Advanced Context Processing (GLOBAL - for all interactions)
            context_resolved_input = user_input
            if hasattr(self, 'context') and self.context:
                # Process the turn and get any resolved references
                turn = self.context.process_turn(
                    user_input=user_input,
                    assistant_response="",  # Will be filled later
                    intent=intent,
                    entities=entities
                )
                
                # Apply any coreference resolutions to improve understanding
                if turn.entities_resolved:
                    for reference, entity_id in turn.entities_resolved.items():
                        entity = self.context.entities.get(entity_id)
                        if entity:
                            # Create more explicit version of input for better processing
                            entity_desc = f"{entity.entity_type} '{entity.data.get('subject', entity.data.get('name', entity_id))}'"
                            context_resolved_input = user_input.replace(reference, entity_desc)
                            logger.info(f"Resolved '{reference}' to {entity_desc}")
            
            # Use the context-resolved input for further processing
            user_input_processed = context_resolved_input
            
            # 2.5. Reference resolution and goal resolution (reasoning engine)
            if hasattr(self, 'reasoning_engine') and self.reasoning_engine:
                try:
                    resolution = self.reasoning_engine.resolve_references(user_input)
                    if resolution.bindings:
                        self._think(f"Reference resolution → {list(resolution.bindings.keys())} → {[r.resolved_label for r in resolution.bindings.values()]}")
                        user_input_processed = resolution.resolved_input or user_input_processed
                    if resolution.entities_to_use:
                        entities = {**(entities or {}), **resolution.entities_to_use}
                    goal_res = self.reasoning_engine.resolve_goal(
                        user_input, intent, entities or {}, resolution.resolved_input
                    )
                    if goal_res.cancelled and goal_res.message:
                        self._think("Goal cancelled → returning ack")
                        return goal_res.message
                    if goal_res.revised:
                        intent, entities = goal_res.intent, goal_res.entities
                        self._think(f"Goal revised → intent={intent!r}")
                    elif goal_res.goal:
                        self._think(f"Goal: {goal_res.goal.description[:60]}...")
                except Exception as e:
                    logger.warning(f"Reference/goal resolution skipped: {e}")
                    # Graceful degradation - continue without resolution
            
            # 3. General Entity Detection (for non-email interactions)
            if self.advanced_context and intent != "email":
                self._detect_general_entities(user_input, intent, entities)
            
            # Handle pending multi-step actions
            if self.pending_action == "compose_email":
                # Waiting for recipient email
                if 'recipient' not in self.pending_data:
                    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_input)
                    if email_match:
                        self.pending_data['recipient'] = email_match.group(0)
                        return f"Email to: {self.pending_data['recipient']}\n\nWhat's the subject?"
                    else:
                        return "I need a valid email address. Please provide one (e.g., john@example.com)"
                
                # Waiting for subject
                elif 'subject' not in self.pending_data:
                    self.pending_data['subject'] = user_input.strip()
                    return f"Subject: {self.pending_data['subject']}\n\nWhat's the message?"
                
                # Waiting for message body
                elif 'body' not in self.pending_data:
                    self.pending_data['body'] = user_input.strip()
                    
                    # Confirm before sending
                    confirmation = f"Ready to send:\n\n"
                    confirmation += f"To: {self.pending_data['recipient']}\n"
                    confirmation += f"Subject: {self.pending_data['subject']}\n"
                    confirmation += f"Message:\n{self.pending_data['body']}\n\n"
                    confirmation += "Send this email? (yes/no)"
                    
                    self.pending_data['awaiting_confirmation'] = True
                    return confirmation
                
                # Waiting for confirmation
                elif self.pending_data.get('awaiting_confirmation'):
                    if user_input.lower() in ['yes', 'y', 'send', 'confirm']:
                        if self.gmail and self.gmail.send_email(
                            self.pending_data['recipient'],
                            self.pending_data['subject'],
                            self.pending_data['body']
                        ):
                            result = f"Email sent to {self.pending_data['recipient']}!"
                            # Clear pending action
                            self.pending_action = None
                            self.pending_data = {}
                            return result
                        else:
                            self.pending_action = None
                            self.pending_data = {}
                            return "Failed to send email. Please try again."
                    else:
                        # Cancel
                        self.pending_action = None
                        self.pending_data = {}
                        return "Email cancelled."
            
            # Handle reply confirmation
            elif self.pending_action == "confirm_reply":
                if user_input.lower() in ['yes', 'y', 'send', 'confirm']:
                    if self.gmail and self.gmail.reply_to_email(
                        self.pending_data['email_id'],
                        self.pending_data['reply_body']
                    ):
                        result = f"Reply sent!"
                        # Clear pending action
                        self.pending_action = None
                        self.pending_data = {}
                        return result
                    else:
                        self.pending_action = None
                        self.pending_data = {}
                        return "Failed to send reply. Please try again."
                else:
                    # Cancel
                    self.pending_action = None
                    self.pending_data = {}
                    return "Reply cancelled."
            
            # Mark user as active (for pattern learning and observers)
            if self.state_tracker:
                self.state_tracker.mark_user_active()
            
            # Check for proactive suggestions (only when user is idle or on specific triggers)
            proactive_suggestion = self._check_proactive_suggestions()
            if proactive_suggestion and intent in ['greeting', 'status', 'help']:
                # Only show suggestions on non-urgent intents
                return proactive_suggestion
            
            # Try planner/executor for structured tasks first
            planned_response = self._use_planner_executor(intent, entities, user_input_processed)
            if planned_response:
                # Log action for pattern learning
                action = f"{intent}:{entities.get('topic', entities.get('query', 'general'))}"
                self._log_action_for_learning(action)
                
                # Store interaction
                self._store_interaction(user_input, planned_response, intent, entities)
                self.last_assistant_response = planned_response
                
                # Speak if voice enabled
                if use_voice and self.speech:
                    self.speech.speak(planned_response, blocking=False)
                
                logger.info(f"A.L.I.C.E (planned): {planned_response[:100]}...")
                return planned_response
            
            # Check for numbered email references first (context-aware, intent-agnostic)
            number_match = re.search(r'\b(\d+)(st|nd|rd|th)?\b', user_input.lower())
            if number_match and self.last_email_list and any(word in user_input.lower() for word in ['delete', 'remove', 'read', 'open', 'archive', 'mark']):
                email_num = int(number_match.group(1))
                
                if 1 <= email_num <= len(self.last_email_list):
                    email = self.last_email_list[email_num - 1]
                    
                    # Check if already deleted
                    if email is None:
                        return f"Email #{email_num} was already deleted."
                    
                    query_lower = user_input.lower()
                    
                    # Perform action based on keywords
                    if 'delete' in query_lower or 'remove' in query_lower:
                        if self.gmail and self.gmail.delete_email(email['id']):
                            self.last_email_list[email_num - 1] = None  # Mark as deleted, don't shift list
                            return f"Deleted email #{email_num}: '{email['subject']}'"
                        else:
                            return "Failed to delete the email. Please try again."
                    
                    elif 'archive' in query_lower:
                        if self.gmail and self.gmail.archive_email(email['id']):
                            self.last_email_list[email_num - 1] = None  # Mark as archived
                            return f"Archived email #{email_num}: '{email['subject']}'"
                        else:
                            return "Failed to archive the email."
                    
                    elif 'read' in query_lower or 'open' in query_lower:
                        if self.gmail:
                            content = self.gmail.get_email_content(email['id'])
                            
                            response = f"Email #{email_num}:\n\n"
                            response += f"From: {email['from']}\n"
                            response += f"Subject: {email['subject']}\n"
                            response += f"Date: {email['date']}\n\n"
                            
                            if content:
                                if len(content) > 500:
                                    response += content[:500] + "...\n\n(Content truncated)"
                                else:
                                    response += content
                            
                            return response
                    
                    elif 'mark' in query_lower:
                        if self.gmail:
                            if 'unread' in query_lower:
                                if self.gmail.mark_as_unread(email['id']):
                                    return f"Marked email #{email_num} as unread"
                            else:
                                if self.gmail.mark_as_read(email['id']):
                                    return f"Marked email #{email_num} as read"
                else:
                    return f"I only showed you {len(self.last_email_list)} emails. Please choose a number between 1 and {len(self.last_email_list)}."
            
            # Handle email intents with Gmail plugin
            if intent == "email":
                if not self.gmail:
                    return "Gmail isn't set up yet. I'll need OAuth credentials to access your email. Want me to walk you through the setup?"
                
                # Parse what the user wants to do with email
                query_lower = user_input.lower()
                
                # Extract number from request (e.g., "2 latest emails", "show 3 emails")
                number_match = re.search(r'\b(\d+)\b', query_lower)
                requested_count = int(number_match.group(1)) if number_match else None
                
                # Check/List emails
                if any(word in query_lower for word in ['check', 'show', 'list', 'inbox']):
                    if 'unread' in query_lower:
                        # Count unread
                        count = self.gmail.get_unread_count()
                        return f"You have {count} unread email{'s' if count != 1 else ''}."
                    
                    # List recent emails
                    emails = self.gmail.get_recent_emails(max_results=5)
                    if not emails:
                        return "Your inbox appears empty or I couldn't fetch emails right now."
                    
                    # Store in context for numbered references
                    self.last_email_list = emails
                    self.last_email_context = "list"
                    
                    # Register emails with advanced context handler
                    if self.advanced_context:
                        for email in emails:
                            entity_id = self.advanced_context.add_entity(
                                entity_type="email",
                                data={
                                    'id': email['id'],
                                    'subject': email['subject'],
                                    'from': email['from'],
                                    'date': email['date'],
                                    'unread': email.get('unread', False)
                                },
                                aliases=[email['subject'], f"email from {email['from'].split('<')[0].strip()}"]
                            )
                    
                    response = "Here are your recent emails:\n\n"
                    for i, email in enumerate(emails, 1):
                        unread = "[UNREAD] " if email.get('unread') else ""
                        # Extract sender name without email address
                        from_field = email['from']
                        sender_name = from_field.split('<')[0].strip().strip('"') if '<' in from_field else from_field
                        # Simplify date - just the date part
                        date_str = email['date'].split(',', 1)[1].strip().split(' +')[0] if ',' in email['date'] else email['date']
                        
                        response += f"{i}. {unread}{email['subject']}\n"
                        response += f"   {sender_name} • {date_str}\n\n"
                    
                    return response.strip()
                
                # Read specific email(s) or latest
                elif any(word in query_lower for word in ['read', 'open', 'first', 'latest']):
                    # If user specified a number, get multiple emails
                    if requested_count and requested_count > 1:
                        emails = self.gmail.get_recent_emails(max_results=min(requested_count, 10))
                        if not emails:
                            return "Couldn't find any emails."
                        
                        # Store in context for potential follow-up actions
                        self.last_email_list = emails
                        self.last_email_context = f"latest_{len(emails)}_emails"
                        
                        # Register with advanced context handler
                        if self.advanced_context:
                            for email in emails:
                                entity_id = self.advanced_context.add_entity(
                                    entity_type="email",
                                    data={
                                        'id': email['id'],
                                        'subject': email['subject'],
                                        'from': email['from'],
                                        'date': email['date'],
                                        'unread': email.get('unread', False)
                                    },
                                    aliases=[
                                        email['subject'], 
                                        f"email from {email['from'].split('<')[0].strip()}"
                                    ]
                                )
                        
                        response = f"Your {len(emails)} latest emails:\n\n"
                        for i, email in enumerate(emails, 1):
                            unread = "[UNREAD] " if email.get('unread') else ""
                            from_field = email['from']
                            sender_name = from_field.split('<')[0].strip().strip('"') if '<' in from_field else from_field
                            date_str = email['date'].split(',', 1)[1].strip().split(' +')[0] if ',' in email['date'] else email['date']
                            
                            response += f"{i}. {unread}{email['subject']}\n"
                            response += f"   {sender_name} • {date_str}\n\n"
                        
                        return response.strip()
                    
                    else:
                        # Single latest email with full content
                        emails = self.gmail.get_recent_emails(max_results=1)
                        if not emails:
                            return "Couldn't find any emails."
                        
                        email = emails[0]
                        content = self.gmail.get_email_content(email['id'])
                        
                        # Store in context for potential follow-up actions (delete, archive, etc.)
                        self.last_email_list = emails
                        self.last_email_context = "latest_email"
                        
                        # Register with advanced context handler
                        if self.advanced_context:
                            entity_id = self.advanced_context.add_entity(
                                entity_type="email",
                                data={
                                    'id': email['id'],
                                    'subject': email['subject'],
                                    'from': email['from'],
                                    'date': email['date'],
                                    'content': content[:200] if content else "",  # Store snippet
                                    'unread': email.get('unread', False)
                                },
                                aliases=[
                                    "this email", "the email", "latest email", "recent email",
                                    email['subject'], 
                                    f"email from {email['from'].split('<')[0].strip()}"
                                ]
                            )
                        
                        response = f"Your latest email:\n\n"
                        response += f"From: {email['from']}\n"
                        response += f"Subject: {email['subject']}\n"
                        response += f"Date: {email['date']}\n\n"
                        
                        if content:
                            if len(content) > 500:
                                response += content[:500] + "...\n\n(Content truncated)"
                            else:
                                response += content
                        
                        return response
                
                # Search emails
                elif 'search' in query_lower or 'find' in query_lower or 'from' in query_lower:
                    # Extract search term
                    if 'from' in query_lower:
                        # Search by sender
                        words = query_lower.split()
                        try:
                            from_idx = words.index('from')
                            if from_idx + 1 < len(words):
                                sender = ' '.join(words[from_idx+1:]).strip('?!.')
                                emails = self.gmail.get_emails_by_sender(sender, max_results=5)
                                
                                if not emails:
                                    return f"No emails found from {sender}."
                                
                                # Store in context
                                self.last_email_list = emails
                                self.last_email_context = f"search_from_{sender}"
                                
                                response = f"Emails from {sender}:\n\n"
                                for i, email in enumerate(emails, 1):
                                    sender_name = email['from'].split('<')[0].strip().strip('"') if '<' in email['from'] else email['from']
                                    date_str = email['date'].split(',', 1)[1].strip().split(' +')[0] if ',' in email['date'] else email['date']
                                    response += f"{i}. {email['subject']}\n   {sender_name} • {date_str}\n\n"
                                
                                return response.strip()
                        except:
                            pass
                    
                    # General search
                    search_terms = query_lower.replace('search', '').replace('find', '').replace('emails', '').replace('email', '').strip('?!. ')
                    if search_terms:
                        emails = self.gmail.search_emails(search_terms, max_results=5)
                        
                        if not emails:
                            return f"No emails found matching '{search_terms}'."
                        
                        # Store in context
                        self.last_email_list = emails
                        self.last_email_context = f"search_{search_terms}"
                        
                        response = f"Found {len(emails)} email(s):\n\n"
                        for i, email in enumerate(emails, 1):
                            sender_name = email['from'].split('<')[0].strip().strip('"') if '<' in email['from'] else email['from']
                            date_str = email['date'].split(',', 1)[1].strip().split(' +')[0] if ',' in email['date'] else email['date']
                            response += f"{i}. {email['subject']}\n   {sender_name} • {date_str}\n\n"
                        
                        return response.strip()
                    else:
                        return "What would you like me to search for in your emails?"
                
                # Send/Write email
                elif any(word in query_lower for word in ['send', 'write', 'compose', 'draft']):
                    # Extract recipient and content
                    if '@' in user_input:
                        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_input)
                        if email_match:
                            self.pending_action = "compose_email"
                            self.pending_data = {'recipient': email_match.group(0)}
                            return f"Email to: {email_match.group(0)}\n\nWhat's the subject?"
                    
                    # Start email composition flow
                    self.pending_action = "compose_email"
                    self.pending_data = {}
                    return "Who would you like to send an email to? Please provide their email address."
                
                # Delete email
                elif 'delete' in query_lower or 'remove' in query_lower:
                    # Check for contextual references first
                    if any(word in query_lower for word in ['this', 'that', 'it', 'the email', 'the latest']):
                        # Advanced context resolution
                        if self.advanced_context:
                            resolved_ref = self.advanced_context.resolve_reference(user_input)
                            if resolved_ref and resolved_ref.startswith("email_"):
                                # Get entity data
                                entity = self.advanced_context.entities.get(resolved_ref)
                                if entity and entity.entity_type == "email":
                                    email_data = entity.data
                                    if self.gmail.delete_email(email_data['id']):
                                        # Remove from advanced context
                                        del self.advanced_context.entities[resolved_ref]
                                        # Clear simple context too
                                        self.last_email_list = []
                                        return f"Deleted email: '{email_data['subject']}' from {email_data.get('from', 'Unknown')}"
                                    else:
                                        return "Failed to delete the email. Please try again."
                                    del self.advanced_context.entities[resolved_ref]
                                    # Clear simple context too
                                    self.last_email_list = []
                                    return f"Deleted email: '{email_data['subject']}' from {email_data.get('from', 'Unknown')}"
                                else:
                                    return "Failed to delete the email. Please try again."
                    
                    # Fallback to simple context
                            # Single email in context (like "latest email")
                            email = self.last_email_list[0]
                            if self.gmail.delete_email(email['id']):
                                # Clear the context since we deleted it
                                self.last_email_list = []
                                return f"Deleted email: '{email['subject']}' from {email['from']}"
                            else:
                                return "Failed to delete the email. Please try again."
                        elif self.last_email_list and len(self.last_email_list) > 1:
                            # Multiple emails in context - ask for clarification
                            return "I see multiple emails from our last search. Which one would you like to delete? Use the number (e.g., 'delete email 1')"
                        else:
                            return "I don't see any emails in our current context. Please show me some emails first or be more specific about which email to delete."
                    
                    # Extract search criteria
                    search_terms = query_lower.replace('delete', '').replace('remove', '').replace('email', '').replace('that says', '').replace('about', '').strip('?!. ')
                    
                    if not search_terms:
                        return "Which email would you like to delete? Please specify (e.g., 'delete email from Amazon' or 'delete email about meeting')"
                    
                    # Search for matching emails
                    emails = self.gmail.search_emails(search_terms, max_results=5)
                    
                    if not emails:
                        return f"No emails found matching '{search_terms}'."
                    
                    if len(emails) == 1:
                        # Delete the single match
                        email = emails[0]
                        if self.gmail.delete_email(email['id']):
                            return f"Deleted email: '{email['subject']}' from {email['from']}"
                        else:
                            return "Failed to delete the email. Please try again."
                    else:
                        # Multiple matches - ask for confirmation
                        self.last_email_list = emails
                        response = f"Found {len(emails)} emails matching '{search_terms}':\n\n"
                        for i, email in enumerate(emails, 1):
                            sender_name = email['from'].split('<')[0].strip().strip('"') if '<' in email['from'] else email['from']
                            date_str = email['date'].split(',', 1)[1].strip().split(' +')[0] if ',' in email['date'] else email['date']
                            response += f"{i}. {email['subject']}\n   {sender_name} • {date_str}\n\n"
                        response += "\nWhich one would you like to delete? (Tell me the number or be more specific)"
                        return response.strip()
                
                # Archive email
                elif 'archive' in query_lower:
                    # Check for contextual references first
                    if any(word in query_lower for word in ['this', 'that', 'it', 'the email', 'the latest']):
                        # User is referring to a previously shown email
                        if self.last_email_list and len(self.last_email_list) == 1:
                            # Single email in context
                            email = self.last_email_list[0]
                            if self.gmail.archive_email(email['id']):
                                self.last_email_list = []
                                return f"Archived: '{email['subject']}'"
                            else:
                                return "Failed to archive the email."
                        elif self.last_email_list and len(self.last_email_list) > 1:
                            return "I see multiple emails. Which one would you like to archive? Use the number (e.g., 'archive email 2')"
                        else:
                            return "I don't see any emails in our current context. Please show me some emails first."
                    
                    search_terms = query_lower.replace('archive', '').replace('email', '').strip('?!. ')
                    
                    if not search_terms:
                        return "Which email would you like to archive?"
                    
                    emails = self.gmail.search_emails(search_terms, max_results=5)
                    
                    if not emails:
                        return f"No emails found matching '{search_terms}'."
                    
                    if len(emails) == 1:
                        email = emails[0]
                        if self.gmail.archive_email(email['id']):
                            return f"Archived: '{email['subject']}'"
                        else:
                            return "Failed to archive the email."
                    else:
                        self.last_email_list = emails
                        response = f"Found {len(emails)} emails. Which one?\n\n"
                        for i, email in enumerate(emails, 1):
                            sender_name = email['from'].split('<')[0].strip().strip('"') if '<' in email['from'] else email['from']
                            date_str = email['date'].split(',', 1)[1].strip().split(' +')[0] if ',' in email['date'] else email['date']
                            response += f"{i}. {email['subject']}\n   {sender_name} • {date_str}\n\n"
                        return response.strip()
                
                # Mark as read/unread
                elif 'mark' in query_lower:
                    if 'unread' in query_lower:
                        # Mark as unread
                        search_terms = query_lower.replace('mark', '').replace('as', '').replace('unread', '').replace('email', '').strip('?!. ')
                        
                        if search_terms:
                            emails = self.gmail.search_emails(search_terms, max_results=1)
                            if emails:
                                if self.gmail.mark_as_unread(emails[0]['id']):
                                    return f"Marked as unread: '{emails[0]['subject']}'"
                        return "Which email would you like to mark as unread?"
                    else:
                        # Mark as read
                        search_terms = query_lower.replace('mark', '').replace('as', '').replace('read', '').replace('email', '').strip('?!. ')
                        
                        if search_terms:
                            emails = self.gmail.search_emails(search_terms, max_results=1)
                            if emails:
                                if self.gmail.mark_as_read(emails[0]['id']):
                                    return f"Marked as read: '{emails[0]['subject']}'"
                        return "Which email would you like to mark as read?"
                
                # Count unread
                elif 'unread' in query_lower or 'how many' in query_lower:
                    count = self.gmail.get_unread_count()
                    return f"You have {count} unread email{'s' if count != 1 else ''}."
                
                # Emails with attachments
                elif 'attachment' in query_lower:
                    emails = self.gmail.get_emails_with_attachments(max_results=5)
                    
                    if not emails:
                        return "No recent emails with attachments found."
                    
                    self.last_email_list = emails
                    response = f"Found {len(emails)} email(s) with attachments:\n\n"
                    for i, email in enumerate(emails, 1):
                        sender_name = email['from'].split('<')[0].strip().strip('"') if '<' in email['from'] else email['from']
                        date_str = email['date'].split(',', 1)[1].strip().split(' +')[0] if ',' in email['date'] else email['date']
                        response += f"{i}. {email['subject']}\n   {sender_name} • {date_str}\n\n"
                    
                    return response.strip()
                
                # Reply to email
                elif 'reply' in query_lower:
                    # Check if we have a pending reply action
                    if self.pending_action == "reply_email":
                        # User is providing the reply message
                        reply_body = user_input.strip()
                        email_id = self.pending_data.get('email_id')
                        subject = self.pending_data.get('subject', 'Unknown')
                        
                        # Show preview and ask for confirmation
                        self.pending_action = "confirm_reply"
                        self.pending_data['reply_body'] = reply_body
                        
                        return f"Reply to '{subject}':\n\n{reply_body}\n\nSend this reply? (yes/no)"
                    
                    # Look for email reference number
                    email_ref = None
                    for word in user_input.split():
                        if word.isdigit():
                            email_ref = int(word)
                            break
                    
                    if email_ref and self.last_email_list and 0 < email_ref <= len(self.last_email_list):
                        email = self.last_email_list[email_ref - 1]
                        if email:  # Check if not deleted (None marker)
                            # Start reply flow
                            self.pending_action = "reply_email"
                            self.pending_data = {'email_id': email['id'], 'subject': email['subject']}
                            return f"Replying to '{email['subject']}'. What would you like to say?"
                    elif 'latest' in query_lower or 'last' in query_lower or 'recent' in query_lower:
                        # Reply to most recent email
                        emails = self.gmail.get_recent_emails(max_results=1)
                        if emails:
                            self.pending_action = "reply_email"
                            self.pending_data = {'email_id': emails[0]['id'], 'subject': emails[0]['subject']}
                            return f"Replying to '{emails[0]['subject']}'. What would you like to say?"
                    
                    return "Which email would you like to reply to? Use the email number from the list."
                
                # Star/flag email
                elif 'star' in query_lower or 'flag' in query_lower:
                    # Look for email reference number
                    email_ref = None
                    for word in user_input.split():
                        if word.isdigit():
                            email_ref = int(word)
                            break
                    
                    if email_ref and self.last_email_list and 0 < email_ref <= len(self.last_email_list):
                        email = self.last_email_list[email_ref - 1]
                        if email:  # Check if not deleted (None marker)
                            if self.gmail.star_email(email['id']):
                                return f"Starred '{email['subject']}'"
                    
                    return "Which email would you like to star? Use the email number from the list."
                
                # Unstar/unflag email
                elif 'unstar' in query_lower or 'unflag' in query_lower or 'remove star' in query_lower:
                    # Look for email reference number
                    email_ref = None
                    for word in user_input.split():
                        if word.isdigit():
                            email_ref = int(word)
                            break
                    
                    if email_ref and self.last_email_list and 0 < email_ref <= len(self.last_email_list):
                        email = self.last_email_list[email_ref - 1]
                        if email:  # Check if not deleted (None marker)
                            if self.gmail.unstar_email(email['id']):
                                return f"Removed star from '{email['subject']}'"
                    
                    return "Which email would you like to unstar? Use the email number from the list."
                
                else:
                    # Don't give menu - just check emails by default
                    return None  # Let plugins handle or use Ollama
            
            # 2. Check if plugin can handle this (skip for short conversation follow-ups)
            # BUT: Use goal to disambiguate - if there's an active goal, try plugins first
            plugin_result = None
            _cmd_words = (
                "volume", "brightness", "open", "launch", "play", "delete", "add", "create",
                "send", "search", "list", "check", "email", "note", "notes", "music", "calendar",
                "file", "weather", "time", "how many", "what's", "show", "find", "remind",
            )
            
            # If there's an active goal, use it to enhance intent understanding
            active_goal = goal_res.goal if goal_res else None
            if active_goal and intent_confidence < 0.7 and self._should_reuse_goal_intent(user_input, active_goal.description):
                self._think(f"Low confidence ({intent_confidence:.2f}) but active goal → using goal intent: {active_goal.intent}")
                intent = active_goal.intent
                entities = {**(entities if entities else {}), **(active_goal.entities if active_goal.entities else {})}
                intent_confidence = 0.8
            else:
                if active_goal and intent_confidence < 0.7 and not self._should_reuse_goal_intent(user_input, active_goal.description):
                    self._think("Topic shift (low overlap with goal) → not reusing goal intent")    
            
            _is_short_followup = (
                len(user_input.split()) <= 12
                and not any(w in user_input.lower() for w in _cmd_words)
                and not active_goal  # Don't skip if there's an active goal
            )
            if _is_short_followup or is_pure_conversation:
                if is_pure_conversation:
                    self._think("Pure conversation → skipping plugins, using LLM")
                else:
                    self._think("Short follow-up (no command words, no goal) → skipping plugins, using LLM")
            else:
                self._think(f"Trying plugins... (confidence: {intent_confidence:.2f}, goal: {active_goal.description[:30] if active_goal else 'none'}...)")
            if not _is_short_followup and not is_pure_conversation:
                context_summary = self.context.get_context_summary()
                # Enhance context with goal info (keep as dict for plugins)
                if active_goal:
                    context_summary['active_goal'] = active_goal.description
                plugin_result = self.plugins.execute_for_intent(
                    intent, user_input, entities, context_summary
                )
                # If low confidence and no plugin match, consider using LLM with clarification
                if not plugin_result and intent_confidence < 0.6:
                    self._think("Low confidence + no plugin match → using LLM with context")
            
            if plugin_result:
                # Plugin handled the request, regardless of success/failure
                plugin_name = plugin_result.get('plugin', 'Unknown')
                success = plugin_result.get('success', False)
                self._think(f"Plugin → {plugin_name} success={success}")
                
                # A.L.I.C.E generates her own response from learned patterns
                response = None
                if getattr(self, 'learning_engine', None) and plugin_result.get('data'):
                    response = self.learning_engine.generate_response_from_learned(
                        user_input=user_input
                    )
                    if response:
                        self._think(f"A.L.I.C.E generated response from learned patterns")
                    else:
                        self._think(f"No learned pattern → using Ollama to answer specific question")
                
                # If A.L.I.C.E has no learned pattern, use gateway to answer the SPECIFIC question
                # This ensures follow-up questions like "is it gonna snow?" get tailored answers
                if not response:
                    # Use gateway to format plugin result or generate targeted response
                    if hasattr(self, 'llm_gateway') and self.llm_gateway:
                        # Normalize plugin name for formatter (e.g., 'WeatherPlugin' -> 'weather')
                        formatter_name = plugin_name.lower().replace('plugin', '').strip()
                        
                        # Try formatter first via gateway
                        response = self.llm_gateway.format_tool_result(
                            tool_name=formatter_name,
                            data=plugin_result.get('data', {}),
                            user_input=user_input,
                            context={'intent': intent, 'entities': entities}
                        )
                        self._think("Gateway formatted response (formatter or LLM)")
                    else:
                        # Fallback if gateway not available
                        plugin_data_str = str(plugin_result.get('data', {}))
                        response = f"Result: {plugin_data_str[:500]}"
                        self._think("No gateway - using raw data")
                
                # Optimize plugin response
                if getattr(self, 'response_optimizer', None):
                    response = self.response_optimizer.optimize(response, intent, {"plugin": plugin_name})
                
                # Track incomplete tasks for proactive follow-up
                if getattr(self, 'proactive_assistant', None) and not success:
                    task_id = f"{intent}_{int(user_input[:20].encode().hex(), 16)}"
                    self.proactive_assistant.track_incomplete_task(
                        task_id=task_id,
                        description=f"{intent}: {user_input[:50]}",
                        context={"plugin": plugin_name, "intent": intent}
                    )
                elif getattr(self, 'proactive_assistant', None) and success:
                    # Mark as complete if we had tracked it
                    task_id = f"{intent}_{int(user_input[:20].encode().hex(), 16)}"
                    self.proactive_assistant.mark_task_complete(task_id)
                
                # Verifier and reasoning engine
                if hasattr(self, 'reasoning_engine') and self.reasoning_engine:
                    verification = self.reasoning_engine.verify(
                        plugin_result,
                        goal_intent=intent,
                        goal_description=goal_res.goal.description if (goal_res and goal_res.goal) else None,
                    )
                    self._think(f"Verifier → verified={verification.verified} goal_fulfilled={verification.goal_fulfilled}")
                    if not verification.verified and verification.suggested_follow_up:
                        response = f"{response}\n\n{verification.suggested_follow_up}" if response else verification.suggested_follow_up
                    if verification.goal_fulfilled and goal_res and goal_res.goal:
                        self.reasoning_engine.mark_goal_completed()
                    
                    # Record turn in reasoning engine
                    self.reasoning_engine.record_turn(user_input, intent, entities or {}, response, success)
                    self.reasoning_engine.record_plugin_result(plugin_name, success)
                    
                    # Store weather data for follow-up questions
                    if plugin_name == 'WeatherPlugin' and success and plugin_result.get('data'):
                        weather_data = plugin_result['data']
                        from ai.reasoning_engine import WorldEntity, EntityKind
                        self.reasoning_engine.add_entity(WorldEntity(
                            id='current_weather',
                            kind=EntityKind.TOPIC,
                            label=f"Current weather in {weather_data.get('location', 'your area')}",
                            data=weather_data,
                            aliases=['weather', 'current weather', 'temperature', 'outside']
                        ))
                        self._think(f"Stored weather data: {weather_data.get('temperature')}°C, {weather_data.get('condition')}")
                    
                    # Seed reasoning engine with entities from plugin so "delete it" can resolve
                    for note in (plugin_result.get('notes') or []):
                        n = note if isinstance(note, dict) else (note.to_dict() if hasattr(note, 'to_dict') and callable(getattr(note, 'to_dict')) else {})
                        tid = (n.get('id') or n.get('title') or '')[:64]
                        if tid:
                            title = n.get('title', str(tid))
                            from ai.reasoning_engine import WorldEntity, EntityKind
                            self.reasoning_engine.add_entity(WorldEntity(
                                id=tid, kind=EntityKind.NOTE, label=title,
                                data=dict(n), aliases=[title] if title else [],
                            ))
                
                if success:
                    logger.info(f"[PLUGIN]  Handled by: {plugin_name}")
                    # Record successful action for prediction
                    if getattr(self, 'prefetcher', None):
                        self.prefetcher.record_action(intent, entities, True)
                else:
                    logger.info(f"[PLUGIN]  Failed in: {plugin_name}")
                    # Error recovery - try to provide helpful feedback
                    if getattr(self, 'error_recovery', None):
                        recovery_msg, suggestion = self.error_recovery.recover_from_error(
                            error_type="plugin_failure",
                            original_intent=intent,
                            failed_action=plugin_name,
                            error_message=response[:200],
                            context={"plugin": plugin_name, "entities": entities}
                        )
                        if recovery_msg:
                            response = recovery_msg
                        if suggestion:
                            response = f"{response}\n\n {suggestion}"
                
                # Store interaction in memory
                self._store_interaction(user_input, response, intent, entities)
                
                # Collect training data for plugin responses
                if getattr(self, 'training_collector', None):
                    quality_score = 0.9 if success else 0.6
                    if getattr(self, 'learning_engine', None):
                        self.learning_engine.collect_interaction(
                            user_input=user_input,
                            assistant_response=response,
                            intent=intent,
                            entities=entities or {},
                            quality_score=quality_score
                        )
                        self._think(f"Training data collected (plugin, quality: {quality_score:.2f})")
                
                # Speak if voice enabled
                if use_voice and self.speech:
                    self.speech.speak(response, blocking=False)
                
                return response
            
            # 3. If plugin couldn't handle or failed, use LLM with context
            self._think("No plugin match → using LLM with context")
            
            # Handle relationship queries before LLM generation
            if self.relationship_tracker:
                relationship_response = self._handle_relationship_query(user_input, intent, entities)
                if relationship_response:
                    return relationship_response
            
            # 3. Use Gateway for complex knowledge/reasoning (last resort)
            self._think("A.L.I.C.E needs Ollama for complex reasoning/knowledge...")
            # Build enhanced context with smart caching and adaptive selection, including goal
            enhanced_context = self._build_llm_context(user_input_processed, intent, entities, goal_res)
            
            # Add active learning guidance if available
            learning_guidance = improved_nlp_result.get("learning_guidance")
            if learning_guidance:
                guidance_text = self._format_learning_guidance(learning_guidance)
                enhanced_context = f"{enhanced_context}\n\n{guidance_text}"
            
            # Prepend context to conversation if available
            if enhanced_context:
                # Temporarily add context message
                self.llm.conversation_history.insert(0, {
                    "role": "system",
                    "content": f"Context: {enhanced_context}"
                })
            
            # Get LLM response via gateway using the context-resolved input
            # Enhance user input with goal context if available
            llm_input = user_input_processed
            if goal_res and goal_res.goal:
                # Add goal context to help LLM understand what user is trying to accomplish
                goal_note = f"\n[Context: You're helping the user accomplish: {goal_res.goal.description}. Keep this goal in mind when responding.]"
                llm_input = user_input_processed + goal_note
            
            try:
                llm_response = self.llm_gateway.request(
                    prompt=llm_input,
                    call_type=LLMCallType.GENERATION,
                    use_history=True,
                    user_input=user_input,
                    context={'intent': intent, 'entities': entities, 'goal': goal_res.goal if goal_res else None}
                )
                
                if llm_response.success and llm_response.response:
                    response = llm_response.response
                else:
                    # Gateway denied or error - provide fallback
                    response = llm_response.response or "I don't have enough training data to answer that yet. Keep interacting with me so I can learn!"
            
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                response = "I'm having trouble connecting to my language model. Please make sure Ollama is running and try again."
            
            # Optimize response for clarity and user preference
            if getattr(self, 'response_optimizer', None):
                response = self.response_optimizer.optimize(
                    response,
                    intent,
                    {
                        "entities": entities,
                        "goal": goal_res.goal.description if (goal_res and goal_res.goal) else None,
                    },
                )
            
            # NOTE: Don't cache LLM responses - we want fresh answers each time for variety
            # Cache is only used for learned conversational patterns (which rotate automatically)

            # Learning engine can optionally evaluate quality
            # (This section can be expanded later for quality checks)
            
            # Update the context handler (if still exists)
                # Update the turn with the response
                turn.assistant_response = response
                
                # Extract any entities mentioned in the response and track them
                self._track_response_entities(response, intent)
            
            # Remove context message and prune old history if too long
            if enhanced_context and self.llm.conversation_history:
                if self.llm.conversation_history[0]["role"] == "system":
                    self.llm.conversation_history.pop(0)
            
            # Smart history pruning - keep most relevant turns
            if len(self.llm.conversation_history) > self.llm.config.max_history:
                # Keep system prompt, last 5 turns, and recent relevant turns
                to_keep = []
                # Keep first (system prompt if exists)
                if self.llm.conversation_history and self.llm.conversation_history[0].get("role") == "system":
                    to_keep.append(self.llm.conversation_history[0])
                # Keep last 8 turns (most recent context)
                to_keep.extend(self.llm.conversation_history[-8:])
                self.llm.conversation_history = to_keep
            
            # 4. Store interaction
            self._store_interaction(user_input, response, intent, entities)
            
            # 4.5. Collect training data - learn from this interaction
            if getattr(self, 'training_collector', None):
                # Calculate quality score based on success, length, etc.
                quality_score = 1.0
                if plugin_result:
                    quality_score = 0.9 if plugin_result.get('success', False) else 0.6
                elif len(response) > 20:  # Substantial response
                    quality_score = 0.8
                
                # Collect for training
                if getattr(self, 'learning_engine', None):
                    self.learning_engine.collect_interaction(
                        user_input=user_input,
                        assistant_response=response,
                        intent=intent,
                        entities=entities or {},
                        quality_score=quality_score
                    )
                    self._think(f"Training data collected (quality: {quality_score:.2f})")
            
            # Store response for active learning
            self.last_assistant_response = response
            
            # Log action for pattern learning
            action = f"{intent}:{entities.get('topic', entities.get('query', 'general'))}"
            self._log_action_for_learning(action)
            
            # 5. Speak if voice enabled
            if use_voice and self.speech:
                self.speech.speak(response, blocking=False)
            
            logger.info(f"A.L.I.C.E: {response[:100]}...")
            return response
            
        except Exception as e:
            logger.error(f"[ERROR] Error processing input: {e}")
            error_response = "I apologize, but I encountered an error processing your request."
            
            if use_voice and self.speech:
                self.speech.speak(error_response, blocking=False)
            
            return error_response
    
    def _track_response_entities(self, response: str, intent: str):
        """Track entities mentioned in assistant responses"""
        if not self.advanced_context:
            return
            
        # Track files mentioned
        file_patterns = [
            r"created? (?:a )?file (?:called )?['\"]([^'\"]+)['\"]",
            r"(?:file|document) ['\"]([^'\"]+)['\"]",
            r"saved? (?:to|as) ['\"]([^'\"]+)['\"]"
        ]
        
        for pattern in file_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                filename = match.group(1)
                self.advanced_context.add_entity(
                    entity_type="file",
                    data={"name": filename, "mentioned_in_response": True},
                    aliases=["the file", "this file", filename]
                )
        
        # Track people mentioned
        person_patterns = [
            r"(?:from|by|to) ([A-Z][a-z]+ [A-Z][a-z]+)",  # Full names
            r"(?:from|by|to) ([A-Z][a-z]+)",  # First names
        ]
        
        for pattern in person_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                person_name = match.group(1)
                self.advanced_context.add_entity(
                    entity_type="person",
                    data={"name": person_name, "mentioned_in_response": True},
                    aliases=["this person", person_name]
                )
        
        # Track topics mentioned
        if intent in ["weather", "time", "calculation"]:
            topic_name = intent.replace("_", " ")
            self.advanced_context.add_entity(
                entity_type="topic",
                data={"name": topic_name, "intent": intent},
                aliases=["this topic", "that", topic_name]
            )
    
    def _detect_general_entities(self, user_input: str, intent: str, entities: Dict):
        """Detect and track general entities from user input"""
        if not self.advanced_context:
            return
            
        # Detect file mentions
        file_patterns = [
            r"(?:file|document) (?:called |named )?['\"]([^'\"]+)['\"]",
            r"create (?:a )?file ['\"]([^'\"]+)['\"]",
            r"(?:open|read|edit) ['\"]([^'\"]+)['\"]"
        ]
        
        for pattern in file_patterns:
            matches = re.finditer(pattern, user_input, re.IGNORECASE)
            for match in matches:
                filename = match.group(1)
                self.advanced_context.add_entity(
                    entity_type="file",
                    data={"name": filename, "mentioned_by_user": True},
                    aliases=["the file", "this file", "that file", filename]
                )
        
        # Detect person mentions
        person_patterns = [
            r"(?:tell|ask|message|email|call) ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            r"(?:with|from|to) ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)"
        ]
        
        for pattern in person_patterns:
            matches = re.finditer(pattern, user_input, re.IGNORECASE)
            for match in matches:
                person_name = match.group(1)
                if person_name.lower() not in ["alice", "you", "me", "i"]:  # Skip self-references
                    self.advanced_context.add_entity(
                        entity_type="person",
                        data={"name": person_name, "mentioned_by_user": True},
                        aliases=["this person", "they", "them", person_name]
                    )
        
        # Detect location mentions
        location_patterns = [
            r"(?:in|at|to|from) ([A-Z][a-z]+(?:,?\s+[A-Z][a-z]+)*)",
            r"weather (?:in|at|for) ([A-Z][a-z]+(?:,?\s+[A-Z][a-z]+)*)"
        ]
        
        for pattern in location_patterns:
            matches = re.finditer(pattern, user_input, re.IGNORECASE)
            for match in matches:
                location = match.group(1)
                # Simple validation - avoid common words
                if location.lower() not in ["the", "this", "that", "there", "here", "now", "today"]:
                    self.advanced_context.add_entity(
                        entity_type="location",
                        data={"name": location, "mentioned_by_user": True},
                        aliases=["there", "this place", "that location", location]
                    )
        
        # Detect task/topic mentions based on intent
        if intent in ["file_operation", "system_control", "weather", "time"]:
            topic_data = {
                "intent": intent,
                "user_input": user_input[:100],  # Store snippet
                "mentioned_by_user": True
            }
            
            self.advanced_context.add_entity(
                entity_type="topic",
                data=topic_data,
                aliases=["this", "that", intent.replace("_", " ")]
            )
    
    def _get_recent_conversation_summary(self) -> str:
        """Get a summary of the last few conversation exchanges"""
        if not self.conversation_summary:
            return ""
        
        # Get last 3 exchanges
        recent = self.conversation_summary[-3:]
        summary_parts = []
        
        for exchange in recent:
            # Create concise summary
            user_text = exchange['user'][:50] + "..." if len(exchange['user']) > 50 else exchange['user']
            summary_parts.append(f"User asked: {user_text}")
        
        return " | ".join(summary_parts)
    
    def _get_active_context(self) -> str:
        """Get currently active context (email lists, pending actions, etc.)"""
        context_parts = []
        
        # Track email context
        if self.last_email_list:
            num_emails = sum(1 for e in self.last_email_list if e is not None)
            if num_emails > 0:
                context_parts.append(f"User is viewing {num_emails} emails from their inbox")
        
        # Track pending actions
        if self.pending_action:
            context_parts.append(f"In progress: {self.pending_action.replace('_', ' ')}")
        
        # Track referenced items
        if self.referenced_items:
            refs = ", ".join(f"{k}: {v}" for k, v in list(self.referenced_items.items())[-3:])
            context_parts.append(f"Recently referenced: {refs}")
        
        # Track topics
        if self.conversation_topics:
            topics = ", ".join(self.conversation_topics[-3:])
            context_parts.append(f"Current topics: {topics}")
        
        return " | ".join(context_parts) if context_parts else ""
    
    def _load_conversation_state(self):
        """Load conversation state from previous session"""
        import pickle
        import os
        
        state_file = "data/conversation_state.pkl"
        
        if os.path.exists(state_file):
            try:
                with open(state_file, 'rb') as f:
                    state = pickle.load(f)
                    
                    # Restore conversation summary (only recent ones)
                    if 'conversation_summary' in state:
                        self.conversation_summary = state['conversation_summary'][-5:]  # Last 5 only
                    
                    # Restore topics
                    if 'conversation_topics' in state:
                        self.conversation_topics = state['conversation_topics'][-5:]
                    
                    # Restore referenced items
                    if 'referenced_items' in state:
                        self.referenced_items = state['referenced_items']
                    
                    logger.info("[OK] Previous conversation context restored")
            except Exception as e:
                logger.warning(f"[WARNING] Could not load conversation state: {e}")
        
        # Load advanced context state if available
        if self.advanced_context:
            advanced_state_file = "data/advanced_context_state.pkl"
            if os.path.exists(advanced_state_file):
                try:
                    self.advanced_context.load_state(advanced_state_file)
                    logger.info("[OK] Advanced context state restored")
                except Exception as e:
                    logger.warning(f"[WARNING] Could not load advanced context state: {e}")
    
    def _save_conversation_state(self):
        """Save conversation state for next session"""
        import pickle
        import os
        
        os.makedirs("data", exist_ok=True)
        state_file = "data/conversation_state.pkl"
        
        try:
            state = {
                'conversation_summary': self.conversation_summary[-10:],  # Keep last 10
                'conversation_topics': self.conversation_topics[-10:],
                'referenced_items': self.referenced_items,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(state_file, 'wb') as f:
                pickle.dump(state, f)
            
            # Save context state if available
            if self.context:
                context_state_file = "data/context_state.pkl"
                self.context.save_state(context_state_file)
            
            logger.info("[OK] Conversation state saved")
        except Exception as e:
            logger.warning(f"[WARNING] Could not save conversation state: {e}")
    
    def _store_interaction(
        self,
        user_input: str,
        response: str,
        intent: str,
        entities: Dict
    ):
        """Store interaction in memory and context"""
        try:
            # Process with unified context engine
            if self.context:
                turn = self.context.process_turn(
                    user_input=user_input,
                    assistant_response=response,
                    intent=intent,
                    entities=entities
                )
            
            # Add to conversation summarizer for intelligent context management
            if self.summarizer:
                # Extract entity list from entities dict
                entity_list = []
                if entities:
                    for entity_type, entity_values in entities.items():
                        if isinstance(entity_values, list):
                            entity_list.extend(entity_values)
                        elif entity_values:
                            entity_list.append(str(entity_values))
                
                # Get sentiment from NLP result if available
                sentiment = getattr(self, '_last_sentiment', None)
                
                self.summarizer.add_turn(
                    user_input=user_input,
                    assistant_response=response,
                    intent=intent,
                    entities=entity_list,
                    sentiment=sentiment
                )
            
            # Extract and store entity relationships
            if self.relationship_tracker:
                try:
                    # Extract relationships from user input
                    relationships = self.relationship_tracker.process_text(user_input)
                    logger.debug(f"Extracted {len(relationships)} relationships from user input")
                    
                    # Also process assistant response for relationship context
                    if response and len(response) < 500:  # Only process shorter responses
                        assistant_relationships = self.relationship_tracker.process_text(response)
                        logger.debug(f"Extracted {len(assistant_relationships)} relationships from assistant response")
                except Exception as e:
                    logger.error(f"Error extracting relationships: {e}")
            
            # Add to conversation summary (keep last 10)
            self.conversation_summary.append({
                'user': user_input,
                'assistant': response[:200],  # Truncate for summary
                'intent': intent,
                'timestamp': datetime.now().isoformat()
            })
            if len(self.conversation_summary) > 10:
                self.conversation_summary.pop(0)
            
            # Track topics from intent and entities
            if intent and intent != "unknown":
                if intent not in self.conversation_topics:
                    self.conversation_topics.append(intent)
            
            # Track entities as references
            if entities:
                for entity_type, entity_values in entities.items():
                    if entity_values:
                        # Store most recent reference
                        self.referenced_items[entity_type] = entity_values[0] if isinstance(entity_values, list) else entity_values
            
            # Limit tracking size
            if len(self.conversation_topics) > 10:
                self.conversation_topics = self.conversation_topics[-10:]
            if len(self.referenced_items) > 15:
                # Keep most recent 15
                keys = list(self.referenced_items.keys())[-15:]
                self.referenced_items = {k: self.referenced_items[k] for k in keys}
            
            # Update context
            entity_list = []
            for entity_type, values in entities.items():
                entity_list.extend(values)
            
            self.context.update_conversation(user_input, response, intent, entity_list)
            
            # Store in episodic memory with enhanced metadata (unless privacy mode)
            if not self.privacy_mode:
                self.memory.store_memory(
                    content=f"User: {user_input} | Assistant: {response}",
                    memory_type="episodic",
                    context={
                        "intent": intent,
                        "entities": entities,
                        "topics": self.conversation_topics[-3:],
                        "has_email_context": bool(self.last_email_list),
                        "pending_action": self.pending_action
                    },
                    importance=0.6,
                    tags=["conversation", intent]
                )
                
                # Check if periodic consolidation is needed
                self.memory.periodic_consolidation_check()
            else:
                logger.info("[PRIVACY] Episodic memory storage skipped (privacy mode enabled)")
            
        except Exception as e:
            logger.warning(f"[WARNING] Error storing interaction: {e}")
    
    def run_interactive(self, skip_welcome=False):
        """Run interactive console mode
        
        Args:
            skip_welcome: If True, skip welcome banner and greeting (for alice.py)
        """
        self.running = True
        
        if not skip_welcome:
            print("\n" + "=" * 80)
            print("A.L.I.C.E - Your Personal AI Assistant")
            print("=" * 80)
            print(f"\nHello {self.context.user_prefs.name}! I'm ALICE, your advanced AI assistant.")
            print("I'm here to help you with anything you need.\n")
            print("Commands:")
            print("   /help      - Show available commands")
            print("   /voice     - Toggle voice mode")
            print("   /clear     - Clear conversation history")
            print("   /memory    - Show memory statistics")
            print("   /plugins   - List available plugins")
            print("   /location  - Set or view your location")
            print("   /status    - Show system status")
            print("   /save      - Save current state")
            print("   /summary   - Get conversation summary")
            print("   /context   - Show current context")
            print("   /topics    - List conversation topics")
            print("   /entities  - Show tracked entities")
            print("   /relationships - Show entity relationships")
            print()
            print("Debug Commands:")
            print("   /correct   - Correct my last response")
            
            # Proactive morning briefing
            if getattr(self, 'proactive_assistant', None):
                briefing = self.proactive_assistant.get_morning_briefing()
                if briefing:
                    print("\n" + briefing)
            print("   /feedback  - Rate my last response")
            print("   /learning  - Show learning statistics")
            print("   exit       - End conversation")
            print("=" * 80)
            
            # Greet user
            greeting = self._get_greeting()
            print(f"\nA.L.I.C.E: {greeting}\n")
            
            if self.speech and self.voice_enabled:
                self.speech.speak(greeting, blocking=False)
        
        while self.running:
            try:
                # Get input
                user_input = input(f"\n{self.context.user_prefs.name}: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    continue
                
                # Handle exit
                if user_input.lower() in ['exit', 'quit', 'goodbye', 'bye']:
                    farewell = self._get_farewell()
                    print(f"\nA.L.I.C.E: {farewell}\n")
                    
                    if self.speech and self.voice_enabled:
                        self.speech.speak(farewell, blocking=True)
                    
                    self.shutdown()
                    break
                
                # Process input
                response = self.process_input(user_input, use_voice=self.voice_enabled)
                print(f"\nA.L.I.C.E: {response}")
                
            except KeyboardInterrupt:
                farewell = self._get_farewell()
                print(f"\n\nA.L.I.C.E: {farewell}")
                self.shutdown()
                break
            except Exception as e:
                logger.error(f"[ERROR] Error in interactive loop: {e}")
                print(f"\n[ERROR] Error: {e}")
    
    def run_voice_mode(self):
        """Run voice-activated mode"""
        if not self.speech:
            logger.error("[ERROR] Voice mode not available - speech engine not initialized")
            return
        
        self.running = True
        
        print("\n" + "=" * 80)
        print("Voice Mode Activated")
        print("=" * 80)
        print(f"\nListening for wake words: {', '.join(self.speech.config.wake_words)}")
        print("Say the wake word followed by your command.")
        print("Press Ctrl+C to exit voice mode.\n")
        
        def handle_voice_command(command: str):
            """Handle voice command"""
            if command.lower() in ['exit', 'quit', 'goodbye']:
                print("\nExiting voice mode...")
                self.speech.speak("Exiting voice mode. Goodbye!")
                self.speech.stop_listening()
                self.running = False
                return
            
            print(f"\nYou said: {command}")
            response = self.process_input(command, use_voice=True)
            print(f"A.L.I.C.E: {response}\n")
        
        # Start listening for wake word
        try:
            self.speech.listen_for_wake_word(handle_voice_command, background=False)
        except KeyboardInterrupt:
            print("\n\nExiting voice mode...")
            self.speech.stop_listening()
    
    def _handle_command(self, command: str):
        """Handle system commands"""
        cmd = command.lower().strip()
        
        if cmd == '/help':
            print("\nAvailable Commands:")
            print("   /help              - Show this help message")
            print("   /voice             - Toggle voice mode on/off")
            print("   /clear             - Clear conversation history")
            print("   /memory            - Show memory system statistics")
            print("   /plugins           - List all available plugins")
            print("   /status            - Show system status, gateway stats, and LLM usage")
            print("   /location [City]   - Set or view your location")
            print("   /save              - Save current state manually")
            print("   /summary           - Get conversation summary")
            print("   /context           - Show current context")
            print("   /topics            - List conversation topics")
            print("   /entities          - Show tracked entities")
            print("   /relationships     - Show entity relationships")
            print()
            print("Memory Management:")
            print("   /mem-list [type]   - List memories (types: episodic, semantic, procedural, document)")
            print("   /mem-search <query>- Search memories by semantic similarity")
            print("   /mem-delete <id>   - Delete a specific memory by ID")
            print("   /patterns          - Show proposed patterns awaiting approval")
            print()
            print("Debug Commands:")
            print("   /correct [type]    - Correct A.L.I.C.E's last response")
            print("   /feedback [rating] - Rate A.L.I.C.E's last response (1-5)")
            print("   /learning          - Show active learning statistics")
            print("   exit               - End conversation and exit")
        
        elif cmd == '/voice':
            if self.speech:
                self.voice_enabled = not self.voice_enabled
                status = "enabled" if self.voice_enabled else "disabled"
                print(f"\n[OK] Voice mode {status}")
            else:
                print("\n[ERROR] Voice engine not available")
        
        elif cmd == '/clear':
            self.llm.clear_history()
            self.context.clear_short_term_memory()
            print("\n[OK] Conversation history cleared")
        
        elif cmd == '/memory':
            stats = self.memory.get_statistics()
            print("\n Memory Statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        
        elif cmd == '/plugins':
            plugins = self.plugins.get_all_plugins()
            print("\nAvailable Plugins:")
            for plugin in plugins:
                status = "[ON]" if plugin['enabled'] else "[PAUSED]"
                print(f"   {status} {plugin['name']}: {plugin['description']}")
        
        elif cmd == '/status':
            print("\n🔧 System Status:")
            status = self.context.get_system_status()
            print(f"   LLM Model: {status.get('llm_model', 'N/A')}")
            print(f"   Voice: {'Enabled' if status.get('voice_enabled') else 'Disabled'}")
            print(f"   Privacy Mode: {'Enabled' if self.privacy_mode else 'Disabled'}")
            print(f"   Plugins: {len(self.plugins.plugins)}")
            print(f"   Capabilities: {len(status.get('capabilities', []))}")
            
            memory_stats = self.memory.get_statistics()
            print(f"   Total Memories: {memory_stats['total_memories']}")
            if self.privacy_mode:
                print(f"   ⚠️  Episodic memory storage is DISABLED")
            
            # LLM Gateway Statistics
            if hasattr(self, 'llm_gateway') and self.llm_gateway:
                print("\n📊 LLM Gateway Statistics:")
                gateway_stats = self.llm_gateway.get_statistics()
                print(f"   Total Requests: {gateway_stats['total_requests']}")
                
                # Advanced breakdown
                print(f"   ✓ Self Handlers: {gateway_stats['self_handlers']} ({gateway_stats.get('self_handler_percentage', 0)}%)")
                print(f"   💡 Pattern Hits: {gateway_stats['pattern_hits']} ({gateway_stats.get('pattern_hit_percentage', 0)}%)")
                print(f"   🔧 Tool Calls: {gateway_stats['tool_calls']} ({gateway_stats.get('tool_call_percentage', 0)}%)")
                print(f"   🧠 RAG Lookups: {gateway_stats['rag_lookups']} ({gateway_stats.get('rag_lookup_percentage', 0)}%)")
                print(f"   📝 Formatter Usage: {gateway_stats['formatter_calls']} ({gateway_stats.get('formatter_percentage', 0)}%)")
                print(f"   🤖 LLM Fallback: {gateway_stats['llm_calls']} ({gateway_stats.get('llm_fallback_percentage', 0)}%)")
                print(f"   ✗ Policy Denials: {gateway_stats['policy_denials']} ({gateway_stats.get('denial_percentage', 0)}%)")
                
                if gateway_stats['by_type']:
                    print("\n   By Call Type:")
                    for call_type, count in sorted(gateway_stats['by_type'].items(), key=lambda x: x[1], reverse=True):
                        print(f"      {call_type}: {count}")
            
            # Conversational Engine Statistics
            if hasattr(self, 'conversational_engine') and self.conversational_engine:
                print("\n💬 Conversational Engine:")
                if hasattr(self.conversational_engine, 'pattern_count'):
                    print(f"   Learned Patterns: {self.conversational_engine.pattern_count}")
                if hasattr(self.conversational_engine, 'learned_greetings'):
                    print(f"   Learned Greetings: {len(self.conversational_engine.learned_greetings) if self.conversational_engine.learned_greetings else 0}")
                print(f"   Status: Active")
        
        elif cmd.startswith('/location'):
            # Set location manually: /location City, Country
            parts = command.split(maxsplit=1)
            if len(parts) > 1:
                location_parts = parts[1].split(',')
                city = location_parts[0].strip()
                country = location_parts[1].strip() if len(location_parts) > 1 else None
                self.context.user_prefs.set_location(city, country)
                self.context.save_context()
                print(f"\n[OK] Location set to: {self.context.user_prefs.location}")
            else:
                current = self.context.user_prefs.location or "Not set"
                print(f"\nCurrent location: {current}")
                print("Usage: /location City, Country")
        
        elif cmd == '/save':
            self._save_conversation_state()
            self.context.save_context()
            self.memory._save_memories()
            print("\n[OK] Conversation state saved successfully")
        
        elif cmd == '/summary':
            if self.summarizer:
                try:
                    summary_text = self.summarizer.get_conversation_summary()
                    print("\n Conversation Summary:")
                    print("=" * 50)
                    print(summary_text)
                    print("=" * 50)
                except Exception as e:
                    print(f"\n[ERROR] Failed to get summary: {e}")
            else:
                print("\n[ERROR] Conversation summarizer not available")
        
        elif cmd == '/context':
            if self.summarizer:
                try:
                    context_summary = self.summarizer.get_context_summary()
                    print("\n Current Context:")
                    print("=" * 50)
                    print(context_summary)
                    print("=" * 50)
                except Exception as e:
                    print(f"\n[ERROR] Failed to get context: {e}")
            else:
                print("\n[ERROR] Conversation summarizer not available")
        
        elif cmd == '/topics':
            if self.summarizer:
                try:
                    context = self.summarizer.get_detailed_context()
                    topics = context.get('frequent_topics', [])
                    print("\n🏷️ Conversation Topics:")
                    print("=" * 50)
                    if topics:
                        for i, topic in enumerate(topics, 1):
                            print(f"   {i}. {topic.title()}")
                    else:
                        print("   No topics identified yet.")
                    print("=" * 50)
                except Exception as e:
                    print(f"\n[ERROR] Failed to get topics: {e}")
            else:
                print("\n[ERROR] Conversation summarizer not available")
        
        elif cmd == '/entities':
            if self.relationship_tracker:
                try:
                    stats = self.relationship_tracker.get_statistics()
                    print("\n👥 Tracked Entities:")
                    print("=" * 50)
                    print(f"Total entities: {stats['total_entities']}")
                    
                    if stats['most_connected_entities']:
                        print("\nMost connected entities:")
                        for entity, count in stats['most_connected_entities']:
                            print(f"   • {entity.title()}: {count} connections")
                    
                    if stats['entity_types']:
                        print(f"\nEntity types: {', '.join(stats['entity_types'].keys())}")
                    
                    print("=" * 50)
                except Exception as e:
                    print(f"\n[ERROR] Failed to get entities: {e}")
            else:
                print("\n[ERROR] Entity relationship tracker not available")
        
        elif cmd == '/relationships':
            if self.relationship_tracker:
                try:
                    stats = self.relationship_tracker.get_statistics()
                    print("\n🔗 Entity Relationships:")
                    print("=" * 50)
                    print(f"Total relationships: {stats['total_relationships']}")
                    
                    if stats['relationship_types']:
                        print("\nRelationship types:")
                        for rel_type, count in stats['relationship_types'].items():
                            print(f"   • {rel_type.replace('_', ' ').title()}: {count}")
                    
                    if stats['recent_relationships']:
                        print(f"\nRecent relationships:")
                        for rel in stats['recent_relationships'][:5]:
                            source = rel['source_entity'].title()
                            target = rel['target_entity'].title()
                            rel_type = rel['relationship_type'].replace('_', ' ')
                            confidence = rel['confidence']
                            print(f"   • {source} {rel_type} {target} (confidence: {confidence:.2f})")
                    
                    print("=" * 50)
                except Exception as e:
                    print(f"\n[ERROR] Failed to get relationships: {e}")
            else:
                print("\n[ERROR] Entity relationship tracker not available")
        
        elif cmd.startswith('/mem-list'):
            # List memories with optional filter by type
            parts = command.split(maxsplit=1)
            memory_type = parts[1].strip() if len(parts) > 1 else None
            
            if memory_type and memory_type not in ['episodic', 'semantic', 'procedural', 'document']:
                print(f"\n[ERROR] Invalid memory type: {memory_type}")
                print("   Valid types: episodic, semantic, procedural, document")
                return
            
            memories = []
            if not memory_type or memory_type == 'episodic':
                memories.extend([('episodic', m) for m in self.memory.episodic_memory])
            if not memory_type or memory_type == 'semantic':
                memories.extend([('semantic', m) for m in self.memory.semantic_memory])
            if not memory_type or memory_type == 'procedural':
                memories.extend([('procedural', m) for m in self.memory.procedural_memory])
            if not memory_type or memory_type == 'document':
                memories.extend([('document', m) for m in self.memory.document_memory])
            
            # Sort by timestamp (most recent first)
            memories.sort(key=lambda x: x[1].timestamp, reverse=True)
            
            title = f"{'All' if not memory_type else memory_type.title()} Memories"
            print(f"\n🧠 {title}:")
            print("=" * 70)
            
            if not memories:
                print("   No memories found.")
            else:
                for i, (mem_type, mem) in enumerate(memories[:20], 1):  # Show max 20
                    content_preview = mem.content[:60] + "..." if len(mem.content) > 60 else mem.content
                    importance = "★" * int(mem.importance * 5)  # Convert to 0-5 stars
                    print(f"   {i}. [{mem_type[:3].upper()}] {content_preview}")
                    print(f"      ID: {mem.id} | Importance: {importance} | Access: {mem.access_count}x")
                    print(f"      Tags: {', '.join(mem.tags) if mem.tags else 'none'}")
                    print()
                
                if len(memories) > 20:
                    print(f"   ... and {len(memories) - 20} more memories")
            
            print("=" * 70)
        
        elif cmd.startswith('/mem-search'):
            # Search memories by content
            parts = command.split(maxsplit=1)
            if len(parts) < 2:
                print("\n[ERROR] Usage: /mem-search <query>")
                return
            
            query = parts[1].strip()
            results = self.memory.recall_memory(query, top_k=10, min_similarity=0.5)
            
            print(f"\n🔍 Memory Search Results for: '{query}'")
            print("=" * 70)
            
            if not results:
                print("   No matching memories found.")
            else:
                for i, result in enumerate(results, 1):
                    content_preview = result['content'][:60] + "..." if len(result['content']) > 60 else result['content']
                    similarity = result['similarity']
                    importance = "★" * int(result['importance'] * 5)
                    
                    print(f"   {i}. {content_preview}")
                    print(f"      ID: {result['id']} | Type: {result['type']}")
                    print(f"      Similarity: {similarity:.2f} | Importance: {importance}")
                    print(f"      Tags: {', '.join(result['tags']) if result['tags'] else 'none'}")
                    print()
            
            print("=" * 70)
        
        elif cmd.startswith('/mem-delete'):
            # Delete memory by ID
            parts = command.split(maxsplit=1)
            if len(parts) < 2:
                print("\n[ERROR] Usage: /mem-delete <memory_id>")
                return
            
            memory_id = parts[1].strip()
            
            # Find and delete the memory
            deleted = False
            for memory_list in [self.memory.episodic_memory, self.memory.semantic_memory,
                              self.memory.procedural_memory, self.memory.document_memory]:
                for i, mem in enumerate(memory_list):
                    if mem.id == memory_id:
                        memory_type = mem.type
                        content_preview = mem.content[:60] + "..." if len(mem.content) > 60 else mem.content
                        
                        # Remove from memory list
                        memory_list.pop(i)
                        
                        # Remove from vector store
                        self.memory.vector_store.delete(memory_id)
                        
                        # Save changes
                        self.memory._save_memories()
                        self.memory.vector_store.save(self.memory.vector_store_path)
                        
                        deleted = True
                        print(f"\n✓ Deleted {memory_type} memory:")
                        print(f"   ID: {memory_id}")
                        print(f"   Content: {content_preview}")
                        break
                
                if deleted:
                    break
            
            if not deleted:
                print(f"\n[ERROR] Memory not found: {memory_id}")
        
        elif cmd == '/patterns':
            # Show proposed patterns awaiting approval
            if not hasattr(self, 'pattern_miner'):
                try:
                    from ai.pattern_miner import PatternMiner
                    self.pattern_miner = PatternMiner()
                except Exception as e:
                    print(f"\n[ERROR] Pattern miner not available: {e}")
                    return
            
            stats = self.pattern_miner.get_pattern_stats()
            print(f"\n\ud83d\udccb Pattern Learning System:")
            print("=" * 70)
            print(f"   Total Proposals: {stats['total_proposals']}")
            print(f"   Pending Approval: {stats['pending_approval']}")
            print(f"   Approved: {stats['approved']}")
            print(f"   Rejected: {stats['rejected']}")
            if stats['total_proposals'] > 0:
                approval_rate = stats['approval_rate']
                print(f"   Approval Rate: {approval_rate:.1%}")
            
            print()
            
            # Show pending patterns
            if stats['pending_patterns']:
                print("Pending Patterns Awaiting Your Approval:")
                print("-" * 70)
                for i, pattern in enumerate(stats['pending_patterns'][:5], 1):
                    print(f"\n   {i}. [{pattern['intent']}] Pattern {pattern['id']}")
                    print(f"      Examples: {pattern['example_inputs'][:2]}")
                    print(f"      Template: {pattern['proposed_template'][:80]}...")
                    print(f"      Cluster Size: {pattern['cluster_size']} similar interactions")
                    print(f"      Confidence: {pattern['confidence']:.1%}")
                    print(f"      Approve: /patterns approve {pattern['id']}")
                    print(f"      Reject:  /patterns reject {pattern['id']}")
                
                if len(stats['pending_patterns']) > 5:
                    print(f"\n   ... and {len(stats['pending_patterns']) - 5} more pending patterns")
            else:
                print("   \u2713 No pending patterns. All proposed patterns have been reviewed.")
            
            print("\n" + "=" * 70)
        
        elif cmd.startswith('/patterns approve'):
            # Approve a specific pattern
            parts = command.split(maxsplit=2)
            if len(parts) < 3:
                print("\n[ERROR] Usage: /patterns approve <pattern_id>")
                return
            
            pattern_id = parts[2].strip()
            
            if not hasattr(self, 'pattern_miner'):
                from ai.pattern_miner import PatternMiner
                self.pattern_miner = PatternMiner()
            
            if self.pattern_miner.approve_pattern(pattern_id):
                print(f"\n\u2713 Pattern {pattern_id} approved and will be used for future interactions")
            else:
                print(f"\n[ERROR] Pattern {pattern_id} not found")
        
        elif cmd.startswith('/patterns reject'):
            # Reject a specific pattern
            parts = command.split(maxsplit=2)
            if len(parts) < 3:
                print("\n[ERROR] Usage: /patterns reject <pattern_id>")
                return
            
            pattern_id = parts[2].strip()
            
            if not hasattr(self, 'pattern_miner'):
                from ai.pattern_miner import PatternMiner
                self.pattern_miner = PatternMiner()
            
            if self.pattern_miner.reject_pattern(pattern_id):
                print(f"\n\u2713 Pattern {pattern_id} rejected")
            else:
                print(f"\n[ERROR] Pattern {pattern_id} not found")
        
        elif cmd.startswith('/correct'):
            self._handle_correction_command(cmd)
        
        elif cmd.startswith('/feedback'):
            self._handle_feedback_command(cmd)
        
        elif cmd == '/learning':
            self._handle_learning_stats_command()
        
        else:
            print(f"\n[ERROR] Unknown command: {command}")
            print("   Type /help for available commands")
    
    def _get_greeting(self) -> str:
        """Generate context-aware greeting using ALICE's conversational abilities"""
        name = self.context.user_prefs.name
        hour = datetime.now().hour
        
        # Determine time of day context
        if 5 <= hour < 12:
            time_context = "morning"
        elif 12 <= hour < 17:
            time_context = "afternoon"
        elif 17 <= hour < 22:
            time_context = "evening"
        else:
            time_context = "late night"
        
        # Try conversational engine first (learned greetings)
        if hasattr(self, 'conversational_engine') and self.conversational_engine:
            # Use learned greeting patterns if available
            if hasattr(self.conversational_engine, 'learned_greetings') and self.conversational_engine.learned_greetings:
                if hasattr(self.conversational_engine, '_pick_non_repeating'):
                    return self.conversational_engine._pick_non_repeating(self.conversational_engine.learned_greetings)
        
        # Fallback to Gateway for natural greeting generation
        prompt = f"""Generate a brief, natural greeting for the user who just started the session.
Context:
- User's name: {name}
- Time of day: {time_context}
- This is the opening greeting

Generate only the greeting (1 sentence), no other text. Be friendly and offer to help."""
        
        try:
            llm_response = self.llm_gateway.request(
                prompt=prompt,
                call_type=LLMCallType.CHITCHAT,
                use_history=False,
                user_input="greeting"
            )
            if llm_response.success and llm_response.response:
                greeting = llm_response.response.strip().strip('"').strip("'")
                return greeting
            else:
                # Policy denied - use simple greeting
                return f"Hey {name}, how can I help?"
        except Exception as e:
            # Ultimate fallback
            return f"Hey {name}, how can I help?"
    
    def _get_farewell(self) -> str:
        """Generate context-aware farewell using ALICE's conversational abilities"""
        name = self.context.user_prefs.name
        hour = datetime.now().hour
        
        # Determine time of day context
        if 5 <= hour < 12:
            time_context = "morning/day"
        elif 12 <= hour < 17:
            time_context = "afternoon"
        elif 17 <= hour < 22:
            time_context = "evening"
        else:
            time_context = "late night"
        
        # Try conversational engine first (learned farewells)
        if hasattr(self, 'conversational_engine') and self.conversational_engine:
            # Check if there are learned farewell patterns
            try:
                if hasattr(self, 'learning_engine') and self.learning_engine:
                    examples = self.learning_engine.get_high_quality_examples()
                    farewell_responses = []
                    for ex in examples:
                        user_text = ex.get('user_input', '') if isinstance(ex, dict) else getattr(ex, 'user_input', '')
                        response_text = ex.get('assistant_response', '') if isinstance(ex, dict) else getattr(ex, 'assistant_response', '')
                        if any(word in user_text.lower() for word in ['bye', 'goodbye', 'exit', 'quit']):
                            if response_text and len(response_text) < 80:
                                farewell_responses.append(response_text)
                    
                    if farewell_responses and hasattr(self.conversational_engine, '_pick_non_repeating'):
                        return self.conversational_engine._pick_non_repeating(farewell_responses)
            except:
                pass
        
        # Fallback to Gateway for natural farewell generation
        prompt = f"""Generate a brief, natural farewell for the user.
Context:
- User's name: {name}
- Time: {time_context}
- The user is ending the session

Generate only the farewell (1 sentence), no other text. Be warm and friendly."""
        
        try:
            llm_response = self.llm_gateway.request(
                prompt=prompt,
                call_type=LLMCallType.CHITCHAT,
                use_history=False,
                user_input="farewell"
            )
            if llm_response.success and llm_response.response:
                farewell = llm_response.response.strip().strip('"').strip("'")
                return farewell
            else:
                # Policy denied - use simple farewell
                return f"Take care, {name}!"
        except Exception as e:
            # Ultimate fallback
            return f"Take care, {name}!"
    
    def _handle_relationship_query(self, user_input: str, intent: str, entities: Dict[str, Any]) -> Optional[str]:
        """Handle relationship queries like 'who does John work for?' or 'tell me about Sarah'"""
        query_lower = user_input.lower()
        
        # Patterns for relationship queries
        relationship_query_patterns = [
            r'who does (\w+) work (?:for|with)',
            r'where does (\w+) live',
            r'who is (\w+)(?:\s+to\s+(\w+))?',
            r'tell me about (\w+)',
            r'what do you know about (\w+)',
            r"(\w+)(?:'s)?\s+(?:relationship|connection)\s+(?:with|to)\s+(\w+)",
            r'how are (\w+) and (\w+) (?:related|connected)',
            r'(\w+) and (\w+) relationship',
        ]
        
        for pattern in relationship_query_patterns:
            match = re.search(pattern, query_lower)
            if match:
                entity_name = match.group(1)
                second_entity = match.group(2) if len(match.groups()) > 1 and match.group(2) else None
                
                # Get relationships for the entity
                relationships = self.relationship_tracker.get_entity_relationships(entity_name)
                
                if not relationships:
                    return f"I don't have any information about relationships involving {entity_name.title()}."
                
                # Format relationships
                response_parts = [f"Here's what I know about {entity_name.title()}:"]
                
                # Group relationships by type
                by_type = defaultdict(list)
                for rel in relationships:
                    if rel.source_entity == entity_name.lower():
                        by_type[rel.relationship_type].append(f"{rel.relationship_type.replace('_', ' ')} {rel.target_entity.title()}")
                    else:
                        by_type[rel.relationship_type].append(f"is {rel.relationship_type.replace('_', ' ')} by {rel.source_entity.title()}")
                
                for rel_type, connections in by_type.items():
                    if len(connections) == 1:
                        response_parts.append(f"• {connections[0]}")
                    else:
                        response_parts.append(f"• {rel_type.replace('_', ' ')}: {', '.join([c.replace(rel_type.replace('_', ' '), '').strip() for c in connections])}")
                
                # If asking about specific relationship between two entities
                if second_entity:
                    specific_rels = [
                        rel for rel in relationships
                        if (rel.source_entity == second_entity.lower() or rel.target_entity == second_entity.lower())
                    ]
                    if specific_rels:
                        response_parts.append(f"\nConnection with {second_entity.title()}:")
                        for rel in specific_rels:
                            if rel.source_entity == entity_name.lower():
                                response_parts.append(f"• {entity_name.title()} {rel.relationship_type.replace('_', ' ')} {rel.target_entity.title()}")
                            else:
                                response_parts.append(f"• {rel.source_entity.title()} {rel.relationship_type.replace('_', ' ')} {entity_name.title()}")
                
                return "\n".join(response_parts)
        
        # Check if user is asking for general relationship information
        if any(word in query_lower for word in ['relationships', 'connections', 'network']):
            stats = self.relationship_tracker.get_statistics()
            if stats['total_relationships'] == 0:
                return "I haven't tracked any entity relationships yet. Have a conversation mentioning people, places, or things and I'll start building a relationship map!"
            
            response_parts = [
                f"I've tracked {stats['total_relationships']} relationships between {stats['total_entities']} entities.",
            ]
            
            if stats['most_connected_entities']:
                response_parts.append("\nMost connected entities:")
                for entity, count in stats['most_connected_entities'][:3]:
                    response_parts.append(f"• {entity.title()}: {count} connections")
            
            return "\n".join(response_parts)
        
        return None
    
    def _handle_correction_command(self, command: str):
        """Handle correction commands"""
        if not self.last_user_input or not self.last_assistant_response:
            print("\n[ERROR] No previous interaction to correct")
            return
        
        parts = command.split(" ", 1)
        correction_type = parts[1] if len(parts) > 1 else ""
        
        print(f"\n Correction Mode")
        print("=" * 50)
        print(f"Last input: {self.last_user_input}")
        print(f"Last response: {self.last_assistant_response[:200]}...")
        print()
        
        if not correction_type:
            print("Available correction types:")
            print("   intent     - Correct intent classification")
            print("   entity     - Correct entity extraction")
            print("   response   - Correct response quality")
            print("   sentiment  - Correct sentiment analysis")
            print("   factual    - Correct factual error")
            print()
            correction_type = input("Correction type: ").strip().lower()
        
        if correction_type == "intent":
            print(f"Current intent: {self.last_intent}")
            new_intent = input("Correct intent: ").strip()
            if new_intent:
                self.learning_manager.record_correction(
                    CorrectionType.INTENT_CLASSIFICATION,
                    self.last_user_input,
                    self.last_intent,
                    new_intent,
                    f"User corrected intent from '{self.last_intent}' to '{new_intent}'",
                    {"original_nlp_result": self.last_nlp_result}
                )
                print(f" Recorded intent correction: {self.last_intent} → {new_intent}")
        
        elif correction_type == "entity":
            print(f"Current entities: {self.last_entities}")
            print("Enter correct entities (JSON format):")
            try:
                new_entities_input = input("Correct entities: ").strip()
                if new_entities_input:
                    new_entities = eval(new_entities_input)  # Simple eval for demo
                    self.learning_manager.record_correction(
                        CorrectionType.ENTITY_EXTRACTION,
                        self.last_user_input,
                        self.last_entities,
                        new_entities,
                        "User corrected entity extraction",
                        {"original_nlp_result": self.last_nlp_result}
                    )
                    print(" Recorded entity correction")
            except Exception as e:
                print(f"[ERROR] Invalid entity format: {e}")
        
        elif correction_type == "response":
            print("Enter the correct response:")
            correct_response = input("Correct response: ").strip()
            if correct_response:
                self.learning_manager.record_correction(
                    CorrectionType.RESPONSE_QUALITY,
                    self.last_user_input,
                    self.last_assistant_response,
                    correct_response,
                    "User provided better response",
                    {"intent": self.last_intent}
                )
                print(" Recorded response quality correction")
        
        elif correction_type == "sentiment":
            print(f"Current sentiment: {self.last_nlp_result.get('sentiment', {})}")
            new_sentiment = input("Correct sentiment (positive/negative/neutral): ").strip()
            if new_sentiment:
                self.learning_manager.record_correction(
                    CorrectionType.SENTIMENT_ANALYSIS,
                    self.last_user_input,
                    self.last_nlp_result.get('sentiment'),
                    new_sentiment,
                    f"User corrected sentiment to {new_sentiment}",
                    {"original_nlp_result": self.last_nlp_result}
                )
                print(f" Recorded sentiment correction: {new_sentiment}")
        
        elif correction_type == "factual":
            print("Describe the factual error:")
            error_description = input("Error description: ").strip()
            correct_fact = input("Correct fact: ").strip()
            if error_description and correct_fact:
                self.learning_manager.record_correction(
                    CorrectionType.FACTUAL_ERROR,
                    self.last_user_input,
                    error_description,
                    correct_fact,
                    f"User corrected factual error: {error_description}",
                    {"response": self.last_assistant_response}
                )
                print(" Recorded factual correction")
        
        else:
            print(f"[ERROR] Unknown correction type: {correction_type}")
    
    def _handle_feedback_command(self, command: str):
        """Handle feedback commands"""
        if not self.last_user_input or not self.last_assistant_response:
            print("\n[ERROR] No previous interaction to rate")
            return
        
        parts = command.split(" ", 1)
        rating_str = parts[1] if len(parts) > 1 else ""
        
        try:
            rating = int(rating_str) if rating_str else None
        except ValueError:
            rating = None
        
        print(f"\n Feedback Mode")
        print("=" * 50)
        print(f"Last input: {self.last_user_input}")
        print(f"Last response: {self.last_assistant_response[:200]}...")
        print()
        
        if rating is None:
            rating = int(input("Rate this response (1-5): ").strip())
        
        if 1 <= rating <= 5:
            feedback_type = FeedbackType.POSITIVE if rating >= 4 else FeedbackType.NEGATIVE
            comment = input("Additional comment (optional): ").strip()
            suggestion = ""
            
            if rating <= 3:
                suggestion = input("How could I improve? ").strip()
            
            self.learning_manager.record_feedback(
                feedback_type,
                self.last_user_input,
                self.last_assistant_response,
                rating,
                comment,
                suggestion,
                {"intent": self.last_intent, "entities": self.last_entities}
            )
            
            print(f"] Thank you for the {rating}/5 rating!")
            if suggestion:
                print(f" I'll work on: {suggestion}")
        else:
            print("[ERROR] Rating must be between 1 and 5")
    
    def _handle_learning_stats_command(self):
        """Handle learning statistics command"""
        print(f"\n Active Learning Statistics")
        print("=" * 50)
        
        try:
            stats = self.learning_manager.get_learning_stats()
            
            print(f"Total corrections: {stats['total_corrections']}")
            print(f"Learning patterns: {stats['total_patterns']}")
            print(f"User feedback entries: {stats['total_feedback']}")
            print(f"Applied patterns: {stats['applied_patterns']}")
            print(f"Recent corrections (7 days): {stats['recent_corrections']}")
            print(f"Average user rating: {stats['average_user_rating']:.1f}/5.0")
            
            if stats['correction_types']:
                print("\nCorrection types:")
                for corr_type, count in stats['correction_types'].items():
                    print(f"   • {corr_type.replace('_', ' ').title()}: {count}")
            
            suggestions = self.learning_manager.suggest_improvements()
            if suggestions:
                print("\n Improvement suggestions:")
                for suggestion in suggestions:
                    print(f"   • {suggestion}")
            
            print("=" * 50)
            
        except Exception as e:
            print(f"[ERROR] Failed to get learning statistics: {e}")
    
    def _format_learning_guidance(self, guidance: Dict[str, Any]) -> str:
        """Format learning guidance for LLM context"""
        guidance_parts = ["Learning guidance:"]
        
        if guidance.get("preferred_words"):
            preferred = ", ".join(guidance["preferred_words"][:5])
            guidance_parts.append(f"Consider using these words: {preferred}")
        
        if guidance.get("avoid_words"):
            avoid = ", ".join(guidance["avoid_words"][:5])
            guidance_parts.append(f"Avoid using these words: {avoid}")
        
        if guidance.get("style_improvement"):
            guidance_parts.append(f"Style note: {guidance['style_improvement']}")
        
        return "\n".join(guidance_parts)
    
    def shutdown(self):
        """Gracefully shutdown ALICE"""
        logger.info(" Shutting down ALICE...")
        
        # Save conversation state
        self._save_conversation_state()
        
        # Save context and memory
        self.context.save_context()
        self.memory._save_memories()
        
        # Stop event-driven systems
        if self.observer_manager:
            self.observer_manager.stop_all()
            logger.info("[OK] Observers stopped")
        
        if self.system_monitor:
            self.system_monitor.stop_monitoring()
            logger.info("[OK] System monitor stopped")
        
        if self.state_tracker:
            self.state_tracker.stop_monitoring()
            logger.info("[OK] State tracker stopped")
        
        if self.pattern_learner:
            self.pattern_learner._save_patterns()
            logger.info("[OK] Patterns saved")
        
        if getattr(self, 'proactive_assistant', None):
            self.proactive_assistant.stop()
            logger.info("[OK] Proactive assistant stopped")
        
        # Stop voice if active
        if self.speech:
            self.speech.stop_listening()
        
        self.running = False
        logger.info("[OK] ALICE shutdown complete")


# Main entry point
def main():
    """Main entry point for A.L.I.C.E"""
    import argparse
    
    parser = argparse.ArgumentParser(description="A.L.I.C.E - Advanced AI Assistant")
    parser.add_argument("--voice", action="store_true", help="Enable voice interaction")
    parser.add_argument("--voice-only", action="store_true", help="Run in voice-only mode")
    parser.add_argument("--model", default="llama3", help="LLM model to use")
    parser.add_argument("--name", default="User", help="Your name")
    
    args = parser.parse_args()
    
    try:
        # Initialize ALICE
        alice = ALICE(
            voice_enabled=args.voice or args.voice_only,
            llm_model=args.model,
            user_name=args.name
        )
        
        # Run appropriate mode
        if args.voice_only:
            alice.run_voice_mode()
        else:
            alice.run_interactive()
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        logger.error(f"[ERROR] Fatal error: {e}")
        print(f"\n[ERROR] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
