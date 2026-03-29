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
import json
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings

import sys
import logging
import re
import html
import time
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from ai.planning.goal_from_llm import get_goal_from_llm, GoalJSON
from ai.infrastructure.policy import get_policy_decision, PolicyDecision
from ai.infrastructure.rbac import get_rbac_engine, AccessRequest
from ai.infrastructure.approval_ledger import get_approval_ledger
from ai.roadmap import get_roadmap_completion_stack
from ai.optimization.runtime_thresholds import get_tool_path_confidence, get_goal_path_confidence, get_ask_threshold
from ai.integration.git_manager import get_git_manager
from ai.integration.build_runner import get_build_runner
from ai.integration.operator_workflow import OperatorWorkflowOrchestrator

# Add project root to path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Import ALICE components
from ai.core.nlp_processor import NLPProcessor
from ai.core.llm_engine import LocalLLMEngine, LLMConfig
from ai.memory.context_engine import get_context_engine
from ai.memory.memory_system import MemorySystem
from ai.memory.conversation_summarizer import ConversationSummarizer
from ai.models.entity_relationship_tracker import EntityRelationshipTracker
from ai.learning.active_learning_manager import ActiveLearningManager, CorrectionType, FeedbackType
from ai.plugins.email_plugin import GmailPlugin
from ai.plugins.plugin_system import (
    PluginManager, WeatherPlugin, TimePlugin,
    SystemControlPlugin, WebSearchPlugin
)
from ai.plugins.file_operations_plugin import FileOperationsPlugin
from ai.plugins.memory_plugin import MemoryPlugin
from ai.plugins.document_plugin import DocumentPlugin
from ai.plugins.calendar_plugin import CalendarPlugin
from ai.plugins.notes_plugin import NotesPlugin
from ai.plugins.maps_plugin import MapsPlugin
from ai.planning.task_executor import TaskExecutor
from speech.speech_engine import SpeechEngine, SpeechConfig

# New anticipatory AI systems
from ai.infrastructure.event_bus import get_event_bus, EventType, EventPriority
from ai.infrastructure.system_state import get_state_tracker, SystemStatus
from ai.infrastructure.observers import get_observer_manager
from ai.learning.pattern_learner import get_pattern_learner
from ai.optimization.system_monitor import get_system_monitor
from ai.planning.task_planner import get_planner
from ai.planning.plan_executor import initialize_executor, get_executor
from ai.planning.planner import ReasoningPlanner
from ai.planning.task import PersistentTaskQueue, Task, TaskStatus as QueueTaskStatus
from ai.core.reasoning_engine import get_reasoning_engine, WorldEntity, EntityKind
from ai.planning.proactive_assistant import (
    get_proactive_assistant,
    parse_reminder_time,
    make_reminder_id,
)
from ai.infrastructure.error_recovery import get_error_recovery
from ai.memory.smart_context_cache import get_context_cache
from ai.memory.adaptive_context_selector import get_context_selector
from ai.memory.predictive_prefetcher import get_prefetcher
from ai.memory.conversation_state import get_conversation_state_tracker
from ai.context_resolver import get_context_resolver
from ai.optimization.response_optimizer import get_response_optimizer
from ai.learning.self_reflection import get_self_reflection
from ai.learning.learning_engine import get_learning_engine
from ai.core.conversational_engine import get_conversational_engine, ConversationalContext
from ai.core.llm_gateway import get_llm_gateway, LLMGateway
from ai.core.llm_policy import LLMCallType
from ai.core.response_self_critic import get_response_self_critic
from ai.core.executive_controller import get_executive_controller
from ai.core.reflection_engine import get_reflection_engine
from ai.core.response_planner import get_response_planner
from ai.core.goal_tracker import get_goal_tracker
from ai.core.response_quality_tracker import get_response_quality_tracker
from ai.core.cognitive_orchestrator import get_cognitive_orchestrator
from ai.core.activity_monitor import ActivityMonitor
from ai.core.temporal_reasoner import TemporalReasoner
from ai.core.adaptive_intent_calibrator import AdaptiveIntentCalibrator
from ai.core.context_intent_refiner import ContextIntentRefiner
from ai.core.constraint_preference_extractor import ConstraintPreferenceExtractor
from ai.core.multi_step_reasoning_engine import MultiStepReasoningEngine
from ai.core.episodic_memory_engine import EpisodicMemoryEngine
from ai.core.proactive_interruption_manager import ProactiveInterruptionManager
from ai.core.adaptive_response_style import AdaptiveResponseStyle
from ai.core.causal_inference_engine import CausalInferenceEngine
from ai.core.hypothetical_scenario_generator import HypotheticalScenarioGenerator
from ai.core.decision_constraint_solver import DecisionConstraintSolver
from ai.core.semantic_memory_index import SemanticMemoryIndex
from ai.core.memory_consolidator import MemoryConsolidator
from ai.core.cross_session_pattern_detector import CrossSessionPatternDetector
from ai.core.system_design_response_guard import SystemDesignResponseGuard
from ai.core.unified_action_engine import ActionRequest, get_unified_action_engine
from ai.core.world_state_memory import get_world_state_memory
from ai.core.execution_journal import get_execution_journal
from ai.core.bounded_autonomy_manager import AutonomyLoop, get_bounded_autonomy_manager
from ai.learning.phrasing_learner import PhrasingLearner
from ai.core.response_formulator import get_response_formulator
from ai.lab_simulator import LabSimulator
from ai.red_team_tester import RedTeamTester

# Continuous learning system
from ai.learning.realtime_logger import get_realtime_logger
from ai.learning.continuous_learning import get_continuous_learning_loop
from ai.learning.response_quality_checker import get_quality_checker

# Automated learning system (Ollama-driven)
from ai.training.ollama_evaluator import get_ollama_evaluator
from ai.training.autolearn import get_autolearn
from ai.training.async_evaluation import get_async_evaluator

# Advanced learning, testing, and telemetry
from ai.learning.pattern_miner import PatternMiner
from ai.training.synthetic_corpus_generator import SyntheticCorpusGenerator
from ai.memory.multimodal_context import MultimodalContext

# Analytics and memory management
from ai.analytics.memory_growth_monitor import get_memory_growth_monitor
from ai.analytics.usage_analytics import get_usage_analytics
from ai.memory.embedding_manager import get_embedding_manager
from ai.memory.bg_embedding_generator import get_bg_embedding_generator
from ai.memory.memory_pruner import get_memory_pruner

# Production Infrastructure
from ai.infrastructure.cache_manager import get_cache_manager, initialize_cache
from ai.infrastructure.metrics_collector import get_metrics_collector, initialize_metrics
from ai.infrastructure.structured_logging import get_structured_logger, configure_logging
from ai.infrastructure.task_queue import get_task_queue, initialize_task_queue
from ai.infrastructure.database_pool import get_connection_pool, initialize_database, DatabaseConfig, DatabaseType
from ai.infrastructure.runtime_flags import is_enabled
from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from ai.reasoning.routing_decision_logger import RoutingDecisionLogger, RoutingDecisionType

# ===== 10 TIER IMPROVEMENTS (LAZY IMPORT UNDER QUARANTINE FLAGS) =====

# Foundation Systems - Response Variance, Personality Evolution, Context Graph
from ai.foundation_integration import FoundationIntegration
from tools.auditing.startup_doctor import StartupDoctor

# NLP perception & policy layer
from ai.core.nlp_processor import Perception
from ai.core.interaction_policy import InteractionPolicy
from ai.learning.learning_engine import get_nlp_error_logger

# Area 1-8: advanced intelligence components (merged into existing modules)
from ai.core.intent_classifier import get_bayesian_router, IntentCandidate
from ai.memory.context_graph import get_world_graph
from ai.core.interaction_policy import get_knob_bandit
from ai.memory.memory_system import get_memory_replay
from ai.core.failure_taxonomy import get_self_debugger, TurnPostmortem
from ai.plugins.plugin_system import get_capability_graph
from ai.learning.pattern_miner import get_habit_miner, get_htn_planner, HTNMethod
from ai.infrastructure.metrics_collector import get_adaptive_controller

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
        self.user_name = user_name
        self.strict_no_llm = llm_policy == "strict"
        self.running = False
        
        # ===== PRODUCTION INFRASTRUCTURE INITIALIZATION =====
        logger.info("=" * 80)
        logger.info("Initializing Production Infrastructure")
        logger.info("=" * 80)
        
        # 0.1: Structured Logging System
        configure_logging(
            level='DEBUG' if debug else 'INFO',
            enable_json=not debug,  # Use JSON in production, human-readable in debug
            log_dir='logs'
        )
        self.structured_logger = get_structured_logger('alice')
        self.structured_logger.info("Structured logging initialized", component='infrastructure')
        
        # 0.2: Redis Cache Manager
        try:
            self.cache = initialize_cache(
                redis_host='localhost',
                redis_port=6379,
                default_ttl=3600
            )
            cache_stats = self.cache.get_stats()
            self.structured_logger.info(
                f"Cache initialized: {cache_stats['backend']}",
                component='cache',
                backend=cache_stats['backend']
            )
        except Exception as e:
            self.structured_logger.warning(f"Cache initialization failed: {e}, using fallback", component='cache')
            self.cache = get_cache_manager()  # Fallback to in-memory
        
        # 0.3: Prometheus Metrics Collector
        self.metrics = initialize_metrics(enable_prometheus=True)
        self.structured_logger.info("Metrics collector initialized", component='metrics')
        
        # 0.4: Async Task Queue
        try:
            self.task_queue = initialize_task_queue(
                broker_url='redis://localhost:6379/0',
                backend_url='redis://localhost:6379/1',
                num_workers=4
            )
            queue_stats = self.task_queue.get_stats()
            self.structured_logger.info(
                f"Task queue initialized: {queue_stats['backend']}",
                component='task_queue',
                workers=queue_stats['workers']
            )
        except Exception as e:
            self.structured_logger.warning(f"Task queue initialization failed: {e}", component='task_queue')
            self.task_queue = get_task_queue()  # Fallback to thread pool
        
        # 0.5: Database Connection Pool (optional - for future PostgreSQL migration)
        try:
            db_config = DatabaseConfig(
                db_type=DatabaseType.SQLITE,
                database="data/alice.db",
                min_connections=2,
                max_connections=10
            )
            self.db_pool = initialize_database(db_config)
            self.structured_logger.info("Database pool initialized", component='database')
        except Exception as e:
            self.structured_logger.warning(f"Database pool initialization failed: {e}", component='database')
            self.db_pool = None
        
        self.structured_logger.info("Production infrastructure ready", component='infrastructure')
        
        # ===== END PRODUCTION INFRASTRUCTURE =====
        
        # ===== FOUNDATION SYSTEMS INITIALIZATION =====
        logger.info("=" * 80)
        logger.info("Initializing Foundation Systems")
        logger.info("=" * 80)
        
        # Initialize foundation systems (Response Variance, Personality Evolution, Context Graph)
        # These will be used alongside existing systems during migration
        self.foundations = None  # Will be initialized after LLM engine is ready
        self.foundation_mode = "parallel"  # "parallel", "primary", or "exclusive"
        self.structured_logger.info("Foundation systems will initialize after LLM engine", component='foundations')
        
        logger.info("=" * 80)
        # ===== END FOUNDATION SYSTEMS =====
        
        # Conversation state for context-aware operations
        self.last_email_list = []  # Store last displayed email list
        self.last_email_context = None  # Store context of last email operation
        self.last_code_file = None  # Store last code file for follow-up queries
        self.pending_action = None  # Track multi-step actions (e.g., composing email)
        self.pending_data = {}  # Store data for pending actions
        self.pending_operator_actions = {}
        
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
        self.last_interaction = None
        # Turn index at which the active intent domain last changed.
        # Used to compute turn_distance for FollowUpResolver decay.
        self._last_intent_turn = 0
        
        # Session-level response cache for fast conversational responses
        # Caches (input_normalized, intent) -> response for quick lookup
        self._response_cache = {}
        self._cache_max_size = 50  # Keep last 50 responses per session
        
        # Conversation summarizer for intelligent context management
        self.summarizer = None  # Will be initialized after LLM engine
        
        # Advanced context handler
        self.advanced_context = None  # Will be initialized after NLP processor
        self.context_resolver = None
        
        # Event-driven architecture (anticipatory AI)
        self.event_bus = None
        self.state_tracker = None
        self.observer_manager = None
        self.pattern_learner = None
        self.system_monitor = None
        self.planner = None
        self.plan_executor = None
        self.reasoning_planner = None
        self.persistent_task_queue = None
        self.activity_monitor = None
        self.temporal_reasoner = None
        self.adaptive_intent_calibrator = None
        self.context_intent_refiner = None
        self.constraint_preference_extractor = None
        self.multi_step_reasoning_engine = None
        self.episodic_memory_engine = None
        self.proactive_interruption_manager = None
        self.adaptive_response_style = None
        self.causal_inference_engine = None
        self.hypothetical_scenario_generator = None
        self.decision_constraint_solver = None
        self.semantic_memory_index = None
        self.memory_consolidator = None
        self.cross_session_pattern_detector = None
        self.system_design_response_guard = None
        self._episodic_turn_counter = 0
        self._last_routed_intent = ""
        self._last_routed_confidence = 0.0
        
        logger.info("=" * 80)
        logger.info("Initializing A.L.I.C.E - Advanced Linguistic Intelligence Computer Entity")
        logger.info("=" * 80)
        
        # Initialize components
        try:
            # Set up logging context for this session
            self.structured_logger.set_context(user=user_name, session_id=f"{time.time()}")
            
            # 1. NLP Processor
            logger.info("Loading NLP processor...")
            self.nlp = NLPProcessor()
            # Pre-warm the semantic classifier so the first user query has no cold-start delay
            try:
                self.nlp._ensure_semantic_classifier()
                logger.info("Semantic classifier pre-warmed.")
            except Exception:
                pass  # non-fatal; will lazy-load on first real query
            # Shared session objects from NLP stack
            self.dialogue_memory = getattr(self.nlp, 'dialogue_memory', None)
            self.fp_store = getattr(self.nlp, '_fp_store', None)

            # NLP perception & policy layer (sit between NLP and routing)
            self.perception = Perception()
            self.interaction_policy = InteractionPolicy()
            self.nlp_error_logger = get_nlp_error_logger()
            self.context_resolver = get_context_resolver()

            # Area 1–8: advanced intelligence components
            try:
                self.bayesian_router = get_bayesian_router()
                # Seed cost matrix from stored NLP error log so the router
                # benefits from past correction history on every startup.
                try:
                    _drained = self.bayesian_router.drain_for_cost_update(
                        "memory/nlp_errors.jsonl"
                    )
                    if _drained:
                        logger.info(
                            "[BayesianRouter] Seeded cost matrix from %d error log entries",
                            _drained,
                        )
                except Exception as _drain_err:
                    logger.debug("BayesianRouter cost-matrix seeding skipped: %s", _drain_err)
                self.world_graph = get_world_graph(
                    persistence_path="memory/world_graph.json"
                )
                self.knob_bandit = get_knob_bandit()
                self.memory_replay = get_memory_replay()
                self.self_debugger = get_self_debugger()
                self.capability_graph = get_capability_graph()
                self.habit_miner = get_habit_miner()
                self.adaptive_controller = get_adaptive_controller()
                self.htn_planner = get_htn_planner()
            except Exception as _adv_err:
                logger.warning("Advanced components init failed (non-fatal): %s", _adv_err)
                for _attr in ("bayesian_router", "world_graph", "knob_bandit",
                              "memory_replay", "self_debugger", "capability_graph",
                              "habit_miner", "adaptive_controller", "htn_planner"):
                    if not hasattr(self, _attr):
                        setattr(self, _attr, None)

            # Seed hand-crafted HTN decomposition methods for common workflows
            try:
                if getattr(self, 'htn_planner', None) is not None:
                    _htn = self.htn_planner
                    _htn.add_method(HTNMethod(
                        goal="research:topic",
                        subtasks=["web:search", "notes:create"],
                        priority=1.0,
                    ))
                    _htn.add_method(HTNMethod(
                        goal="calendar:plan_meeting",
                        subtasks=["calendar:query", "calendar:create_event"],
                        priority=1.0,
                    ))
                    _htn.add_method(HTNMethod(
                        goal="notes:research_note",
                        subtasks=["web:search", "notes:create", "notes:append"],
                        priority=0.9,
                    ))
            except Exception as _htn_seed_err:
                logger.debug("HTN method seeding skipped: %s", _htn_seed_err)
            
            # 1.5. Unified Context Engine (combines context_manager + advanced_context_handler)
            logger.info("Loading unified context engine...")
            self.context = get_context_engine()
            self.context.user_prefs.name = user_name
            # Backward compatibility: older code paths reference advanced_context
            self.advanced_context = self.context
            
            # 2. Reasoning Engine (combines world_state + reference_resolver + goal_resolver + verifier)
            logger.info("Loading reasoning engine...")
            self.reasoning_engine = get_reasoning_engine(user_name)
            self.response_self_critic = get_response_self_critic()
            self.executive_controller = get_executive_controller()
            self.reflection_engine = get_reflection_engine()
            self.response_planner = get_response_planner()
            self.goal_tracker = get_goal_tracker()
            self.response_quality_tracker = get_response_quality_tracker()
            self.cognitive_orchestrator = get_cognitive_orchestrator(
                goal_tracker=self.goal_tracker,
                reflection_engine=self.reflection_engine,
                response_quality_tracker=self.response_quality_tracker,
                executive_controller=self.executive_controller,
                conversation_state_tracker=getattr(self, 'conversation_state_tracker', None),
                response_planner=self.response_planner,
                tick_interval_seconds=5.0,
                reasoning_importance_threshold=0.74,
            )
            self.cognitive_orchestrator.start()
            self._internal_reasoning_state = {}
            self._exec_should_store_memory = True
            self._last_exec_gate_eval = {}
            self._last_goal_tracker_topic = ""
            self.error_recovery = get_error_recovery()
            self.context_cache = get_context_cache()
            self.context_selector = get_context_selector()
            self.prefetcher = get_prefetcher(self.reasoning_engine)
            self.response_optimizer = get_response_optimizer(self.reasoning_engine)
            # Point self-reflection to workspace root so code-analysis requests
            # can resolve files like app/alice.py (not only ai/*).
            self.self_reflection = get_self_reflection(PROJECT_ROOT)
            logger.info("[OK] Reasoning engine initialized - unified entity/goal/verification tracking")
            
            # 2.9. Unified Learning Engine - collect and learn from interactions
            self.learning_engine = get_learning_engine()
            logger.info("[OK] Learning engine initialized - A.L.I.C.E will learn from your interactions")

            # 2.9.1. Real-time Continuous Learning System
            logger.info("Initializing continuous learning system...")
            self.realtime_logger = get_realtime_logger()
            self.continuous_learning = get_continuous_learning_loop(
                learning_engine=self.learning_engine,
                realtime_logger=self.realtime_logger,
                check_interval_hours=6,
                auto_start=True
            )
            logger.info("[OK] Continuous learning active - Alice learns 24/7 from real-time errors")

            # 2.9.2. Failure taxonomy for retraining signal
            try:
                from ai.core.failure_taxonomy import get_failure_taxonomy
                self.failure_taxonomy = get_failure_taxonomy()
                logger.info("[OK] Failure taxonomy ready - failed turns will be classified for retraining")
            except ImportError:
                self.failure_taxonomy = None

            # 3. Memory System (needs to be before conversational engine)
            logger.info("Loading memory system...")
            self.memory = MemorySystem()
            
            # 3.5. Conversational engine - A.L.I.C.E's own conversational logic (no Ollama)
            self.conversational_engine = get_conversational_engine(
                memory_system=self.memory,
                world_state=self.reasoning_engine
            )
            logger.info("[OK] Conversational engine initialized - A.L.I.C.E thinks independently")

            # 3.6. Phrasing Learner - Progressive independence from Ollama
            logger.info("Loading phrasing learner...")
            self.phrasing_learner = PhrasingLearner(storage_path="data/learned_phrasings.jsonl")
            logger.info("[OK] Phrasing learner ready - Alice will learn from Ollama and become independent")

            # 3.7. Knowledge Engine - Alice's own intelligence and learning
            logger.info("Initializing knowledge engine...")
            from ai.core.knowledge_engine import KnowledgeEngine
            self.knowledge_engine = KnowledgeEngine(storage_path="data/knowledge")
            stats = self.knowledge_engine.get_stats()
            logger.info(f"[OK] Knowledge engine ready - Alice has learned {stats['total_entities']} entities, {stats['total_relationships']} relationships")

            # 3.8. Conversation Context Manager - Multi-turn conversation tracking
            logger.info("Initializing conversation context manager...")
            from ai.memory.conversation_context import get_context_manager
            self.conversation_context = get_context_manager(max_turns=50, context_window=10)
            logger.info("[OK] Conversation context manager ready - Alice can now track 'it', 'that', and conversation flow")
            self.conversation_state_tracker = get_conversation_state_tracker(max_chain=8, max_depth=5)
            logger.info("[OK] Conversation state tracker ready - topic/goal/depth/question-chain tracking enabled")
            if getattr(self, 'cognitive_orchestrator', None) is not None:
                self.cognitive_orchestrator.conversation_state_tracker = self.conversation_state_tracker

            # 3.9. User Profile Engine - Deep user modeling and preference learning
            logger.info("Initializing user profile engine...")
            from ai.learning.user_profile_engine import get_profile_engine
            self.user_profile = get_profile_engine(storage_path="data/user_profiles")
            profile_summary = self.user_profile.get_profile_summary()
            logger.info(f"[OK] User profile loaded - {profile_summary.get('interactions', 0)} interactions, learning preferences")

            # 3.10. Persistent Goal System - Long-running goals across sessions
            logger.info("Initializing persistent goal system...")
            from ai.planning.persistent_goals import get_goal_system
            self.goal_system = get_goal_system(storage_path="data/goals")
            active_goals = self.goal_system.get_active_goals()
            logger.info(f"[OK] Goal system ready - {len(active_goals)} active goals")

            # Resume goals from previous session
            if active_goals:
                logger.info(f"Resuming {len(active_goals)} goals from previous session")
                for goal in active_goals[:3]:
                    logger.info(f"  - {goal.title} ({int(goal.progress * 100)}% complete)")

            # 3.11. Proactive Intelligence Loop - Background monitoring and insights
            logger.info("Initializing proactive intelligence...")
            from ai.planning.proactive_intelligence import get_proactive_intelligence
            self.proactive_intelligence = get_proactive_intelligence(check_interval=60)

            # Inject dependencies
            self.proactive_intelligence.inject_dependencies(
                goal_system=self.goal_system,
                profile_engine=self.user_profile,
                context_manager=self.conversation_context
            )

            # Start proactive loop
            self.proactive_intelligence.start()
            logger.info("[OK] Proactive intelligence active - Alice will monitor and help proactively")

            # 3.12. Autonomous Agent System - Multi-step goal execution
            logger.info("Initializing autonomous agent system...")
            from ai.planning.autonomous_agent import AutonomousAgent
            from ai.planning.autonomous_execution_loop import AutonomousExecutionLoop

            self.autonomous_agent = AutonomousAgent(
                goal_system=self.goal_system,
                llm_engine=None,  # Will be set after LLM engine loads
                plugin_system=self.plugins if hasattr(self, 'plugins') else None
            )

            self.execution_loop = AutonomousExecutionLoop(
                goal_system=self.goal_system,
                autonomous_agent=self.autonomous_agent
            )

            logger.info("[OK] Autonomous agent ready - Alice can work on multi-step goals independently")

            # 3.13. Capability Registry - Alice knows what she can do (in code, not prompts)
            logger.info("Initializing capability registry...")
            self._init_capabilities_registry()
            logger.info("[OK] Capability registry ready - Alice knows her own capabilities")

            # 4. LLM Engine
            logger.info("🧠 Loading LLM engine...")
            llm_config = LLMConfig(model=llm_model)
            self.llm = LocalLLMEngine(llm_config)

            # 4.0. Foundation Systems Integration
            logger.info("🎯 Initializing Foundation Systems...")
            try:
                self.foundations = FoundationIntegration(
                    llm_generator=self.llm,
                    phrasing_learner=getattr(self, 'phrasing_learner', None)
                )
                self.structured_logger.info(
                    "Foundation systems initialized",
                    component='foundations',
                    mode=self.foundation_mode
                )
                logger.info("[OK] Foundation systems ready - Response Variance, Personality Evolution, Context Graph active")
            except Exception as e:
                logger.error(f"[ERROR] Foundation systems initialization failed: {e}")
                self.foundations = None
                self.structured_logger.error(f"Foundation systems failed: {e}", component='foundations')

            # 4.0.5. ===== 10 TIER IMPROVEMENTS INITIALIZATION =====
            logger.info("🚀 Initializing 10 Tier Improvements (quarantine-aware)...")
            try:
                # Tier 1: High-Impact Wins
                logger.info("  ├─ Tier 1: Initializing long-session coherence...")
                if is_enabled("session_summarizer"):
                    from ai.memory.session_summarizer import SessionSummarizer
                    self.session_summarizer = SessionSummarizer(summarize_every_n_turns=5)
                
                logger.info("  ├─ Tier 1: Initializing capability constraints...")
                if is_enabled("capability_constraints"):
                    from ai.infrastructure.capability_constraints import CapabilityConstraintsLedger
                    self.capability_constraints = CapabilityConstraintsLedger()
                
                logger.info("  ├─ Tier 1: Initializing result quality scorer...")
                if is_enabled("result_quality_scorer"):
                    from ai.core.result_quality_scorer import ResultQualityScorer
                    self.result_quality_scorer = ResultQualityScorer()
                
                logger.info("  ├─ Tier 1: Initializing goal alignment tracker...")
                if is_enabled("goal_alignment_tracker"):
                    from ai.learning.goal_alignment_tracker import GoalAlignmentTracker
                    self.goal_alignment_tracker = GoalAlignmentTracker()
                
                # Tier 2: Personality & Agency
                logger.info("  ├─ Tier 2: Initializing tone trajectory engine...")
                if is_enabled("tone_trajectory_engine"):
                    from ai.learning.tone_trajectory_engine import ToneTrajectoryEngine
                    self.tone_trajectory_engine = ToneTrajectoryEngine()
                
                logger.info("  ├─ Tier 2: Initializing pattern-based nudger...")
                if is_enabled("pattern_based_nudger"):
                    from ai.proactivity.pattern_based_nudger import PatternBasedNudger
                    self.pattern_nudger = PatternBasedNudger()
                
                # Tier 3: Deep System Knowledge
                logger.info("  ├─ Tier 3: Initializing system state API...")
                if is_enabled("system_state_api"):
                    from ai.introspection.system_state_api import SystemStateAPI
                    self.system_state_api = SystemStateAPI()
                
                logger.info("  ├─ Tier 3: Initializing weak-spot detector...")
                if is_enabled("weak_spot_detector"):
                    from ai.learning.weak_spot_detector import WeakSpotDetector
                    self.weak_spot_detector = WeakSpotDetector()
                
                # Tier 4: Mission-Aligned Execution 
                logger.info("  ├─ Tier 4: Initializing multi-goal arbitrator...")
                if is_enabled("multi_goal_arbitrator"):
                    from ai.goals.multi_goal_arbitrator import MultiGoalArbitrator
                    self.multi_goal_arbitrator = MultiGoalArbitrator()
                
                logger.info("  └─ Tier 4: Initializing routing decision logger...")
                if is_enabled("routing_decision_logger"):
                    self.routing_decision_logger = RoutingDecisionLogger()
                
                self.structured_logger.info(
                    "All 10 tier improvements initialized successfully",
                    component='tier_improvements',
                    active_systems=10
                )
                logger.info("[OK] All 10 Tier Improvements active - JARVIS-like capabilities enabled")
            except Exception as e:
                logger.error(f"[ERROR] Tier improvements initialization failed: {e}")
                import traceback
                traceback.print_exc()
                self.structured_logger.error(f"Tier improvements failed: {e}", component='tier_improvements')
                # Don't fail startup if improvements fail - these are enhancements
            # ===== END 10 TIER IMPROVEMENTS =====

            # Inject LLM engine into autonomous agent now that it's loaded
            if hasattr(self, 'autonomous_agent'):
                self.autonomous_agent.llm_engine = self.llm

            # 4.1. LLM Gateway - Single entry point for all LLM calls
            logger.info(" Initializing LLM Gateway with policy enforcement...")
            self.llm_gateway = get_llm_gateway(
                llm_engine=self.llm,
                learning_engine=self.learning_engine
            )
            logger.info("[OK] LLM Gateway active - all calls now policy-gated")

            # 4.1.0 Contract Runtime Boundaries and Pipeline
            try:
                self.runtime_boundaries = build_runtime_boundaries(self)
                self.contract_pipeline = ContractPipeline(self.runtime_boundaries)
                self.structured_logger.info(
                    "Contract runtime boundaries initialized",
                    component='contracts',
                    boundaries=['routing', 'memory', 'tools', 'response']
                )
            except Exception as e:
                self.runtime_boundaries = None
                self.contract_pipeline = None
                logger.error(f"[ERROR] Contract runtime initialization failed: {e}")

            # 4.1.1. Response Formulator - Transform plugin data into natural responses
            logger.info("Initializing response formulator...")
            self.response_formulator = get_response_formulator(
                phrasing_learner=self.phrasing_learner,
                llm_gateway=self.llm_gateway
            )
            stats = self.response_formulator.get_stats()
            logger.info(f"[OK] Response formulator ready - Alice can independently formulate {stats['independent_actions']}/{stats['total_templates']} action types")

            # 4.1.2. Automated Learning System - Ollama evaluates, Alice learns
            logger.info("Initializing automated learning system...")

            # Ollama Evaluator - scores every response 0-100
            self.ollama_evaluator = get_ollama_evaluator(llm_engine=self.llm)
            logger.info("[OK] Ollama evaluator ready - automated feedback active")

            # Async Evaluation Wrapper - non-blocking evaluation
            self.async_evaluator = get_async_evaluator(
                ollama_evaluator=self.ollama_evaluator,
                response_formulator=self.response_formulator,
                realtime_logger=self.realtime_logger
            )
            logger.info("[OK] Async evaluator ready - user experience won't be blocked")

            # AutoLearn - 6-hour automated learning cycles
            self.autolearn = get_autolearn(
                ollama_evaluator=self.ollama_evaluator,
                learning_engine=self.learning_engine,
                response_formulator=self.response_formulator,
                realtime_logger=self.realtime_logger,
                check_interval_hours=6,
                auto_start=True
            )
            logger.info("[OK] AutoLearn active - Alice will improve every 6 hours automatically")
            
            # 4.2. Configure LLM Policy based on startup flag
            if self.llm_policy != "default":
                logger.info(f"  Configuring LLM policy: {self.llm_policy}")
                from ai.core.llm_policy import configure_minimal_policy
                
                if self.llm_policy == "minimal":
                    configure_minimal_policy()
                    logger.info("[OK] Minimal policy active - patterns-first, LLM only for generation")
                elif self.llm_policy == "strict":
                    logger.info("[OK] Strict policy active - deterministic phrasing for open-ended responses")
                    configure_minimal_policy()

            # 4.3. Runtime safety and verification guards
            self.rbac_engine = get_rbac_engine()
            self.action_engine = get_unified_action_engine()
            self.approval_ledger = get_approval_ledger()
            self.roadmap_stack = get_roadmap_completion_stack()
            self.world_state_memory = get_world_state_memory(storage_path="data/world_state.json")
            self.execution_journal = get_execution_journal(storage_path="data/action_journal.jsonl")
            self.autonomy_manager = get_bounded_autonomy_manager(storage_path="data/autonomy_loops.json")
            self._init_bounded_autonomy_loops()
            logger.info("[OK] RBAC and unified action guards active")
            
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
            self.action_engine.bind_plugin_manager(self.plugins)
            
            # 6. Task Executor
            logger.info(" Loading task executor...")
            self.executor = TaskExecutor(safe_mode=True)

            # 6.1. Operator integrations
            self.git_manager = get_git_manager(PROJECT_ROOT)
            self.build_runner = get_build_runner(PROJECT_ROOT)
            self.operator_workflow = OperatorWorkflowOrchestrator(self.git_manager, self.build_runner)
            
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

            # Advanced reasoning planner + persistent task queue
            self.reasoning_planner = ReasoningPlanner()
            self.persistent_task_queue = PersistentTaskQueue("data/planning/runtime_tasks.json")
            self.persistent_task_queue.register_handler("execute_plan", self._execute_plan_queue_task)
            self.persistent_task_queue.start_background_loop(tick_seconds=0.2)
            logger.info("[OK] Reasoning planner and persistent queue wired")
            
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

            # 10. Analytics and Memory Management
            logger.info("Initializing analytics and memory management...")

            # Usage analytics - track interaction patterns
            self.usage_analytics = get_usage_analytics()
            logger.info("[OK] Usage analytics ready")

            # Memory growth monitor - track memory size over time
            self.memory_growth_monitor = get_memory_growth_monitor()
            logger.info("[OK] Memory growth monitor ready")

            # Background embedding generator - async embedding generation
            self.bg_embedding_generator = get_bg_embedding_generator(
                embedding_manager=get_embedding_manager()
            )
            self.bg_embedding_generator.start()
            logger.info("[OK] Background embedding generator started")

            # Memory pruner - automatic memory lifecycle management
            self.memory_pruner = get_memory_pruner()
            logger.info("[OK] Memory pruner ready")

            logger.info("=" * 80)
            logger.info("[OK] A.L.I.C.E initialized successfully!")
            logger.info("=" * 80)

            self.startup_health_report = None
            self._run_startup_doctor()
            
            # Store system capabilities in context
            self.context.update_system_status("capabilities", self.plugins.get_capabilities())
            self.context.update_system_status("voice_enabled", voice_enabled)
            self.context.update_system_status("llm_model", llm_model)
            
            # Load previous conversation state if available
            self._load_conversation_state()
            
        except Exception as e:
            logger.error(f"[ERROR] Initialization failed: {e}")
            raise

    def _run_startup_doctor(self) -> None:
        """Run profile-based startup diagnostics and persist a health summary."""
        enabled_raw = os.getenv("ALICE_STARTUP_DOCTOR", "1").strip().lower()
        if enabled_raw in {"0", "false", "off", "no"}:
            logger.info("[INFO] Startup doctor disabled by ALICE_STARTUP_DOCTOR")
            return

        profile = os.getenv("ALICE_STARTUP_PROFILE", "fast").strip().lower() or "fast"
        try:
            doctor = StartupDoctor(root_dir=PROJECT_ROOT)
            report = doctor.run(profile=profile)
            self.startup_health_report = report
            logger.info(
                "[OK] Startup doctor completed | profile=%s status=%s score=%s",
                report.get("profile"),
                report.get("status"),
                report.get("health_score"),
            )
        except Exception as exc:
            # Startup diagnostics must never prevent ALICE from booting.
            logger.warning("[WARNING] Startup doctor failed: %s", exc)

    def _init_capabilities_registry(self):
        """
        Alice knows what she can do - in CODE, not prompts.
        This is programmatic self-awareness of her capabilities.
        """
        self.capabilities = {
            'codebase_access': {
                'available': True,
                'type': 'read-only',
                'scope': ['workspace/**/*'],
                'operations': ['list', 'read', 'search', 'analyze'],
                'description': "I can read and analyze my own Python codebase across all directories",
                'examples': [
                    "list files in ai directory",
                    "read llm_engine.py",
                    "search for 'reasoning' in codebase"
                ]
            },
            'email_access': {
                'available': self.gmail is not None if hasattr(self, 'gmail') else False,
                'type': 'read-write',
                'provider': 'Gmail',
                'operations': ['read', 'send', 'search', 'delete', 'compose'],
                'description': "I can read, send, and manage your Gmail emails",
                'examples': [
                    "read my emails",
                    "send email to john@example.com",
                    "search emails from Alice"
                ]
            },
            'calendar': {
                'available': True,
                'operations': ['create', 'read', 'update', 'delete'],
                'description': "I can manage your calendar events",
                'examples': [
                    "what's on my calendar today",
                    "add meeting at 3pm",
                    "show tomorrow's schedule"
                ]
            },
            'weather': {
                'available': True,
                'operations': ['get_current', 'get_forecast'],
                'description': "I can check weather conditions and forecasts",
                'examples': [
                    "what's the weather",
                    "weather forecast for tomorrow",
                    "will it rain today"
                ]
            },
            'file_operations': {
                'available': True,
                'operations': ['read', 'write', 'list', 'search'],
                'description': "I can manage files on your system",
                'examples': [
                    "create a file named notes.txt",
                    "read document.txt",
                    "list files in downloads"
                ]
            },
            'notes': {
                'available': True,
                'operations': ['create', 'read', 'update', 'delete', 'search'],
                'description': "I can take and manage notes for you",
                'examples': [
                    "take a note: meeting at 3pm",
                    "show my notes",
                    "search notes for 'project'"
                ]
            },
            'maps': {
                'available': True,
                'operations': ['get_directions', 'search_location'],
                'description': "I can provide directions and location information",
                'examples': [
                    "directions to the airport",
                    "where is the nearest coffee shop",
                    "how do I get to Times Square"
                ]
            },
            'time': {
                'available': True,
                'operations': ['get_time', 'get_date', 'set_timer', 'set_reminder'],
                'description': "I can tell time and set timers/reminders",
                'examples': [
                    "what time is it",
                    "what's today's date",
                    "set a timer for 10 minutes"
                ]
            },
            'web_search': {
                'available': True,
                'operations': ['search'],
                'description': "I can search the web for information",
                'examples': [
                    "search for Python tutorials",
                    "look up quantum physics",
                    "find news about AI"
                ]
            },
            'memory': {
                'available': True,
                'operations': ['store', 'recall', 'search'],
                'description': "I maintain memory of our conversations and can recall information",
                'examples': [
                    "remember my favorite color is blue",
                    "what did I tell you about my project",
                    "do you remember my birthday"
                ]
            },
            'reasoning': {
                'available': True,
                'operations': ['analyze', 'solve', 'debug', 'explain'],
                'description': "I can reason about problems and provide logical analysis",
                'examples': [
                    "help me debug this code",
                    "explain how neural networks work",
                    "analyze this situation"
                ]
            },
            'self_reflection': {
                'available': True,
                'operations': ['read_own_code', 'analyze_capabilities', 'explain_architecture'],
                'description': "I can read and understand my own source code",
                'examples': [
                    "how do you process natural language",
                   "explain your memory system",
                    "what's in your reasoning engine"
                ]
            }
        }
        self.activity_monitor = ActivityMonitor()
        self.temporal_reasoner = TemporalReasoner()
        self.adaptive_intent_calibrator = AdaptiveIntentCalibrator()
        self.context_intent_refiner = ContextIntentRefiner()
        self.constraint_preference_extractor = ConstraintPreferenceExtractor()
        self.multi_step_reasoning_engine = MultiStepReasoningEngine()
        self.episodic_memory_engine = EpisodicMemoryEngine()
        self.proactive_interruption_manager = ProactiveInterruptionManager()
        self.adaptive_response_style = AdaptiveResponseStyle()
        self.causal_inference_engine = CausalInferenceEngine()
        self.hypothetical_scenario_generator = HypotheticalScenarioGenerator()
        self.decision_constraint_solver = DecisionConstraintSolver()
        self.semantic_memory_index = SemanticMemoryIndex()
        self.memory_consolidator = MemoryConsolidator()
        self.cross_session_pattern_detector = CrossSessionPatternDetector()
        self.system_design_response_guard = SystemDesignResponseGuard()

    def _select_tone(self, intent: str, context: Any, user_input: str) -> str:
        """
        Alice's personality is CODED here.
        She decides her own tone based on the situation.
        This is Alice's emotional intelligence - coded, not prompted.

        Args:
            intent: The classified intent
            context: Conversational context
            user_input: Original user input

        Returns:
            Tone identifier for phrasing
        """
        # ── Interaction policy overrides (mood-driven) ────────────────────────
        # If the perception layer detected a strong mood this turn, honour it
        # before falling back to Alice's content-based tone selection.
        policy = getattr(self, '_last_policy', None)
        if policy is not None:
            if policy.tone == "empathetic":
                return "calm and understanding"
            if policy.tone == "direct":
                return "professional and precise"
            if policy.tone == "encouraging":
                return "casual and friendly"

        # ── Response knob bandit override ─────────────────────────────────────
        # Consult the learned style policy.  High empathy → calm tone; high
        # directness → precise tone.  Only overrides when the bandit has enough
        # signal (arm.count check is inside propose()).
        try:
            if getattr(self, 'knob_bandit', None) is not None:
                _sentiment_label = getattr(self, '_last_sentiment', None) or 'neutral'
                _topic = (self.conversation_topics[-1]
                          if getattr(self, 'conversation_topics', None) else "")
                _key, _knobs = self.knob_bandit.propose(
                    intent=intent,
                    sentiment=_sentiment_label,
                    topic=_topic,
                )
                self._knob_context_key = _key
                self._last_knobs = _knobs
                if _knobs.empathy_level > 0.70:
                    return "calm and understanding"
                if _knobs.directness > 0.75 and _knobs.formality > 0.6:
                    return "professional and precise"
        except Exception:
            pass

        # Alice's core personality (consistent and reliable)
        base_tone = "warm and helpful"

        # Adjust based on context (Alice's situational awareness)
        if intent.startswith('error:') or intent.startswith('problem:'):
            return "professional and supportive"

        elif user_input.isupper() or user_input.count('!') > 2:
            # User seems upset or excited - de-escalate with calm tone
            return "calm and understanding"

        elif intent.startswith('casual:') or intent in ['greeting', 'chitchat', 'farewell']:
            return "casual and friendly"

        elif intent.startswith('technical:') or intent.startswith('code:'):
            return "professional and precise"

        elif intent.startswith('creative:') or 'write' in intent or 'create' in intent:
            return "enthusiastic and supportive"

        else:
            return base_tone

    def _formulate_response(
        self,
        user_input: str,
        intent: str,
        entities: Dict[str, Any],
        context: Any
    ) -> Dict[str, Any]:
        """
        Alice's core intelligence formulates a structured response.
        This is WHERE ALICE THINKS - using her coded logic, not LLM delegation.

        Process:
        1. Understand what user wants (reasoning)
        2. Check Alice's knowledge/memory
        3. Decide response type and content
        4. Return structured thought (not natural language yet)

        Args:
            user_input: User's message
            intent: Classified intent
            entities: Extracted entities
            context: Conversational context (can have plugin_data)

        Returns:
            Structured response dict with:
            - type: Response type (capability_answer, knowledge_answer, etc.)
            - content: Main content/data
            - reasoning: Chain of reasoning (optional)
            - confidence: Confidence score
        """
        # Step 0: Check if plugin provided data - formulate based on that + user question
        plugin_data = getattr(context, 'plugin_data', None) if hasattr(context, 'plugin_data') else context.get('plugin_data') if isinstance(context, dict) else None

        if plugin_data:
            # Alice analyzes plugin data in context of user's question
            return self._formulate_from_plugin_data(user_input, intent, entities, plugin_data)
        # Step 0.5: Self-analysis requests - Alice should read her own code and formulate real insights
        input_lower = user_input.lower()
        if self._is_location_query(user_input):
            location_payload = self._build_location_payload()
            return {
                'type': 'location_report',
                **location_payload,
                'confidence': 0.98,
                'source': 'context_engine',
            }

        self_analysis_phrases = [
            'go through your code', 'analyze your code', 'review your code',
            'look at your code', 'read your code', 'check your code',
            'analyze yourself', 'review yourself', 'improvements',
            'what can we improve', 'what can you improve', 'suggest improvements'
        ]

        # Check if this is a comprehensive self-analysis request
        if any(phrase in input_lower for phrase in self_analysis_phrases):
            # This should trigger actual code reading, not LLM hallucination
            return {
                'type': 'self_analysis_needed',
                'query': user_input,
                'confidence': 0.9
            }

        # Step 1: Check if Alice can answer from her own knowledge
        # Alice's knowledge engine - learns from every interaction
        can_answer, confidence = self.knowledge_engine.can_answer_independently(user_input, intent)
        if can_answer and confidence > 0.7:
            # Alice knows this! Answer from her own knowledge
            return {
                'type': 'knowledge_answer',
                'question': user_input,
                'intent': intent,
                'confidence': confidence,
                'source': 'alice_knowledge'  # Alice's own learning, not Ollama
            }

        # Step 2: Capability questions - Alice knows her own capabilities from registry
        if 'capability' in intent or any(phrase in user_input.lower() for phrase in [
            'can you', 'do you have access', 'are you able to', 'can you access'
        ]):
            # Extract what capability is being asked about
            capability_key = self._identify_capability_from_input(user_input)

            if capability_key and capability_key in self.capabilities:
                capability = self.capabilities[capability_key]
                return {
                    'type': 'capability_answer',
                    'can_do': capability['available'],
                    'details': capability.get('description', ''),
                    'operations': capability.get('operations', []),
                    'examples': capability.get('examples', []),
                    'confidence': 0.95
                }

        # Step 2: Check Alice's memory for relevant information
        try:
            relevant_memories = self.memory.recall_relevant(user_input, top_k=3)
        except:
            relevant_memories = []

        # Step 3: Knowledge questions - Alice might need to query her knowledge tool
        if intent.startswith('question:') and 'knowledge' in intent:
            return {
                'type': 'knowledge_query_needed',
                'query': user_input,
                'confidence': 0.7
            }

        # Step 4: Reasoning/analysis tasks - Alice uses her reasoning engine
        if intent.startswith('reasoning:') or intent.startswith('analyze:'):
            reasoning_chain = self._apply_reasoning(user_input, entities, context)
            return {
                'type': 'reasoning_result',
                'conclusion': reasoning_chain[-1] if reasoning_chain else "Analysis complete",
                'reasoning': reasoning_chain,
                'confidence': 0.8
            }

        # Step 5: General conversational response
        return {
            'type': 'general_response',
            'content': user_input,  # Will be processed by conversational engine
            'memories': relevant_memories,
            'confidence': 0.6
        }

    def _formulate_from_plugin_data(
        self,
        user_input: str,
        intent: str,
        entities: Dict[str, Any],
        plugin_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Alice formulates an intelligent response based on plugin data + user's question.
        This is Alice THINKING about the data, not just formatting it.

        Example:
        - User: "should I wear a layer?"
        - Plugin: {temperature: -19.2, condition: "clear"}
        - Alice thinks: "Very cold → yes, definitely wear layers"
        - Returns: {type: 'weather_advice', ...}

        Args:
            user_input: User's original question
            intent: Classified intent
            entities: Extracted entities
            plugin_data: Data from plugin execution

        Returns:
            Structured thought about how to answer based on data
        """
        input_lower = user_input.lower()
        umbrella_aliases = ['umbrella', 'umbrela', 'umberella', 'umbralla']

        if self._is_location_query(user_input):
            return {
                'type': 'location_report',
                **self._build_location_payload(),
                'confidence': 0.98,
                'source': 'context_engine',
            }

        # Detect weather by CONTENT as well as intent — NLP sometimes misfires
        # (e.g. intent='music:pause') but the WeatherPlugin still succeeds.
        _is_weather_data = (
            intent.startswith('weather')
            or 'forecast' in plugin_data
            or ('temperature' in plugin_data and 'condition' in plugin_data)
        )

        # Weather-related formulations
        if _is_weather_data:
            temp = plugin_data.get('temperature')
            condition = plugin_data.get('condition', '').lower()
            location = plugin_data.get('location', '')
            forecast = plugin_data.get('forecast')  # Check for forecast data

            # Round temperature to whole number for cleaner display
            if temp is not None:
                temp = round(temp)

            # Check if we recently gave weather info (avoid repetition)
            # "Still" only makes sense when the immediately preceding turn was
            # also a weather response — not whenever weather appeared anywhere.
            recent_weather_given = False
            if hasattr(self, 'conversation_summary') and self.conversation_summary:
                last_turn = self.conversation_summary[-1]
                if last_turn.get('intent', '').startswith('weather'):
                    recent_weather_given = True

            # User asking about clothing/layers/what to wear
            if any(word in input_lower for word in ['wear', 'layer', 'coat', 'jacket', 'dress', 'clothing', 'bring', 'scarf', 'hat', 'gloves', 'boots', 'sweater', 'hoodie'] + umbrella_aliases):
                # Detect what specific item they're asking about
                clothing_item = None
                item_keywords = {
                    'scarf': ['scarf', 'scarves'],
                    'coat': ['coat'],
                    'jacket': ['jacket'],
                    'hat': ['hat', 'beanie', 'toque'],
                    'gloves': ['glove', 'gloves', 'mitten', 'mittens'],
                    'boots': ['boot', 'boots'],
                    'sweater': ['sweater', 'jumper'],
                    'hoodie': ['hoodie', 'sweatshirt'],
                    'umbrella': umbrella_aliases,
                    'layers': ['layer', 'layers']
                }

                for item, keywords in item_keywords.items():
                    if any(kw in input_lower for kw in keywords):
                        clothing_item = item
                        break

                # Handle forecast-based advice (weekly forecast)
                if forecast and isinstance(forecast, list) and len(forecast) > 0:
                    # Analyze the week's temperature range
                    all_temps = []
                    for day in forecast:
                        if 'low' in day:
                            all_temps.append(round(day['low']))
                        if 'high' in day:
                            all_temps.append(round(day['high']))

                    if all_temps:
                        min_temp = min(all_temps)
                        max_temp = max(all_temps)

                        # Just pass the data - let A.L.I.C.E formulate the response
                        return {
                            'type': 'weather_advice',
                            'temperature': min_temp,
                            'temp_range': f"{min_temp}°C to {max_temp}°C",
                            'location': location,
                            'is_forecast': True,
                            'clothing_item': clothing_item,  # Pass the item for context
                            'user_question': user_input,  # Let LLM see original question
                            'confidence': 0.95
                        }

                # Handle current weather advice
                elif temp is not None:
                    # Just pass the data - let A.L.I.C.E formulate the response
                    return {
                        'type': 'weather_advice',
                        'temperature': temp,
                        'condition': condition,
                        'location': location,
                        'clothing_item': clothing_item,  # Pass the item for context
                        'user_question': user_input,  # Let LLM see original question
                        'is_followup': recent_weather_given,
                        'confidence': 0.95
                    }

            # User asking about specific conditions (rain, snow, etc.)
            elif any(word in input_lower for word in ['rain', 'snow', 'storm', 'sunny', 'cloud']):
                # Check if question word suggests yes/no answer
                is_question = any(word in input_lower for word in ['will', 'is', 'going to', 'gonna'])

                if is_question:
                    # Alice provides yes/no answer with reasoning
                    if 'rain' in input_lower:
                        will_rain = 'rain' in condition or 'drizzle' in condition or 'shower' in condition
                        return {
                            'type': 'weather_prediction',
                            'answer': 'yes' if will_rain else 'no',
                            'condition': condition,
                            'location': location,
                            'confidence': 0.9
                        }
                    elif 'snow' in input_lower:
                        will_snow = 'snow' in condition
                        return {
                            'type': 'weather_prediction',
                            'answer': 'yes' if will_snow else 'no',
                            'condition': condition,
                            'location': location,
                            'confidence': 0.9
                        }

            # General weather query - provide comprehensive info
            if temp is not None:
                return {
                    'type': 'weather_report',
                    'temperature': temp,
                    'condition': condition,
                    'location': location,
                    'full_data': plugin_data,
                    'is_followup': recent_weather_given,
                    'confidence': 0.9
                }

            # Forecast data (no current temperature) — Alice formats directly
            if forecast and isinstance(forecast, list) and len(forecast) > 0:
                return {
                    'type': 'weather_forecast',
                    'forecast': forecast,
                    'location': location,
                    'user_input': user_input,
                    'confidence': 0.9
                }

            # No data at all
            return {
                'type': 'operation_failure',
                'operation': 'weather_lookup',
                'error': 'no data returned',
                'confidence': 0.5
            }

        # Note/file operations — detect by intent OR by the action the plugin returned
        # (NLP occasionally misfires but the plugin still succeeds; always trust the action)
        _NOTE_ACTIONS = {
            'count_notes', 'list_notes', 'get_note_content', 'summarize_note',
            'create_note', 'delete_note', 'edit_note', 'append_note', 'add_to_note',
            'search_notes', 'search_notes_content', 'list_archived_notes',
            'pin_note', 'unpin_note', 'archive_note', 'unarchive_note',
            'get_note_title', 'link_notes', 'set_priority', 'set_category',
        }
        _action_from_data = plugin_data.get('action', plugin_data.get('operation', 'unknown'))
        if intent.startswith('note') or intent.startswith('file') or _action_from_data in _NOTE_ACTIONS:
            action = _action_from_data
            success = plugin_data.get('success', False)

            if action == 'count_notes':
                return {
                    'type': 'notes_count',
                    'total': plugin_data.get('total', 0),
                    'todos': plugin_data.get('todos', 0),
                    'ideas': plugin_data.get('ideas', 0),
                    'meetings': plugin_data.get('meetings', 0),
                    'pinned': plugin_data.get('pinned', 0),
                    'archived': plugin_data.get('archived', 0),
                    'confidence': 0.95, 
                }
            if action == 'list_notes':
                return {
                    'type': 'notes_listing',
                    'note_count': plugin_data.get('count', 0),
                    'notes': plugin_data.get('notes', []),
                    'has_more': plugin_data.get('has_more', False),
                    'confidence': 0.95
                }
            if action == 'get_note_content':
                return {
                    'type': 'note_content',
                    'title': plugin_data.get('note_title', ''),
                    'content': plugin_data.get('content', ''),
                    'tags': plugin_data.get('tags', []),
                    'confidence': 0.95
                }
            if action == 'summarize_note':
                return {
                    'type': 'note_summary',
                    'title': plugin_data.get('note_title', ''),
                    'summary': plugin_data.get('summary', {}),
                    'confidence': 0.95
                }

            if success:
                return {
                    'type': 'operation_success',
                    'operation': action,
                    'details': plugin_data,
                    'user_question': user_input,
                    'confidence': 0.95
                }
            else:
                return {
                    'type': 'operation_failure',
                    'operation': action,
                    'error': plugin_data.get('error', 'Operation failed'),
                    'user_question': user_input,
                    'confidence': 0.9
                }

        # Calendar events
        elif intent.startswith('calendar') or intent.startswith('schedule'):
            events = plugin_data.get('events', [])
            if events:
                return {
                    'type': 'calendar_info',
                    'event_count': len(events),
                    'events': events,
                    'confidence': 0.95
                }

        # Generic plugin response - Alice provides what she can
        return {
            'type': 'plugin_result',
            'data': plugin_data,
            'confidence': 0.7
        }

    def _apply_reasoning(
        self,
        user_input: str,
        entities: Dict[str, Any],
        context: Any
    ) -> list:
        """
        Alice's reasoning logic (coded, not delegated to LLM).
        Returns chain of reasoning steps.

        Args:
            user_input: User's message
            entities: Extracted entities
            context: Conversational context

        Returns:
            List of reasoning steps
        """
        reasoning_chain = []

        # Example: Debugging logic
        if 'debug' in user_input.lower() or 'error' in user_input.lower():
            reasoning_chain.append("User needs debugging help")
            reasoning_chain.append(f"Looking for error patterns in context")

            # Alice would apply her debugging rules here
            reasoning_chain.append("Analyzing recent conversation for code context")

        # Example: Explanation logic
        elif 'how' in user_input.lower() or 'why' in user_input.lower():
            reasoning_chain.append("User wants an explanation")
            reasoning_chain.append("Need to provide logical breakdown")

        # Default: general analysis
        else:
            reasoning_chain.append("Analyzing user request")
            reasoning_chain.append("Formulating response based on context")

        return reasoning_chain

    def _identify_capability_from_input(self, user_input: str) -> Optional[str]:
        """
        Identify which capability the user is asking about.

        Args:
            user_input: User's message

        Returns:
            Capability key or None
        """
        input_lower = user_input.lower()

        # Map input patterns to capability keys
        capability_patterns = {
            'codebase_access': ['code', 'codebase', 'source', 'file', 'python'],
            'email_access': ['email', 'gmail', 'mail', 'inbox'],
            'calendar': ['calendar', 'schedule', 'event', 'meeting'],
            'weather': ['weather', 'forecast', 'temperature'],
            'file_operations': ['file', 'document', 'folder'],
            'notes': ['note', 'notes'],
            'maps': ['directions', 'map', 'location', 'navigate'],
            'time': ['time', 'date', 'timer', 'reminder'],
            'web_search': ['search', 'look up', 'find'],
            'memory': ['remember', 'recall', 'memory'],
            'reasoning': ['analyze', 'think', 'reason', 'debug'],
            'self_reflection': ['how do you', 'explain your', 'your system']
        }

        # Find best match
        for capability_key, patterns in capability_patterns.items():
            if any(pattern in input_lower for pattern in patterns):
                return capability_key

        return None

    def _is_location_query(self, user_input: str) -> bool:
        """Detect explicit location requests that should bypass LLM phrasing."""
        text = str(user_input or "").lower()
        location_phrases = [
            'what is my location',
            'where am i',
            'where are we',
            'my location',
            'current location',
            'what city am i in',
            'what country am i in',
            'what is my city',
        ]
        return any(p in text for p in location_phrases)

    def _build_location_payload(self) -> Dict[str, Any]:
        """Build deterministic location data from context engine state."""
        prefs = getattr(getattr(self, 'context', None), 'user_prefs', None)
        city = str(getattr(prefs, 'city', '') or '').strip()
        country = str(getattr(prefs, 'country', '') or '').strip()
        location = str(getattr(prefs, 'location', '') or '').strip()
        timezone = str(getattr(prefs, 'timezone', '') or '').strip()

        if not location:
            location = ', '.join([x for x in [city, country] if x])

        return {
            'location_known': bool(location),
            'location': location,
            'city': city,
            'country': country,
            'timezone': timezone,
        }

    def _clamp_final_response(
        self,
        response: str,
        *,
        tone: str = "helpful",
        response_type: str = "",
        route: str = "unknown",
        user_input: str = "",
    ) -> str:
        """Final clamp for user-facing responses, especially LLM-originated text."""
        text = str(response or "").strip()
        if not text:
            return "I need one more detail to answer correctly."

        # Normalize shell quoting and whitespace noise.
        text = text.strip().strip('"').strip("'")
        text = re.sub(r"\s+", " ", text).strip()

        # Remove common filler/over-softening prefixes.
        text = re.sub(
            r"^(?:sure|of course|absolutely|definitely|certainly|no problem|happy to help)[:,.!]?\s+",
            "",
            text,
            flags=re.IGNORECASE,
        )

        # Reject meta-assistant leakage and replace with Alice-safe fallback.
        lower = text.lower()
        if "as an ai" in lower or "language model" in lower:
            if response_type == "clarification_prompt":
                return "Please clarify the exact outcome you want so I can route this correctly."
            return "I can help with that. Tell me the exact result you want."

        # Tone-aware trim: keep professional tones concise.
        max_chars = {
            "clarification_prompt": 140,
            "operation_success": 160,
            "operation_failure": 180,
            "knowledge_answer": 220,
            "wake_word_ack": 80,
            "farewell": 100,
        }.get(response_type, 320)

        if "professional" in tone or "precise" in tone:
            max_chars = min(max_chars, 220)

        if len(text) > max_chars:
            text = text[: max_chars - 3].rstrip() + "..."

        # Enforce one-line concise output for micro conversational routes.
        if route in {"wake_word", "farewell", "greeting", "ollama_phrase_micro"}:
            text = text.split("\n", 1)[0].strip()

        return text or "I need one more detail to answer correctly."

    def _alice_direct_phrase(self, response_type: str, alice_response: Dict[str, Any]) -> Optional[str]:
        """
        Alice phrases structured/factual responses entirely on her own.
        Returns None only for truly open-ended types that benefit from Ollama.
        Ollama is NEVER in control here — it is only a teacher when Alice lacks
        phrasing experience for a novel conversational pattern.
        """
        # ── Notes ────────────────────────────────────────────────────────────
        if response_type == 'notes_count':
            total = alice_response.get('total', 0)
            extras = []
            if alice_response.get('todos'):    extras.append(f"{alice_response['todos']} to-do")
            if alice_response.get('ideas'):    extras.append(f"{alice_response['ideas']} idea")
            if alice_response.get('meetings'): extras.append(f"{alice_response['meetings']} meeting")
            if alice_response.get('pinned'):   extras.append(f"{alice_response['pinned']} pinned")
            if alice_response.get('archived'): extras.append(f"{alice_response['archived']} archived")
            detail = f" ({', '.join(extras)})" if extras else ""
            if total == 0:
                return "You don't have any notes yet."
            return f"You have {total} note{'s' if total != 1 else ''}{detail}."

        if response_type == 'notes_listing':
            notes = alice_response.get('notes', [])
            count = alice_response.get('note_count', len(notes))
            if not notes:
                return "You don't have any notes yet."
            titles = [n.get('title', 'Untitled') if isinstance(n, dict) else str(n) for n in notes[:10]]
            has_more = alice_response.get('has_more', False)
            lines = [f"**You have {count} note{'s' if count != 1 else ''}:**", ""]
            for i, t in enumerate(titles, 1):
                lines.append(f"{i}. {t}")
            if has_more:
                lines.append(f"\n*…and {count - len(titles)} more.*")
            return "\n".join(lines)

        if response_type == 'note_content':
            title = alice_response.get('title', 'that note')
            content_body = alice_response.get('content', '')
            tags = alice_response.get('tags', [])
            header = f"## {title}\n" if title else ""
            tag_line = f"\n---\n*Tags: {', '.join(tags)}*" if tags else ""
            # Don't repeat the title as body — if a note's content was never
            # filled in and equals its title, show a polite empty-note message.
            body = content_body if content_body and content_body.strip() != title.strip() else ""
            if not body:
                return f"{header.strip()}\n\n*(This note has no content yet.)*".strip()
            return f"{header}\n{body}{tag_line}".strip()

        if response_type == 'note_summary':
            title = alice_response.get('title', 'that note')
            summary = alice_response.get('summary', {})
            if isinstance(summary, dict):
                lines = [f"## Summary: {title}", ""]
                if summary.get('word_count'):
                    lines.append(f"- **{summary['word_count']} words**")
                if summary.get('key_points'):
                    lines.append("\n**Key points:**")
                    for pt in summary['key_points'][:5]:
                        lines.append(f"- {pt}")
                return "\n".join(lines) if len(lines) > 2 else f"Summary of **{title}** — no structured data available."
            return f"**{title}:** {summary}"

        if response_type == 'operation_success':
            op = str(alice_response.get('operation') or 'operation').replace('_', ' ').strip()
            details = alice_response.get('details', {}) if isinstance(alice_response.get('details', {}), dict) else {}
            subject = str(details.get('note_title') or details.get('title') or details.get('name') or '').strip()
            if subject:
                return f"Done: {op} for '{subject}'."
            return f"Done: {op}."

        if response_type == 'operation_failure':
            op = str(alice_response.get('operation') or 'operation').replace('_', ' ').strip()
            err = str(alice_response.get('error') or '').strip()
            if err:
                return f"I couldn't complete {op}: {err}."
            return f"I couldn't complete {op}."

        if response_type == 'knowledge_answer':
            answer = str(alice_response.get('answer') or alice_response.get('content') or '').strip()
            if answer:
                return self._clamp_final_response(
                    answer,
                    tone='professional and precise',
                    response_type='knowledge_answer',
                    route='direct',
                )
            question = str(alice_response.get('question') or '').strip()
            if question:
                return f"I need one concrete fact target to answer: {question}"
            return "I need a specific fact-focused question to answer directly."

        if response_type == 'operation_status':
            status = str(alice_response.get('status') or 'in progress').strip().lower()
            detail = str(alice_response.get('detail') or '').strip()
            msg = f"Status: {status}."
            if detail:
                msg += f" {detail}"
            return msg

        if response_type == 'location_report':
            location_known = bool(alice_response.get('location_known'))
            location = str(alice_response.get('location', '') or '').strip()
            city = str(alice_response.get('city', '') or '').strip()
            country = str(alice_response.get('country', '') or '').strip()
            timezone = str(alice_response.get('timezone', '') or '').strip()

            if location_known:
                parts = [f"Your current location is **{location}**."]
                if timezone:
                    parts.append(f"Timezone: **{timezone}**.")
                return " ".join(parts)

            fallback = []
            if city:
                fallback.append(city)
            if country:
                fallback.append(country)
            if fallback:
                return f"I have partial location data: **{', '.join(fallback)}**."
            return "I don't have a reliable location set right now. You can tell me your city to set it explicitly."

        # ── Weather ──────────────────────────────────────────────────────────
        def _format_temp(value):
            """Format temperature with proper minus sign (− instead of -)"""
            if value is None:
                return None
            # Convert to numeric value
            if isinstance(value, str):
                # Replace any Unicode minus with hyphen for parsing
                value = value.replace("−", "-")
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    return str(value)  # Return as-is if can't parse
            
            rounded = int(round(float(value)))
            if rounded < 0:
                # Use proper Unicode minus sign (U+2212) instead of ASCII hyphen-minus (U+002D)
                return f"−{abs(rounded)}"
            return str(rounded)
        
        if response_type == 'weather_report':
            temp = alice_response.get('temperature')
            condition = alice_response.get('condition', '')
            location = alice_response.get('location', '')
            is_followup = alice_response.get('is_followup', False)
            
            loc_str = f" in {location}" if location else ""
            cond_cap = condition.capitalize() if condition else ""
            temp_str = f"{_format_temp(temp)}°C" if temp is not None else ""
            
            if is_followup:
                return (f"Still **{cond_cap}**{loc_str} {temp_str}." if temp is not None
                        else f"Still **{cond_cap}**{loc_str}.")
            return (f"**{temp_str}** {cond_cap}{loc_str}." if temp is not None
                    else f"**{cond_cap}**{loc_str}.")

        if response_type == 'weather_advice':
            raw_temp = alice_response.get('temperature')
            condition_raw = str(alice_response.get('condition', '') or '').lower().strip()
            location = str(alice_response.get('location', '') or '').strip()
            item = str(alice_response.get('clothing_item', 'jacket') or 'jacket').lower().strip()
            rainy = alice_response.get('rainy')
            force_yes_no = bool(alice_response.get('force_yes_no'))

            # Normalize frequent condition typos from upstream providers.
            _cond_fixups = {
                'drizzleing': 'drizzling',
                'drizzeling': 'drizzling',
                'drizzel': 'drizzle',
                'lgiht': 'light',
            }
            condition = condition_raw
            for bad, good in _cond_fixups.items():
                condition = condition.replace(bad, good)

            def _item_phrase(it: str) -> str:
                if it == 'umbrella':
                    return 'an umbrella'
                if it.startswith(('a ', 'an ')):
                    return it
                return f"an {it}" if it[:1] in {'a', 'e', 'i', 'o', 'u'} else f"a {it}"

            rainy_by_condition = any(w in condition for w in ['rain', 'drizzle', 'drizzl', 'shower', 'storm', 'thunder'])
            if rainy is None:
                rainy = rainy_by_condition

            temp_val = None
            if raw_temp is not None:
                try:
                    temp_val = float(raw_temp)
                except (TypeError, ValueError):
                    temp_val = None

            def _recommendation() -> tuple[bool, str]:
                if item == 'umbrella':
                    return bool(rainy), (
                        f"{condition} is expected" if condition else "rain is expected"
                    ) if rainy else (
                        f"it is {condition}" if condition else "rain is not expected"
                    )

                threshold = {
                    'coat': 10.0,
                    'jacket': 14.0,
                    'layers': 16.0,
                    'hoodie': 15.0,
                    'sweater': 15.0,
                    'scarf': 8.0,
                    'hat': 7.0,
                    'gloves': 5.0,
                    'boots': 6.0,
                }.get(item, 14.0)

                if temp_val is None:
                    return True, (f"it is {condition}" if condition else "conditions look cool")
                return temp_val <= threshold, f"it is about {_format_temp(temp_val)}°C"

            should_bring, reason = _recommendation()
            place = f" in {location}" if location else ""
            item_text = _item_phrase(item)

            if force_yes_no:
                if should_bring:
                    return f"Yes, bring {item_text} because {reason}{place}."
                return f"No, you likely do not need {item_text} because {reason}{place}."

            if should_bring:
                return f"Bring {item_text}; {reason}{place}."
            return f"You likely do not need {item_text}; {reason}{place}."

        if response_type == 'weather_prediction':
            answer = alice_response.get('answer', '').capitalize()
            condition = alice_response.get('condition', '')
            return f"{answer}. Current condition: {condition}." if condition else f"{answer}."

        if response_type == 'weather_forecast':
            from datetime import datetime as _dt, timedelta as _td
            forecast = alice_response.get('forecast', [])
            location = alice_response.get('location', '')
            raw_input = alice_response.get('user_input', '').lower()

            if not forecast:
                return "I don't have forecast data available right now."

            wants_weekend = 'weekend' in raw_input
            wants_tomorrow = 'tomorrow' in raw_input
            _days_kw = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                        'friday': 4, 'saturday': 5, 'sunday': 6}
            wants_specific_day = next((d for d in _days_kw if d in raw_input), None)

            loc_str = f" for {location}" if location else ''
            today = _dt.now().date()

            def _day_label(d) -> str:
                is_today = d == today
                is_tmr   = d == today + _td(days=1)
                return 'Today' if is_today else ('Tomorrow' if is_tmr else d.strftime('%A'))

            def _fmt_table_row(d, day: dict) -> str:
                label = _day_label(d)
                high  = day.get('high')
                low   = day.get('low')
                cond  = day.get('condition', 'unknown').title()
                temp  = f"{_format_temp(low)}° to {_format_temp(high)}°C" if (high is not None and low is not None) else "—"
                return f"| {label} | {cond} | {temp} |"

            TABLE_SEP = "| --- | --- | --- |"

            # ── Specific weekday ─────────────────────────────────────────────
            if wants_specific_day:
                target_wd = _days_kw[wants_specific_day]
                for day in forecast:
                    try:
                        d = _dt.strptime(day.get('date', ''), '%Y-%m-%d').date()
                        if d.weekday() == target_wd:
                            high, low = day.get('high'), day.get('low')
                            cond = day.get('condition', 'unknown').title()
                            temp = f"{_format_temp(low)}° to {_format_temp(high)}°C" if (high is not None and low is not None) else ''
                            return f"**{_day_label(d)}{loc_str}** {cond}{', ' + temp if temp else ''}"
                    except Exception:
                        pass

            # ── Weekend: two-row table ───────────────────────────────────────
            if wants_weekend:
                weekend_days = []
                for day in forecast:
                    try:
                        d = _dt.strptime(day.get('date', ''), '%Y-%m-%d').date()
                        if d.weekday() in (5, 6):
                            weekend_days.append((d, day))
                    except Exception:
                        pass
                if weekend_days:
                    lines = [
                        f"**Weekend forecast{loc_str}**", "",
                        "| Day | Condition | Temp |",
                        TABLE_SEP,
                    ]
                    for d, day in weekend_days:
                        lines.append(_fmt_table_row(d, day))
                    return '\n'.join(lines)

            # ── Tomorrow: single bold line ───────────────────────────────────
            if wants_tomorrow:
                tomorrow_str = (today + _td(days=1)).strftime('%Y-%m-%d')
                for day in forecast:
                    if day.get('date') == tomorrow_str:
                        d = _dt.strptime(tomorrow_str, '%Y-%m-%d').date()
                        high, low = day.get('high'), day.get('low')
                        cond = day.get('condition', 'unknown').title()
                        temp = f"{_format_temp(low)}° to {_format_temp(high)}°C" if (high is not None and low is not None) else ''
                        return f"**Tomorrow{loc_str}** {cond}{', ' + temp if temp else ''}"

            # ── Full 7-day table ─────────────────────────────────────────────
            lines = [
                f"**7-day forecast{loc_str}**", "",
                "| Day | Condition | Temp |",
                TABLE_SEP,
            ]
            for day in forecast[:7]:
                try:
                    d = _dt.strptime(day.get('date', ''), '%Y-%m-%d').date()
                    lines.append(_fmt_table_row(d, day))
                except Exception:
                    pass
            return '\n'.join(lines) if len(lines) > 4 else f"Forecast{loc_str} available but couldn't be formatted."

        # ── Capability ───────────────────────────────────────────────────────
        if response_type == 'capability_answer':
            can_do = alice_response.get('can_do', False)
            details = alice_response.get('details', '')
            ops = alice_response.get('operations', [])
            base = "Yes, I can do that." if can_do else "No, I can't do that."
            if details:
                base += f" {details}"
            if ops:
                base += f" Available operations: {', '.join(ops[:5])}."
            return base

        # ── Self-analysis / code ─────────────────────────────────────────────
        if response_type == 'self_analysis':
            total_files = alice_response.get('total_files', 0)
            analyzed = alice_response.get('analyzed_files', [])
            points = alice_response.get('architecture_points', [])
            lines = [f"I have {total_files} Python files in my codebase."]
            for f in analyzed[:5]:
                lines.append(f"  • {f['path']}: {f.get('lines', 0)} lines")
            for p in points[:5]:
                lines.append(f"  • {p}")
            return "\n".join(lines)

        if response_type == 'code_explanation':
            name = alice_response.get('file_name', 'file')
            lines = alice_response.get('lines', 0)
            module_type = alice_response.get('module_type', 'code')
            preview = alice_response.get('content_preview', '')[:200]
            return f"{name} is a {module_type} file with {lines} lines.{(' ' + preview) if preview else ''}"

        if response_type == 'codebase_listing':
            heading = str(alice_response.get('heading') or 'My codebase')
            total_files = int(alice_response.get('total_files', 0) or 0)
            display_limit = int(alice_response.get('display_limit', 25) or 25)
            files = list(alice_response.get('files') or [])

            lines = [f" **{heading}** ({total_files} files):", ""]
            for item in files[:display_limit]:
                path = str(item.get('path', '')).strip()
                module_type = str(item.get('module_type', 'unknown')).strip()
                if not path:
                    continue
                lines.append(f"- `{path}` ({module_type})")

            if total_files > display_limit:
                lines.append(
                    f"\n... and {total_files - display_limit} more files. Ask me about any file to see its details."
                )

            return "\n".join(lines)

        if response_type == 'code_summaries':
            items = list(alice_response.get('items') or [])
            total_files = int(alice_response.get('total_files', len(items)) or len(items))

            lines = [f" **File Summaries** ({total_files} files):", ""]
            for item in items[:25]:
                summary = str(item.get('summary') or '').strip()
                if summary:
                    lines.append(summary)
                    lines.append("")

            return "\n".join(lines).rstrip()

        if response_type == 'clarification_prompt':
            options = [
                str(x).strip()
                for x in list(alice_response.get('options') or [])[:3]
                if str(x).strip()
            ]
            pronouns = [
                str(x).strip()
                for x in list(alice_response.get('pronouns') or [])[:2]
                if str(x).strip()
            ]
            if options:
                return f"Do you mean {', '.join(options[:-1]) + ' or ' + options[-1] if len(options) > 1 else options[0]}?"
            if pronouns:
                return f"What does '{pronouns[0]}' refer to in your request?"
            return "Please clarify the exact outcome you want so I can route this correctly."

        # Open-ended types — return None to let Ollama assist Alice
        return None

    def _generate_natural_response(
        self,
        alice_response: Dict[str, Any],
        tone: str,
        context: Any,
        user_input: str
    ) -> str:
        """
        Generate natural language response.  Alice is always in control.

        Priority order:
        1. Alice's learned phrasing patterns  (fastest — zero LLM)
        2. Alice's built-in structured phrasing (_alice_direct_phrase)
        3. Ollama assists Alice for open-ended/conversational types only,
           then Alice LEARNS from it so she needs Ollama less each time.

        Ollama never overrides Alice.  It is strictly a teacher, not a speaker.
        """
        response_type = alice_response.get('type')
        content = alice_response.get('content') or alice_response

        # ── Step 1: Alice phrases it directly from structured data ───────────
        # This MUST come first so context-aware filters (weekend, tomorrow, etc.)
        # always produce the correct, fresh answer.  Learned patterns are only
        # used for open-ended types where _alice_direct_phrase returns None.
        direct = self._alice_direct_phrase(response_type, alice_response)
        if direct:
            logger.info(f"[ALICE] Phrased '{response_type}' directly (no LLM needed)")
            return self._clamp_final_response(
                direct,
                tone=tone,
                response_type=str(response_type or ''),
                route='direct',
                user_input=user_input,
            )

        # ── Step 2: Learned pattern for open-ended types (Alice independent) ─
        if self.phrasing_learner.can_phrase_myself(alice_response, tone):
            natural_response = self.phrasing_learner.phrase_myself(alice_response, tone)
            logger.info(f"[ALICE] Phrased '{response_type}' from learned pattern")
            return self._clamp_final_response(
                natural_response,
                tone=tone,
                response_type=str(response_type or ''),
                route='learned_pattern',
                user_input=user_input,
            )

        if getattr(self, 'strict_no_llm', False):
            logger.info(f"[ALICE] Strict mode fallback for '{response_type}'")
            if response_type == 'knowledge_answer':
                return "I can answer in strict mode, but I need a more specific target question."
            if response_type == 'reasoning_result':
                return str(alice_response.get('conclusion') or 'Reasoning complete.')
            if response_type == 'operation_success':
                op = str(alice_response.get('operation') or 'operation').replace('_', ' ')
                return f"Completed: {op}."
            if response_type == 'operation_failure':
                op = str(alice_response.get('operation') or 'operation').replace('_', ' ')
                err = str(alice_response.get('error') or '').strip()
                return f"Failed: {op}. {err}".strip()
            if response_type == 'clarification_prompt':
                return "Please clarify the exact outcome you want so I can route this safely."
            return self._clamp_final_response(
                str(content),
                tone=tone,
                response_type=str(response_type or ''),
                route='strict_no_llm',
                user_input=user_input,
            )

        # ── Step 3: Open-ended — Alice asks Ollama for phrasing help ─────────
        # Ollama is a teacher here, not a speaker. Alice learns from the reply.
        logger.info(f"[ALICE] Asking Ollama to help phrase '{response_type}' (Alice will learn from this)")

        # Build a concise prompt from the structured content so Ollama can help
        # Alice phrasing it naturally — she provides the facts, Ollama provides fluency.
        if response_type == 'knowledge_answer':
            question = alice_response.get('question', '')
            intent = alice_response.get('intent', '')
            entities_known = [e for e in self.knowledge_engine.entities.values()
                              if e.name.lower() in question.lower()]
            relationships: list = []
            for ent in entities_known:
                relationships.extend(self.knowledge_engine.get_relationships_for_entity(ent.name))
            if entities_known:
                entity_info = ", ".join([f"{e.name} ({e.type})" for e in entities_known[:3]])
                content_str = f"From my knowledge: {entity_info}"
                if relationships:
                    rel = relationships[0]
                    content_str += f". {rel['subject']} {rel['predicate']} {rel['object']}"
            else:
                topic = intent.split(':')[0] if ':' in intent else intent
                content_str = f"Based on what I've learned about {topic}, I can help with: {question}"

        elif response_type == 'reasoning_result':
            conclusion = alice_response.get('conclusion', '')
            content_str = f"Conclusion: {conclusion}"

        elif response_type == 'operation_success':
            op = alice_response.get('operation', 'operation')
            details = alice_response.get('details', {})
            title = (details.get('note_title') or details.get('title', '')) if isinstance(details, dict) else ''
            title_str = f" '{title}'" if title else ''
            user_q = alice_response.get('user_question', '')
            content_str = (
                f"Action completed successfully: {op.replace('_', ' ')}{title_str}. "
                f"User asked: '{user_q}'. Confirm this briefly and naturally in one sentence."
            )

        elif response_type == 'operation_failure':
            op = alice_response.get('operation', 'that')
            error = alice_response.get('error', '')
            user_q = alice_response.get('user_question', '')
            error_str = f" Reason: {error}" if error and error not in ('Operation failed', 'unknown') else ''
            content_str = (
                f"Action failed: {op.replace('_', ' ')}.{error_str} "
                f"User asked: '{user_q}'. Explain this briefly and naturally, and offer to help in one sentence."
            )

        elif response_type == 'weather_advice':
            # Give Ollama the bare facts so it can phrase the advice naturally.
            raw_temp = alice_response.get('temperature')
            condition = alice_response.get('condition', '')
            location = alice_response.get('location', '')
            item = alice_response.get('clothing_item', 'jacket')
            rainy = alice_response.get('rainy')  # set for umbrella queries
            force_yes_no = bool(alice_response.get('force_yes_no'))
            user_q = alice_response.get('user_question', '')
            loc_str = f" in {location}" if location else ""
            temp_str = f"{round(raw_temp)}°C{loc_str}" if raw_temp is not None else f"unknown temperature{loc_str}"
            if rainy is False:
                content_str = (
                    f"Current weather: {temp_str}, {condition}. "
                    f"User asked: '{user_q}'. No rain is expected — advise on whether an umbrella is needed."
                )
            else:
                content_str = (
                    f"Current weather: {temp_str}, {condition}. "
                    f"User asked: '{user_q}'. Give a short, direct answer about whether to bring/wear '{item}'."
                )
            if force_yes_no:
                content_str += " Start with exactly 'Yes,' or 'No,' and then one brief reason."

        elif response_type == 'clarification_prompt':
            payload = {
                'reason': str(alice_response.get('reason') or '').strip().lower(),
                'options': [
                    str(x).strip()
                    for x in list(alice_response.get('options') or [])[:5]
                    if str(x).strip()
                ],
                'details': [
                    str(x).strip()
                    for x in list(alice_response.get('details') or [])[:3]
                    if str(x).strip()
                ],
                'pronouns': [
                    str(x).strip()
                    for x in list(alice_response.get('pronouns') or [])[:3]
                    if str(x).strip()
                ],
                'user_input': user_input,
            }
            content_str = (
                "Generate one concise clarification question for the user from this structured payload: "
                + json.dumps(payload, ensure_ascii=True)
            )

        else:
            # general_response and any other open-ended types
            content_str = str(content)

        thought_content = alice_response

        try:
            phrase_call_type = (
                LLMCallType.PHRASE_MICRO
                if len(content_str) <= 160
                else LLMCallType.PHRASE_STRUCTURED
            )
            phrasing_context = {
                'alice_thought': content_str,
                'structured_payload': content_str,
                'tone': tone,
                'user_name': '',
                'allow_user_name': False,
            }
            llm_response = self.llm_gateway.request(
                prompt=content_str,
                call_type=phrase_call_type,
                context=phrasing_context,
                user_input=user_input,
            )
            if llm_response.success and llm_response.response:
                natural_response = self._clamp_final_response(
                    llm_response.response,
                    tone=tone,
                    response_type=str(response_type or ''),
                    route=(
                        'ollama_phrase_micro'
                        if phrase_call_type == LLMCallType.PHRASE_MICRO
                        else 'ollama_phrase_structured'
                    ),
                    user_input=user_input,
                )
                # Alice learns from Ollama's phrasing so she needs it less next time
                self.phrasing_learner.record_phrasing(
                    alice_thought=thought_content,
                    ollama_phrasing=natural_response,
                    context={
                        'tone': tone,
                        'intent': context.current_intent if hasattr(context, 'current_intent') else 'unknown',
                        'user_input': user_input,
                        'source': 'ollama_teacher',
                    },
                )
                if self.phrasing_learner.can_phrase_myself(thought_content, tone):
                    self._think(f"Alice learned '{response_type}' — can now phrase independently!")
                # Prepend an empathy phrase when the interaction policy asks for it
                # (applies only to open-ended/conversational Ollama-assisted responses).
                _pol = getattr(self, '_last_policy', None)
                if _pol is not None and getattr(_pol, 'add_empathy_prefix', False):
                    _mood = getattr(getattr(self, '_last_perception', None), 'inferred_mood', '')
                    _pfx = (
                        "I'm sorry to hear that — "
                        if _mood == 'frustrated'
                        else "Of course — "
                    )
                    natural_response = _pfx + natural_response
                return self._clamp_final_response(
                    natural_response,
                    tone=tone,
                    response_type=str(response_type or ''),
                    route='ollama_post_policy',
                    user_input=user_input,
                )
            else:
                logger.warning("[ALICE] Ollama phrasing failed — using Alice's direct fallback")
                return self._clamp_final_response(
                    content_str,
                    tone=tone,
                    response_type=str(response_type or ''),
                    route='alice_fallback',
                    user_input=user_input,
                )
        except Exception as e:
            logger.error(f"[ALICE] Error in Ollama phrasing: {e}")
            return self._clamp_final_response(
                content_str,
                tone=tone,
                response_type=str(response_type or ''),
                route='alice_fallback_error',
                user_input=user_input,
            )

    def _register_plugins(self):
        """Register all available plugins"""
        from ai.plugins.rag_indexer_plugin import RAGIndexerPlugin

        # Register NotesPlugin early to ensure it handles note commands before calendar
        self.plugins.register_plugin(NotesPlugin())
        self.plugins.register_plugin(WeatherPlugin())
        self.plugins.register_plugin(MapsPlugin())
        self.plugins.register_plugin(TimePlugin())
        self.plugins.register_plugin(FileOperationsPlugin())
        self.plugins.register_plugin(MemoryPlugin(self.memory))  # Pass existing memory system
        self.plugins.register_plugin(SystemControlPlugin())
        self.plugins.register_plugin(WebSearchPlugin())
        self.plugins.register_plugin(DocumentPlugin())
        self.plugins.register_plugin(CalendarPlugin())
        self.plugins.register_plugin(RAGIndexerPlugin(self.memory))  # RAG document indexer

        logger.info(f"[OK] Registered {len(self.plugins.plugins)} plugins")
    
    def _handle_observer_notification(self, message: str, priority: EventPriority):
        """
        Handle notifications from background observers
        Display/speak them appropriately
        """
        priority_label = {
            EventPriority.LOW: "",
            EventPriority.NORMAL: "",
            EventPriority.HIGH: "",
            EventPriority.CRITICAL: ""
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
        
        merged_suggestions: List[str] = []
        if suggestions:
            # Store pattern ID for tracking acceptance
            pattern, suggestion_text = suggestions[0]
            self._last_suggestion_pattern = pattern.pattern_id
            merged_suggestions.append(str(suggestion_text))
        if self.activity_monitor:
            proactive = self.activity_monitor.proactive_suggestions()
            if proactive:
                merged_suggestions.extend(proactive)
        if self.proactive_interruption_manager:
            selected = self.proactive_interruption_manager.select(merged_suggestions)
            if selected:
                return f" {selected[0]}"
        elif merged_suggestions:
            return f" {merged_suggestions[0]}"
        
        return None

    def _track_activity_signal(self, intent: str, user_input: str) -> None:
        """Track lightweight user activity classes for proactive monitoring."""
        if not self.activity_monitor:
            return
        intent = (intent or "").lower()
        text = (user_input or "").lower()

        if intent.startswith("email:") or "email" in text:
            self.activity_monitor.observe("email")
        if intent.startswith("learning:") or "study" in text or "learn" in text:
            self.activity_monitor.observe("study")
        if "error" in text or "traceback" in text or "debug" in text:
            self.activity_monitor.observe("debug")

    def _collect_secondary_intents(self, nlp_result: Any) -> List[Dict[str, Any]]:
        """Extract secondary/compound intents from NLP modifiers."""
        if not nlp_result or not getattr(nlp_result, 'parsed_command', None):
            return []
        modifiers = getattr(nlp_result.parsed_command, 'modifiers', {}) or {}
        secondary = modifiers.get('secondary_intents', []) or []
        if secondary:
            return [dict(item) for item in secondary if isinstance(item, dict)]

        compound = modifiers.get('compound_frames', []) or []
        out: List[Dict[str, Any]] = []
        for frame in compound:
            if not isinstance(frame, dict):
                continue
            plugin = str(frame.get('plugin') or '').strip()
            action = str(frame.get('action') or '').strip()
            if plugin:
                out.append(
                    {
                        'intent': f"{plugin}:{action or 'general'}",
                        'confidence': float(frame.get('confidence', 0.6) or 0.6),
                        'text': str(frame.get('slot_evidence') or ''),
                    }
                )
        return out

    def _execute_secondary_intents(
        self,
        secondary_intents: List[Dict[str, Any]],
        entities: Dict[str, Any],
        context_summary: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Execute secondary compound intents sequentially using plugin dispatch."""
        if not secondary_intents or not getattr(self, 'plugins', None):
            return []

        outcomes: List[Dict[str, Any]] = []
        for item in secondary_intents[:3]:
            sec_intent = str(item.get('intent') or '').strip()
            sec_text = str(item.get('text') or '').strip() or sec_intent
            if not sec_intent:
                continue
            try:
                result = self.plugins.execute_for_intent(sec_intent, sec_text, entities, context_summary)
                outcomes.append(
                    {
                        'intent': sec_intent,
                        'success': bool(result and result.get('success')),
                        'plugin': (result or {}).get('plugin', ''),
                    }
                )
            except Exception:
                outcomes.append({'intent': sec_intent, 'success': False, 'plugin': ''})
        return outcomes

    def _apply_response_style_constraints(self, response: str) -> str:
        """Apply user preference constraints and adaptive verbosity formatting."""
        if not response:
            return response
        prefs = dict((getattr(self, '_internal_reasoning_state', {}) or {}).get('response_preferences', {}) or {})
        if self.adaptive_response_style:
            try:
                return self.adaptive_response_style.apply_constraints(response, prefs)
            except Exception:
                return response
        return response

    def _handle_advanced_reasoning_queries(self, user_input: str) -> Optional[str]:
        """Handle advanced reasoning prompts directly for transparency and speed."""
        text = str(user_input or '').strip()
        lowered = text.lower()

        if self.system_design_response_guard:
            direct = self.system_design_response_guard.direct_answer(text)
            if direct:
                return direct

        if self.causal_inference_engine and any(k in lowered for k in ('why did', 'root cause', 'cause of', 'why is this failing')):
            analysis = self.causal_inference_engine.infer(text)
            causes = analysis.get('likely_causes', [])
            checks = analysis.get('recommended_checks', [])
            return (
                "Likely causes:\n- "
                + "\n- ".join(causes)
                + "\n\nNext checks:\n- "
                + "\n- ".join(checks)
            )

        if self.hypothetical_scenario_generator and any(k in lowered for k in ('what if', 'scenario', 'hypothetical')):
            scenarios = self.hypothetical_scenario_generator.generate(text, max_scenarios=3)
            if scenarios:
                lines = ["Hypothetical outcomes:"]
                for sc in scenarios:
                    lines.append(f"- {sc.get('name')}: {sc.get('impact')}")
                return "\n".join(lines)

        if self.decision_constraint_solver and 'choose between' in lowered:
            options = [
                {'name': 'option_a', 'speed': 0.9, 'quality': 0.7, 'risk': 0.4},
                {'name': 'option_b', 'speed': 0.7, 'quality': 0.9, 'risk': 0.3},
            ]
            ranked = self.decision_constraint_solver.solve(
                options,
                soft_weights={'quality': 0.55, 'speed': 0.30, 'risk': -0.15},
            )
            if ranked:
                top = ranked[0]
                return f"Constraint analysis suggests {top.get('name')} (score={float(top.get('constraint_score', 0.0)):.2f})."

        return None

    def _handle_operator_request(self, user_input: str) -> Optional[str]:
        """Handle safe operator commands for repository and build/test workflows."""
        text = str(user_input or "").strip()
        lowered = text.lower()

        if lowered.startswith("operator reject "):
            parts = text.split(maxsplit=2)
            if len(parts) < 3:
                return "Usage: operator reject <approval_id>"
            approval_id = parts[2].strip()
            if getattr(self, 'approval_ledger', None):
                self.approval_ledger.reject(
                    approval_id=approval_id,
                    confirmation_text=text,
                    actor="user",
                )
            self.pending_operator_actions.pop(approval_id, None)
            return f"Approval {approval_id} rejected."

        if lowered.startswith("operator approve "):
            parts = text.split(maxsplit=2)
            if len(parts) < 3:
                return "Usage: operator approve <approval_id>"
            approval_id = parts[2].strip()
            pending = self.pending_operator_actions.get(approval_id)
            if not pending:
                return f"No pending operator action found for {approval_id}."

            if getattr(self, 'approval_ledger', None):
                rec = self.approval_ledger.confirm(
                    approval_id=approval_id,
                    confirmation_text=text,
                    actor="user",
                )
                if rec is None:
                    self.pending_operator_actions.pop(approval_id, None)
                    return f"Approval {approval_id} is no longer valid (expired or missing)."

            action = pending.get("action")
            if action == "controlled_commit":
                if not getattr(self, 'operator_workflow', None):
                    return "Operator workflow is not initialized."
                message = pending.get("commit_message") or "operator commit"
                result = self.operator_workflow.run_controlled_commit_workflow(message)
                self.pending_operator_actions.pop(approval_id, None)
                return result.render()

            self.pending_operator_actions.pop(approval_id, None)
            return f"Approved {approval_id}, but no executable action payload was found."

        if any(k in lowered for k in ("operator workflow", "repo health", "repository health check", "run health workflow")):
            if not getattr(self, 'operator_workflow', None):
                return "Operator workflow is not initialized."
            include_tests = any(k in lowered for k in ("with tests", "and tests", "full"))
            if getattr(self, 'roadmap_stack', None):
                def _branch_handler(_step, _state):
                    res = self.git_manager.current_branch()
                    return {"success": res.success, "output": res.output or res.error}

                def _status_handler(_step, _state):
                    res = self.git_manager.status_short()
                    return {"success": res.success, "output": res.output or res.error}

                def _build_handler(_step, _state):
                    res = self.build_runner.run_python_build()
                    return {"success": res.success, "output": res.output or res.error}

                def _tests_handler(_step, _state):
                    res = self.build_runner.run_python_tests()
                    return {"success": res.success, "output": res.output or res.error}

                steps = [
                    {"name": "branch", "tool": "git_branch"},
                    {"name": "status", "tool": "git_status", "depends_on": ["branch"]},
                    {"name": "build", "tool": "py_build", "depends_on": ["status"]},
                ]
                if include_tests:
                    steps.append({"name": "tests", "tool": "py_tests", "depends_on": ["build"]})

                handlers = {
                    "git_branch": _branch_handler,
                    "git_status": _status_handler,
                    "py_build": _build_handler,
                    "py_tests": _tests_handler,
                }
                chain_results = self.roadmap_stack.chain_engine.run(steps, handlers)
                failed = next((r for r in chain_results if not r.get("success", False) and not r.get("skipped", False)), None)
                if failed:
                    replan = self.roadmap_stack.replanner.replan(
                        [str(s.get("name")) for s in steps],
                        str(failed.get("name")),
                        str(failed.get("error") or failed.get("output") or "unknown"),
                    )
                    self._internal_reasoning_state["operator_replan"] = replan

            wf = self.operator_workflow.run_repo_health_workflow(include_tests=include_tests)
            return wf.render()

        if lowered.startswith("git "):
            if not getattr(self, 'git_manager', None):
                return "Git manager is not initialized."

            if lowered.startswith("git status"):
                status = self.git_manager.status_short()
                if not status.success:
                    return f"Git status failed: {status.error or status.output}"
                return status.output or "Working tree is clean."

            if lowered.startswith("git diff"):
                diff_res = self.git_manager.diff_unstaged()
                if not diff_res.success:
                    return f"Git diff failed: {diff_res.error or diff_res.output}"
                out = diff_res.output or "No unstaged diff."
                return "\n".join(out.splitlines()[:200])

            if lowered.startswith("git log"):
                log_res = self.git_manager.recent_commits(limit=8)
                if not log_res.success:
                    return f"Git log failed: {log_res.error or log_res.output}"
                return log_res.output or "No commits found."

            if lowered.startswith("git branch"):
                branch_res = self.git_manager.current_branch()
                if not branch_res.success:
                    return f"Git branch lookup failed: {branch_res.error or branch_res.output}"
                return f"Current branch: {branch_res.output}"

            if lowered.startswith("git commit"):
                return "For controlled writes, use: operator commit <message>."

            return "Only safe git reads are enabled right now: git status, git diff, git log, git branch."

        if lowered.startswith("operator commit "):
            message = text[len("operator commit "):].strip()
            if not message:
                return "Provide a commit message: operator commit <message>."
            if not getattr(self, 'approval_ledger', None):
                return "Approval ledger is not initialized."
            req = self.approval_ledger.create_request(
                action="controlled_commit",
                scope="write",
                summary=f"Commit all current repository changes with message: {message}",
            )
            self.pending_operator_actions[req.approval_id] = {
                "action": "controlled_commit",
                "commit_message": message,
                "created_at": req.created_at,
            }
            return (
                "Approval required for high-impact action.\n"
                f"- approval_id: {req.approval_id}\n"
                f"- summary: {req.summary}\n"
                f"- expires_in_seconds: {int(req.expires_at - req.created_at)}\n"
                f"Reply with: operator approve {req.approval_id}"
            )

        if any(k in lowered for k in ("run tests", "run test suite", "pytest", "test project")):
            if not getattr(self, 'build_runner', None):
                return "Build runner is not initialized."
            test_res = self.build_runner.run_python_tests()
            body = test_res.output or test_res.error
            body = "\n".join((body or "").splitlines()[:220])
            if test_res.success:
                return body or "Tests passed."
            return f"Tests failed (exit={test_res.exit_code}):\n{body}"

        if any(k in lowered for k in ("run build", "build project", "build check", "compile project")):
            if not getattr(self, 'build_runner', None):
                return "Build runner is not initialized."
            build_res = self.build_runner.run_python_build()
            body = build_res.output or build_res.error
            body = "\n".join((body or "").splitlines()[:200])
            if build_res.success:
                return body or "Build check passed."
            return f"Build check failed (exit={build_res.exit_code}):\n{body}"

        return None

    def _validate_tool_invocation_schema(
        self,
        *,
        intent: Any,
        user_input: Any,
        entities: Any,
        context_summary: Any,
    ) -> Optional[str]:
        if not isinstance(intent, str) or not intent.strip():
            return "intent must be a non-empty string"
        if not isinstance(user_input, str) or not user_input.strip():
            return "user_input must be a non-empty string"
        if entities is not None and not isinstance(entities, dict):
            return "entities must be a dict"
        if not isinstance(context_summary, dict):
            return "context_summary must be a dict"
        return None

    def _handle_explain_command(self, user_input: str) -> Optional[str]:
        """Expose compact reasoning trace for transparency commands."""
        text = (user_input or "").strip().lower()
        if text not in {"/explain", "explain reasoning", "why did you say that", "why that response"}:
            return None

        rs = dict(getattr(self, '_internal_reasoning_state', {}) or {})
        if not rs:
            return "I do not have a recent reasoning trace yet. Ask me something first, then run /explain."

        lines = ["Reasoning trace:"]
        lines.append(f"1) Intent: {rs.get('user_intent', 'unknown')} (confidence={float(rs.get('confidence', 0.0)):.2f})")
        lines.append(f"2) Plausibility: {float(rs.get('intent_plausibility', 1.0)):.2f}")
        candidates = rs.get('intent_candidates', []) or []
        if candidates:
            ranked_intents = sorted(
                candidates,
                key=lambda c: float(c.get('score', c.get('confidence', 0.0)) or 0.0),
                reverse=True,
            )[:3]
            alt = ", ".join(
                f"{c.get('intent', 'unknown')} ({float(c.get('score', c.get('confidence', 0.0)) or 0.0):.2f})"
                for c in ranked_intents
            )
            lines.append(f"3) Candidate intents: {alt}")
        scores = rs.get('decision_scores', {}) or {}
        if scores:
            top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
            top_text = ", ".join(f"{k}={float(v):.2f}" for k, v in top)
            lines.append(f"4) Top routes: {top_text}")
        runtime_controls = rs.get('runtime_controls', {}) or {}
        if runtime_controls:
            lines.append(
                "5) Tipping factors: "
                + ", ".join(
                    [
                        f"routing={runtime_controls.get('routing_preference', 'balanced')}",
                        f"allow_tools={bool(runtime_controls.get('allow_tools', True))}",
                        f"thinking_depth={int(runtime_controls.get('thinking_depth', 1) or 1)}",
                    ]
                )
            )
        if rs.get('reasoning_planner', {}):
            rp = rs.get('reasoning_planner', {}) or {}
            lines.append(f"6) Plan: id={rp.get('plan_id', 'n/a')} critical_path={rp.get('critical_path', 'n/a')}")
        if rs.get('learning_decision'):
            lines.append(f"7) Learning decision: {rs.get('learning_decision')}")
        return "\n".join(lines)

    def _promote_learning_goal_intent(
        self, user_input: str, intent: str, entities: Dict[str, Any]
    ) -> tuple[str, Dict[str, Any]]:
        """Promote study-oriented requests into explicit planner intents."""
        text = (user_input or "").strip()
        text_lower = text.lower()
        entities = dict(entities or {})

        study_markers = (
            "help me study",
            "teach me",
            "i want to learn",
            "help me learn",
            "study ",
            "learn ",
        )
        if not any(m in text_lower for m in study_markers):
            return intent, entities

        topic_match = re.search(
            r"(?:help\s+me\s+study|help\s+me\s+learn|i\s+want\s+to\s+learn|teach\s+me|study|learn)\s+(?:about\s+)?(.+)$",
            text_lower,
            re.IGNORECASE,
        )
        topic = topic_match.group(1).strip(" .?!") if topic_match else ""
        if topic:
            entities["topic"] = topic
        entities.setdefault("query", text)

        self._think(f"Reasoning goal detected → planning study flow for: {entities.get('topic', 'topic')}")
        return "learning:study_topic", entities
    
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
            'create_note', 'question', 'conversation:question',
            'study_topic', 'learning:study_topic'
        ]
        
        if intent not in plannable_intents:
            return None
        
        try:
            if self.reasoning_planner and self.persistent_task_queue:
                reasoning_task = self.reasoning_planner.create_task_representation(
                    query,
                    context={"intent": intent, "entities": dict(entities or {})},
                )
                reasoning_plan = self.reasoning_planner.create_plan(reasoning_task)
                self._internal_reasoning_state["reasoning_planner"] = {
                    "task_id": reasoning_task.task_id,
                    "plan_id": reasoning_plan.plan_id,
                    "critical_path": self.reasoning_planner.estimate_critical_path(reasoning_plan),
                    "trace": self.reasoning_planner.debug_trace_view(reasoning_plan),
                }

                queued_task = self.persistent_task_queue.create_task(
                    kind="execute_plan",
                    payload={
                        "intent": intent,
                        "entities": dict(entities or {}),
                    },
                    priority=2,
                    max_attempts=2,
                )

                queue_result = self._await_queue_task_result(queued_task.task_id, timeout_seconds=4.0)
                if queue_result and queue_result.get("success"):
                    all_results = queue_result.get("all_results", {}) or {}
                    if intent in ("study_topic", "learning:study_topic"):
                        explain = all_results.get(1, "")
                        example = all_results.get(2, "")
                        check = all_results.get(3, "")
                        deeper = all_results.get(4, "")

                        parts = []
                        if explain:
                            parts.append(f"1) Concept\n{explain}")
                        if example:
                            parts.append(f"2) Example\n{example}")
                        if check:
                            parts.append(f"3) Check\n{check}")
                        if deeper:
                            parts.append(f"4) Next Step\n{deeper}")
                        if parts:
                            return "\n\n".join(parts)
                    return queue_result.get("result")

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
                if intent in ('study_topic', 'learning:study_topic'):
                    all_results = result.get('all_results', {}) or {}
                    explain = all_results.get(1, '')
                    example = all_results.get(2, '')
                    check = all_results.get(3, '')
                    deeper = all_results.get(4, '')

                    parts = []
                    if explain:
                        parts.append(f"1) Concept\n{explain}")
                    if example:
                        parts.append(f"2) Example\n{example}")
                    if check:
                        parts.append(f"3) Check\n{check}")
                    if deeper:
                        parts.append(f"4) Next Step\n{deeper}")
                    if parts:
                        return "\n\n".join(parts)
                return result.get('result')
            else:
                logger.error(f"Plan execution failed: {result.get('error')}")
                return None
        
        except Exception as e:
            logger.error(f"Planner/executor error: {e}")
            return None

    def _execute_plan_queue_task(self, task: Task) -> Dict[str, Any]:
        """Execute one queued planning task with the legacy planner/executor path."""
        payload = dict(task.payload or {})
        intent = str(payload.get("intent") or "question")
        entities = dict(payload.get("entities") or {})
        context = {
            'user_prefs': vars(self.context.user_prefs),
            'system_state': self.state_tracker.get_status().value if self.state_tracker else 'unknown'
        }

        plan = self.planner.create_plan(intent, entities, context)
        if not self.planner.validate_plan(plan):
            raise ValueError(f"Invalid plan for intent {intent}")

        result = self.plan_executor.execute(plan)
        if not result.get("success"):
            raise RuntimeError(str(result.get("error") or "Unknown plan execution failure"))

        return {
            "success": True,
            "plan_id": plan.plan_id,
            "result": result.get("result"),
            "all_results": result.get("all_results", {}),
        }

    def _await_queue_task_result(self, task_id: str, timeout_seconds: float = 4.0) -> Optional[Dict[str, Any]]:
        """Wait briefly for a queued task to finish and return normalized result."""
        if not self.persistent_task_queue:
            return None

        deadline = time.time() + max(0.1, float(timeout_seconds or 4.0))
        while time.time() < deadline:
            tasks = self.persistent_task_queue.list_tasks()
            queued = next((t for t in tasks if t.task_id == task_id), None)
            if queued is None:
                return None
            if queued.status == QueueTaskStatus.COMPLETED:
                payload = queued.result if isinstance(queued.result, dict) else {}
                return {
                    "success": True,
                    "result": payload.get("result"),
                    "all_results": payload.get("all_results", {}),
                }
            if queued.status == QueueTaskStatus.FAILED:
                logger.error(f"Queued planner task failed: {queued.error}")
                return None
            time.sleep(0.05)
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
        
        # 0. ACTIVE GOAL - only when this turn is task-directed
        if goal_res and goal_res.goal and self._should_attach_goal_context(user_input, intent):
            goal = goal_res.goal
            goal_context = f"ACTIVE GOAL: The user is trying to {goal.description}. "
            goal_context += f"Intent: {goal.intent}. "
            if goal.entities:
                # Convert entity values to strings (might be Entity objects)
                goal_ents = ", ".join(
                    f"{k}={getattr(v, 'value', v) if hasattr(v, 'value') else str(v)}" 
                    for k, v in list(goal.entities.items())[:3]
                )
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

        if self.system_design_response_guard and self.system_design_response_guard.is_architecture_query(user_input):
            context_parts.append(self.system_design_response_guard.guidance_text())
            context_types.append("architecture_policy")

        # 1.1 Episodic + semantic recall to preserve continuity on long sessions.
        if self.episodic_memory_engine:
            try:
                episodic_hits = self.episodic_memory_engine.recall_similar(user_input, limit=2)
                if episodic_hits:
                    snippets = [
                        f"- {item.get('intent', 'unknown')}: {str(item.get('user_input', ''))[:90]}"
                        for item in episodic_hits
                    ]
                    context_parts.append("Relevant prior episodes:\n" + "\n".join(snippets))
                    context_types.append("episodic")
            except Exception:
                pass
        if self.semantic_memory_index:
            try:
                semantic_hits = self.semantic_memory_index.search(user_input, limit=2)
                if semantic_hits:
                    snippets = [f"- {item.get('text', '')}" for item in semantic_hits]
                    context_parts.append("Semantic memory matches:\n" + "\n".join(snippets))
                    context_types.append("semantic")
            except Exception:
                pass
        
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

        if getattr(self, 'conversation_state_tracker', None):
            state_context = self.conversation_state_tracker.format_for_prompt()
            if state_context:
                context_parts.append(state_context)
                context_types.append("conversation_state")

        if getattr(self, '_internal_reasoning_state', None):
            try:
                _rs = self._internal_reasoning_state
                _lines = [
                    "Internal reasoning state (system-only):",
                    f"- user_intent: {_rs.get('user_intent', 'unknown')}",
                    f"- topic: {_rs.get('topic', 'unknown') or 'unknown'}",
                    f"- confidence: {float(_rs.get('confidence', 0.0)):.2f}",
                    f"- intent_plausibility: {float(_rs.get('intent_plausibility', 1.0)):.2f}",
                    f"- conversation_goal: {_rs.get('conversation_goal', 'general_assistance')}",
                    f"- user_goal: {_rs.get('user_goal', 'none') or 'none'}",
                    f"- depth_level: {int(_rs.get('depth_level', 0))}",
                ]
                _plan = _rs.get('plan', []) or []
                if _plan:
                    _lines.append(f"- plan: {' | '.join(str(p) for p in _plan[:4])}")
                _scores = _rs.get('decision_scores', {}) or {}
                if _scores:
                    _top = sorted(_scores.items(), key=lambda kv: kv[1], reverse=True)[:4]
                    _lines.append(
                        "- decision_scores: "
                        + ", ".join(f"{k}={float(v):.2f}" for k, v in _top)
                    )
                if _rs.get('learning_decision'):
                    _lines.append(f"- learning_decision: {_rs.get('learning_decision')}")
                _runtime_controls = _rs.get('runtime_controls', {}) or {}
                if _runtime_controls:
                    _lines.append(
                        "- runtime_controls: "
                        + ", ".join(
                            [
                                f"routing_preference={_runtime_controls.get('routing_preference', 'balanced')}",
                                f"thinking_depth={int(_runtime_controls.get('thinking_depth', 1) or 1)}",
                                f"allow_tools={bool(_runtime_controls.get('allow_tools', True))}",
                                f"max_tool_hops={int(_runtime_controls.get('max_tool_hops', 1) or 1)}",
                            ]
                        )
                    )
                context_parts.append("\n".join(_lines))
                context_types.append("reasoning_state")
            except Exception:
                pass

        # 3a. Response plan injection (structure constraints for LLM)
        _rp_data = (getattr(self, '_internal_reasoning_state', {}) or {}).get('response_plan', {})
        if _rp_data:
            try:
                _rp_lines = [
                    "Response plan (internal):",
                    f"- type: {_rp_data.get('response_type', 'factual')}",
                    f"- strategy: {_rp_data.get('strategy', 'answer_directly')}",
                ]
                _outline = _rp_data.get('outline', []) or []
                if _outline:
                    _rp_lines.append(f"- outline: {' -> '.join(str(o) for o in _outline[:4])}")
                _rp_constraints = _rp_data.get('constraints', []) or []
                if _rp_constraints:
                    _rp_lines.append(f"- constraints: {'; '.join(str(c) for c in _rp_constraints[:4])}")
                _rp_sections = _rp_data.get('required_sections', []) or []
                if _rp_sections:
                    _rp_lines.append(f"- required_sections: {'; '.join(str(s) for s in _rp_sections[:5])}")
                _rp_depth = int(_rp_data.get('plan_depth', 1) or 1)
                _rp_lines.append(f"- plan_depth: {_rp_depth}")
                _goal_ctx = _rp_data.get('goal_context', '')
                if _goal_ctx:
                    _rp_lines.append(f"- goal_context: {_goal_ctx}")
                _fmt = _rp_data.get('format_hint', '')
                if _fmt:
                    _rp_lines.append(f"- format: {_fmt}")
                context_parts.append("\n".join(_rp_lines))
                context_types.append("response_plan")
            except Exception:
                pass

        # 3ab. User response preferences / constraints extracted from utterance
        _prefs = (getattr(self, '_internal_reasoning_state', {}) or {}).get('response_preferences', {}) or {}
        if _prefs:
            try:
                _plines = [
                    "Response preferences (internal):",
                    f"- format: {_prefs.get('format', 'default')}",
                    f"- detail: {_prefs.get('detail', 'normal')}",
                ]
                _constraints = _prefs.get('constraints', []) or []
                if _constraints:
                    _plines.append(f"- constraints: {', '.join(str(c) for c in _constraints[:5])}")
                if _prefs.get('max_words'):
                    _plines.append(f"- max_words: {int(_prefs.get('max_words'))}")
                context_parts.append("\n".join(_plines))
                context_types.append("response_preferences")
            except Exception:
                pass

        # 3b. Goal steering injection (keeps response aligned with active goal)
        if getattr(self, 'goal_tracker', None):
            try:
                _goal_injection = self.goal_tracker.get_goal_prompt_injection()
                if _goal_injection:
                    context_parts.append(_goal_injection)
                    context_types.append("goal_steering")
                # Append next-step suggestion when goal is achieved
                if self.goal_tracker.is_goal_achieved():
                    _nxt = self.goal_tracker.get_next_step_suggestion()
                    if _nxt:
                        context_parts.append(_nxt)
                        context_types.append("goal_next_step")
            except Exception:
                pass

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
        
        # 6. Relevant memories (RAG) - weighted by relevance/recency/confidence/source
        # Keep memory injection conservative on conversational turns to avoid random replies.
        _memory_trigger_words = (
            "remember",
            "what did we",
            "what have we",
            "earlier",
            "last time",
            "my preference",
            "you said",
        )
        _is_conversational_turn = intent in {
            "conversation:ack",
            "conversation:general",
            "conversation:help",
            "conversation:question",
            "conversation:meta_question",
            "greeting",
            "farewell",
            "thanks",
            "status_inquiry",
        }
        _should_fetch_memory = (
            (not _is_conversational_turn and len(user_input.split()) >= 3)
            or any(trigger in user_input.lower() for trigger in _memory_trigger_words)
        )
        if _should_fetch_memory:
            try:
                _weighted_memories = self.memory.recall_memory_weighted(
                    user_input,
                    top_k=3,
                    min_similarity=0.35,
                )
                if _weighted_memories:
                    _lines = ["Relevant information from memory (weighted):"]
                    for _i, _m in enumerate(_weighted_memories, 1):
                        _ts = str(_m.get('timestamp', ''))[:10]
                        _ws = float(_m.get('weighted_score', 0.0))
                        _lines.append(
                            f"{_i}. {_m.get('content', '')} (score {_ws:.2f}, from {_ts})"
                        )
                    context_parts.append("\n".join(_lines))
                    context_types.append("memory")
            except Exception:
                memory_context = self.memory.get_context_for_llm(user_input, max_memories=5)
                if memory_context:
                    context_parts.append(memory_context)
                    context_types.append("memory")
        
        # 7. System capabilities (only if relevant)
        if intent and any(cap in intent for cap in ['note', 'email', 'calendar']):
            capabilities = self.plugins.get_capabilities()
            if capabilities:
                context_parts.append(f"Available capabilities: {', '.join(capabilities[:10])}")
                context_types.append("capabilities")

        # 8. Notes context injection (Feature #2) — surface relevant note snippets
        # when the query is notes-related or explicitly asks about personal knowledge.
        _notes_trigger_words = [
            'note', 'notes', 'wrote', 'saved', 'remember', 'wrote down',
            'reminder', 'todo', 'task', 'idea', 'meeting notes',
        ]
        if any(w in user_input.lower() for w in _notes_trigger_words):
            try:
                notes_plugin = getattr(self.plugins, '_plugin_instances', {}).get('notes')
                if notes_plugin is None:
                    # Try iterating over registered plugins
                    for _p in (getattr(self.plugins, 'plugins', None) or []):
                        if hasattr(_p, 'get_note_context_snippet'):
                            notes_plugin = _p
                            break
                if notes_plugin is not None and hasattr(notes_plugin, 'get_note_context_snippet'):
                    notes_snippet = notes_plugin.get_note_context_snippet(user_input, max_chars=500)
                    if notes_snippet:
                        context_parts.append(notes_snippet)
                        context_types.append("notes")
                        self._think("Notes context snippet injected into LLM context")
            except Exception as _ne:
                import logging as _log
                _log.getLogger(__name__).debug(f"Notes context injection skipped: {_ne}")

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

    def _self_critique_and_regenerate(
        self,
        user_input: str,
        intent: str,
        entities: Dict[str, Any],
        response: str,
        goal_res: Any = None,
    ) -> str:
        """Second-pass quality check with one-shot regeneration on failure."""
        if not response or not getattr(self, 'response_self_critic', None):
            return response

        memory_snapshot = None
        try:
            if getattr(self, 'reasoning_engine', None):
                memory_snapshot = self.reasoning_engine.snapshot()
        except Exception:
            memory_snapshot = None

        critique = self.response_self_critic.assess(
            user_input=user_input,
            intent=intent,
            entities=entities or {},
            response=response,
            memory_snapshot=memory_snapshot,
        )
        if critique.passed:
            return self._clamp_final_response(
                response,
                tone='professional and precise',
                response_type='knowledge_answer' if 'question' in str(intent or '') else 'general_response',
                route='self_critique_pass',
                user_input=user_input,
            )

        self._think(
            "Self-critique failed -> regenerating once "
            f"({', '.join(critique.fail_reasons[:3])})"
        )

        if not getattr(self, 'llm_gateway', None):
            return response

        try:
            regen_prompt = (
                "Revise this draft answer so it matches intent/topic, avoids unsupported claims, "
                "and stays consistent with memory snapshot. Keep it concise.\n\n"
                f"User input: {user_input}\n"
                f"Intent: {intent}\n"
                f"Entities: {entities or {}}\n"
                f"Memory snapshot: {memory_snapshot or {}}\n"
                f"Draft answer: {response}\n"
                f"Failures: {', '.join(critique.fail_reasons)}\n"
                "Return only the revised answer."
            )
            regen = self.llm_gateway.request(
                prompt=regen_prompt,
                call_type=LLMCallType.PHRASE_STRUCTURED,
                use_history=False,
                user_input=user_input,
                context={
                    'structured_payload': regen_prompt,
                    'intent': intent,
                    'entities': entities or {},
                    'goal': goal_res.goal if (goal_res and getattr(goal_res, 'goal', None)) else None,
                    'self_critique': critique.fail_reasons,
                },
            )
            if regen.success and regen.response:
                revised = self._clamp_final_response(
                    regen.response.strip(),
                    tone='professional and precise',
                    response_type='general_response',
                    route='self_critique_regen',
                    user_input=user_input,
                )
                critique2 = self.response_self_critic.assess(
                    user_input=user_input,
                    intent=intent,
                    entities=entities or {},
                    response=revised,
                    memory_snapshot=memory_snapshot,
                )
                if critique2.passed or len(critique2.fail_reasons) < len(critique.fail_reasons):
                    return revised
        except Exception as e:
            logger.debug(f"Self-critique regeneration failed: {e}")

        return response
    
    def _think(self, msg: str) -> None:
        """Emit a thinking-step line when debug mode is on (dev mode)."""
        if getattr(self, 'debug', False):
            print("  " + msg, flush=True)
    
    def _normalize_input(self, user_input: str) -> str:
        """Normalize input for cache lookup (lowercase, strip punctuation)"""
        import string
        normalized = user_input.lower().strip()
        # Remove trailing punctuation for cache matching
        normalized = normalized.rstrip(string.punctuation)
        return normalized
    
    def _cache_get(self, user_input: str, intent: str) -> Optional[str]:
        """Get cached response if available and fresh (within 5 minutes)"""
        cache_key = (self._normalize_input(user_input), intent)
        cached_data = self._response_cache.get(cache_key)

        if cached_data:
            response, timestamp = cached_data
            # Check if cache is still fresh (5 minutes = 300 seconds)
            if time.time() - timestamp < 300:
                return response
            else:
                # Expired, remove from cache
                del self._response_cache[cache_key]

        return None

    def _cache_put(self, user_input: str, intent: str, response: str) -> None:
        """Store response in cache with timestamp"""
        cache_key = (self._normalize_input(user_input), intent)
        self._response_cache[cache_key] = (response, time.time())

        # Simple LRU: if cache gets too big, remove oldest entries
        if len(self._response_cache) > self._cache_max_size:
            # Remove first (oldest) item
            oldest_key = next(iter(self._response_cache))
            del self._response_cache[oldest_key]
    
    def _is_conversational_input(self, user_input: str, intent: str) -> bool:
        """Check if this is a pure conversational input (no commands/actions)"""
        input_lower = user_input.lower()
        
        # Only narrow intents should hit the fast conversational path.
        # Broad buckets like conversation:general/question stay on normal routing
        # to avoid intercepting knowledge/tool-adjacent queries.
        conversational_intents = [
            'conversation:ack',
            'greeting',
            'farewell',
            'thanks',
            'status_inquiry',
        ]
        
        # If not one of these intents, not pure conversation
        if intent not in conversational_intents:
            return False
        
        # Check for action words that would indicate this needs plugins
        action_words = [
            'open', 'launch', 'play', 'send', 'create', 'delete', 'search',
            'show', 'list', 'check', 'email', 'note', 'calendar',
            'weather', 'time', 'find', 'remind', 'file', 'document'
        ]
        
        if any(word in input_lower for word in action_words):
            return False
        
        return True

    def _has_explicit_action_cue(self, user_input: str) -> bool:
        """Return True when the utterance clearly asks for an action/tool operation."""
        if not user_input:
            return False

        text = user_input.lower().strip()
        action_pattern = (
            r"\b(open|launch|play|send|create|make|delete|remove|search|find|"
            r"show|list|check|read|write|edit|update|move|copy|archive|"
            r"schedule|set|remind|email|calendar|note|notes|file|files|"
            r"weather|forecast|temperature|time|date)\b"
        )
        return bool(re.search(action_pattern, text))

    def _should_attach_goal_context(self, user_input: str, intent: str) -> bool:
        """Attach active-goal context only for task-directed turns."""
        conversational_intents = {
            'conversation:general',
            'conversation:ack',
            'conversation:help',
            'conversation:question',
            'conversation:meta_question',
            'greeting',
            'farewell',
            'thanks',
            'status_inquiry',
        }
        if intent in conversational_intents:
            return self._has_explicit_action_cue(user_input)
        return True

    def _is_wake_word_only_input(self, user_input: str) -> bool:
        """Detect short wake-word nudges like 'alice' with no task attached."""
        if not user_input:
            return False

        normalized = user_input.strip().lower()
        compact = re.sub(r"[^a-z]", "", normalized)
        if compact not in {"alice", "aliceai"}:
            return False

        tokens = re.findall(r"\b[a-z']+\b", normalized)
        return len(tokens) <= 2

    def _wake_word_acknowledgment(self, user_name: str) -> Optional[str]:
        """Generate wake-word acknowledgment via learned phrasing, with Ollama as teacher."""
        wake_thought = {
            'type': 'wake_word_ack',
            'data': {
                'user_input': 'alice',
                'user_name': user_name,
            },
        }

        # 1) Alice tries to phrase independently from learned patterns.
        if self.phrasing_learner:
            try:
                if self.phrasing_learner.can_phrase_myself(wake_thought, 'friendly'):
                    learned = self.phrasing_learner.phrase_myself(wake_thought, 'friendly')
                    if learned:
                        return learned
            except Exception:
                pass

        # 2) If Alice does not know yet, ask Ollama for a short line, then learn it.
        if getattr(self, 'llm_gateway', None):
            prompt = (
                "User said only your wake word. "
                "Reply with one short, natural acknowledgment (max 9 words). "
                "Friendly, casual, human. "
                "No quotes, no emoji, no explanation. "
                f"You may include the user's name if useful: {user_name!r}."
            )
            try:
                llm_response = self.llm_gateway.request(
                    prompt=prompt,
                    call_type=LLMCallType.CHITCHAT,
                    use_history=False,
                    user_input="alice"
                )
                if llm_response.success and llm_response.response:
                    response = self._clamp_final_response(
                        llm_response.response.strip().strip('"').strip("'"),
                        tone='casual and friendly',
                        response_type='wake_word_ack',
                        route='wake_word',
                        user_input='alice',
                    )
                    if response:
                        if self.phrasing_learner:
                            self.phrasing_learner.record_phrasing(
                                alice_thought=wake_thought,
                                ollama_phrasing=response,
                                context={
                                    'tone': 'friendly',
                                    'intent': 'wake_word_ack',
                                    'user_input': 'alice',
                                }
                            )
                        return response
            except Exception:
                pass

        # 3) Last non-hardcoded fallback: use any learned greeting pattern.
        return self._learned_greeting_response(
            user_input='alice',
            user_name=user_name,
            asked_how=False,
        )

    def _is_issue_report_input(self, user_input: str) -> bool:
        """Detect reflective problem-report text that should stay conversational."""
        if not user_input:
            return False

        text = user_input.strip().lower()
        if len(text.split()) < 8:
            return False

        issue_markers = [
            "missed intent",
            "missed intents",
            "misread",
            "wrong answer",
            "wrong route",
            "misclass",
            "intent",
            "conversational",
            "routing",
            "capabilitygraph",
            "capability graph",
            "recommendation",
            "recommendations",
        ]
        reflective_markers = [
            "i am trying",
            "i'm trying",
            "my project",
            "my ai",
            "we need to fix",
            "there are",
            "it keeps",
        ]
        explicit_command_markers = [
            "create note",
            "save note",
            "delete note",
            "open note",
            "send email",
            "check email",
            "create event",
            "delete event",
            "what's the weather",
            "show reminders",
            "set reminder",
            "open file",
            "delete file",
        ]

        if any(marker in text for marker in explicit_command_markers):
            return False

        has_issue_signal = any(marker in text for marker in issue_markers)
        has_reflective_signal = any(marker in text for marker in reflective_markers)
        return has_issue_signal and has_reflective_signal

    def _is_explicit_greeting_input(self, user_input: str) -> bool:
        """Return True only when the utterance clearly looks like a greeting."""
        if not user_input:
            return False

        normalized = user_input.strip().lower()
        tokens = re.findall(r"\b[a-z']+\b", normalized)
        if not tokens:
            return False

        greeting_words = {
            "hi",
            "hey",
            "hello",
            "yo",
            "hiya",
            "sup",
            "morning",
            "afternoon",
            "evening",
        }
        polite_words = {"there", "alice", "gabriel", "good", "howdy"}
        non_greeting_action_words = {
            "open",
            "launch",
            "play",
            "send",
            "create",
            "delete",
            "search",
            "show",
            "list",
            "check",
            "email",
            "note",
            "calendar",
            "weather",
            "time",
            "find",
            "remind",
            "file",
            "document",
        }

        if any(token in non_greeting_action_words for token in tokens):
            return False

        if len(tokens) > 6:
            return False

        has_greeting = any(token in greeting_words for token in tokens)
        if not has_greeting:
            return False

        return all(token in greeting_words or token in polite_words for token in tokens)

    def _learned_greeting_response(
        self,
        user_input: str,
        user_name: str,
        asked_how: bool,
        time_context: Optional[str] = None,
    ) -> Optional[str]:
        """Get a greeting from learned patterns (no hardcoded greeting text)."""
        try:
            if hasattr(self, 'conversational_engine') and self.conversational_engine:
                if hasattr(self.conversational_engine, 'learned_greetings') and self.conversational_engine.learned_greetings:
                    if hasattr(self.conversational_engine, '_pick_non_repeating'):
                        picked = self.conversational_engine._pick_non_repeating(
                            self.conversational_engine.learned_greetings
                        )
                        if picked:
                            return picked
        except Exception:
            pass

        if self.phrasing_learner:
            greeting_thought = {
                'type': 'greeting',
                'data': {
                    'user_input': user_input,
                    'user_name': user_name,
                    'asked_how': asked_how,
                    'time_context': time_context,
                },
            }
            try:
                if self.phrasing_learner.can_phrase_myself(greeting_thought, 'friendly'):
                    return self.phrasing_learner.phrase_myself(greeting_thought, 'friendly')
            except Exception:
                pass

        return None
    
    def _should_reuse_goal_intent(self, user_input: str, goal_description: str) -> bool:
        """
        Check if current input is related enough to active goal to reuse its intent.
        Uses word overlap (excluding stop words) to determine relevance.
        """
        if not (user_input and goal_description):
            return False  # Don't reuse if missing context

        input_lower = user_input.lower()

        # Never let a notes:create goal override an append/add-to operation.
        # "add X to the list/note" is always an append, not a create.
        if any(w in input_lower for w in ['add', 'put', 'append', 'include']) and \
           any(phrase in input_lower for phrase in ['to the list', 'to my list', 'to the note',
                                                     'to my note', 'on the list', 'on my list']):
            return False

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
        py_file_match = re.search(r'([a-zA-Z0-9_/\\]+\.py)', input_lower)
        has_py_file = py_file_match is not None

        def _render_codebase_listing(
            files: List[Dict[str, Any]],
            heading: str = "My codebase",
            limit: int = 25,
        ) -> str:
            """Render codebase listing via Alice's response pipeline (no inline text assembly)."""
            listing = [
                {
                    'path': f.get('path', ''),
                    'module_type': f.get('module_type', 'unknown'),
                }
                for f in files[: max(1, int(limit or 25))]
            ]
            return self._generate_natural_response(
                {
                    'type': 'codebase_listing',
                    'heading': heading,
                    'total_files': len(files),
                    'display_limit': max(1, int(limit or 25)),
                    'files': listing,
                },
                'helpful',
                None,
                user_input,
            )

        def _render_code_summaries(summaries: Dict[str, str]) -> str:
            """Render file summaries via Alice's response pipeline (no inline text assembly)."""
            items = [
                {
                    'path': str(path),
                    'summary': str(summary),
                }
                for path, summary in list((summaries or {}).items())
            ]
            return self._generate_natural_response(
                {
                    'type': 'code_summaries',
                    'total_files': len(items),
                    'items': items,
                },
                'helpful',
                None,
                user_input,
            )

        def _is_code_access_capability_request(text: str) -> bool:
            """Detect capability questions about ALICE reading her own code without brittle phrase lists."""
            normalized = (text or "").lower()
            normalized = normalized.replace("acess", "access")
            normalized = re.sub(r"\s+", " ", normalized).strip()

            # Scope must indicate ALICE's internal/source code.
            has_self_scope = bool(
                re.search(
                    r"\b(?:your|internal|own|source)\b.*\b(?:code|codebase|source\s+code)\b",
                    normalized,
                )
                or re.search(r"\b(?:internal|own)\s+code\b", normalized)
            )

            if not has_self_scope:
                return False

            # Capability/visibility verbs and question framing.
            has_capability_verb = bool(
                re.search(
                    r"\b(?:see|access|read|inspect|view|open)\b",
                    normalized,
                )
            )
            has_question_frame = bool(
                re.search(
                    r"\b(?:can|could|would|do|are)\s+you\b",
                    normalized,
                )
                or normalized.endswith("?")
            )

            return has_capability_verb and has_question_frame

        # Follow-up bridge: after a code-access capability answer, resolve
        # short pronoun requests like "list it to me" to code listing.
        _ctx_action = self.code_context.get('last_action')
        _ctx_ts = self.code_context.get('timestamp')
        _ctx_fresh = bool(
            _ctx_ts and isinstance(_ctx_ts, datetime) and (datetime.now() - _ctx_ts).total_seconds() <= 180
        )
        _followup_text = input_lower.strip().rstrip('?!.,')
        _is_code_list_followup = bool(
            re.match(
                r"^(?:please\s+)?(?:yes\s+)?(?:list|show)\s+(?:it|them)(?:\s+to\s+me|\s+for\s+me)?$",
                _followup_text,
            )
        )
        if (
            _ctx_action in ('code_access_confirmed', 'code_list_offered')
            and _ctx_fresh
            and _is_code_list_followup
        ):
            files = self.self_reflection.list_codebase()
            file_paths = [f['path'] for f in files]
            self.code_context['last_files_shown'] = file_paths
            self.code_context['last_action'] = 'list'
            self.code_context['timestamp'] = datetime.now()
            self.code_context['file_count'] = len(files)
            return _render_codebase_listing(files, heading="My codebase", limit=25)
        
        # Smart follow-up detection: user asking for summaries after listing files
        if (not has_py_file) and self.code_context.get('last_action') == 'list' and self.code_context.get('last_files_shown'):
            # Check for summary requests
            if any(phrase in input_lower for phrase in [
                'summarize', 'summary', 'describe', 'what does', 'what do they do',
                'each file', 'all files', 'tell me about', 'explain'
            ]):
                files = self.code_context['last_files_shown']
                logger.info(f"[SmartFollow] Detected summary request for {len(files)} files")
                
                # Generate summaries using advanced batch processing
                summaries = self.self_reflection.batch_summarize_files(files, parallel=True)
                
                # Update context
                self.code_context['last_action'] = 'summary'
                self.code_context['timestamp'] = datetime.now()

                return _render_code_summaries(summaries)
        
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
                        return f"**{code_file.name}** ({code_file.lines} lines, {code_file.module_type}):\n\n```python\n{code_file.content[:2000]}...\n```"
        
        # Only handle EXPLICIT requests to show/list code files
        # Questions about capabilities should go to LLM for natural responses
        if not has_py_file and any(phrase in input_lower for phrase in [
            'show me your code', 'show me code', 'show your code', 'show code',
            'show me all', 'your codebase', 'show me internal',
            'list files', 'show files', 'what files', 'list your code',
            'list your internal code', 'list internal code', 'show your internal code'
        ]):
            # EXCLUDE notes-related queries (they should be handled by notes plugin)
            if any(word in input_lower for word in ['note', 'notes', 'memo']):
                return None  # Let notes plugin handle it

            # A.L.I.C.E understands: show means LIST the codebase
            files = self.self_reflection.list_codebase()

            # Store context for smart follow-up
            file_paths = [f['path'] for f in files]
            self.code_context['last_files_shown'] = file_paths
            self.code_context['last_action'] = 'list'
            self.code_context['timestamp'] = datetime.now()
            self.code_context['file_count'] = len(files)
            return _render_codebase_listing(files, heading="My codebase", limit=25)

        # Capability question: "are you able to see your internal code?"
        # Route to self-reflection directly instead of generic LLM conversation.
        if not has_py_file and _is_code_access_capability_request(user_input):
            capability = self.capabilities.get('codebase_access', {})
            self.code_context['last_action'] = 'code_access_confirmed'
            self.code_context['timestamp'] = datetime.now()
            self.code_context['file_count'] = 0
            self.code_context['last_files_shown'] = []
            return self._generate_natural_response(
                {
                    'type': 'capability_answer',
                    'can_do': bool(capability.get('available', True)),
                    'details': capability.get('description', ''),
                    'operations': capability.get('operations', []),
                    'examples': capability.get('examples', []),
                    'confidence': 0.95,
                },
                'helpful',
                None,
                user_input,
            )

        # Read file request - flexible matching for EXPLICIT read/show commands
        if has_py_file:
            file_path = py_file_match.group(1).strip("'\"")

            # Explicit file discovery requests (e.g., "find code.py").
            if any(
                phrase in input_lower
                for phrase in [
                    'find', 'locate', 'where is', 'where\'s', 'look for', 'search for'
                ]
            ):
                code_file = self.self_reflection.read_file(file_path)
                if code_file:
                    self.last_code_file = code_file.path
                    return (
                        f"Found `{file_path}` at `{code_file.path}` "
                        f"({code_file.lines} lines, {code_file.module_type}).\n"
                        f"Say 'analyze it' or 'show it' if you want details."
                    )

                files = self.self_reflection.list_codebase()
                result = f"Couldn't find `{file_path}`. Closest available files:\n\n"
                shown = 0
                needle = Path(file_path).name.lower().replace('.py', '')
                for f in files:
                    name = f.get('name', '').lower()
                    if needle and needle in name:
                        result += f"- `{f['path']}`\n"
                        shown += 1
                        if shown >= 10:
                            break
                if shown == 0:
                    for f in files[:10]:
                        result += f"- `{f['path']}`\n"
                return result

            # Only handle explicit "read" or "show" or "summarize" commands, not questions
            if any(word in input_lower for word in ['read', 'show', 'summarize', 'summary', 'tell me about', 'what does', 'describe']):
                code_file = self.self_reflection.read_file(file_path)
                if code_file:
                    # Return the file content or explanation based on request
                    if 'summarize' in input_lower or 'summary' in input_lower or 'tell me about' in input_lower or 'what does' in input_lower or 'describe' in input_lower:
                        # USE CODE ANALYZER - Actually parse the file instead of guessing
                        try:
                            from ai.core.code_intelligence import get_code_analyzer
                            from ai.core.fact_checker import get_fact_checker

                            analyzer = get_code_analyzer()
                            fact_checker = get_fact_checker()

                            # Analyze the Python code with AST parsing
                            analysis = analyzer.analyze_python_code(code_file.content)

                            if analysis.get('valid'):
                                # Build factual response from actual code analysis
                                result = f"**{code_file.name}** ({code_file.lines} lines)\n\n"

                                # Module docstring
                                if analysis.get('docstrings'):
                                    result += f"{analysis['docstrings'][0]}\n\n"

                                # Functions
                                if analysis.get('functions'):
                                    result += f"**Functions** ({len(analysis['functions'])}):\n"
                                    for func in analysis['functions'][:10]:
                                        args_str = ', '.join(func['args'])
                                        result += f"- `{func['name']}({args_str})` - {func['lines']} lines"
                                        if func.get('docstring'):
                                            doc_preview = func['docstring'].split('\n')[0][:80]
                                            result += f": {doc_preview}"
                                        result += f" (complexity: {func['complexity']})\n"
                                    if len(analysis['functions']) > 10:
                                        result += f"  ... and {len(analysis['functions']) - 10} more\n"
                                    result += "\n"

                                # Classes
                                if analysis.get('classes'):
                                    result += f"**Classes** ({len(analysis['classes'])}):\n"
                                    for cls in analysis['classes'][:5]:
                                        result += f"- `{cls['name']}`"
                                        if cls.get('bases'):
                                            result += f" (extends {', '.join(cls['bases'])})"
                                        result += f" - {cls['method_count']} methods\n"
                                        if cls.get('docstring'):
                                            doc_preview = cls['docstring'].split('\n')[0][:80]
                                            result += f"  {doc_preview}\n"
                                    if len(analysis['classes']) > 5:
                                        result += f"  ... and {len(analysis['classes']) - 5} more\n"
                                    result += "\n"

                                # Imports
                                if analysis.get('imports'):
                                    result += f"**Dependencies**: {', '.join(analysis['imports'][:8])}\n"
                                    if len(analysis['imports']) > 8:
                                        result += f"  ... and {len(analysis['imports']) - 8} more\n"
                                    result += "\n"

                                # Code quality metrics
                                metrics = analysis.get('metrics', {})
                                if metrics:
                                    result += f"**Metrics**:\n"
                                    result += f"- Code quality score: {analysis.get('quality_score', 0.0):.2f}/1.0\n"
                                    result += f"- Average complexity: {metrics.get('avg_complexity', 0):.1f}\n"
                                    result += f"- Documentation ratio: {metrics.get('doc_ratio', 0):.0%}\n"

                                # Design patterns
                                if analysis.get('patterns'):
                                    result += f"\n**Design Patterns**: {', '.join(analysis['patterns'])}\n"

                                # Improvement suggestions
                                suggestions = analyzer.suggest_improvements(analysis)
                                if suggestions:
                                    result += f"\n**Suggestions**:\n"
                                    for suggestion in suggestions[:3]:
                                        result += f"- {suggestion}\n"

                                return result
                            else:
                                # Fall back to self-reflection summary when the
                                # rich analyzer cannot fully parse a valid file.
                                fallback_summary = self.self_reflection.generate_file_summary(code_file.path)
                                if fallback_summary:
                                    return fallback_summary
                                return f"`{code_file.path}` - {code_file.lines} lines, {code_file.module_type}"

                        except Exception as e:
                            logger.error(f"Error in code analysis: {e}")
                            # Fallback: use self-reflection summary before giving up.
                            try:
                                fallback_summary = self.self_reflection.generate_file_summary(code_file.path)
                                if fallback_summary:
                                    return fallback_summary
                            except Exception:
                                pass
                            return f"`{code_file.path}` - {code_file.lines} lines, {code_file.module_type}"
                    else:
                        # Show full file content
                        return f"**{code_file.name}** ({code_file.lines} lines, {code_file.module_type}):\n\n```python\n{code_file.content[:2000]}...\n```"
                # File not found - just list what's available
                files = self.self_reflection.list_codebase()
                result = f"Couldn't find `{file_path}`. Available files:\n\n"
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

            return _render_codebase_listing(files, heading="Codebase Structure", limit=20)
        
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
            result = "[LEARNING] **Learning Status**:\n\n"
            result += f"[OK] Total interactions: {stats['total_examples']}\n"
            result += f"[OK] High-quality examples: {stats['high_quality']}\n"
            result += f"[OK] Ready for fine-tuning: {'Yes' if stats['should_finetune'] else 'No (need 50+ examples)'}\n"
            
            if stats['examples_by_intent']:
                result += "\n[DETAILS] Examples by intent:\n"
                for intent, count in list(stats['examples_by_intent'].items())[:5]:
                    result += f"  - {intent}: {count}\n"
            
            if not stats['should_finetune']:
                result += "\n[INFO] Keep using A.L.I.C.E! I'm learning from every interaction."
            
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
                    return f"[OK] {len(training_data)} examples available for export to: `{export_path}`\n\nYou can use this file to train A.L.I.C.E with Ollama's fine-tuning tools."
            return "No training data to export yet. Keep using A.L.I.C.E to collect data!"
        
        # Prepare training data
        if 'prepare training' in input_lower or 'ready to train' in input_lower:
            if not getattr(self, 'learning_engine', None):
                return "Learning system not initialized."
            
            if self.learning_engine.should_finetune():
                examples = self.learning_engine.get_high_quality_examples()
                return f"[OK] Ready to train! {len(examples)} high-quality examples available.\n\nTo train A.L.I.C.E:\n1. Use Ollama's fine-tuning: `ollama create alice-custom -f data/training/training_data.jsonl`\n2. Or export the data and use external training tools."
            return "[ERROR] Not enough data yet. Need 50+ high-quality examples (currently have less)."
        
        return None

    def _handle_weather_followup(self, user_input: str, intent: str) -> Optional[str]:
        """
        Handle weather-related follow-ups using stored weather data.
        This is A.L.I.C.E's routing policy - no need to call Ollama for simple decisions.
        
        NOTE: This should be called REGARDLESS of intent, because weekday mentions
        should ALWAYS trigger forecast follow-up if a forecast is in context.
        """
        input_lower = user_input.lower()
        umbrella_aliases = ['umbrella', 'umbrela', 'umberella', 'umbralla']

        weekday_keywords = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        time_range_keywords = [
            'rest of the week', 'rest of week',
            'this weekend', 'this week', 'next week', 'weekend', 'tomorrow', 'tonight'
        ]

        # Check for time-range follow-up mentions (e.g., "what about this week?")
        mentioned_time_range = None
        for time_range in time_range_keywords:
            if time_range in input_lower:
                mentioned_time_range = time_range
                break

        if mentioned_time_range:
            self._think(f"Detected weather time-range follow-up: {mentioned_time_range} (intent was {intent})")
            if not hasattr(self, 'reasoning_engine') or not self.reasoning_engine:
                self._think("Reasoning engine not available for weather follow-up")
                return None

            forecast_entity = self.reasoning_engine.get_entity('weather_forecast')
            if not forecast_entity or not forecast_entity.data:
                self._think(f"No forecast entity found for '{mentioned_time_range}' follow-up")
                return None

            forecast_data = forecast_entity.data
            self._think(
                f"Using stored forecast data for time-range '{mentioned_time_range}' with {len(forecast_data.get('forecast', []))} days"
            )

            from ai.models.simple_formatters import WeatherFormatter
            try:
                result = WeatherFormatter.format(
                    forecast_data,
                    entities={'TIME_RANGE': [mentioned_time_range]}
                )
                if result:
                    self._think(f"Forecast formatter returned: {result[:60]}...")
                    logger.info(f"Weather follow-up (stored, time-range) → {result[:60]}...")
                    return result
                self._think(f"Formatter returned None for time-range '{mentioned_time_range}'")
                return None
            except Exception as e:
                logger.error(f"Error formatting forecast for time-range '{mentioned_time_range}': {e}", exc_info=True)
                self._think(f"Error formatting forecast: {e}")
                return None
        
        # Check for weekday mentions (indicates a forecast follow-up like "is that wednesday?")
        mentioned_day = None
        for day in weekday_keywords:
            if day in input_lower:
                mentioned_day = day
                break
        
        if mentioned_day:
            self._think(f"Detected weekday mention: {mentioned_day} (intent was {intent})")
            if not hasattr(self, 'reasoning_engine') or not self.reasoning_engine:
                self._think("Reasoning engine not available for weather follow-up")
                return None
            
            # Try to get stored forecast data
            forecast_entity = self.reasoning_engine.get_entity('weather_forecast')
            if not forecast_entity:
                self._think(f"No forecast entity found for '{mentioned_day}' follow-up - may be first weather query")
                return None
            
            if not forecast_entity.data:
                self._think(f"Forecast entity exists but has no data")
                return None

            forecast_data = forecast_entity.data
            self._think(f"Using stored forecast data for {mentioned_day} query: {type(forecast_data)} with {len(forecast_data.get('forecast', []))} days")
            
            from ai.models.simple_formatters import WeatherFormatter
            try:
                result = WeatherFormatter.format(
                    forecast_data,
                    entities={'TIME_RANGE': [mentioned_day]}
                )
                if result:
                    self._think(f"Forecast formatter returned: {result[:60]}...")
                    logger.info(f"Weather follow-up (stored) → {result[:60]}...")
                    return result
                else:
                    self._think(f"Formatter returned None for {mentioned_day}")
                    return None
            except Exception as e:
                logger.error(f"Error formatting forecast for {mentioned_day}: {e}", exc_info=True)
                self._think(f"Error formatting forecast: {e}")
                return None
        
        # Check if this is a clothing/outside question — answer from stored weather data
        # NOTE: Run this regardless of intent (NLP may have already classified as weather:*)
        clothing_items = [
            'jacket', 'coat', 'layer', 'wear', 'bring', 'outside', 'go out',
            'scarf', 'hat', 'gloves', 'boots', 'sweater', 'hoodie',
        ] + umbrella_aliases
        weather_condition_indicators = ['rain', 'snow', 'cold', 'warm', 'hot', 'freeze', 'thunderstorm', 'hail']
        weather_question_indicators = clothing_items + weather_condition_indicators

        if any(re.search(r'\b' + re.escape(kw) + r'\b', input_lower)
               for kw in weather_question_indicators):
            if not hasattr(self, 'reasoning_engine') or not self.reasoning_engine:
                return None

            current_entity = self.reasoning_engine.get_entity('current_weather')
            forecast_entity = self.reasoning_engine.get_entity('weather_forecast')
            has_current = bool(current_entity and current_entity.data)
            has_forecast = bool(forecast_entity and forecast_entity.data)
            if not has_current and not has_forecast:
                return None

            def _entity_age_score(ent) -> float:
                try:
                    return ent.created_at.timestamp()
                except Exception:
                    return 0.0

            # Prefer whichever weather snapshot is newer; this avoids stale
            # current-weather entities overriding a fresh forecast context.
            weather_entity = None
            if has_current and has_forecast:
                weather_entity = (
                    current_entity
                    if _entity_age_score(current_entity) >= _entity_age_score(forecast_entity)
                    else forecast_entity
                )
            else:
                weather_entity = current_entity if has_current else forecast_entity

            # We have weather data — A.L.I.C.E answers directly using her own reasoning
            wd = weather_entity.data
            location = wd.get('location', 'your area')
            condition = (wd.get('condition') or '').lower()
            temp = wd.get('temperature')

            # Forecast entities do not carry a scalar temperature. Derive a
            # representative value from the first available day.
            if temp is None and isinstance(wd.get('forecast'), list) and wd.get('forecast'):
                d0 = wd['forecast'][0]
                low = d0.get('low')
                high = d0.get('high')
                if low is not None and high is not None:
                    temp = round((float(low) + float(high)) / 2)
                if not condition:
                    condition = (d0.get('condition') or '').lower()

            # Detect if this is a follow-up (user already saw the weather report this turn)
            is_follow_up = False
            if hasattr(self, 'conversation_summary') and self.conversation_summary:
                last_turn = self.conversation_summary[-1]
                is_follow_up = last_turn.get('intent', '').startswith('weather')

            # ── Umbrella — condition-driven, not temperature-driven ───────────
            if any(re.search(r'\b' + re.escape(alias) + r'\b', input_lower) for alias in umbrella_aliases):
                rainy = any(w in condition for w in ['rain', 'drizzle', 'shower', 'storm'])
                _adv_loc = '' if is_follow_up else location
                force_yes_no = bool(
                    re.search(r'\b(should|do|does|is|am|are)\b', input_lower)
                    or 'or no' in input_lower
                )
                if rainy:
                    return self._generate_natural_response({
                        'type': 'weather_advice',
                        'temperature': temp,
                        'condition': condition,
                        'location': _adv_loc,
                        'clothing_item': 'umbrella',
                        'force_yes_no': force_yes_no,
                        'user_question': user_input,
                    }, 'helpful', None, user_input)
                # No rain — report the facts and let Alice/Ollama advise
                return self._generate_natural_response({
                    'type': 'weather_advice',
                    'temperature': temp,
                    'condition': condition,
                    'location': _adv_loc,
                    'clothing_item': 'umbrella',
                    'rainy': False,
                    'force_yes_no': force_yes_no,
                    'user_question': user_input,
                }, 'helpful', None, user_input)

            # ── Clothing items — delegate entirely to weather_advice formatter ─
            # (handles single-item AND X-or-Y choice questions)
            if any(w in input_lower for w in clothing_items):
                if temp is None:
                    return None
                _warmth_rank = {
                    'parka': 6, 'coat': 5, 'anorak': 5,
                    'jacket': 4, 'sweater': 3, 'hoodie': 3, 'cardigan': 3,
                    'layer': 3, 'shirt': 2, 'blouse': 2, 'tshirt': 1, 't-shirt': 1,
                }
                _item_words = {
                    'coat': 'coat', 'jacket': 'jacket', 'layer': 'layers',
                    'scarf': 'scarf', 'hat': 'hat', 'gloves': 'gloves',
                    'boots': 'boots', 'sweater': 'sweater', 'hoodie': 'hoodie',
                }
                # "should i wear X or Y?" — pick warmer/lighter based on temperature
                _choice_m = re.search(
                    r'\b(?:wear|put\s+on)\s+(?:a\s+)?(\w+)\s+or\s+(?:a\s+)?(\w+)\b',
                    input_lower,
                )
                if _choice_m:
                    opt_a, opt_b = _choice_m.group(1), _choice_m.group(2)
                    rank_a = _warmth_rank.get(opt_a, 2)
                    rank_b = _warmth_rank.get(opt_b, 2)
                    # below 15°C → pick the warmer item; at/above 15°C → pick the lighter item
                    recommended = (opt_a if rank_a >= rank_b else opt_b) if temp < 15 \
                                  else (opt_a if rank_a <= rank_b else opt_b)
                else:
                    recommended = next(
                        (label for word, label in _item_words.items() if word in input_lower),
                        'jacket',
                    )
                return self._generate_natural_response({
                    'type': 'weather_advice',
                    'temperature': temp,
                    'condition': condition,
                    'location': '' if is_follow_up else location,
                    'clothing_item': recommended,
                    'force_yes_no': bool(
                        re.search(r'\b(should|do|does|is|am|are)\b', input_lower)
                        or 'or no' in input_lower
                    ),
                    'user_question': user_input,
                }, 'helpful', None, user_input)

            # ── Cold/warm/hot/freeze condition questions ───────────────────────
            # Report the current conditions — let the facts answer the question.
            if any(w in input_lower for w in ['cold', 'warm', 'hot', 'freeze']):
                if temp is None:
                    return None
                return self._alice_direct_phrase('weather_report', {
                    'temperature': wd.get('temperature'),
                    'condition': condition,
                    'location': '' if is_follow_up else location,
                    'is_followup': is_follow_up,
                })

        return None

    def _normalize_weather_intent_for_time_range(
        self,
        user_input: str,
        intent: str,
        intent_confidence: float,
    ) -> tuple[str, float]:
        """Promote weather:current to weather:forecast for explicit time-range weather requests."""
        if str(intent or "") != "weather:current":
            return intent, float(intent_confidence or 0.0)

        text = (user_input or "").lower()
        has_weather_scope = any(
            token in text
            for token in (
                "weather",
                "forecast",
                "temperature",
                "rain",
                "snow",
                "outside",
            )
        )
        if not has_weather_scope:
            return intent, float(intent_confidence or 0.0)

        time_range_cues = (
            "rest of the week",
            "rest of week",
            "this week",
            "next week",
            "weekend",
            "weekly",
            "next few days",
            "coming days",
            "for the week",
            "for this week",
        )
        if any(cue in text for cue in time_range_cues):
            if hasattr(self, "_think"):
                self._think(
                    f"Weather time-range cue detected → promoting {intent!r} to 'weather:forecast'"
                )
            return "weather:forecast", max(float(intent_confidence or 0.0), 0.9)

        return intent, float(intent_confidence or 0.0)

    def _apply_confidence_cascade(self, intent: str, confidence: float, nlp_result) -> dict:
        """
        Confidence cascade policy:
          >= 0.85 → execute directly
          >= 0.65 → execute with a low-confidence prefix marker
          >= 0.45 → ask for clarification
          <  0.45 → surface top-2 interpretations
        """
        if confidence >= 0.85:
            return {"action": "execute", "confidence": confidence}
        if confidence >= 0.65:
            return {
                "action": "execute_low_conf",
                "confidence": confidence,
                "marker": "I'm not 100% sure, but I'll try—",
            }
        if confidence >= 0.45:
            return {
                "action": "clarify",
                "confidence": confidence,
                "question": "Could you clarify what you'd like me to do?",
            }
        # < 0.45: surface top-2 candidates
        scores = getattr(nlp_result, 'plugin_scores', {}) or {}
        top2 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
        options = [f"{p} ({v:.0%})" for p, v in top2] if top2 else [intent]
        return {
            "action": "interpret",
            "confidence": confidence,
            "options": options,
            "question": f"Did you mean: {' or '.join(options)}?",
        }

    def _executive_apply_response_gate(
        self,
        *,
        user_input: str,
        intent: str,
        response: str,
        route: str,
    ) -> str:
        """Final executive authority before a response is sent to the user."""
        if not getattr(self, 'executive_controller', None):
            return response

        _gate_context = dict(getattr(self, '_internal_reasoning_state', {}) or {})
        if getattr(self, 'goal_tracker', None):
            try:
                _gate_context['goal_alignment'] = self.goal_tracker.goal_alignment_score(response)
            except Exception:
                _gate_context['goal_alignment'] = 1.0

        evaluation = self.executive_controller.evaluate_response(
            user_input=user_input,
            intent=intent,
            response=response,
            route=route,
            context=_gate_context,
        )
        self._last_exec_gate_eval = evaluation
        score = float(evaluation.get("score", 0.0))
        self._think(
            f"Executive response gate ({route}) → score={score:.2f} "
            f"accepted={evaluation.get('accepted', False)}"
        )

        if evaluation.get("accepted", False):
            return response

        fallback_action = evaluation.get("fallback_action", "safe_reply")
        if fallback_action == "clarify":
            return "I want to make sure I answer correctly. Do you want an explanation, a direct action, or a quick search?"
        return "I may be off-track. Let me answer more directly: tell me the exact outcome you want and I will do that."

    def _run_executive_reflection(
        self,
        *,
        user_input: str,
        intent: str,
        response: str,
        route: str,
        prior_confidence: float,
    ) -> None:
        """Post-response reflection loop that updates executive routing weights."""
        if not getattr(self, 'reflection_engine', None) or not getattr(self, 'executive_controller', None):
            return
        try:
            gate_eval = getattr(self, '_last_exec_gate_eval', {}) or {}

            # Track response quality first so reflection can consume it.
            _turn_quality = None
            if getattr(self, 'response_quality_tracker', None):
                try:
                    _goal_align = 1.0
                    if getattr(self, 'goal_tracker', None):
                        _goal_align = self.goal_tracker.goal_alignment_score(response)
                    _topic_hint = str((getattr(self, '_internal_reasoning_state', {}) or {}).get('topic', '') or '')
                    _turn_quality = self.response_quality_tracker.track_turn(
                        user_input=user_input,
                        response=response,
                        intent=intent,
                        gate_accepted=bool(gate_eval.get('accepted', True)),
                        goal_alignment=_goal_align,
                        topic_hint=_topic_hint,
                    )
                    self._internal_reasoning_state["turn_quality"] = _turn_quality.as_dict()
                except Exception as _qt_err:
                    logger.debug(f"[QualityTracker] {_qt_err}")

            _quality_payload = _turn_quality.as_dict() if _turn_quality else {}
            _failure_type = _quality_payload.get("failure_type", "none")
            reflection = self.reflection_engine.reflect(
                user_input=user_input,
                intent=intent,
                response=response,
                route=route,
                gate_accepted=bool(gate_eval.get('accepted', True)),
                decision_scores=(getattr(self, '_internal_reasoning_state', {}) or {}).get('decision_scores', {}),
                prior_confidence=float(prior_confidence or 0.0),
                quality_metrics=_quality_payload,
                failure_type=str(_failure_type),
            ).as_dict()
            self.executive_controller.apply_reflection(reflection)
            self._internal_reasoning_state["reflection"] = reflection
            self._think(
                "Executive reflection → "
                f"score={float(reflection.get('success_score', 0.0)):.2f}, "
                f"relevant={reflection.get('was_relevant', False)}"
            )
            if _turn_quality and _turn_quality.failure_type != "none":
                self._think(
                    f"Quality tracker → failure={_turn_quality.failure_type}, "
                    f"relevance={_turn_quality.relevance:.2f}, "
                    f"topic={_turn_quality.topic_adherence:.2f}"
                )

        except Exception as e:
            logger.debug(f"[ExecutiveReflection] {e}")

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
            # ===== PRODUCTION METRICS & LOGGING =====
            start_time = time.time()
            route_taken = 'unknown'
            success = False
            self._exec_should_store_memory = True
            self._internal_reasoning_state = {}
            self._last_exec_gate_eval = {}

            # Per-turn quality tracking flags (read by _store_interaction)
            self._last_plugin_called = None
            self._last_had_stored_data = False
            
            # Set logging context
            self.structured_logger.set_context(
                user_input=user_input[:100],  # Truncate for privacy
                timestamp=datetime.utcnow().isoformat()
            )
            
            self.structured_logger.info("Processing input", input_length=len(user_input))
            logger.info(f"User: {user_input}")
            is_location_query = self._is_location_query(user_input)

            # Canonical path first when enabled: avoid side-path bypasses.
            if is_enabled("contract_pipeline") and getattr(self, 'contract_pipeline', None):
                try:
                    _pipeline_result = self.contract_pipeline.run_turn(
                        user_input=user_input,
                        user_id=str(self.user_name),
                        turn_number=int(getattr(self, '_turn_count', 0) or 0),
                    )
                    if _pipeline_result and _pipeline_result.handled and _pipeline_result.response_text:
                        _meta = dict(_pipeline_result.metadata or {})
                        self.structured_logger.info(
                            "Canonical pipeline handled request",
                            component='pipeline',
                            trace_id=str(_meta.get('trace_id') or ''),
                            route=str(_meta.get('route') or ''),
                            intent=str(_meta.get('intent') or ''),
                            verification_reason=str(((_meta.get('verification') or {}).get('reason')) or ''),
                        )
                        return _pipeline_result.response_text
                except Exception as e:
                    logger.debug(f"[ContractPipeline] Fallback to legacy path due to: {e}")
            
            # ===== DISTRIBUTED CACHE CHECK =====
            # Check if we have a cached response for this exact input
            cache_key = f"response_{hash(user_input)}"
            cached_response = None
            if not is_location_query:
                cached_response = self.cache.get('responses', cache_key)
            if cached_response:
                duration = time.time() - start_time
                self.metrics.track_cache('get', 'hit')
                self.metrics.track_request('cached', True, duration, 'cache')
                self.structured_logger.info(
                    "Cache hit - returning cached response",
                    duration_ms=round(duration * 1000, 2)
                )
                self._think("Distributed cache hit → ultra-fast response")
                return cached_response
            else:
                self.metrics.track_cache('get', 'miss')
            # ===== END CACHE CHECK =====

            # Hard deterministic location path: never delegate to LLM.
            if is_location_query:
                _location_response = self._alice_direct_phrase(
                    'location_report',
                    self._build_location_payload(),
                )
                if _location_response:
                    return _location_response
            
            logger.info(f"User: {user_input}")
            
            # ===== 10 TIER IMPROVEMENTS - INPUT PROCESSING =====
            # Record this turn for long-session coherence
            if hasattr(self, 'session_summarizer'):
                try:
                    self.session_summarizer.record_turn(
                        turn_number=getattr(self, '_turn_count', 1),
                        user_input=user_input[:500],  # Truncate for memory
                        internal_state={
                            'timestamp': datetime.utcnow().isoformat(),
                            'user': self.user_name
                        }
                    )
                except Exception as e:
                    logger.debug(f"[SessionSummarizer] Error recording turn: {e}")
            
            # Start routing decision logging
            routing_decision_id = None
            if hasattr(self, 'routing_decision_logger'):
                try:
                    _routing = self.routing_decision_logger.log_decision(
                        user_input=user_input,
                        classified_intent="unknown",
                        intent_confidence=0.0,
                        decision_type=RoutingDecisionType.REASONING,
                        candidates_considered=[
                            {
                                'name': 'bootstrap',
                                'type': 'routing',
                                'score': 0.0,
                                'reasoning': 'initial routing placeholder',
                                'pros': [],
                                'cons': [],
                            }
                        ],
                        winning_candidate='bootstrap',
                        winning_score=0.0,
                        decision_reasoning='Initial routing placeholder before downstream routing completes.',
                        factors_used=['user_input_received'],
                        uncertainty_level=1.0,
                        turn_number=getattr(self, '_turn_count', 0),
                    )
                    routing_decision_id = _routing.decision_id
                except Exception as e:
                    logger.debug(f"[RoutingDecisionLogger] Error logging decision: {e}")
            # ===== END 10 TIER INPUT PROCESSING =====
            
            # ===== FOUNDATION FEEDBACK LEARNING =====
            # Learn from previous interaction based on user's current input
            if self.foundations and hasattr(self, '_last_interaction'):
                try:
                    last_input = self._last_interaction.get('input')
                    last_response = self._last_interaction.get('response')
                    if last_input and last_response:
                        # Current user input serves as implicit feedback
                        self.foundations.learn_from_feedback(
                            user_id=self.user_name,
                            user_input=last_input,
                            alice_response=last_response,
                            user_reaction=user_input
                        )
                        self._think("Foundation learning from previous interaction")
                except Exception as e:
                    logger.debug(f"Foundation feedback learning error: {e}")
            # ===== END FOUNDATION FEEDBACK =====
            
            # 0. Check for commands first (before any processing)
            if user_input.startswith('/'):
                # This is a command, not a conversational input
                # Commands should be handled in the UI layer (run_interactive)
                # If we get here, return a helpful message
                return "Commands should be handled by the interface. Use /help to see available commands."

            # 1. NLP Processing
            _nlp_input = user_input
            _context_resolution = None
            if getattr(self, 'context_resolver', None):
                _pre_state = {
                    'current_topic': '',
                    'last_subject': '',
                    'last_intent': str(getattr(self, 'last_intent', '') or ''),
                    'active_goal': '',
                    'referenced_entities': [],
                    'last_entities': dict(getattr(self, 'last_entities', {}) or {}),
                }

                if getattr(self, 'conversation_state_tracker', None):
                    try:
                        _cs = self.conversation_state_tracker.get_state_summary()
                        _pre_state['current_topic'] = str(_cs.get('conversation_topic', '') or '')
                        _pre_state['active_goal'] = str(
                            _cs.get('user_goal') or _cs.get('conversation_goal') or ''
                        )
                    except Exception:
                        pass

                if getattr(self, 'conversation_context', None):
                    try:
                        _ctx = self.conversation_context.get_context_summary()
                        _pre_state['referenced_entities'] = list(_ctx.get('salient_entities', []) or [])
                        _last_file = str(_ctx.get('last_file') or '')
                        _last_concept = str(_ctx.get('last_concept') or '')
                        _last_object = str(_ctx.get('last_object') or '')
                        _pre_state['last_subject'] = _last_file or _last_concept or _last_object
                    except Exception:
                        pass

                _last_files = (
                    (getattr(self, 'code_context', {}) or {}).get('last_files_shown', [])
                    if isinstance(getattr(self, 'code_context', {}), dict)
                    else []
                )
                if isinstance(_last_files, list) and _last_files:
                    _pre_state['last_subject'] = str(_last_files[0])
                    refs = list(_pre_state.get('referenced_entities', []) or [])
                    refs.extend(str(x) for x in _last_files[:5])
                    _pre_state['referenced_entities'] = refs

                _context_resolution = self.context_resolver.resolve(user_input, _pre_state)
                if _context_resolution.needs_clarification:
                    return self._generate_natural_response(
                        {
                            'type': 'clarification_prompt',
                            'reason': 'ambiguous_reference',
                            'pronouns': list(_context_resolution.unresolved_pronouns or []),
                            'options': list(_context_resolution.clarification_options or []),
                        },
                        'helpful',
                        None,
                        user_input,
                    )
                if _context_resolution.rewritten_input != user_input:
                    self._think(
                        f"Context resolve: {user_input!r} -> {_context_resolution.rewritten_input!r}"
                    )

            if _context_resolution and _context_resolution.rewritten_input:
                _nlp_input = _context_resolution.rewritten_input
            nlp_start = time.time()
            nlp_result = self.nlp.process(_nlp_input)
            intent = nlp_result.intent
            entities = nlp_result.entities
            if _context_resolution and _context_resolution.resolved_bindings:
                entities = dict(entities or {})
                entities['resolved_bindings'] = dict(_context_resolution.resolved_bindings)
                if 'resolved_reference' not in entities:
                    entities['resolved_reference'] = next(
                        iter(_context_resolution.resolved_bindings.values()),
                        '',
                    )
            sentiment = nlp_result.sentiment
            nlp_duration = time.time() - nlp_start
            _response_prefs = (
                self.constraint_preference_extractor.extract(user_input)
                if self.constraint_preference_extractor
                else {}
            )
            _response_style = (
                self.adaptive_response_style.derive_style(
                    intent=intent,
                    sentiment=sentiment,
                    preferences=_response_prefs,
                )
                if self.adaptive_response_style
                else {}
            )
            
            # Track NLP metrics
            self.metrics.record_histogram('nlp_duration', nlp_duration)
            intent_confidence = getattr(nlp_result, 'intent_confidence', 0.5)
            self.metrics.track_intent_confidence(intent, intent_confidence)

            # ── 1.1  Perception layer ─────────────────────────────────────────────
            # Build a unified "what is the user actually trying to do?" object that
            # captures mood, ambiguity, and follow-up domain in one place.
            perception = self.perception.build(
                nlp_result,
                last_intent=self.last_intent,
                conversation_topics=self.conversation_topics,
            )

            # ── 1.2  Interaction policy ───────────────────────────────────────────
            # Derive tone/length/clarification settings from the user's inferred mood.
            urgency_level = getattr(nlp_result, 'urgency_level', 'low')
            _frame_info = getattr(nlp_result, 'frame_result', None)
            policy = self.interaction_policy.derive(
                perception.inferred_mood, sentiment, urgency_level,
                intent=intent,
                intent_conf=intent_confidence,
                frame_name=getattr(_frame_info, 'frame_name', None) if _frame_info else None,
                frame_conf=getattr(_frame_info, 'confidence', 0.0) if _frame_info else 0.0,
                n_slots=len(getattr(_frame_info, 'slots', {}) or {}) if _frame_info else 0,
                session_id=getattr(self, '_session_id', ''),
            )
            if perception.inferred_mood not in ("neutral", "positive"):
                self._think(
                    f"Mood={perception.inferred_mood!r} → "
                    f"skip_clarification={policy.skip_clarification}, tone={policy.tone!r}"
                )

            # ── 1.3  Follow-up resolution ─────────────────────────────────────────
            # Foundation 2: NLPProcessor is the primary follow-up authority.
            # Use main-level resolver only as a legacy fallback when metadata
            # is unavailable.
            _modifiers = (
                getattr(nlp_result, 'parsed_command', {}) or {}
            ).get('modifiers', {})
            _followup_meta = _modifiers.get('followup_resolution') if isinstance(_modifiers, dict) else None
            _followup_applied = False

            if isinstance(_followup_meta, dict) and _followup_meta.get('was_followup'):
                _resolved_intent = str(_followup_meta.get('resolved_intent') or intent)
                _resolved_conf = float(_followup_meta.get('confidence', intent_confidence) or intent_confidence)
                _reason = str(_followup_meta.get('reason') or 'nlp_followup_resolution')
                _domain = str(_followup_meta.get('domain') or '')
                self._think(
                    f"Follow-up detected [{_reason}]: "
                    f"{intent} → {_resolved_intent} "
                    f"(confidence {intent_confidence:.2f} → {_resolved_conf:.2f})"
                )
                self.nlp_error_logger.log_followup_resolved(
                    user_input=user_input,
                    nlp_intent=intent,
                    resolved_intent=_resolved_intent,
                    nlp_confidence=intent_confidence,
                    domain=_domain,
                    reason=_reason,
                    session_id=getattr(self, '_session_id', None),
                )
                intent = _resolved_intent
                intent_confidence = _resolved_conf
                _followup_applied = True

            if not _followup_applied:
                self._think("No NLP follow-up metadata this turn; continuing without secondary follow-up override.")

            # 1.35 Context-aware intent refinement: same surface words can map
            # to different intents based on recent topic and domain continuity.
            if self.context_intent_refiner:
                try:
                    _recent_topic = ""
                    if hasattr(self.nlp, 'conversation_memory') and self.nlp.conversation_memory:
                        _recent_topic = self.nlp.conversation_memory.latest_topic()
                    _refined = self.context_intent_refiner.refine(
                        user_input=user_input,
                        intent=intent,
                        confidence=float(intent_confidence or 0.0),
                        recent_topic=_recent_topic,
                        last_intent=self.last_intent,
                    )
                    _new_intent = str(_refined.get('intent') or intent)
                    _new_conf = float(_refined.get('confidence', intent_confidence) or intent_confidence)
                    if _new_intent != intent or abs(_new_conf - float(intent_confidence or 0.0)) >= 0.01:
                        self._think(
                            f"Context refine [{_refined.get('reason', 'n/a')}] → {intent!r} ({float(intent_confidence or 0.0):.2f}) "
                            f"to {_new_intent!r} ({_new_conf:.2f})"
                        )
                    intent, intent_confidence = _new_intent, _new_conf
                except Exception:
                    pass

                # ── Slot inheritance ──────────────────────────────────────────
                # When a follow-up is detected, the NLP processor sees only the
                # current short utterance so slots like city/song/note-title
                # that were extracted on the PREVIOUS turn are lost.  Merge them
                # in — current-turn values always win, prior values fill gaps.
                _prior_entities = getattr(self.nlp.context, 'last_entities', {}) if hasattr(self.nlp, 'context') else {}
                if _prior_entities:
                    _merged_entities: Dict = {}
                    for _k, _v in _prior_entities.items():
                        _merged_entities[_k] = _v
                    for _k, _v in entities.items():
                        _merged_entities[_k] = _v  # current turn wins
                    entities = _merged_entities

            # ── 1.4  Validation / clarification gate ─────────────────────────────
            # Now validates the *resolved* intent — follow-up corrections are
            # already applied above so mis-classified intents won't fire spurious
            # "Missing required" prompts.
            # Feature flag check for clarification prompts
            if hasattr(self.nlp, 'feature_flags') and self.nlp.feature_flags:
                show_clarification = self.nlp.feature_flags.is_enabled("nlp_validation_prompts")
            else:
                show_clarification = True  # Default enabled

            # Policy suppresses clarification when the user is frustrated or urgent.
            if policy.skip_clarification:
                show_clarification = False
                self.nlp_error_logger.log_clarification_skip(
                    user_input=user_input,
                    intent=intent,
                    confidence=intent_confidence,
                    mood=perception.inferred_mood,
                    reason="interaction_policy",
                    session_id=getattr(self, '_session_id', None),
                )
            # Also suppress if follow-up resolution changed the intent — the
            # original validation issues no longer apply to the new intent.
            if followup.was_followup:
                show_clarification = False

            validation_score = getattr(nlp_result, 'validation_score', 1.0)
            validation_issues = getattr(nlp_result, 'validation_issues', [])
            has_critical_issue = any("Missing required" in issue for issue in validation_issues)

            if show_clarification and (validation_score < 0.50 and has_critical_issue):
                self.metrics.track_clarification_prompt("validation_low")
                clarification_parts = []
                if validation_issues:
                    critical_issues = [i for i in validation_issues if "Missing required" in i]
                    if critical_issues:
                        clarification_parts.extend(critical_issues[:2])
                return self._generate_natural_response(
                    {
                        'type': 'clarification_prompt',
                        'reason': 'validation_low',
                        'details': clarification_parts,
                    },
                    'helpful',
                    None,
                    user_input,
                )

            # Coreference ambiguity clarification
            if hasattr(self.nlp.context, 'pending_clarification'):
                pending = self.nlp.context.pending_clarification
                if pending.get('type') == 'ambiguity' and show_clarification:
                    self.metrics.track_clarification_prompt("ambiguity")
                    candidates = pending.get('candidates', [])
                    if len(candidates) > 1:
                        self.nlp.context.pending_clarification = {}
                        return self._generate_natural_response(
                            {
                                'type': 'clarification_prompt',
                                'reason': 'ambiguity_matches',
                                'options': list(candidates[:5]),
                            },
                            'helpful',
                            None,
                            user_input,
                        )

            self._think(f"NLP → intent={intent!r} confidence={intent_confidence}")
            self.structured_logger.debug(
                "NLP processing complete",
                intent=intent,
                confidence=intent_confidence,
                entities_count=len(entities),
                duration_ms=round(nlp_duration * 1000, 2),
                validation_score=validation_score,
                validation_issues_count=len(validation_issues),
            )
            if entities:
                self._think(f"     entities={str(entities)[:120]}...")

            # Normalize weather time-range asks so first-turn queries like
            # "weather for the rest of the week" route to forecast, not current.
            intent, intent_confidence = self._normalize_weather_intent_for_time_range(
                user_input=user_input,
                intent=intent,
                intent_confidence=float(intent_confidence or 0.0),
            )

            # ── 1.5  Reference resolution ─────────────────────────────────────────
            if hasattr(self, 'conversation_context'):
                pronouns = ['it', 'that', 'this', 'them', 'he', 'she', 'they']
                words = user_input.lower().split()
                for pronoun in pronouns:
                    if pronoun in words:
                        resolved = self.conversation_context.resolve_reference(pronoun)
                        if resolved:
                            # Domain-aware guard: don't apply a cross-domain entity
                            # resolution when the current intent is already anchored to
                            # a specific domain.  E.g. after a notes query, 'it' should
                            # not resolve to the last weather entity.
                            _intent_domain = intent.split(':')[0] if ':' in intent else intent
                            _resolved_lower = resolved.lower()
                            _weather_entity = any(
                                kw in _resolved_lower
                                for kw in ('weather', 'forecast', 'temperature',
                                           'rain', 'sunny', 'cloud', '°c', '°f')
                            )
                            _skip = _weather_entity and _intent_domain in (
                                'notes', 'email', 'calendar', 'reminder'
                            )
                            if not _skip:
                                self._think(f"Reference resolution: '{pronoun}' → '{resolved}'")
                                if 'resolved_reference' not in entities:
                                    entities['resolved_reference'] = resolved
                        break

            
            # FAST PATH: Check cache and conversational shortcuts
            # Keep adjusted follow-up confidence if it was updated above.

            # ── REMINDER FAST PATH ────────────────────────────────────────────────
            # Handle reminder:set / reminder:list / reminder:cancel before any
            # plugin dispatch or LLM fallback so they are always crisp and instant.
            if intent == "reminder:set":
                proactive = getattr(self, "proactive_assistant", None)
                if proactive:
                    trigger = parse_reminder_time(user_input)
                    if trigger is None and self.temporal_reasoner:
                        _tt = self.temporal_reasoner.parse_temporal_task(user_input)
                        if _tt and _tt.when_iso:
                            try:
                                trigger = datetime.fromisoformat(_tt.when_iso)
                            except Exception:
                                trigger = None
                    if trigger is None:
                        from datetime import timedelta
                        trigger = datetime.now() + timedelta(minutes=30)
                    # Extract what to be reminded about: text after "remind me to/about"
                    match = re.search(
                        r"remind\s+me\s+(?:to|about|that)?\s+(.+?)(?:\s+(?:in|at|tomorrow|tonight|on)\s+|$)",
                        user_input.lower(),
                    )
                    if match:
                        subject = match.group(1).strip()
                    else:
                        subject = re.sub(
                            r"\b(?:remind\s+me|set\s+a?\s*reminder)\b",
                            "",
                            user_input,
                            flags=re.IGNORECASE,
                        ).strip() or user_input

                    rid = make_reminder_id()
                    proactive.add_reminder(
                        reminder_id=rid,
                        message=f"Reminder: {subject}",
                        trigger_time=trigger,
                    )
                    try:
                        time_str = trigger.strftime("%I:%M %p").lstrip("0") or trigger.strftime("%I:%M %p")
                    except Exception:
                        time_str = str(trigger)
                    response = f"Got it — I'll remind you to {subject} at {time_str}."
                    self._store_interaction(user_input, response, intent, entities)
                    return response
                else:
                    return "Reminder system is not available right now."

            if intent == "reminder:list":
                proactive = getattr(self, "proactive_assistant", None)
                if proactive:
                    pending = proactive.list_reminders()
                    if not pending:
                        return "You have no pending reminders."
                    lines = []
                    for r in pending[:10]:
                        try:
                            t = r.trigger_time.strftime("%a %b %d at %I:%M %p")
                        except Exception:
                            t = str(r.trigger_time)
                        lines.append(f"• {r.message}  ({t})")
                    return "Your pending reminders:\n" + "\n".join(lines)
                else:
                    return "Reminder system is not available right now."

            if intent == "reminder:cancel":
                proactive = getattr(self, "proactive_assistant", None)
                if proactive:
                    kw_match = re.search(
                        r"(?:cancel|delete|remove)\s+(?:the\s+)?reminder\s+(?:about|for|to)?\s*(.+)",
                        user_input.lower(),
                    )
                    keyword = kw_match.group(1).strip() if kw_match else user_input
                    if proactive.cancel_reminder(keyword):
                        return f"Done — I've cancelled the reminder about \"{keyword}\"."
                    else:
                        return f"I couldn't find a reminder matching \"{keyword}\". Use 'show reminders' to see what's pending."
                else:
                    return "Reminder system is not available right now."
            # ─────────────────────────────────────────────────────────────────────

            # 0.54 CORRECTION/DISAGREEMENT GUARD: if the user is expressing that
            # ALICE got something wrong, don't re-invoke a plugin — let the LLM
            # handle it conversationally with an empathetic tone.
            _correction_markers = (
                "you are wrong", "you're wrong", "that's wrong", "that is wrong",
                "wrong answer", "not right", "that's not right", "that is not right",
                "you got it wrong", "that was wrong", "that's incorrect", "that is incorrect",
            )
            _input_for_correction = user_input.strip().lower().rstrip("?!")
            if any(_input_for_correction == m or _input_for_correction.startswith(m)
                   for m in _correction_markers):
                self._think("Correction/disagreement detected → skipping plugins, using LLM")
                _prev_intent = str(getattr(self, '_last_routed_intent', '') or intent)
                if getattr(self, 'cognitive_orchestrator', None):
                    try:
                        self.cognitive_orchestrator.ingest_user_feedback(
                            user_input=user_input,
                            previous_intent=_prev_intent,
                            corrected_intent="",
                            severity=0.9,
                        )
                    except Exception:
                        pass
                if getattr(self, 'executive_controller', None):
                    try:
                        # Immediate effect: bias next routing turn toward clarification/LLM.
                        self.executive_controller.apply_reflection(
                            {
                                "routing_adjustments": {
                                    "clarify": 0.08,
                                    "llm": 0.05,
                                    "tools": -0.06,
                                    "search": -0.03,
                                }
                            }
                        )
                    except Exception:
                        pass
                if self.adaptive_intent_calibrator:
                    self.adaptive_intent_calibrator.record_feedback(
                        _prev_intent,
                        was_correct=False,
                    )
                intent = "conversation:general"
                intent_confidence = 0.9

            # 0.545 ISSUE-REPORT GUARD: long reflective debugging statements
            # (about intents/routing/recommendations) should not be tool-routed.
            if self._is_issue_report_input(user_input):
                _tool_domains = (
                    "notes:", "email:", "calendar:", "file_operations:",
                    "memory:", "reminder:", "system:", "weather:",
                )
                if any(intent.startswith(domain) for domain in _tool_domains):
                    self._think("Issue-report text detected → forcing conversational routing")
                    intent = "conversation:general"
                    intent_confidence = min(float(intent_confidence or 0.5), 0.55)

            # 0.546 LOW-CONFIDENCE TOOL GUARD: if NLP predicts a tool domain but
            # the utterance has no explicit action cues, keep it conversational.
            _tool_domains = (
                "notes:", "email:", "calendar:", "file_operations:",
                "memory:", "reminder:", "system:", "weather:", "time:",
            )
            if (
                any(intent.startswith(domain) for domain in _tool_domains)
                and float(intent_confidence or 0.0) < 0.82
                and not self._has_explicit_action_cue(user_input)
            ):
                self._think("Low-confidence tool intent without action cues → conversational fallback")
                intent = "conversation:general"
                intent_confidence = min(float(intent_confidence or 0.5), 0.58)

            # 0.547 WAKE-WORD ACK: short nudges like "alice" should feel
            # immediate and natural, not trigger tools or verbose LLM output.
            if self._is_wake_word_only_input(user_input):
                self._think("Wake-word-only input → concise conversational acknowledgment")
                _user_name = getattr(getattr(self, 'context', None), 'user_prefs', None)
                _user_name = getattr(_user_name, 'name', '') if _user_name else ''
                response = self._wake_word_acknowledgment(_user_name)
                if response:
                    self._store_interaction(user_input, response, 'conversation:general', entities)
                    if use_voice and self.speech:
                        self.speech.speak(response, blocking=False)
                    logger.info(f"A.L.I.C.E: {response[:100]}...")
                    return response
                # If no learned or teacher phrasing is available, continue normal
                # conversational flow rather than returning a hardcoded line.
                intent = 'conversation:general'
                intent_confidence = min(float(intent_confidence or 0.5), 0.6)

            # 0.55 PRIORITY WEATHER FOLLOW-UP PATH: handle weather context BEFORE
            # conversational fast path so weather follow-ups don't depend on LLM.
            weather_followup_early = None
            weather_followup_indicators = [
                'weather', 'week', 'weekend', 'tomorrow', 'tonight',
                'umbrella', 'jacket', 'coat', 'layer', 'wear', 'outside',
                'rain', 'snow', 'cold', 'warm', 'forecast',
                'scarf', 'hat', 'gloves', 'boots', 'sweater', 'hoodie',
            ]
            _wfu_input_lower = user_input.lower()
            # Only attempt weather fast-path when the NLP intent is NOT already
            # resolved to a different high-confidence domain (notes, email, calendar, etc.)
            _non_weather_domains = ('notes:', 'email:', 'calendar:', 'file_operations:',
                                    'memory:', 'reminder:', 'system:', 'conversation:')
            _intent_is_non_weather = any(intent.startswith(d) for d in _non_weather_domains)
            _wfu_hit = (
                not _intent_is_non_weather
                and (
                    'weather' in intent.lower()
                    or any(
                        re.search(r'\b' + re.escape(kw) + r'\b', _wfu_input_lower)
                        for kw in weather_followup_indicators
                    )
                )
            )
            if _wfu_hit:
                weather_followup_early = self._handle_weather_followup(user_input, intent)
            if weather_followup_early:
                self._last_had_stored_data = True   # fast-path used stored data; no plugin needed
                self._think("Weather follow-up (early) → answering from stored data (no LLM)")
                self._store_interaction(user_input, weather_followup_early, intent, entities)
                if use_voice and self.speech:
                    self.speech.speak(weather_followup_early, blocking=False)
                logger.info(f"A.L.I.C.E: {weather_followup_early[:100]}...")
                return weather_followup_early
            
            # 0.5 Check if cached response exists (with 5-minute expiry for variation)
            # Cache conversational intents for faster responses to repeated questions
            cacheable_intents = [
                'conversation:ack',
                'conversation:general',
                'farewell'
                # Removed 'status_inquiry' - we want varied responses to "how are you?"
            ]
            if intent in cacheable_intents:
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

                # If this is a true greeting utterance, respond directly via gateway
                explicit_greeting_input = self._is_explicit_greeting_input(user_input)
                if intent == "greeting" and not explicit_greeting_input:
                    self._think("Intent labeled greeting, but input is not explicit greeting → continue normal routing")
                    intent = "conversation:general"
                    intent_confidence = min(intent_confidence, 0.55)

                if intent == "greeting" and explicit_greeting_input and getattr(self, "llm_gateway", None):
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
                        response = self._clamp_final_response(
                            llm_response.response,
                            tone='casual and friendly',
                            response_type='greeting',
                            route='greeting',
                            user_input=user_input,
                        )
                    else:
                        # No hardcoded greetings: use learned greetings if LLM unavailable/denied
                        response = self._learned_greeting_response(
                            user_input=user_input,
                            user_name=user_name,
                            asked_how=asked_how,
                        ) or llm_response.response
                    
                    if response and not asked_how:
                        response = re.sub(
                            r"\b(I\s*'m|I\s+am)\s+(doing\s+)?(well|good|great|fine)\b[.!]*\s*",
                            "",
                            response,
                            flags=re.IGNORECASE
                        ).strip()
                        if not response and getattr(self, "llm_gateway", None):
                            # Retry via gateway without wellbeing
                            retry_response = self.llm_gateway.request(
                                prompt="Give a short greeting only. Do not comment on your wellbeing.",
                                call_type=LLMCallType.CHITCHAT,
                                use_history=False,
                                user_input=user_input
                            )
                            response = (
                                self._clamp_final_response(
                                    retry_response.response,
                                    tone='casual and friendly',
                                    response_type='greeting',
                                    route='greeting',
                                    user_input=user_input,
                                )
                                if retry_response.success and retry_response.response
                                else None
                            )
                    if response:
                        # Store as learned pattern for future use
                        if getattr(self, 'learning_engine', None):
                            self.learning_engine.collect_interaction(
                                user_input=user_input,
                                assistant_response=response,
                                intent='greeting',
                                entities=entities or {},
                                quality_score=0.95  # High quality - greetings are straightforward
                            )

                        # LEARN phrasing so A.L.I.C.E can greet independently
                        if self.phrasing_learner:
                            self.phrasing_learner.record_phrasing(
                                alice_thought={
                                    'type': 'greeting',
                                    'data': {
                                        'user_input': user_input,
                                        'user_name': user_name,
                                        'asked_how': asked_how
                                    }
                                },
                                ollama_phrasing=response,
                                context={
                                    'tone': 'friendly',
                                    'intent': 'greeting',
                                    'user_input': user_input
                                }
                            )

                        # Don't add to conversational_engine.learned_greetings
                        # Always use LLM for greetings to get fresh, varied responses
                        self._think("LLM → greeting response (will be learned)")

                        # Cache the greeting response (5-minute expiry for variation)
                        self._cache_put(user_input, intent, response)

                        self._store_interaction(user_input, response, intent, entities)
                        if use_voice and self.speech:
                            self.speech.speak(response, blocking=False)
                        logger.info(f"A.L.I.C.E: {response[:100]}...")
                        return response

            # Store for active learning
            self.last_user_input = user_input
            self.last_nlp_result = vars(nlp_result).copy()  # Convert to dict for compatibility
            # Track turn index when intent last changed (for follow-up decay)
            _prev_intent = getattr(self, 'last_intent', '')
            _new_domain = intent.split(':')[0] if ':' in intent else intent
            _old_domain = _prev_intent.split(':')[0] if ':' in _prev_intent else _prev_intent
            if _new_domain != _old_domain:
                self._last_intent_turn = getattr(getattr(self.nlp, 'context', None), 'turn_index', 0)
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
                original_intent_before_al = intent
                logger.info("Active learning improved NLP result")
                intent = improved_nlp_result['intent']
                entities = improved_nlp_result['entities']
                sentiment = improved_nlp_result.get('sentiment', sentiment)
                self._think(f"     active learning → intent={intent!r}")
                # Log as a training signal so the NLP layer can self-tune
                if intent != original_intent_before_al:
                    self.nlp_error_logger.log_intent_override(
                        user_input=user_input,
                        original_intent=original_intent_before_al,
                        corrected_intent=intent,
                        original_confidence=intent_confidence,
                        corrected_confidence=improved_nlp_result.get('intent_confidence', intent_confidence),
                        reason="active_learning",
                        session_id=getattr(self, '_session_id', None),
                    )

            # ── BayesianIntentRouter: minimise expected regret across candidates ─
            # Build candidate list from primary intent + plugin_score runners-up,
            # then let the router apply its cost matrix before final routing.
            try:
                if getattr(self, 'bayesian_router', None) is not None and intent:
                    _ps = getattr(nlp_result, 'plugin_scores', {}) or {}
                    _candidates: List[IntentCandidate] = [
                        IntentCandidate(
                            intent=intent,
                            confidence=float(intent_confidence or 0.5),
                        )
                    ]
                    # Derive the plugin prefix of the primary intent (e.g. "notes"
                    # from "notes:search") so we can skip redundant :general
                    # alternatives that would silently downgrade a specific intent.
                    _primary_plugin = intent.split(':')[0] if ':' in intent else intent
                    for _alt_plugin, _alt_score in sorted(
                        _ps.items(), key=lambda x: x[1], reverse=True
                    )[:3]:
                        # Normalise bare plugin name to a colon-separated intent
                        _alt_intent = (
                            _alt_plugin if ':' in _alt_plugin
                            else f"{_alt_plugin}:general"
                        )
                        # Skip alternatives that are just a :general downgrade of
                        # the already-specific primary intent (e.g. don't add
                        # notes:general when intent is already notes:search).
                        _alt_plugin_prefix = _alt_plugin.split(':')[0]
                        if _alt_plugin_prefix == _primary_plugin and _alt_intent.endswith(':general'):
                            continue
                        if _alt_intent != intent:
                            _candidates.append(
                                IntentCandidate(
                                    intent=_alt_intent,
                                    confidence=float(_alt_score),
                                )
                            )
                    _rd = self.bayesian_router.decide(
                        _candidates,
                        user_priors=getattr(self.user_profile, 'get_intent_priors', lambda: {})(),
                    )
                    if _rd.intent != intent:
                        # Never downgrade a high-confidence specific intent to
                        # a bare `:general` catch-all — let the NLP win.
                        _is_downgrade = (
                            _rd.intent.endswith(':general')
                            and not intent.endswith(':general')
                            and intent_confidence >= 0.85
                        )
                        if not _is_downgrade:
                            self._think(
                                f"BayesianRouter overrides {intent!r} → {_rd.intent!r} "
                                f"(expected_regret={_rd.expected_regret:.3f})"
                            )
                            intent = _rd.intent
                    intent_confidence = _rd.calibrated_confidence
            except Exception as _br_err:
                logger.debug("BayesianIntentRouter skipped: %s", _br_err)

            # Adaptive confidence calibration (correction-driven, no full retrain).
            if self.adaptive_intent_calibrator:
                intent_confidence = self.adaptive_intent_calibrator.calibrate(
                    intent,
                    float(intent_confidence or 0.0),
                )

            _secondary_intents = self._collect_secondary_intents(nlp_result)
            _turn_reasoning_plan = None
            if self.multi_step_reasoning_engine:
                try:
                    _turn_reasoning_plan = self.multi_step_reasoning_engine.plan_turn(
                        user_input=user_input,
                        primary_intent=intent,
                        primary_confidence=float(intent_confidence or 0.0),
                        secondary_intents=_secondary_intents,
                    )
                except Exception:
                    _turn_reasoning_plan = None

            # Store policy hints for downstream tone/length use
            self._last_policy = policy
            self._last_perception = perception
            self._last_intent_confidence = intent_confidence
            self._last_routed_intent = intent
            self._last_routed_confidence = float(intent_confidence or 0.0)

            # World graph: persist entities/topics from this resolved turn
            try:
                if self.world_graph is not None:
                    _wg_entities = {
                        k: str(v) for k, v in (entities or {}).items()
                        if isinstance(v, (str, int, float)) and v
                    }
                    _wg_topics = list(self.conversation_topics[-5:]) if self.conversation_topics else []
                    self.world_graph.update_from_turn(
                        intent=intent,
                        entities=_wg_entities,
                        sentiment=self._last_sentiment if hasattr(self, '_last_sentiment') else None,
                        topics=_wg_topics,
                    )
            except Exception as _wg_err:
                logger.debug("WorldGraph update skipped: %s", _wg_err)
            # Store sentiment for use in conversation summarizer
            self._last_sentiment = sentiment['category'] if sentiment else None
            
            logger.info(f"Intent: {intent}, Sentiment: {sentiment['category']}")
            self._track_activity_signal(intent, user_input)

            explain_response = self._handle_explain_command(user_input)
            if explain_response:
                return explain_response

            advanced_reasoning_response = self._handle_advanced_reasoning_queries(user_input)
            if advanced_reasoning_response:
                return advanced_reasoning_response

            operator_response = self._handle_operator_request(user_input)
            if operator_response:
                return operator_response
            
            # 1.5.5. Check for code/self-reflection requests EARLY (before reasoning/plugins)
            effective_input = _nlp_input
            code_response = self._handle_code_request(effective_input, entities)
            if code_response:
                self._think("Code request detected → handled directly")

                # Keep shared conversation state in sync even on early returns
                # from direct code handling paths.
                _code_entities = dict(entities or {})
                _code_intent = "code:request"
                if self.code_context.get('last_action') == 'code_access_confirmed':
                    _code_entities.setdefault('resolved_reference', 'internal code')
                    _code_entities.setdefault('subject', 'codebase')
                    _code_entities.setdefault('topic', 'internal code')
                self._store_interaction(effective_input, code_response, _code_intent, _code_entities)

                # Store in training data
                if getattr(self, 'learning_engine', None):
                    self.learning_engine.collect_interaction(
                        user_input=effective_input,
                        assistant_response=code_response,
                        intent='code:request',
                        entities=_code_entities,
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
                        self._store_interaction(user_input, response, intent, entities)
                        return response
                    else:
                        self._think("No learned patterns yet → will use Ollama")
            
            # 1.5.8. Weather follow-up routing (use stored weather data, no LLM)
            # Skip if the early fast-path already attempted this (avoids duplicate detection logs)
            weather_followup = None if _wfu_hit else self._handle_weather_followup(user_input, intent)
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
                        cancel_ack_intents = {
                            "conversation:ack",
                            "cancel",
                            "task:cancel",
                            "email:cancel",
                        }
                        if intent in cancel_ack_intents:
                            self._think("Goal cancelled → returning ack")
                            return goal_res.message
                        self._think(f"Goal cancelled → continuing with intent={intent!r}")
                    if goal_res.revised:
                        intent, entities = goal_res.intent, goal_res.entities
                        self._think(f"Goal revised → intent={intent!r}")
                    elif goal_res.goal:
                        self._think(f"Goal: {goal_res.goal.description[:60]}...")
                except Exception as e:
                    logger.warning(f"Reference/goal resolution skipped: {e}")
                    # Graceful degradation - continue without resolution
            
            # 3. General Entity Detection (for non-email interactions)
            if self.advanced_context and not (intent and intent.startswith("email")):
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
            intent, entities = self._promote_learning_goal_intent(
                user_input_processed, intent, entities
            )
            planned_response = None
            try:
                _pre_conv_state = {}
                if getattr(self, 'conversation_state_tracker', None):
                    _pre_conv_state = self.conversation_state_tracker.get_state_summary()
                _pre_exec_state = self.executive_controller.build_state(
                    user_input=user_input_processed,
                    intent=intent,
                    confidence=float(intent_confidence or 0.0),
                    entities=entities or {},
                    conversation_state=_pre_conv_state,
                )
                _pre_scores = self.executive_controller.score_decisions(
                    _pre_exec_state,
                    is_pure_conversation=False,
                    has_explicit_action_cue=bool(self._has_explicit_action_cue(user_input_processed)),
                    has_active_goal=bool(goal_res and goal_res.goal),
                    force_plugins_for_notes=False,
                )
                _plan_allowed = self.executive_controller.should_use_planner(_pre_exec_state, _pre_scores)
                if _plan_allowed:
                    planned_response = self._use_planner_executor(intent, entities, user_input_processed)
                    if planned_response:
                        self._internal_reasoning_state = {
                            **_pre_exec_state.as_dict(),
                            "decision_scores": _pre_scores,
                            "planner_path": True,
                        }
            except Exception:
                planned_response = self._use_planner_executor(intent, entities, user_input_processed)
            if planned_response:
                planned_response = self._executive_apply_response_gate(
                    user_input=user_input,
                    intent=intent,
                    response=planned_response,
                    route="llm",
                )
                # Log action for pattern learning
                action = f"{intent}:{entities.get('topic', entities.get('query', 'general'))}"
                self._log_action_for_learning(action)
                
                # Store interaction
                self._store_interaction(user_input, planned_response, intent, entities)
                self.last_assistant_response = planned_response

                self._run_executive_reflection(
                    user_input=user_input,
                    intent=intent,
                    response=planned_response,
                    route="llm",
                    prior_confidence=float(intent_confidence or 0.0),
                )
                
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
                            content = self._clean_email_body(content)
                            
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
            if intent and intent.startswith("email"):
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
                        content = self._clean_email_body(content)
                        
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
                "send", "search", "list", "check", "email", "note", "notes", "calendar",
                "file", "weather", "time", "how many", "what's", "show", "find", "remind",
            )
            
            # If there's an active goal, use it to enhance intent understanding
            active_goal = goal_res.goal if goal_res else None
            intent_candidates = getattr(nlp_result, 'intent_candidates', []) or []
            intent_plausibility = float(getattr(nlp_result, 'intent_plausibility', 1.0) or 1.0)
            if (
                active_goal
                and intent_confidence < 0.7
                and intent_plausibility >= 0.60
                and self._should_reuse_goal_intent(user_input, active_goal.description)
            ):
                self._think(f"Low confidence ({intent_confidence:.2f}) but active goal → using goal intent: {active_goal.intent}")
                intent = active_goal.intent
                entities = {**(entities if entities else {}), **(active_goal.entities if active_goal.entities else {})}
                intent_confidence = 0.8
            else:
                if active_goal and intent_confidence < 0.7 and not self._should_reuse_goal_intent(user_input, active_goal.description):
                    self._think("Topic shift (low overlap with goal) → not reusing goal intent")    
                elif active_goal and intent_confidence < 0.7 and intent_plausibility < 0.60:
                    self._think("Low plausibility detected → not reusing goal intent")
            
            _is_short_followup = (
                len(user_input.split()) <= 12
                and not any(w in user_input.lower() for w in _cmd_words)
                and not active_goal  # Don't skip if there's an active goal
            )
            notes_plugin = None
            if hasattr(self, 'plugins') and getattr(self.plugins, 'plugins', None):
                notes_plugin = self.plugins.plugins.get('Notes Plugin')

            has_note_context = bool(getattr(notes_plugin, 'last_note_id', None)) if notes_plugin else False
            force_plugins_for_note_followup = bool(
                has_note_context
                and re.search(r"\b(?:it|this|that|the\s+note|my\s+note)\b", user_input.lower())
                and re.search(r"\b(?:what|read|show|in|inside|content|contents|contains?)\b", user_input.lower())
            )
            force_plugins_for_explicit_note_read = bool(
                re.search(r"\b(?:read|show|open|what(?:'s|\s+is)?|inside|content|contents?)\b", user_input.lower())
                and re.search(r"\bnotes?\b", user_input.lower())
            )
            force_plugins_for_notes = force_plugins_for_note_followup or force_plugins_for_explicit_note_read

            _conversation_state = {}
            if getattr(self, 'conversation_state_tracker', None):
                _conversation_state = self.conversation_state_tracker.get_state_summary()
            _cognitive_guidance = {}
            if getattr(self, 'cognitive_orchestrator', None):
                try:
                    _cognitive_guidance = self.cognitive_orchestrator.get_runtime_guidance()
                except Exception:
                    _cognitive_guidance = {}

            if _cognitive_guidance:
                _conversation_state = {
                    **(_conversation_state or {}),
                    "route_bias": _cognitive_guidance.get("route_bias", "balanced"),
                    "tool_budget": _cognitive_guidance.get("tool_budget", 1),
                    "planner_hint": _cognitive_guidance.get("planner_hint", ""),
                    "planner_depth": _cognitive_guidance.get("thinking_depth", 1),
                }
                _fb = (_cognitive_guidance.get("feedback_adjustments", {}) or {}).get(
                    str(intent or "").lower(),
                    {},
                )
                _penalty = float(_fb.get("penalty", 0.0) or 0.0)
                if _penalty > 0.0:
                    intent_confidence = max(0.0, float(intent_confidence or 0.0) - _penalty)
                    self._think(f"Immediate feedback penalty applied ({_penalty:.2f}) → confidence={intent_confidence:.2f}")

            _entities_for_exec = dict(entities or {})
            _entities_for_exec["_intent_plausibility"] = float(intent_plausibility or 0.0)
            _executive_state = self.executive_controller.build_state(
                user_input=user_input,
                intent=intent,
                confidence=float(intent_confidence or 0.0),
                entities=_entities_for_exec,
                conversation_state=_conversation_state,
            )

            _pre_response_plan = None
            if getattr(self, 'response_planner', None):
                try:
                    _pre_response_plan = self.response_planner.plan(
                        user_input=user_input,
                        intent=intent,
                        reasoning_state=_executive_state.as_dict(),
                        conversation_state=_conversation_state,
                    )
                    _executive_state.planner_depth = max(
                        int(_executive_state.planner_depth or 1),
                        int(getattr(_pre_response_plan, 'plan_depth', 1) or 1),
                    )
                    if getattr(_pre_response_plan, 'strategy', '') in (
                        'guided_explanation',
                        'incremental_teaching',
                        'expand',
                    ):
                        _executive_state.planner_hint = 'increase_structure_depth'
                    if (
                        getattr(_pre_response_plan, 'response_type', '') in ('instruction', 'debugging', 'troubleshooting')
                        and bool(self._has_explicit_action_cue(user_input))
                    ):
                        _executive_state.route_bias = 'tool_first'
                except Exception:
                    _pre_response_plan = None

            _pre_route_guard = self.executive_controller.should_preempt_for_plausibility(
                _executive_state,
                has_explicit_action_cue=bool(self._has_explicit_action_cue(user_input)),
                intent_candidates=intent_candidates,
            )
            if _pre_route_guard.get("block"):
                self._internal_reasoning_state = {
                    **_executive_state.as_dict(),
                    "intent_candidates": intent_candidates,
                    "intent_plausibility": intent_plausibility,
                    "pre_route_guard": _pre_route_guard,
                    "cognitive_guidance": _cognitive_guidance,
                    "response_preferences": _response_prefs,
                    "response_style": _response_style,
                    "secondary_intents": _secondary_intents,
                }
                if _turn_reasoning_plan is not None:
                    self._internal_reasoning_state["reasoning_planner"] = _turn_reasoning_plan.as_dict()
                self._think(f"Pre-route plausibility guard → {_pre_route_guard.get('reason', 'unknown')}")
                return _pre_route_guard.get("question") or (
                    "I need one clarifying detail before I route this request."
                )

            _decision_scores = self.executive_controller.score_decisions(
                _executive_state,
                is_pure_conversation=bool(is_pure_conversation),
                has_explicit_action_cue=bool(self._has_explicit_action_cue(user_input)),
                has_active_goal=bool(active_goal),
                force_plugins_for_notes=bool(force_plugins_for_notes),
            )
            _runtime_controls = self.executive_controller.derive_runtime_controls(
                _executive_state,
                _decision_scores,
            )
            self._internal_reasoning_state = {
                **_executive_state.as_dict(),
                "decision_scores": _decision_scores,
                "intent_candidates": intent_candidates,
                "intent_plausibility": intent_plausibility,
                "runtime_controls": _runtime_controls,
                "cognitive_guidance": _cognitive_guidance,
                "response_preferences": _response_prefs,
                "response_style": _response_style,
                "secondary_intents": _secondary_intents,
            }
            if _turn_reasoning_plan is not None:
                self._internal_reasoning_state["reasoning_planner"] = _turn_reasoning_plan.as_dict()
            if _pre_response_plan is not None:
                self._internal_reasoning_state["response_plan"] = _pre_response_plan.as_dict()
            _executive_decision = self.executive_controller.decide(
                _executive_state,
                is_pure_conversation=bool(is_pure_conversation),
                has_explicit_action_cue=bool(self._has_explicit_action_cue(user_input)),
                has_active_goal=bool(active_goal),
                force_plugins_for_notes=bool(force_plugins_for_notes),
            )
            _learning_decision = self.executive_controller.decide_learning(
                relevance=max(_decision_scores.get("llm", 0.0), _decision_scores.get("tools", 0.0), _decision_scores.get("search", 0.0)),
                confidence=float(intent_confidence or 0.0),
                novelty=0.70 if (_executive_state.topic or _executive_state.user_goal) else 0.40,
                risk=0.60 if _executive_decision.action in ("ask_clarification", "ignore") else 0.20,
            )
            self._internal_reasoning_state["learning_decision"] = _learning_decision
            self._exec_should_store_memory = _learning_decision in ("store", "temporary") and bool(_executive_decision.store_memory)

            # Response planning: decide structure before LLM is called
            if getattr(self, 'response_planner', None):
                try:
                    _resp_plan = _pre_response_plan or self.response_planner.plan(
                        user_input=user_input,
                        intent=intent,
                        reasoning_state=self._internal_reasoning_state,
                        conversation_state=_conversation_state,
                    )
                    self._internal_reasoning_state["response_plan"] = _resp_plan.as_dict()
                    self._think(
                        f"Response plan → type={_resp_plan.response_type}, "
                        f"strategy={_resp_plan.strategy}"
                    )
                    if _resp_plan.strategy == "ask_guiding_question":
                        return self.response_planner.guiding_question(_resp_plan, user_input)
                except Exception as _rp_err:
                    logger.debug(f"[ResponsePlanner] {_rp_err}")

            self._think(
                f"Executive decision → {_executive_decision.action} "
                f"({_executive_decision.reason})"
            )
            self._think(
                "Executive scores → "
                + ", ".join(f"{k}:{v:.2f}" for k, v in sorted(_decision_scores.items(), key=lambda kv: kv[1], reverse=True)[:4])
            )

            if _executive_decision.action == "ask_clarification":
                _uncertainty = self.executive_controller.format_candidate_uncertainty(
                    intent_candidates,
                    limit=3,
                )
                if _uncertainty:
                    return (
                        f"{_executive_decision.clarification_question or 'Can you clarify your intended outcome?'} "
                        f"{_uncertainty}"
                    )
                return _executive_decision.clarification_question or (
                    "Can you clarify what outcome you want so I can route this correctly?"
                )
            if _executive_decision.action == "defer":
                return "I am not confident enough to answer this safely yet. Can you add one more detail so I can continue?"
            if _executive_decision.action == "ignore":
                return "I need a bit more context to act on that."

            _tool_veto = self.executive_controller.should_veto_tool_execution(
                user_input=user_input,
                intent=intent,
                confidence=float(intent_confidence or 0.0),
                intent_plausibility=float(intent_plausibility or 0.0),
                intent_candidates=intent_candidates,
            )
            self._internal_reasoning_state["tool_veto"] = _tool_veto
            if _tool_veto.get("veto"):
                self._think(f"Executive veto → {_tool_veto.get('reason', 'unknown')}")
                return _tool_veto.get("question") or (
                    "I need a little clarification before I trigger a tool."
                )

            _force_skip_plugins = _executive_decision.action in ("use_llm", "answer_direct")
            _force_try_plugins = _executive_decision.action in ("use_plugin", "search")
            if not bool(_runtime_controls.get("allow_tools", True)):
                _force_skip_plugins = True
            if _runtime_controls.get("routing_preference") == "tool_first" and bool(_runtime_controls.get("allow_tools", True)):
                _force_try_plugins = True
            _should_try_plugins = (
                (_force_try_plugins or ((not _is_short_followup and not is_pure_conversation) or force_plugins_for_notes))
                and not _force_skip_plugins
            )

            if getattr(self, 'roadmap_stack', None):
                _route_choice = "tool" if _should_try_plugins else "clarify"
                _route_ok, _route_reason = self.roadmap_stack.route_contracts.validate(
                    route=_route_choice,
                    confidence=float(intent_confidence or 0.0),
                )
                if not _route_ok and _should_try_plugins:
                    self._think(f"Route-contract guard switched to clarify path: {_route_reason}")
                    _should_try_plugins = False

            if not _should_try_plugins:
                if is_pure_conversation:
                    self._think("Pure conversation → skipping plugins, using LLM")
                else:
                    self._think("Short follow-up (no command words, no goal) → skipping plugins, using LLM")
            else:
                self._think(f"Trying plugins... (confidence: {intent_confidence:.2f}, goal: {active_goal.description[:30] if active_goal else 'none'}...)")
            if _should_try_plugins:
                if getattr(self, 'roadmap_stack', None):
                    _allowed, _reason = self.roadmap_stack.secondary_safety.validate(
                        action_text=user_input,
                        has_approval=("confirm" in user_input.lower() or "operator approve" in user_input.lower()),
                    )
                    if not _allowed:
                        self._think(f"Secondary safety blocked tool path: {_reason}")
                        return "This looks like a high-impact action and needs explicit approval first."
                    if not self.roadmap_stack.rate_limiter.allow(cost=1.0):
                        self._think("Global rate limiter blocked tool path")
                        return "I am temporarily rate-limited for external/tool actions. Please retry shortly."

                if getattr(self, 'rbac_engine', None):
                    _rbac_decision = self.rbac_engine.authorize(
                        request=AccessRequest(
                            intent=intent,
                            user_input=user_input,
                            entities=entities or {},
                        )
                    )
                    if _rbac_decision.allowed and not _rbac_decision.requires_confirmation:
                        if (
                            getattr(self, 'approval_ledger', None)
                            and "confirm" in user_input.lower()
                            and getattr(_rbac_decision, 'required_scope', None) is not None
                        ):
                            self.approval_ledger.note_scope_approval(
                                _rbac_decision.required_scope.value,
                                ttl_seconds=300,
                            )
                    if not _rbac_decision.allowed:
                        self._think(f"RBAC blocked tool path: {_rbac_decision.reason}")
                        return "I am not allowed to run that action under the current permission tier."
                    if _rbac_decision.requires_confirmation:
                        if (
                            getattr(self, 'approval_ledger', None)
                            and getattr(_rbac_decision, 'required_scope', None) is not None
                            and self.approval_ledger.is_scope_approved(_rbac_decision.required_scope.value)
                        ):
                            self._think("Approval ledger scope-memory satisfied confirmation requirement")
                        else:
                            self._think("RBAC requires explicit confirmation for this action")
                            return _rbac_decision.reason

                # Confidence cascade: gate plugin execution on intent confidence
                _cascade = self._apply_confidence_cascade(intent, intent_confidence or 0.0, nlp_result)
                if _cascade["action"] == "clarify":
                    return _cascade["question"]
                if _cascade["action"] == "interpret":
                    return _cascade["question"]
                _low_conf_prefix = _cascade.get("marker", "")
                context_summary = self.context.get_context_summary()
                # Enhance context with goal info (keep as dict for plugins)
                if active_goal:
                    context_summary['active_goal'] = active_goal.description
                if nlp_result is not None:
                    context_summary['nlp'] = {
                        'intent': intent,
                        'intent_confidence': float(intent_confidence or 0.0),
                        'intent_candidates': intent_candidates,
                        'intent_plausibility': float(intent_plausibility or 0.0),
                        'parsed_command': getattr(nlp_result, 'parsed_command', {}) or {},
                        'plugin_scores': getattr(nlp_result, 'plugin_scores', {}) or {},
                        'token_debug': getattr(nlp_result, 'token_debug', [])[:30],
                        'dialogue_memory': self.dialogue_memory.entity_chain_dict() if self.dialogue_memory else [],
                    }
                # HTN planning: surface sub-task sequence for complex intents
                try:
                    if getattr(self, 'htn_planner', None) is not None and intent:
                        _plan = self.htn_planner.decompose(intent)
                        if len(_plan) > 1:
                            self._think(
                                f"HTN plan for {intent!r}: "
                                + " → ".join(_plan[:4])
                                + (" …" if len(_plan) > 4 else "")
                            )
                except Exception:
                    pass

                # CapabilityGraph: consult success-weighted routing before dispatch
                try:
                    if getattr(self, 'capability_graph', None) is not None and intent:
                        _best_cap = self.capability_graph.best_plugin_for_intent(intent)
                        if _best_cap:
                            self._think(
                                f"CapabilityGraph recommends {_best_cap!r} for {intent!r}"
                            )
                except Exception:
                    pass

                _schema_error = self._validate_tool_invocation_schema(
                    intent=intent,
                    user_input=user_input,
                    entities=entities,
                    context_summary=context_summary,
                )
                if _schema_error:
                    self._think(f"Tool invocation schema validation blocked dispatch: {_schema_error}")
                    return "I could not safely run that tool request due to invalid request shape. Please rephrase."

                _plugin_name = "unknown"
                _action_name = "unknown"
                if ":" in (intent or ""):
                    _plugin_name, _action_name = intent.split(":", 1)
                    _plugin_name = (_plugin_name or "unknown").strip().lower()
                    _action_name = (_action_name or "unknown").strip().lower()

                # Track plugin execution timing
                plugin_start = time.time()
                action_request = ActionRequest(
                    goal=(active_goal.description if active_goal else user_input),
                    plugin=_plugin_name,
                    action=_action_name,
                    params={
                        **(entities or {}),
                        "_raw_query": user_input,
                        "_approval_token": bool(
                            "confirm" in user_input.lower() or "operator approve" in user_input.lower()
                        ),
                    },
                    source_intent=intent,
                    confidence=float(intent_confidence or 0.0),
                    # RBAC confirmation is already enforced above; do not re-block here.
                    requires_confirmation=False,
                    expected_outcome=(active_goal.description if active_goal else f"Complete {_action_name} successfully"),
                    target_spec={
                        "target": (entities or {}).get("target")
                        or (entities or {}).get("path")
                        or (entities or {}).get("filename")
                        or (entities or {}).get("title")
                        or (entities or {}).get("note_title")
                    },
                    risk_level=(
                        "high"
                        if ('_rbac_decision' in locals() and getattr(_rbac_decision, 'required_scope', None) is not None)
                        else "medium"
                    ),
                    retry_budget=(1 if _action_name in {"read", "list", "search"} else 0),
                    rollback_policy=("manual" if _action_name in {"delete", "update", "append"} else "none"),
                )
                action_result = self.action_engine.execute(action_request)
                self.action_engine.apply_state_updates(self, action_result)

                plugin_result = None
                if action_result.status != "not_handled":
                    plugin_result = action_result.to_plugin_result()

                if plugin_result:
                    _verification = action_result.state_updates.get("verification", {}) or {}
                    if getattr(self, 'roadmap_stack', None):
                        self.roadmap_stack.goal_tracker.record(
                            tool_succeeded=bool(action_result.success),
                            goal_satisfied=bool(action_result.goal_satisfied),
                        )

                plugin_duration = time.time() - plugin_start
                if plugin_result:
                    plugin_name = plugin_result.get('plugin', 'Unknown')
                    action = plugin_result.get('action', 'unknown')
                    success = plugin_result.get('success', False)
                    self.metrics.track_plugin(plugin_name, action, plugin_duration, success)
                    self.structured_logger.debug(
                        "Plugin execution complete",
                        plugin=plugin_name,
                        action=action,
                        success=success,
                        duration_ms=round(plugin_duration * 1000, 2)
                    )
                
                # Failure taxonomy: classify and log failed turns for retraining
                if self.failure_taxonomy and plugin_result:
                    self.failure_taxonomy.classify_and_log(
                        utterance=user_input,
                        intent=intent,
                        plugin_result=plugin_result,
                        nlp_result=vars(nlp_result) if nlp_result else {},
                    )

                # Capability graph + Habit miner: record execution signal
                if plugin_result:
                    _p_name = plugin_result.get('plugin', '')
                    _p_action = plugin_result.get('action', '')
                    _p_success = plugin_result.get('success', False)
                    _p_verified = bool((plugin_result.get('verification') or {}).get('accepted', False))
                    self._last_plugin_result = plugin_result
                    try:
                        if self.capability_graph is not None and _p_verified:
                            self.capability_graph.record_execution(_p_name, intent, _p_success)
                    except Exception:
                        pass
                    try:
                        if self.habit_miner is not None and _p_name and _p_verified:
                            self.habit_miner.update(intent, _p_name, _p_action)
                    except Exception:
                        pass
                # Update dialogue memory with plugin response titles
                if (
                    self.dialogue_memory
                    and plugin_result
                    and plugin_result.get('success')
                    and bool((plugin_result.get('verification') or {}).get('accepted', False))
                ):
                    _dm_titles = plugin_result.get('data', {}).get('titles', []) or []
                    if not _dm_titles:
                        _note_title = plugin_result.get('data', {}).get('note_title')
                        if _note_title:
                            _dm_titles = [_note_title]
                    if _dm_titles:
                        self.dialogue_memory.record_result_set('RESULT_SET', _dm_titles)
                # If low confidence and no plugin match, consider using LLM with clarification
                if not plugin_result and intent_confidence < 0.6:
                    self._think("Low confidence + no plugin match → using LLM with context")
            
            if plugin_result:
                # Plugin handled the request, regardless of success/failure
                plugin_name = plugin_result.get('plugin', 'Unknown')
                success = plugin_result.get('success', False)
                _tool_verified = bool((plugin_result.get('verification') or {}).get('accepted', False))
                self._think(f"Plugin → {plugin_name} success={success}")

                # Foundation 3 execution contract: do not mutate memory/state or
                # continue tool post-processing when verification fails.
                if not _tool_verified:
                    self._think("Tool output not verified → clarification before state updates")
                    response = (
                        plugin_result.get('message')
                        or plugin_result.get('response')
                        or "I could not verify that tool output was reliable. Please clarify or try again."
                    )
                    response = self._executive_apply_response_gate(
                        user_input=user_input,
                        intent=intent,
                        response=response,
                        route="plugin",
                    )
                    if use_voice and self.speech:
                        self.speech.speak(response, blocking=False)
                    return response

                # Tool-based architecture: Alice formulates response, then phrases it
                response = None
                alice_formulation = None  # Initialize to prevent UnboundLocalError

                # Create conversational context for this interaction
                context = ConversationalContext(
                    user_input=user_input,
                    intent=intent,
                    entities=entities,
                    recent_topics=self.conversation_topics[-3:] if self.conversation_topics else [],
                    active_goal=goal_res.goal if goal_res else None,
                    world_state=self.reasoning_engine if hasattr(self, 'reasoning_engine') else None,
                    plugin_data={
                        **(plugin_result.get('data', {}) if isinstance(plugin_result.get('data', {}), dict) else {}),
                        'action': plugin_result.get('action'),
                        'success': plugin_result.get('success', success),
                    }
                )

                # If plugin wants Alice to formulate the response, feed its data
                # back through the standard _formulate_response path so Alice's
                # direct-phrase architecture handles it (never bypasses to Ollama).
                if plugin_result.get('formulate', False):
                    self._think(f"Plugin requested formulation → Alice will learn to respond")

                # Foundation System Integration: Context-aware response generation
                if self.foundations and self.foundation_mode in ["primary", "exclusive"]:
                    try:
                        self._think(f"Foundation systems active (mode={self.foundation_mode}) → generating context-aware response")
                        foundation_result = self.foundations.process_interaction(
                            user_id=self.user_name,
                            user_input=user_input,
                            intent=intent,
                            entities=entities or {},
                            plugin_result=plugin_result
                        )
                        response = foundation_result.get('response')
                        route_taken = 'foundations'
                        self._think(f"Foundation response generated: {response[:100] if response else 'None'}...")
                    except Exception as e:
                        logger.error(f"Foundation system error: {e}", exc_info=True)
                        if self.foundation_mode == "exclusive":
                            response = "I encountered an issue formulating my response."
                        # If mode is "primary", fall through to existing system
                        
                elif self.foundations and self.foundation_mode == "parallel":
                    # Parallel mode: Run foundation system but still use old system for output
                    try:
                        self._think("Foundation parallel mode → running both systems for comparison")
                        foundation_result = self.foundations.process_interaction(
                            user_id=self.user_name,
                            user_input=user_input,
                            intent=intent,
                            entities=entities or {},
                            plugin_result=plugin_result
                        )
                        foundation_response = foundation_result.get('response')
                        logger.info(f"Foundation (parallel): {foundation_response[:100] if foundation_response else 'None'}...")
                    except Exception as e:
                        logger.error(f"Foundation parallel error: {e}", exc_info=True)

                # Existing flow: Formulate response if not already done
                if not response:
                    # Step 1: Alice formulates her response based on plugin data
                    alice_formulation = self._formulate_response(
                    user_input=user_input,
                    intent=intent,
                    entities=entities,
                    context=context
                )

                # Step 1.5: Handle self-analysis requests - Alice actually reads her code
                if alice_formulation and alice_formulation.get('type') == 'self_analysis_needed':
                    self._think("Self-analysis requested → reading actual codebase")

                    # Alice reads her own code
                    files = self.self_reflection.list_codebase()

                    # Analyze key architectural files
                    key_files = [
                        'app/main.py',
                        'ai/conversational_engine.py',
                        'ai/llm_engine.py',
                        'ai/phrasing_learner.py',
                        'ai/self_reflection.py'
                    ]

                    analysis_results = []
                    for file_path in key_files:
                        try:
                            file_info = self.self_reflection.analyze_file_advanced(file_path)
                            if 'error' not in file_info:
                                analysis_results.append({
                                    'path': file_path,
                                    'lines': file_info.get('lines', 0),
                                    'type': file_info.get('module_type', 'unknown'),
                                    'classes': file_info.get('classes', [])[:3],  # Top 3 classes
                                    'functions': file_info.get('functions', [])[:5]  # Top 5 functions
                                })
                        except Exception as e:
                            logger.debug(f"Could not analyze {file_path}: {e}")

                    # Formulate actual insights based on real code structure
                    alice_formulation = {
                        'type': 'self_analysis',
                        'total_files': len(files),
                        'analyzed_files': analysis_results,
                        'architecture_points': [
                            'Tool-based architecture with PhrasingLearner for progressive independence',
                            'Conversational engine handles pattern-based responses without LLM',
                            'Self-reflection system for code awareness',
                            'Plugin system for extensible capabilities',
                            'LLM gateway abstracts Ollama interactions'
                        ],
                        'confidence': 0.95
                    }

                # Step 2: Alice selects tone based on context
                tone = self._select_tone(intent, context, user_input)

                # Step 3: Generate natural response (with learning)
                if alice_formulation and alice_formulation.get('type') != 'general_response':
                    # Alice has a structured response - use learning loop
                    try:
                        response = self._generate_natural_response(
                            alice_response=alice_formulation,
                            tone=tone,
                            context=ConversationalContext(
                                user_input=user_input,
                                intent=intent,
                                entities=entities,
                                recent_topics=self.conversation_topics[-3:] if self.conversation_topics else [],
                                active_goal=goal_res.goal if goal_res else None,
                                world_state=self.reasoning_engine if hasattr(self, 'reasoning_engine') else None
                            ),
                            user_input=user_input
                        )
                        self._think("Alice formulated and phrased response using tool-based architecture")
                    except Exception as e:
                        logger.error(f"Error in tool-based response generation: {e}")
                        response = None

                # Fallback: Alice phrases directly — no Ollama
                if not response:
                    self._think("Response generation failed → Alice uses direct phrase fallback (no Ollama)")
                    if alice_formulation and alice_formulation.get('type') not in ('general_response', None):
                        # Try Alice's structured direct-phrase engine first
                        response = self._alice_direct_phrase(
                            alice_formulation.get('type', ''),
                            alice_formulation
                        )
                    if not response:
                        # Simple inline Alice formatter — no LLM involved
                        action = plugin_result.get('action', intent) if isinstance(plugin_result, dict) else intent
                        data = plugin_result.get('data', {}) if isinstance(plugin_result, dict) else {}
                        if success:
                            subject = (
                                data.get('note_title') or data.get('title') or
                                data.get('name') or data.get('filename') or ''
                            )
                            response = f"Done." if not subject else f"Done — {subject}."
                        else:
                            response = "I wasn't able to complete that."

                # Optimize plugin response
                if getattr(self, 'response_optimizer', None):
                    response = self.response_optimizer.optimize(response, intent, {"plugin": plugin_name})

                # Execute additional intents from compound requests.
                if _secondary_intents:
                    _sec_outcomes = self._execute_secondary_intents(
                        _secondary_intents,
                        entities,
                        context_summary,
                    )
                    if _sec_outcomes and self.multi_step_reasoning_engine:
                        _sec_summary = self.multi_step_reasoning_engine.summarize_outcomes(_sec_outcomes)
                        if _sec_summary:
                            response = f"{response}\n\n{_sec_summary}" if response else _sec_summary

                response = self._apply_response_style_constraints(response)
                
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

                    # Track which plugin ran (used by quality checker to detect unnecessary calls)
                    self._last_plugin_called = plugin_name

                    # Feature #3: Feed conversation turn into notes plugin so
                    # "save this" / "remember that" can capture recent context.
                    try:
                        for _np in (getattr(self.plugins, 'plugins', None) or []):
                            if hasattr(_np, 'record_conversation_turn'):
                                _np.record_conversation_turn("user", user_input)
                                if response:
                                    _np.record_conversation_turn("assistant", str(response)[:400])
                                break
                    except Exception:
                        pass
                    
                    # Store weather data for follow-up questions
                    if plugin_name == 'WeatherPlugin' and success and plugin_result.get('data'):
                        weather_data = plugin_result['data']
                        weather_data = dict(weather_data)
                        weather_data['captured_at'] = datetime.now().isoformat()
                        from ai.core.reasoning_engine import WorldEntity, EntityKind
                        message_code = weather_data.get('message_code')
                        if message_code == 'weather:forecast':
                            self._think(f"Storing weather forecast entity with {len(weather_data.get('forecast', []))} days")
                            self.reasoning_engine.add_entity(WorldEntity(
                                id='weather_forecast',
                                kind=EntityKind.TOPIC,
                                label=f"Weather forecast for {weather_data.get('location', 'your area')}",
                                data=weather_data,
                                aliases=['forecast', 'weather forecast', 'this week', 'next week', 'weekend']
                            ))
                            self._think(f"Weather forecast entity stored successfully")
                            # If the original query mentioned a specific time range,
                            # filter the response now that we have the forecast data.
                            _tr_kw = [
                                'tonight', 'tomorrow', 'this weekend', 'this week', 'next week',
                                'weekend', 'today', 'this evening',
                                'monday', 'tuesday', 'wednesday', 'thursday',
                                'friday', 'saturday', 'sunday',
                            ]
                            _q_lo = user_input.lower()
                            _mentioned_tr = next((k for k in _tr_kw if k in _q_lo), None)
                            if _mentioned_tr:
                                from ai.models.simple_formatters import WeatherFormatter
                                _tr_response = WeatherFormatter.format(
                                    weather_data, entities={'TIME_RANGE': [_mentioned_tr]}
                                )
                                if _tr_response:
                                    self._think(f"Filtered forecast for '{_mentioned_tr}' — replacing full 7-day table")
                                    response = _tr_response
                        else:
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
                            from ai.core.reasoning_engine import WorldEntity, EntityKind
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

                response = self._executive_apply_response_gate(
                    user_input=user_input,
                    intent=intent,
                    response=response,
                    route="plugin",
                )
                
                # Store interaction in memory
                self._store_interaction(user_input, response, intent, entities)

                self._run_executive_reflection(
                    user_input=user_input,
                    intent=intent,
                    response=response,
                    route="tools",
                    prior_confidence=float(intent_confidence or 0.0),
                )
                
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

            # Check for self-analysis requests BEFORE falling through to LLM
            input_lower = user_input.lower()
            if any(phrase in input_lower for phrase in [
                'go through your code', 'analyze your code', 'review your code',
                'look at your code', 'read your code', 'check your code',
                'analyze yourself', 'review yourself', 'improvements',
                'what can we improve', 'what can you improve', 'suggest improvements'
            ]):
                self._think("Self-analysis requested → reading actual codebase")

                # Alice reads her own code
                files = self.self_reflection.list_codebase()

                # Analyze key architectural files
                key_files = [
                    'app/main.py',
                    'ai/conversational_engine.py',
                    'ai/llm_engine.py',
                    'ai/phrasing_learner.py',
                    'ai/self_reflection.py'
                ]

                analysis_results = []
                for file_path in key_files:
                    try:
                        file_info = self.self_reflection.analyze_file_advanced(file_path)
                        if 'error' not in file_info:
                            analysis_results.append({
                                'path': file_path,
                                'lines': file_info.get('lines', 0),
                                'type': file_info.get('module_type', 'unknown'),
                                'classes': file_info.get('classes', [])[:3],
                                'functions': file_info.get('functions', [])[:5]
                            })
                    except Exception as e:
                        logger.debug(f"Could not analyze {file_path}: {e}")

                # Formulate actual insights
                alice_formulation = {
                    'type': 'self_analysis',
                    'total_files': len(files),
                    'analyzed_files': analysis_results,
                    'architecture_points': [
                        'Tool-based architecture with PhrasingLearner for progressive independence',
                        'Conversational engine handles pattern-based responses without LLM',
                        'Self-reflection system for code awareness',
                        'Plugin system for extensible capabilities',
                        'LLM gateway abstracts Ollama interactions'
                    ],
                    'confidence': 0.95
                }

                # Use learning loop to phrase naturally
                tone = self._select_tone(intent, context, user_input)
                response = self._generate_natural_response(
                    alice_response=alice_formulation,
                    tone=tone,
                    context=ConversationalContext(
                        user_input=user_input,
                        intent=intent,
                        entities=entities,
                        recent_topics=self.conversation_topics[-3:] if self.conversation_topics else [],
                        active_goal=goal_res.goal if goal_res else None,
                        world_state=self.reasoning_engine if hasattr(self, 'reasoning_engine') else None
                    ),
                    user_input=user_input
                )

                return response

            # Handle relationship queries before LLM generation
            if self.relationship_tracker:
                relationship_response = self._handle_relationship_query(user_input, intent, entities)
                if relationship_response:
                    return relationship_response
            
            # 3. Alice checks her learned patterns first — only use Ollama if she can't answer
            if self.phrasing_learner:
                _conv_thought = {
                    'type': intent,
                    'data': {'user_input': user_input, 'entities': entities or {}}
                }
                # `context` (ConversationalContext) is only defined in the plugin branch;
                # pass None here — _select_tone handles None gracefully.
                _tone_for_check = self._select_tone(intent, None, user_input) if hasattr(self, '_select_tone') else 'helpful'
                if self.phrasing_learner.can_phrase_myself(_conv_thought, _tone_for_check):
                    _learned_response = self.phrasing_learner.phrase_myself(_conv_thought, tone=_tone_for_check)
                    if _learned_response:
                        self._think(f"Alice answered from learned patterns — Ollama not needed for '{intent}'")
                        if getattr(self, 'response_optimizer', None):
                            _learned_response = self.response_optimizer.optimize(
                                _learned_response, intent, {"entities": entities}
                            )
                        if use_voice and self.speech:
                            self.speech.speak(_learned_response, blocking=False)
                        return _learned_response

            # 4. Use Gateway for complex knowledge/reasoning (last resort — Alice learns from this)
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
            if goal_res and goal_res.goal and self._should_attach_goal_context(user_input_processed, intent):
                # Add goal context to help LLM understand what user is trying to accomplish
                goal_note = f"\n[Context: You're helping the user accomplish: {goal_res.goal.description}. Keep this goal in mind when responding.]"
                llm_input = user_input_processed + goal_note
            
            try:
                # Track LLM execution timing
                llm_start = time.time()
                llm_response = self.llm_gateway.request(
                    prompt=llm_input,
                    call_type=LLMCallType.GENERATION,
                    use_history=True,
                    user_input=user_input,
                    context={'intent': intent, 'entities': entities, 'goal': goal_res.goal if goal_res else None}
                )
                llm_duration = time.time() - llm_start
                
                # Track LLM metrics
                model_name = getattr(self.llm.config, 'model', 'llama3.1')
                self.metrics.track_llm_call(
                    model=model_name,
                    duration=llm_duration,
                    tokens=getattr(llm_response, 'token_count', 0),
                    success=llm_response.success
                )
                self.structured_logger.debug(
                    "LLM generation complete",
                    model=model_name,
                    success=llm_response.success,
                    duration_ms=round(llm_duration * 1000, 2),
                    tokens=getattr(llm_response, 'token_count', 0)
                )
                
                if llm_response.success and llm_response.response:
                    response = self._clamp_final_response(
                        llm_response.response,
                        tone='helpful',
                        response_type='general_response',
                        route='generation',
                        user_input=user_input,
                    )
                    response = self._self_critique_and_regenerate(
                        user_input=user_input,
                        intent=intent,
                        entities=entities or {},
                        response=response,
                        goal_res=goal_res,
                    )

                    # LEARN from Ollama's conversational response
                    # Record so A.L.I.C.E can eventually handle this without Ollama
                    if self.phrasing_learner:
                        self.phrasing_learner.record_phrasing(
                            alice_thought={
                                'type': intent,
                                'data': {
                                    'user_input': user_input,
                                    'entities': entities or {}
                                }
                            },
                            ollama_phrasing=response,
                            context={
                                'tone': 'helpful',
                                'intent': intent,
                                'user_input': user_input
                            }
                        )

                        # Check if Alice can now handle this independently
                        thought_for_check = {
                            'type': intent,
                            'data': {'user_input': user_input, 'entities': entities or {}}
                        }
                        if self.phrasing_learner.can_phrase_myself(thought_for_check, 'helpful'):
                            self._think(f"Alice learned '{intent}' - can now respond independently!")
                        else:
                            # Count learned examples
                            learned_count = 0
                            pattern = f"general:{intent}"
                            if hasattr(self.phrasing_learner, 'learned_patterns') and pattern in self.phrasing_learner.learned_patterns:
                                learned_count = len(self.phrasing_learner.learned_patterns[pattern])
                            self._think(f"Alice learned from this interaction (confidence in '{intent}' topics growing)")
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
            response = self._apply_response_style_constraints(response)

            response = self._executive_apply_response_gate(
                user_input=user_input,
                intent=intent,
                response=response,
                route="llm",
            )
            
            # NOTE: Don't cache LLM responses - we want fresh answers each time for variety
            # Cache is only used for learned conversational patterns (which rotate automatically)

            # Learning engine can optionally evaluate quality
            # (This section can be expanded later for quality checks)
            
            # Update the context handler (if still exists)
            if 'turn' in locals() and turn:
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

            self._run_executive_reflection(
                user_input=user_input,
                intent=intent,
                response=response,
                route="llm",
                prior_confidence=float(intent_confidence if 'intent_confidence' in locals() else 0.0),
            )
            
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

            # Log error-like responses for learning
            expected_domain = intent.split(':')[0] if ':' in intent else intent
            if self._is_error_response(response, expected_domain):
                self._log_error_interaction(
                    user_input=user_input,
                    intent=intent,
                    entities=entities,
                    response=response,
                    error_type="bad_answer" if any(marker in response.lower() for marker in ["i apologize", "error", "sorry"]) else "wrong_domain",
                    actual_route="TOOL" if ("plugin_result" in locals() and plugin_result) else "LLM_FALLBACK"
                )
            
            # Log action for pattern learning
            action = f"{intent}:{entities.get('topic', entities.get('query', 'general'))}"
            self._log_action_for_learning(action)

            # ALICE'S COMPREHENSIVE LEARNING - Learn from EVERY interaction
            # This is where Alice builds her own intelligence
            try:
                _learn_decision = str((getattr(self, '_internal_reasoning_state', {}) or {}).get('learning_decision', 'store'))
                if _learn_decision != "store":
                    self._think(f"Executive learning authority → {_learn_decision} (skip long-term learning)")
                else:
                    learning_result = self.knowledge_engine.learn_from_interaction(
                        user_input=user_input,
                        alice_response=response,
                        intent=intent,
                        entities=entities or {},
                        context={
                            'quality_score': quality_score if 'quality_score' in locals() else 0.5,
                            'route': 'TOOL' if ('plugin_result' in locals() and plugin_result) else 'LLM',
                            'timestamp': datetime.now().isoformat()
                        }
                    )
                    if isinstance(learning_result, dict) and not learning_result.get('stored', True):
                        reasons = learning_result.get('validation', {}).get('reasons', [])
                        self._think(
                            f"Knowledge candidate rejected for '{intent}' "
                            f"(checks failed: {', '.join(reasons) if reasons else 'validation'})"
                        )
                    else:
                        self._think(f"Alice learned from this interaction (confidence in '{intent}' topics growing)")
            except Exception as e:
                logger.error(f"[Learning] Error in knowledge engine: {e}")

            # FOUNDATIONAL SYSTEMS - Track conversation and learn user preferences
            try:
                # Conversation Context - Track this turn for reference resolution
                if hasattr(self, 'conversation_context'):
                    topics = []
                    if getattr(self, 'conversation_state_tracker', None):
                        _state = self.conversation_state_tracker.get_state_summary()
                        _topic = str(_state.get('conversation_topic', '')).strip()
                        if _topic:
                            topics.append(_topic)
                    if intent:
                        topics.append(intent)
                    self.conversation_context.add_turn(
                        user_input=user_input,
                        alice_response=response,
                        intent=intent,
                        entities=entities or {},
                        topics=topics,
                        sentiment=sentiment
                    )

                # User Profile - Learn from this interaction
                if hasattr(self, 'user_profile'):
                    feedback = 'neutral'
                    if 'quality_score' in locals() and quality_score:
                        if quality_score > 0.7:
                            feedback = 'positive'
                        elif quality_score < 0.3:
                            feedback = 'negative'

                    self.user_profile.record_interaction(
                        user_input=user_input,
                        alice_response=response,
                        intent=intent,
                        entities=entities or {},
                        feedback=feedback
                    )

            except Exception as e:
                logger.error(f"[Foundational Systems] Error: {e}")

            # 5. Speak if voice enabled
            if use_voice and self.speech:
                self.speech.speak(response, blocking=False)

            # Log successful interaction to real-time learning
            if hasattr(self, 'realtime_logger'):
                self.realtime_logger.log_success(
                    event_type='successful_response',
                    user_input=user_input,
                    alice_response=response,
                    intent=intent,
                    route=route_taken,
                    confidence=float(intent_confidence if 'intent_confidence' in locals() else 0.0)
                )

            # Queue async evaluation (non-blocking - user gets response immediately)
            if hasattr(self, 'async_evaluator') and self.async_evaluator:
                # Build plugin result structure for evaluation
                eval_plugin_result = plugin_result if 'plugin_result' in locals() and plugin_result else {
                    'action': intent if 'intent' in locals() else 'unknown',
                    'data': entities if 'entities' in locals() else {},
                    'success': True
                }

                # Queue for async evaluation (happens in background after response returned)
                try:
                    alice_confidence = float(intent_confidence if 'intent_confidence' in locals() else 0.5)
                    self.async_evaluator.queue_evaluation(
                        user_input=user_input,
                        alice_response=response,
                        plugin_result=eval_plugin_result,
                        alice_confidence=alice_confidence
                    )
                except Exception as e:
                    logger.debug(f"[AsyncEval] Could not queue evaluation: {e}")

            # Cache response if it's a cacheable intent (with 5-minute expiry)
            if intent in cacheable_intents and response:
                self._cache_put(user_input, intent, response)
                self._think(f"Response cached for {intent}")

            # Analytics - log interaction metrics
            if hasattr(self, 'usage_analytics'):
                try:
                    response_time = (time.time() - locals().get('start_time', time.time())) * 1000  # ms
                    plugin_name = None
                    if 'plugin_result' in locals() and isinstance(plugin_result, dict):
                        plugin_name = plugin_result.get('plugin')
                    llm_used = 'ollama_phrased_response' in locals() or 'llm_response' in locals()
                    cached = intent in cacheable_intents and self._cache_get(user_input, intent) is not None
                    interaction_success = True
                    extra_metadata = None
                    if 'plugin_result' in locals() and isinstance(plugin_result, dict):
                        interaction_success = bool(plugin_result.get('success', True))
                        pdata = plugin_result.get('data', {}) if isinstance(plugin_result.get('data', {}), dict) else {}
                        diagnostics = pdata.get('diagnostics', {}) if isinstance(pdata.get('diagnostics', {}), dict) else {}
                        extra_metadata = {
                            'plugin_action': plugin_result.get('action'),
                            'plugin_error': pdata.get('error'),
                            'message_code': pdata.get('message_code'),
                            'resolution_path': diagnostics.get('resolution_path'),
                        }

                    self.usage_analytics.log_interaction(
                        user_input=user_input,
                        intent=intent,
                        plugin_used=plugin_name,
                        response_time_ms=response_time,
                        success=interaction_success,
                        llm_used=llm_used,
                        cached=cached,
                        extra_metadata=extra_metadata,
                    )
                except Exception as e:
                    logger.debug(f"[Analytics] Could not log interaction: {e}")

            # ── Outcome feedback: close the NLP learning loops ────────────────
            # BayesianIntentRouter: update confidence calibration after each turn
            # so future cost-minimisation uses empirical accuracy, not raw scores.
            # KnobBandit: reward or penalise the proposed style knobs based on
            # whether the plugin (or LLM) actually succeeded this turn.
            try:
                _outcome_success = (
                    interaction_success
                    if 'interaction_success' in locals()
                    else bool('plugin_result' in locals() and (plugin_result or {}).get('success', True))
                )
                _ic_for_outcome = float(getattr(self, '_last_intent_confidence', 0.5))
                if getattr(self, 'bayesian_router', None) is not None:
                    self.bayesian_router.record_outcome(
                        intent, _outcome_success, _ic_for_outcome
                    )
                _knob_key = getattr(self, '_knob_context_key', None)
                if getattr(self, 'knob_bandit', None) is not None and _knob_key is not None:
                    _knob_reward = 1.0 if _outcome_success else -0.5
                    self.knob_bandit.record_outcome(_knob_key, _knob_reward)
                    self._knob_context_key = None  # reset; propose() will set it next turn
            except Exception as _outcome_err:
                logger.debug("[OutcomeFeedback] record_outcome skipped: %s", _outcome_err)

            # Memory Management - periodic monitoring and pruning
            if hasattr(self, 'memory_growth_monitor') and hasattr(self, 'memory_pruner'):
                try:
                    # Take memory snapshot if it's time
                    if self.memory_growth_monitor.should_take_snapshot():
                        self.memory_growth_monitor.take_snapshot(self.memory)

                    # Run memory pruning if it's time
                    if self.memory_pruner.should_run():
                        prune_result = self.memory_pruner.prune_memories(self.memory)
                        if prune_result.get('pruned_count', 0) > 0:
                            logger.info(f"[MemoryPruner] Pruned {prune_result['pruned_count']} old memories")
                except Exception as e:
                    logger.debug(f"[MemoryMgmt] Error in periodic tasks: {e}")

            # ===== PRODUCTION: CACHE & METRICS =====
            # Cache successful responses for future fast retrieval
            if response and len(response) > 10:  # Don't cache trivial/error responses
                try:
                    self.cache.set('responses', cache_key, response, ttl=300)
                    self.metrics.track_cache('set', 'success')
                    self.structured_logger.debug("Response cached", cache_key=cache_key[:50])
                except Exception as cache_err:
                    logger.debug(f"[Cache] Failed to cache response: {cache_err}")
            
            # Track final request metrics
            duration = time.time() - start_time
            route_taken = 'plugin' if 'plugin_result' in locals() and plugin_result else 'llm'
            self.metrics.track_request(
                intent=intent or 'unknown',
                success=True,
                duration=duration,
                route=route_taken
            )
            self.structured_logger.info(
                "Request completed successfully",
                intent=intent,
                route=route_taken,
                duration_ms=round(duration * 1000, 2),
                response_length=len(response)
            )

            logger.info(f"A.L.I.C.E: {response[:100]}...")
            
            # Store interaction for next-turn feedback learning
            if self.foundations:
                self._last_interaction = {
                    'input': user_input,
                    'response': response
                }

            # Adaptive controller: record turn latency + response quality
            try:
                if self.adaptive_controller is not None:
                    self.adaptive_controller.observe_turn(
                        duration_ms=duration * 1000,
                        response_len=len(response),
                        intent=intent or "unknown",
                    )
                    # Safe-mode gate: throttle advanced tooling when the system degrades
                    if self.adaptive_controller.is_degraded():
                        if not getattr(self, '_safe_mode', False):
                            self._safe_mode = True
                            logger.warning(
                                "[AdaptiveController] Degradation detected → safe mode ON"
                            )
                    elif getattr(self, '_safe_mode', False):
                        self._safe_mode = False
                        logger.info(
                            "[AdaptiveController] Recovery detected → safe mode OFF"
                        )
            except Exception:
                pass

            # SelfDebugger: analyse the completed turn for failure patterns
            try:
                if getattr(self, 'self_debugger', None) is not None:
                    _pm = TurnPostmortem(
                        utterance=user_input,
                        intent=intent if 'intent' in dir() else 'unknown',
                        plugin=(
                            plugin_result.get('plugin', '')
                            if 'plugin_result' in locals() and plugin_result
                            else ''
                        ),
                        action=(
                            plugin_result.get('action', '')
                            if 'plugin_result' in locals() and plugin_result
                            else ''
                        ),
                        confidence=float(
                            intent_confidence
                            if 'intent_confidence' in locals()
                            else 0.5
                        ),
                        plugin_success=(
                            plugin_result.get('success', True)
                            if 'plugin_result' in locals() and plugin_result
                            else True
                        ),
                    )
                    _sd_components = {
                        'intent_classifier': getattr(self, 'nlp', None),
                        'nlp_processor': getattr(self, 'nlp', None),
                        'task_planner': getattr(self, 'planner', None),
                    }
                    self.self_debugger.analyse_turn(_pm, _sd_components)
            except Exception:
                pass

            return response
            
        except Exception as e:
            import traceback
            
            # ===== PRODUCTION: ERROR METRICS & LOGGING =====
            error_type = type(e).__name__
            duration = time.time() - start_time if 'start_time' in locals() else 0
            self.metrics.track_error(error_type, 'process_input')
            self.metrics.track_request(
                intent=intent if 'intent' in locals() else 'unknown',
                success=False,
                duration=duration,
                route='error'
            )
            self.structured_logger.exception(
                "Request failed with exception",
                error_type=error_type,
                error_message=str(e),
                user_input=user_input[:100],
                duration_ms=round(duration * 1000, 2)
            )
            
            logger.error(f"[ERROR] Error processing input: {e}")
            logger.error(f"[ERROR] Traceback: {traceback.format_exc()}")

            # Log to real-time learning system
            if hasattr(self, 'realtime_logger'):
                self.realtime_logger.log_error(
                    error_type='processing_error',
                    user_input=user_input,
                    expected=None,
                    actual=str(e),
                    intent=intent if 'intent' in locals() else 'unknown',
                    entities=entities if 'entities' in locals() else {},
                    context={'traceback': traceback.format_exc()},
                    severity='critical'
                )

            error_response = "I apologize, but I encountered an error processing your request."

            # Log processing exception for learning
            self._log_error_interaction(
                user_input=user_input,
                intent=intent if 'intent' in locals() else "unknown",
                entities=entities if 'entities' in locals() else {},
                response=error_response,
                error_type="exception",
                actual_route="LLM_FALLBACK"
            )
            
            if use_voice and self.speech:
                self.speech.speak(error_response, blocking=False)
            
            return error_response

    def _is_error_response(self, response: str, expected_domain: str = None) -> bool:
        """Detect error-like responses that should be logged for learning."""
        if not response:
            return False
        text = response.lower()
        error_markers = [
            "i apologize", "encountered an error", "error", "sorry",
            "i don't know", "i do not know", "not learned", "still learning",
            "i'm not sure", "cannot", "can't"
        ]
        has_error = any(marker in text for marker in error_markers)
        
        # Also detect wrong-domain responses (e.g., weather response to email query)
        if not has_error and expected_domain:
            domain_indicators = {
                'email': ['from', 'subject', 'inbox', 'message', 'sent'],
                'weather': ['temperature', 'forecast', '°c', 'celsius', 'condition', 'humidity'],
                'notes': ['note', 'saved', 'created', 'added', 'reminder'],
                'calendar': ['meeting', 'event', 'appointment', 'schedule', 'time']
            }
            
            expected_indicators = domain_indicators.get(expected_domain, [])
            if expected_indicators and not any(ind in text for ind in expected_indicators):
                # Check if response has OTHER domain's indicators
                for domain, indicators in domain_indicators.items():
                    if domain != expected_domain and any(ind in text for ind in indicators):
                        return True  # Wrong domain detected
        
        return has_error

    def _clean_email_body(self, body: str) -> str:
        if not body:
            return ""
        try:
            body = html.unescape(body)
            # Remove style/script blocks before stripping tags
            body = re.sub(r"<style[\s\S]*?>[\s\S]*?</style>", "", body, flags=re.IGNORECASE)
            body = re.sub(r"<script[\s\S]*?>[\s\S]*?</script>", "", body, flags=re.IGNORECASE)
            body = re.sub(r"<\s*br\s*/?>", "\n", body, flags=re.IGNORECASE)
            body = re.sub(r"</p\s*>", "\n\n", body, flags=re.IGNORECASE)
            body = re.sub(r"<[^>]+>", "", body)

            # Normalize whitespace
            body = re.sub(r"\n{3,}", "\n\n", body)
            body = re.sub(r"[ \t]{2,}", " ", body)

            # Drop CSS-like lines
            cleaned_lines = []
            for line in body.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                # Skip CSS selectors/blocks and property lines
                if "{" in stripped or "}" in stripped:
                    continue
                if re.match(r"^[\w\-]+\s*:\s*[^;]+;?\s*$", stripped):
                    continue
                # Skip single-number or very short noise lines
                if re.match(r"^\d+$", stripped):
                    continue
                cleaned_lines.append(stripped)

            return "\n".join(cleaned_lines).strip()
        except Exception:
            return body

    def _log_error_interaction(
        self,
        user_input: str,
        intent: str,
        entities: Dict[str, Any],
        response: str,
        error_type: str,
        actual_route: str = "LLM_FALLBACK"
    ) -> None:
        """Log error interactions into training data for correction learning."""
        try:
            training_dir = Path("data/training")
            training_dir.mkdir(parents=True, exist_ok=True)
            output_file = training_dir / "auto_generated.jsonl"

            domain = "unknown"
            if intent and ":" in intent:
                domain = intent.split(":", 1)[0]

            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_input": user_input,
                "actual_intent": intent,
                "actual_route": actual_route,
                "alice_response": response,
                "success": False,
                "success_flag": False,
                "error_type": error_type,
                "domain": domain,
                "llm_used": actual_route == "LLM_FALLBACK"
            }

            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

        except Exception as e:
            logger.warning(f"[LOG] Failed to log error interaction: {e}")
    
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

                    if 'conversation_state_tracker' in state and getattr(self, 'conversation_state_tracker', None):
                        self.conversation_state_tracker.load_state(state['conversation_state_tracker'])

                    # Restore adaptive routing weights (with decay toward neutral)
                    if getattr(self, 'executive_controller', None):
                        self.executive_controller.load_weights("data/executive_routing_weights.json")
                    
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
                'conversation_state_tracker': (
                    self.conversation_state_tracker.to_dict()
                    if getattr(self, 'conversation_state_tracker', None)
                    else {}
                ),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(state_file, 'wb') as f:
                pickle.dump(state, f)

            # Persist adaptive routing weights for cumulative learning
            if getattr(self, 'executive_controller', None):
                self.executive_controller.save_weights("data/executive_routing_weights.json")
            
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
        # Update last interaction (for /correct and /feedback)
        self.last_user_input = user_input
        self.last_assistant_response = response
        self.last_intent = intent
        self.last_entities = entities or {}
        self.last_interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "assistant_response": response,
            "intent": intent,
            "entities": entities or {}
        }
        try:
            if getattr(self, 'conversation_state_tracker', None):
                goal_hint = ""
                if isinstance(entities, dict):
                    for _goal_key in ("goal", "user_goal", "objective"):
                        _goal_val = entities.get(_goal_key)
                        if isinstance(_goal_val, str) and _goal_val.strip():
                            goal_hint = _goal_val.strip()
                            break
                if not goal_hint and getattr(self, 'pending_action', None):
                    goal_hint = str(self.pending_action).replace("_", " ")
                self.conversation_state_tracker.update_state(
                    user_input=user_input,
                    intent=intent or "",
                    entities=entities or {},
                    goal_hint=goal_hint,
                )

            # Goal tracker: update goal progress / drift / completion detection
            if getattr(self, 'goal_tracker', None) and getattr(self, 'conversation_state_tracker', None):
                try:
                    _gt_state = self.conversation_state_tracker.get_state_summary()
                    _current_topic = str(_gt_state.get("conversation_topic", "") or "")
                    _previous_topic = str(getattr(self, "_last_goal_tracker_topic", "") or "")
                    self.goal_tracker.update(
                        user_input=user_input,
                        response=response,
                        user_goal=_gt_state.get("user_goal", ""),
                        conversation_goal=_gt_state.get("conversation_goal", ""),
                        intent=intent or "",
                        current_topic=_current_topic,
                        previous_topic=_previous_topic,
                    )
                    self._last_goal_tracker_topic = _current_topic
                except Exception as _gt_err:
                    logger.debug(f"[GoalTracker] {_gt_err}")

            if getattr(self, 'cognitive_orchestrator', None):
                try:
                    _decision_scores = (
                        (getattr(self, '_internal_reasoning_state', {}) or {}).get('decision_scores', {})
                    )
                    _gate_accepted = bool(
                        (getattr(self, '_last_exec_gate_eval', {}) or {}).get('accepted', True)
                    )
                    _turn_route = 'plugin' if bool(locals().get('plugin_result')) else 'llm'
                    self.cognitive_orchestrator.observe_turn(
                        user_input=user_input,
                        intent=intent or "conversation:general",
                        response=response,
                        gate_accepted=_gate_accepted,
                        route=_turn_route,
                        decision_scores=_decision_scores,
                    )
                except Exception as _co_err:
                    logger.debug(f"[CognitiveOrchestrator] {_co_err}")

            if self.episodic_memory_engine:
                self.episodic_memory_engine.add_episode(
                    user_input=user_input,
                    intent=intent,
                    response=response,
                    entities=entities or {},
                )
                self._episodic_turn_counter += 1

            if self.semantic_memory_index:
                _doc_id = f"turn-{int(time.time() * 1000)}"
                _doc_text = f"{intent} {user_input} {response[:240]}"
                self.semantic_memory_index.add(_doc_id, _doc_text)

            if self.cross_session_pattern_detector:
                self.cross_session_pattern_detector.observe(intent)

            if (
                self.memory_consolidator
                and self.episodic_memory_engine
                and self._episodic_turn_counter > 0
                and self._episodic_turn_counter % 15 == 0
            ):
                _episodes = self.episodic_memory_engine.recall_recent(limit=30)
                _consolidated = self.memory_consolidator.consolidate(_episodes)
                self._internal_reasoning_state = {
                    **(getattr(self, '_internal_reasoning_state', {}) or {}),
                    "memory_consolidation": _consolidated,
                }

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
                            for item in entity_values:
                                if hasattr(item, "value"):
                                    entity_list.append(str(item.value))
                                else:
                                    entity_list.append(str(item))
                        elif entity_values:
                            if hasattr(entity_values, "value"):
                                entity_list.append(str(entity_values.value))
                            else:
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
            
            # Track topics from intent and entities — always move the intent
            # to the END of the list so that topics[-1] always reflects the
            # MOST RECENTLY seen topic, even if that domain was seen before.
            # (The old "if not in" check left stale ordering when topics recycled.)
            if intent and intent != "unknown":
                if intent in self.conversation_topics:
                    self.conversation_topics.remove(intent)
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
                if isinstance(values, list):
                    for item in values:
                        if hasattr(item, "value"):
                            entity_list.append(str(item.value))
                        else:
                            entity_list.append(str(item))
                else:
                    if hasattr(values, "value"):
                        entity_list.append(str(values.value))
                    else:
                        entity_list.append(str(values))
            
            self.context.update_conversation(user_input, response, intent, entity_list)
            
            # Store in episodic memory with enhanced metadata (unless privacy mode)
            if not self.privacy_mode and getattr(self, '_exec_should_store_memory', True):
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
            elif not getattr(self, '_exec_should_store_memory', True):
                logger.info("[Executive] Episodic memory storage skipped by executive policy")
            else:
                logger.info("[PRIVACY] Episodic memory storage skipped (privacy mode enabled)")

            # Quality checker: automatic detection of soft issues (vocab gaps, directness, repetition)
            try:
                _qc = get_quality_checker()
                # conversation_summary[-1] is the turn just appended above; [-2] is the previous turn
                _prev = self.conversation_summary[-2] if len(self.conversation_summary) >= 2 else None
                _qc.analyze(
                    user_input=user_input,
                    response=response,
                    intent=intent,
                    plugin_called=getattr(self, '_last_plugin_called', None),
                    had_stored_data=getattr(self, '_last_had_stored_data', False),
                    previous_turn=_prev,
                )
            except Exception as _qc_err:
                logger.debug("[QualityChecker] %s", _qc_err)

            # Self-debugger: post-turn root-cause analysis
            try:
                if getattr(self, 'self_debugger', None) is not None:
                    _last_res = getattr(self, '_last_plugin_result', {}) or {}
                    _pm = TurnPostmortem(
                        utterance=user_input,
                        intent=intent,
                        plugin=_last_res.get('plugin', ''),
                        action=_last_res.get('action', ''),
                        confidence=getattr(self, '_last_intent_confidence', 0.5),
                        plugin_success=_last_res.get('success', True),
                        plugin_error_code=(
                            _last_res.get('data', {}).get('error')
                            if isinstance(_last_res.get('data'), dict) else None
                        ),
                    )
                    _components = {
                        'intent_classifier': getattr(self.nlp, 'classifier', None),
                        'nlp_processor': self.nlp,
                        'task_planner': getattr(self, 'task_planner', None),
                        'memory_system': getattr(self, 'memory', None),
                        'response_formulator': getattr(self, 'response_formulator', None),
                    }
                    self.self_debugger.analyse_turn(_pm, _components)
            except Exception:
                pass
            
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
            print("   /autolearn - Show automated learning audit report")
            
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
            if greeting:
                print(f"\nA.L.I.C.E: {greeting}\n")

            if greeting and self.speech and self.voice_enabled:
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
            print("   /exit, /quit       - End conversation and exit")
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
            print("Autonomous Mode:")
            print("   /autonomous start  - Start autonomous goal execution")
            print("   /autonomous stop   - Stop autonomous execution")
            print("   /autonomous pause  - Pause autonomous execution")
            print("   /autonomous resume - Resume autonomous execution")
            print("   /autonomous status - Show autonomous mode status")
            print("   /goals             - List all active and completed goals")
            print("   /operator-status   - Show action engine, world state, autonomy, and execution metrics")
            print()
            print("Debug Commands:")
            print("   /correct [type]    - Correct A.L.I.C.E's last response")
            print("   /feedback [rating] - Rate A.L.I.C.E's last response (1-5)")
            print("   /learning          - Show active learning statistics")
            print("   /realtime-status   - Show continuous learning metrics and velocity")
            print("   /formulation       - Show response formulation learning progress")
            print("   /autolearn [days]  - Show automated learning audit report (default: 7 days)")

        elif cmd in {'/exit', '/quit'}:
            farewell = self._get_farewell()
            print(f"\nA.L.I.C.E: {farewell}\n")
            if self.speech and self.voice_enabled:
                self.speech.speak(farewell, blocking=True)
            self.running = False
            self.shutdown()
        
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
            print("\n[STATUS] System Status:")
            status = self.context.get_system_status()
            print(f"   LLM Model: {status.get('llm_model', 'N/A')}")
            print(f"   Voice: {'Enabled' if status.get('voice_enabled') else 'Disabled'}")
            print(f"   Privacy Mode: {'Enabled' if self.privacy_mode else 'Disabled'}")
            print(f"   Plugins: {len(self.plugins.plugins)}")
            print(f"   Capabilities: {len(status.get('capabilities', []))}")
            
            memory_stats = self.memory.get_statistics()
            print(f"   Total Memories: {memory_stats['total_memories']}")
            if self.privacy_mode:
                print(f"   Episodic memory storage is DISABLED")
            
            # LLM Gateway Statistics
            if hasattr(self, 'llm_gateway') and self.llm_gateway:
                print("\n LLM Gateway Statistics:")
                gateway_stats = self.llm_gateway.get_statistics()
                print(f"   Total Requests: {gateway_stats['total_requests']}")
                
                # Advanced breakdown
                print(f"   Self Handlers: {gateway_stats['self_handlers']} ({gateway_stats.get('self_handler_percentage', 0)}%)")
                print(f"   Pattern Hits: {gateway_stats['pattern_hits']} ({gateway_stats.get('pattern_hit_percentage', 0)}%)")
                print(f"   Tool Calls: {gateway_stats['tool_calls']} ({gateway_stats.get('tool_call_percentage', 0)}%)")
                print(f"   RAG Lookups: {gateway_stats['rag_lookups']} ({gateway_stats.get('rag_lookup_percentage', 0)}%)")
                print(f"   Formatter Usage: {gateway_stats['formatter_calls']} ({gateway_stats.get('formatter_percentage', 0)}%)")
                print(f"   LLM Fallback: {gateway_stats['llm_calls']} ({gateway_stats.get('llm_fallback_percentage', 0)}%)")
                print(f"   Multi-LLM Router: {'Enabled' if gateway_stats.get('multi_llm_enabled') else 'Disabled'}")
                print(f"   Multi-LLM Calls: {gateway_stats.get('multi_router_calls', 0)} ({gateway_stats.get('multi_router_percentage', 0)}%)")
                print(f"   Policy Denials: {gateway_stats['policy_denials']} ({gateway_stats.get('denial_percentage', 0)}%)")

                model_roles = gateway_stats.get('model_roles', {}) or {}
                if model_roles:
                    print("\n   Model Roles:")
                    for role, model in sorted(model_roles.items()):
                        print(f"      {role}: {model}")

                last_route = gateway_stats.get('last_route', {}) or {}
                if last_route:
                    print("\n   Last Route:")
                    print(f"      source: {last_route.get('source', 'n/a')}")
                    print(f"      call_type: {last_route.get('call_type', 'n/a')}")
                    if last_route.get('role'):
                        print(f"      role: {last_route.get('role')}")
                    print(f"      model: {last_route.get('model', 'n/a')}")
                
                if gateway_stats['by_type']:
                    print("\n   By Call Type:")
                    for cal_type, count in sorted(gateway_stats['by_type'].items(), key=lambda x: x[1], reverse=True):
                        print(f"      {cal_type}: {count}")
            
            # Conversational Engine Statistics
            if hasattr(self, 'conversational_engine') and self.conversational_engine:
                print("\n Conversational Engine:")
                if hasattr(self.conversational_engine, 'pattern_count'):
                    print(f"   Learned Patterns: {self.conversational_engine.pattern_count}")
                if hasattr(self.conversational_engine, 'learned_greetings'):
                    print(f"   Learned Greetings: {len(self.conversational_engine.learned_greetings) if self.conversational_engine.learned_greetings else 0}")
                print(f"   Status: Active")
        
        elif cmd == '/analyze-learning':
            from ai.learning.learning_insights import LearningInsights
            insights = LearningInsights(lookback_days=30).load()
            print(insights.generate_report())

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
                    print("\nConversation Topics:")
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
                    print("\n Tracked Entities:")
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
                    print("\nEntity Relationships:")
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
            print(f"\n{title}:")
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
            
            print(f"\n Memory Search Results for: '{query}'")
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
                        print(f"\n[OK] Deleted {memory_type} memory:")
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
                    from ai.learning.pattern_miner import PatternMiner
                    self.pattern_miner = PatternMiner()
                except Exception as e:
                    print(f"\n[ERROR] Pattern miner not available: {e}")
                    return
            
            stats = self.pattern_miner.get_pattern_stats()
            print("\n[PATTERNS] Pattern Learning System:")
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
                print("   [OK] No pending patterns. All proposed patterns have been reviewed.")
            
            print("\n" + "=" * 70)
        
        elif cmd.startswith('/patterns approve'):
            # Approve a specific pattern
            parts = command.split(maxsplit=2)
            if len(parts) < 3:
                print("\n[ERROR] Usage: /patterns approve <pattern_id>")
                return
            
            pattern_id = parts[2].strip()
            
            if not hasattr(self, 'pattern_miner'):
                from ai.learning.pattern_miner import PatternMiner
                self.pattern_miner = PatternMiner()
            
            if self.pattern_miner.approve_pattern(pattern_id):
                print(f"\n[OK] Pattern {pattern_id} approved and will be used for future interactions")
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
                from ai.learning.pattern_miner import PatternMiner
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

        elif cmd == '/realtime-status':
            self._handle_realtime_status_command()

        elif cmd == '/formulation':
            self._handle_formulation_status_command()

        elif cmd == '/autolearn' or cmd.startswith('/autolearn '):
            self._handle_autolearn_command(command)

        elif cmd.startswith('/autonomous'):
            self._handle_autonomous_command(command)

        elif cmd == '/goals':
            self._handle_goals_command()

        elif cmd == '/operator-status':
            self._handle_operator_status_command()

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
                greeting_options = self.conversational_engine._unique_candidates(
                    self.conversational_engine.learned_greetings
                ) if hasattr(self.conversational_engine, '_unique_candidates') else self.conversational_engine.learned_greetings
                if hasattr(self.conversational_engine, '_pick_non_repeating') and len(greeting_options) >= 2:
                    return self.conversational_engine._pick_non_repeating(greeting_options)

        learned = self._learned_greeting_response(
            user_input="greeting",
            user_name=name,
            asked_how=False,
            time_context=time_context,
        )
        if learned:
            return learned
        
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
                greeting = self._clamp_final_response(
                    llm_response.response.strip().strip('"').strip("'"),
                    tone='casual and friendly',
                    response_type='greeting',
                    route='greeting',
                    user_input='greeting',
                )
                if self.phrasing_learner:
                    self.phrasing_learner.record_phrasing(
                        alice_thought={
                            'type': 'greeting',
                            'data': {
                                'user_input': 'greeting',
                                'user_name': name,
                                'asked_how': False,
                                'time_context': time_context,
                            },
                        },
                        ollama_phrasing=greeting,
                        context={
                            'tone': 'friendly',
                            'intent': 'greeting',
                            'user_input': 'greeting',
                            'time_context': time_context,
                        },
                    )
                return greeting
        except Exception:
            pass

        # No hardcoded greeting fallback: if no LLM and no learned greeting yet, stay silent.
        return ""
    
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
                farewell = self._clamp_final_response(
                    llm_response.response.strip().strip('"').strip("'"),
                    tone='casual and friendly',
                    response_type='farewell',
                    route='farewell',
                    user_input='farewell',
                )
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

        # Technical domain exclusions - don't treat these as relationship queries
        technical_domains = [
            'neural', 'network', 'algorithm', 'memory', 'consolidation', 'gated', 'recall',
            'learning', 'machine', 'deep', 'training', 'model', 'data', 'code', 'function',
            'class', 'method', 'python', 'javascript', 'api', 'database', 'server', 'client',
            'formula', 'equation', 'math', 'science', 'physics', 'chemistry', 'biology',
            'weather', 'temperature', 'climate', 'file', 'folder', 'directory', 'system'
        ]

        # Self-reference exclusions - these should go to conversational engine
        self_references = ['alice', 'a.l.i.c.e', 'a l i c e', 'you', 'yourself']

        # ONLY match specific relationship patterns - not general knowledge queries
        # Explicit relationship patterns (high confidence these are about entity relationships)
        specific_relationship_patterns = [
            r'who does (\w+) work (?:for|with)',  # "who does John work for"
            r'where does (\w+) live',  # "where does Sarah live"
            r"(\w+)(?:'s)?\s+(?:relationship|connection)\s+(?:with|to)\s+(\w+)",  # "John's relationship with Mary"
            r'how are (\w+) and (\w+) (?:related|connected)',  # "how are John and Mary related"
            r'(\w+) and (\w+) relationship',  # "John and Mary relationship"
        ]

        for pattern in specific_relationship_patterns:
            match = re.search(pattern, query_lower)
            if match:
                entity_name = match.group(1)
                second_entity = match.group(2) if len(match.groups()) > 1 and match.group(2) else None

                # Skip if technical domain or self-reference
                if entity_name in technical_domains or entity_name in self_references:
                    continue
                if second_entity and (second_entity in technical_domains or second_entity in self_references):
                    continue

                # Get relationships for the entity
                relationships = self.relationship_tracker.get_entity_relationships(entity_name)

                if not relationships:
                    # Don't return error for specific patterns - just skip to LLM
                    continue
                
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
        # More specific matching to avoid false positives with technical terms (e.g., "neural networks")
        relationship_info_keywords = [
            'relationship', 'relationships',  # Plural and singular
            'connection between', 'connections between',
            'entity network', 'entity connections',
            'show relationships', 'show connections',
            'tracked relationships', 'tracked connections'
        ]

        # Only trigger if explicitly asking about relationship tracking, not technical terms
        if any(keyword in query_lower for keyword in relationship_info_keywords):
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
        last = self.last_interaction or {}
        last_user_input = last.get("user_input") or self.last_user_input
        last_response = last.get("assistant_response") or self.last_assistant_response
        last_intent = last.get("intent") or self.last_intent
        last_entities = last.get("entities") or self.last_entities
        last_nlp_result = self.last_nlp_result

        if not last_user_input or not last_response:
            print("\n[ERROR] No previous interaction to correct")
            return
        
        parts = command.split(" ", 1)
        correction_type = parts[1] if len(parts) > 1 else ""
        
        print(f"\n Correction Mode")
        print("=" * 50)
        print(f"Last input: {last_user_input}")
        print(f"Last response: {last_response[:200]}...")
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
            print(f"Current intent: {last_intent}")
            # If user already provided details after "/correct intent", use that
            if ' ' in command and correction_type == "intent":
                # Extract from command like "/correct intent The intent should be..."
                remaining_text = command.split("intent", 1)[1].strip()
                new_intent = remaining_text
            else:
                new_intent = input("Correct intent: ").strip()
            
            if new_intent:
                self.learning_manager.record_correction(
                    CorrectionType.INTENT_CLASSIFICATION,
                    last_user_input,
                    last_intent,
                    new_intent,
                    f"User corrected intent from '{last_intent}' to '{new_intent}'",
                    {"original_nlp_result": last_nlp_result}
                )
                print(f" Recorded intent correction: {last_intent}  {new_intent}")
        
        elif correction_type == "entity":
            print(f"Current entities: {last_entities}")
            print("Enter correct entities (JSON format):")
            try:
                new_entities_input = input("Correct entities: ").strip()
                if new_entities_input:
                    new_entities = eval(new_entities_input)  # Simple eval for demo
                    self.learning_manager.record_correction(
                        CorrectionType.ENTITY_EXTRACTION,
                        last_user_input,
                        last_entities,
                        new_entities,
                        "User corrected entity extraction",
                        {"original_nlp_result": last_nlp_result}
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
                    last_user_input,
                    last_response,
                    correct_response,
                    "User provided better response",
                    {"intent": last_intent}
                )
                print(" Recorded response quality correction")
        
        elif correction_type == "sentiment":
            print(f"Current sentiment: {last_nlp_result.get('sentiment', {})}")
            new_sentiment = input("Correct sentiment (positive/negative/neutral): ").strip()
            if new_sentiment:
                self.learning_manager.record_correction(
                    CorrectionType.SENTIMENT_ANALYSIS,
                    last_user_input,
                    last_nlp_result.get('sentiment'),
                    new_sentiment,
                    f"User corrected sentiment to {new_sentiment}",
                    {"original_nlp_result": last_nlp_result}
                )
                print(f" Recorded sentiment correction: {new_sentiment}")
        
        elif correction_type == "factual":
            print("Describe the factual error:")
            error_description = input("Error description: ").strip()
            correct_fact = input("Correct fact: ").strip()
            if error_description and correct_fact:
                self.learning_manager.record_correction(
                    CorrectionType.FACTUAL_ERROR,
                    last_user_input,
                    error_description,
                    correct_fact,
                    f"User corrected factual error: {error_description}",
                    {"response": last_response}
                )
                print(" Recorded factual correction")
        
        else:
            print(f"[ERROR] Unknown correction type: {correction_type}")
    
    def _handle_feedback_command(self, command: str):
        """Handle feedback commands"""
        last = self.last_interaction or {}
        last_user_input = last.get("user_input") or self.last_user_input
        last_response = last.get("assistant_response") or self.last_assistant_response
        last_intent = last.get("intent") or self.last_intent
        last_entities = last.get("entities") or self.last_entities

        if not last_user_input or not last_response:
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
        print(f"Last input: {last_user_input}")
        print(f"Last response: {last_response[:200]}...")
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
                last_user_input,
                last_response,
                rating,
                comment,
                suggestion,
                {"intent": last_intent, "entities": last_entities}
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

    def _handle_realtime_status_command(self):
        """Handle real-time learning status command"""
        print(f"\n Continuous Learning Status")
        print("=" * 50)

        try:
            if not hasattr(self, 'continuous_learning') or not self.continuous_learning:
                print("[ERROR] Continuous learning not initialized")
                return

            if not hasattr(self, 'realtime_logger') or not self.realtime_logger:
                print("[ERROR] Real-time logger not initialized")
                return

            status = self.continuous_learning.get_status()
            velocity = self.realtime_logger.get_learning_velocity()

            print(f"Learning Loop:")
            print(f"   Running: {'Yes' if status['running'] else 'No'}")
            print(f"   Paused: {'Yes' if status['paused'] else 'No'}")
            print(f"   Check Interval: {status['check_interval_hours']} hours")
            print(f"   Last Run: {status['last_run'] or 'Never'}")
            print(f"   Cycles Completed: {status['cycles_completed']}")
            print(f"   Total Corrections Applied: {status['total_corrections_applied']}")

            print(f"\nLearning Velocity:")
            print(f"   Total Errors: {velocity['total_errors']}")
            print(f"   Total Successes: {velocity['total_successes']}")
            print(f"   Success Rate: {velocity['success_rate']:.1%}")
            print(f"   Trend: {velocity['trend'].replace('_', ' ').title()}")

            if velocity['errors_per_hour']:
                recent_errors = velocity['errors_per_hour'][-5:]
                print(f"\n   Recent Error Rate (last {len(recent_errors)} hours):")
                for hour_data in recent_errors:
                    print(f"      {hour_data['hour']}: {hour_data['count']} errors")

            recent_errors = self.realtime_logger.get_recent_errors(count=5)
            if recent_errors:
                print(f"\n   Recent Errors:")
                for err in recent_errors[-3:]:
                    print(f"      [{err['error_type']}] {err['user_input'][:50]}...")

            print("=" * 50)

        except Exception as e:
            print(f"[ERROR] Failed to get real-time status: {e}")
            import traceback
            traceback.print_exc()

    def _handle_formulation_status_command(self):
        """Handle formulation learning status command"""
        print(f"\n Response Formulation Learning")
        print("=" * 50)

        try:
            if not hasattr(self, 'response_formulator') or not self.response_formulator:
                print("[ERROR] Response formulator not initialized")
                return

            stats = self.response_formulator.get_stats()

            print(f"Learning Progress:")
            print(f"   Total Templates: {stats['total_templates']}")
            print(f"   Independent Actions: {stats['independent_actions']}")
            print(f"   Progress: {stats['learning_progress']}")

            if stats['independent_actions'] > 0:
                independence_rate = stats['independent_actions'] / stats['total_templates'] * 100 if stats['total_templates'] > 0 else 0
                print(f"   Independence Rate: {independence_rate:.1f}%")

                print(f"\nActions Alice Can Formulate Independently:")
                for action in list(self.response_formulator.independent_actions)[:10]:
                    print(f"   - {action}")
                if len(self.response_formulator.independent_actions) > 10:
                    remaining = len(self.response_formulator.independent_actions) - 10
                    print(f"   ... and {remaining} more")

            if stats['total_templates'] > stats['independent_actions']:
                print(f"\nStill Learning:")
                learning_actions = set(self.response_formulator.templates.keys()) - self.response_formulator.independent_actions
                for action in list(learning_actions)[:5]:
                    print(f"   - {action}")
                if len(learning_actions) > 5:
                    remaining = len(learning_actions) - 5
                    print(f"   ... and {remaining} more")

            print("\n   Tip: Alice learns by seeing examples. After 3 similar")
            print("        formulations, she can phrase independently!")

            print("=" * 50)

        except Exception as e:
            print(f"[ERROR] Failed to get formulation status: {e}")
            import traceback
            traceback.print_exc()

    def _handle_autolearn_command(self, command: str):
        """Handle automated learning audit command"""
        print(f"\n Automated Learning Audit Report")
        print("=" * 70)

        try:
            if not hasattr(self, 'autolearn') or not self.autolearn:
                print("[ERROR] AutoLearn system not initialized")
                return

            # Parse days parameter (default 7)
            days = 7
            parts = command.strip().split()
            if len(parts) > 1:
                try:
                    days = int(parts[1])
                except ValueError:
                    print(f"[WARNING] Invalid days parameter, using default: 7")

            # Get performance report
            report = self.autolearn.get_performance_report(days=days)

            if 'error' in report:
                print(f"[ERROR] {report['error']}")
                return

            # Display overall statistics
            overall = report['overall_stats']
            print(f"\n Period: Last {report['period_days']} days")
            print(f"\nOverall Performance:")
            print(f"   Total Evaluations: {overall['total']}")
            print(f"   Average Score: {overall['average_score']}/100")
            print(f"   Passing Rate: {overall['passing_rate']}% (score >= 85)")
            print(f"   Failing Rate: {overall['failing_rate']}% (score < 70)")
            if overall.get('critical_failures', 0) > 0:
                print(f"   Critical Failures: {overall['critical_failures']} (score < 50)")

            # AutoLearn statistics
            autolearn_stats = report['autolearn_stats']
            print(f"\nAutoLearn Activity:")
            print(f"   Cycles Run: {autolearn_stats['cycles_run']}")
            print(f"   Total Improvements: {autolearn_stats['total_improvements']}")
            print(f"   Last Run: {autolearn_stats['last_run'] or 'Never'}")

            # Performance by action type
            if overall.get('by_action'):
                print(f"\nPerformance by Action Type:")
                sorted_actions = sorted(
                    overall['by_action'].items(),
                    key=lambda x: x[1]['avg_score'],
                    reverse=True
                )
                for action, stats in sorted_actions[:10]:
                    print(f"   {action}:")
                    print(f"      Count: {stats['count']}")
                    print(f"      Avg Score: {stats['avg_score']:.1f}/100")
                    print(f"      Range: {stats['min_score']}-{stats['max_score']}")
                if len(sorted_actions) > 10:
                    print(f"   ... and {len(sorted_actions) - 10} more action types")

            # Problem areas
            problem_areas = report.get('problem_areas', {})
            if problem_areas:
                print(f"\nProblem Areas (score < 70):")
                sorted_problems = sorted(
                    problem_areas.items(),
                    key=lambda x: x[1]['avg_score']
                )
                for action, data in sorted_problems[:5]:
                    print(f"\n   {action}:")
                    print(f"      Failures: {data['count']}")
                    print(f"      Avg Score: {data['avg_score']:.1f}/100")

                    # Show example failure
                    if data.get('examples'):
                        example = data['examples'][0]
                        print(f"      Example Issue:")
                        print(f"         Input: {example['input'][:60]}...")
                        print(f"         Response: {example['response'][:60]}...")
                        print(f"         Score: {example['score']}/100")
                        print(f"         Issue: {example['issue'][:80]}...")

            # Recommendation
            print(f"\nRecommendation:")
            print(f"   {report['recommendation']}")

            print("\n" + "=" * 70)
            print("\n   Note: Ollama automatically evaluates every response.")
            print("   AutoLearn runs every 6 hours to apply improvements.")
            print("   User only audits these aggregated metrics weekly.")
            print("=" * 70)

        except Exception as e:
            print(f"[ERROR] Failed to get autolearn report: {e}")
            import traceback
            traceback.print_exc()

    def _init_bounded_autonomy_loops(self):
        """Register baseline bounded autonomy loops if missing."""
        if not hasattr(self, 'autonomy_manager') or not self.autonomy_manager:
            return

        baseline = [
            AutonomyLoop(
                name="goal_health",
                permission_level="operator",
                scope="goal monitoring",
                stop_conditions=["manual_stop", "confidence_drop"],
                confidence_threshold=0.6,
                enabled=True,
            ),
            AutonomyLoop(
                name="repo_failure_watch",
                permission_level="operator",
                scope="build/test monitoring",
                stop_conditions=["manual_stop", "error_storm"],
                confidence_threshold=0.7,
                enabled=False,
            ),
        ]

        existing = set((self.autonomy_manager.status() or {}).get("enabled_loops", []))
        existing.update((self.autonomy_manager.status() or {}).get("disabled_loops", []))
        for loop in baseline:
            if loop.name not in existing:
                self.autonomy_manager.register_loop(loop)

    def _handle_operator_status_command(self):
        """Show execution-first operator core status."""
        print("\n Operator Core Status:")
        print("=" * 70)

        world_state = {}
        if hasattr(self, 'world_state_memory') and self.world_state_memory:
            world_state = self.world_state_memory.snapshot()

        journal_summary = {}
        if hasattr(self, 'execution_journal') and self.execution_journal:
            journal_summary = self.execution_journal.summary()

        autonomy_status = {}
        if hasattr(self, 'autonomy_manager') and self.autonomy_manager:
            autonomy_status = self.autonomy_manager.status()

        print("\nUnified Action Engine:")
        print(f"   Bound: {'Yes' if hasattr(self, 'action_engine') and self.action_engine else 'No'}")
        print(f"   Last Goal Satisfied: {bool((self._internal_reasoning_state or {}).get('goal_satisfied', False))}")

        print("\nWorld State:")
        print(f"   Active Task: {world_state.get('active_task') or 'None'}")
        last_tool = world_state.get('last_tool') or {}
        if last_tool:
            print(
                f"   Last Tool: {last_tool.get('plugin', 'n/a')}:{last_tool.get('action', 'n/a')} "
                f"({last_tool.get('status', 'unknown')})"
            )
        else:
            print("   Last Tool: n/a")
        print(f"   Last Successful Target: {world_state.get('last_successful_target') or 'None'}")
        unresolved = world_state.get('unresolved_ambiguity') or []
        print(f"   Unresolved Ambiguity: {len(unresolved)}")

        print("\nExecution Journal:")
        print(f"   Total: {journal_summary.get('total', 0)}")
        print(f"   Success: {journal_summary.get('success', 0)}")
        print(f"   Failed: {journal_summary.get('failed', 0)}")
        print(f"   Retries: {journal_summary.get('retry', 0)}")
        print(f"   Goal Satisfied: {journal_summary.get('goal_satisfied', 0)}")

        print("\nBounded Autonomy:")
        print(f"   Total Loops: {autonomy_status.get('total_loops', 0)}")
        print(f"   Enabled Loops: {', '.join(autonomy_status.get('enabled_loops', [])) or 'none'}")
        print(f"   Disabled Loops: {', '.join(autonomy_status.get('disabled_loops', [])) or 'none'}")
        print("=" * 70)

    def _handle_autonomous_command(self, command: str):
        """Handle autonomous mode commands"""
        if not hasattr(self, 'execution_loop') or not self.execution_loop:
            print("\n[ERROR] Autonomous agent system not available")
            return

        parts = command.lower().split()
        if len(parts) < 2:
            print("\n[ERROR] Usage: /autonomous [start|stop|pause|resume|status]")
            return

        subcommand = parts[1]

        if subcommand == 'start':
            if not self.execution_loop.is_running():
                self.execution_loop.start()
                print("\n[OK] Autonomous mode started - Alice will work on active goals independently")
            else:
                print("\n[INFO] Autonomous mode is already running")

        elif subcommand == 'stop':
            if self.execution_loop.is_running():
                self.execution_loop.stop()
                print("\n[OK] Autonomous mode stopped")
            else:
                print("\n[INFO] Autonomous mode is not running")

        elif subcommand == 'pause':
            self.execution_loop.pause()
            print("\n[OK] Autonomous execution paused")

        elif subcommand == 'resume':
            self.execution_loop.resume()
            print("\n[OK] Autonomous execution resumed")

        elif subcommand == 'status':
            is_running = self.execution_loop.is_running()
            is_paused = self.execution_loop.paused
            active_goals = self.goal_system.get_active_goals() if hasattr(self, 'goal_system') else []
            autonomy_status = self.autonomy_manager.status() if hasattr(self, 'autonomy_manager') and self.autonomy_manager else {}

            print("\n Autonomous Agent Status:")
            print("=" * 50)
            print(f"   Running: {'Yes' if is_running else 'No'}")
            print(f"   Paused: {'Yes' if is_paused else 'No'}")
            print(f"   Active Goals: {len(active_goals)}")
            print(f"   Bounded Loops Enabled: {len(autonomy_status.get('enabled_loops', []))}")

            if active_goals:
                print("\n   Current Goals:")
                for goal in active_goals[:3]:
                    print(f"   - {goal.title} ({int(goal.progress * 100)}% complete)")
                    next_step = goal.get_next_step()
                    if next_step:
                        print(f"     Next: {next_step.description}")

            print("=" * 50)

        else:
            print(f"\n[ERROR] Unknown subcommand: {subcommand}")
            print("   Usage: /autonomous [start|stop|pause|resume|status]")

    def _handle_goals_command(self):
        """Handle goals list command"""
        if not hasattr(self, 'goal_system') or not self.goal_system:
            print("\n[ERROR] Goal system not available")
            return

        active_goals = self.goal_system.get_active_goals()
        completed_goals = self.goal_system.get_completed_goals()

        print("\n Active Goals:")
        print("=" * 70)

        if active_goals:
            for i, goal in enumerate(active_goals, 1):
                print(f"\n{i}. {goal.title} (ID: {goal.goal_id})")
                _status = goal.status.value if hasattr(goal.status, 'value') else str(goal.status)
                print(f"   Status: {_status}")
                print(f"   Progress: {int(goal.progress * 100)}%")
                _created = datetime.fromtimestamp(goal.created_at).strftime('%Y-%m-%d %H:%M')
                print(f"   Created: {_created}")

                if goal.deadline:
                    _deadline = datetime.fromtimestamp(goal.deadline).strftime('%Y-%m-%d')
                    print(f"   Deadline: {_deadline}")

                total_steps = len(goal.steps)
                completed_steps = sum(1 for s in goal.steps if s.status == 'completed')
                print(f"   Steps: {completed_steps}/{total_steps} completed")

                next_step = goal.get_next_step()
                if next_step:
                    print(f"   Next: {next_step.description}")

        else:
            print("   No active goals")

        if completed_goals:
            print(f"\n Completed Goals: {len(completed_goals)}")
            for goal in completed_goals[:3]:
                _completed_ts = goal.completed_at or goal.updated_at
                _completed = datetime.fromtimestamp(_completed_ts).strftime('%Y-%m-%d')
                print(f"   - {goal.title} (completed {_completed})")

        print("=" * 70)

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

        # Stop continuous learning
        if hasattr(self, 'continuous_learning'):
            self.continuous_learning.stop()
            logger.info("[OK] Continuous learning stopped")

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

        if hasattr(self, 'proactive_intelligence') and self.proactive_intelligence:
            self.proactive_intelligence.stop()
            logger.info("[OK] Proactive intelligence stopped")

        if getattr(self, 'cognitive_orchestrator', None):
            self.cognitive_orchestrator.stop()
            logger.info("[OK] Cognitive orchestrator stopped")

        if getattr(self, 'persistent_task_queue', None):
            self.persistent_task_queue.stop_background_loop()
            logger.info("[OK] Persistent task queue stopped")

        if hasattr(self, 'execution_loop') and self.execution_loop:
            self.execution_loop.stop()
            logger.info("[OK] Autonomous execution loop stopped")

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
