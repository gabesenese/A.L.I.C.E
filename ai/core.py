"""
A.L.I.C.E Core - Unified Engine Interfaces

This module provides the single source of truth for A.L.I.C.E's three core engines.
All other modules should import from here rather than individual engine files.

Architecture:
- ContextEngine: User context, conversation state, entity tracking
- ReasoningEngine: World state, goals, reference resolution, verification
- LearningEngine: Pattern learning, interaction collection, response generation

Design Principle: Single Responsibility, Clear Interfaces
"""

from typing import Optional
import logging

# Import unified engines
from ai.context_engine import get_context_engine, ContextEngine
from ai.reasoning_engine import get_reasoning_engine, ReasoningEngine
from ai.learning_engine import get_learning_engine, LearningEngine

# Import supporting systems
from ai.router import get_router, RequestRouter, RoutingDecision
from ai.llm_policy import get_llm_policy, configure_llm_policy, LLMPolicy, LLMCallType
from ai.service_degradation import get_degradation_handler, ServiceDegradationHandler
from ai.models.simple_formatters import FormatterRegistry

logger = logging.getLogger(__name__)


class ALICECore:
    """
    Core system facade providing access to all unified engines.
    
    This is the single interface that main.py should use.
    """
    
    def __init__(self, user_name: str = "User", llm_policy: str = "minimal"):
        """
        Initialize A.L.I.C.E core engines
        
        Args:
            user_name: User's name for personalization
            llm_policy: LLM usage policy ('minimal', 'balanced', 'aggressive')
        """
        logger.info("=" * 80)
        logger.info("Initializing A.L.I.C.E Core Engines")
        logger.info("=" * 80)
        
        # Initialize unified engines
        self.context = get_context_engine()
        self.context.user_prefs.name = user_name
        logger.info("[CORE] Context Engine initialized")
        
        self.reasoning = get_reasoning_engine(user_name)
        logger.info("[CORE] Reasoning Engine initialized")
        
        self.learning = get_learning_engine()
        logger.info("[CORE] Learning Engine initialized")
        
        # Initialize routing and policy
        self.router = get_router()
        logger.info("[CORE] Request Router initialized")
        
        self._configure_llm_policy(llm_policy)
        self.llm_policy = get_llm_policy()
        logger.info(f"[CORE] LLM Policy configured: {llm_policy}")
        
        # Initialize service degradation
        self.degradation = get_degradation_handler()
        logger.info("[CORE] Service Degradation Handler initialized")
        
        # Initialize formatters
        self.formatters = FormatterRegistry
        logger.info("[CORE] Simple Formatters registered")
        
        logger.info("[CORE] All engines ready")
        logger.info("=" * 80)
    
    def _configure_llm_policy(self, policy: str):
        """Configure LLM policy based on preset"""
        policies = {
            'minimal': {
                'max_calls_per_minute': 5,
                'allow_llm_for_chitchat': False,
                'allow_llm_for_tools': False,
                'allow_llm_for_generation': True,
                'require_user_approval': True
            },
            'balanced': {
                'max_calls_per_minute': 15,
                'allow_llm_for_chitchat': False,
                'allow_llm_for_tools': True,
                'allow_llm_for_generation': True,
                'require_user_approval': False
            },
            'aggressive': {
                'max_calls_per_minute': 30,
                'allow_llm_for_chitchat': True,
                'allow_llm_for_tools': True,
                'allow_llm_for_generation': True,
                'require_user_approval': False
            }
        }
        
        config = policies.get(policy, policies['minimal'])
        configure_llm_policy(**config)
    
    def get_stats(self) -> dict:
        """Get statistics from all engines"""
        return {
            'routing': self.router.get_stats(),
            'learning': self.learning.get_statistics(),
            'llm_policy': self.llm_policy.get_stats(),
            'service_health': {
                name: status.is_available
                for name, status in self.degradation.get_all_statuses().items()
            }
        }
    
    def periodic_maintenance(self):
        """Run periodic maintenance tasks"""
        # Memory consolidation happens in memory_system.periodic_consolidation_check()
        # This is called from main.py after each turn
        
        # Check for pattern learning opportunities
        suggestions = self.learning.suggest_pattern_creation(min_occurrences=3)
        if suggestions:
            logger.info(f"[CORE] Found {len(suggestions)} pattern learning opportunities")
            for suggestion in suggestions[:3]:  # Log top 3
                logger.info(f"  - '{suggestion['user_input'][:50]}...' ({suggestion['occurrence_count']}x)")


# Global singleton
_core: Optional[ALICECore] = None


def get_core(user_name: str = "User", llm_policy: str = "minimal") -> ALICECore:
    """
    Get or initialize the core A.L.I.C.E system
    
    Args:
        user_name: User's name
        llm_policy: 'minimal' (default), 'balanced', or 'aggressive'
    
    Returns:
        Initialized ALICECore instance
    """
    global _core
    if _core is None:
        _core = ALICECore(user_name, llm_policy)
    return _core


# Convenience exports for direct access
__all__ = [
    'ALICECore',
    'get_core',
    'ContextEngine',
    'ReasoningEngine', 
    'LearningEngine',
    'RequestRouter',
    'RoutingDecision',
    'LLMPolicy',
    'LLMCallType',
    'ServiceDegradationHandler',
    'FormatterRegistry'
]
