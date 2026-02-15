"""
Facade Layer for A.L.I.C.E
Simplifies access to subsystems and reduces coupling
"""

from ai.facades.plugin_facade import PluginFacade, get_plugin_facade
from ai.facades.memory_facade import MemoryFacade, get_memory_facade
from ai.facades.core_facade import CoreFacade, get_core_facade
from ai.facades.learning_facade import LearningFacade, get_learning_facade
from ai.facades.planning_facade import PlanningFacade, get_planning_facade
from ai.facades.infrastructure_facade import InfrastructureFacade, get_infrastructure_facade
from ai.facades.optimization_facade import OptimizationFacade, get_optimization_facade
from ai.facades.knowledge_facade import KnowledgeFacade, get_knowledge_facade
from ai.facades.speech_facade import SpeechFacade, get_speech_facade
from ai.facades.training_facade import TrainingFacade, get_training_facade

__all__ = [
    # Facade classes
    'PluginFacade',
    'MemoryFacade',
    'CoreFacade',
    'LearningFacade',
    'PlanningFacade',
    'InfrastructureFacade',
    'OptimizationFacade',
    'KnowledgeFacade',
    'SpeechFacade',
    'TrainingFacade',
    # Singleton getters
    'get_plugin_facade',
    'get_memory_facade',
    'get_core_facade',
    'get_learning_facade',
    'get_planning_facade',
    'get_infrastructure_facade',
    'get_optimization_facade',
    'get_knowledge_facade',
    'get_speech_facade',
    'get_training_facade',
]
