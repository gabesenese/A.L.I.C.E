"""
Simulation and Scenarios Package

Provides automated testing and training data generation for A.L.I.C.E
through scripted conversations and teacher-guided learning.
"""

from .scenarios import Scenario, ScenarioStep, ScenarioResult
from .run_scenarios import ScenarioRunner

__all__ = ['Scenario', 'ScenarioStep', 'ScenarioResult', 'ScenarioRunner']
