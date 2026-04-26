"""
Simulation and Scenarios Package

Provides automated testing and training data generation for A.L.I.C.E
through scripted conversations and teacher-guided learning.
"""

from typing import TYPE_CHECKING, Any

from .scenarios import Scenario, ScenarioStep, ScenarioResult

if TYPE_CHECKING:
	from .run_scenarios import ScenarioRunner

__all__ = ['Scenario', 'ScenarioStep', 'ScenarioResult', 'ScenarioRunner']


def __getattr__(name: str) -> Any:
	"""Lazily expose heavy modules to avoid import-time side effects."""
	if name == 'ScenarioRunner':
		from .run_scenarios import ScenarioRunner

		return ScenarioRunner
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
