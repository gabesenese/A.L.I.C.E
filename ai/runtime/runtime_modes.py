from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass(frozen=True)
class RuntimeMode:
    name: str
    enabled_groups: Set[str] = field(default_factory=set)


_BASE_GROUPS = {
    "core_nlp",
    "core_memory",
    "core_llm",
    "contract_pipeline",
    "runtime_boundaries",
    "companion_runtime",
    "local_actions",
    "operator_state",
    "memory_turn_service",
    "core_plugins",
}

_MODE_GROUPS: Dict[str, Set[str]] = {
    "minimal": set(_BASE_GROUPS),
    "dev": set(_BASE_GROUPS) | {"debug_tools"},
    "agentic": set(_BASE_GROUPS) | {"agent_loop", "route_arbiter"},
    "training": set(_BASE_GROUPS) | {"training", "evaluation"},
    "lab": set(_BASE_GROUPS) | {"lab", "red_team"},
    "voice": set(_BASE_GROUPS) | {"voice"},
}


def get_runtime_mode(mode: str = "minimal") -> RuntimeMode:
    normalized = str(mode or "minimal").strip().lower()
    if normalized not in _MODE_GROUPS:
        normalized = "minimal"
    return RuntimeMode(name=normalized, enabled_groups=set(_MODE_GROUPS[normalized]))


def list_runtime_modes() -> List[str]:
    return sorted(_MODE_GROUPS.keys())

