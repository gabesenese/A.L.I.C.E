from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeModeConfig:
    mode: str
    enable_voice: bool
    enable_training: bool
    enable_lab_tools: bool
    enable_background_learning: bool
    enable_proactive_loops: bool
    enable_cognitive_orchestrator: bool
    enable_autonomous_agent: bool
    enable_analytics: bool
    enable_advanced_tiers: bool
    enable_contract_pipeline: bool
    enable_local_actions: bool

    @classmethod
    def for_mode(cls, mode: str) -> "RuntimeModeConfig":
        m = str(mode or "minimal").strip().lower()
        if m == "voice":
            return cls(m, True, False, False, False, False, True, True, True, True, True, True)
        if m == "lab":
            return cls(m, False, False, True, False, True, True, True, True, True, True, True)
        if m == "training":
            return cls(m, False, True, False, True, False, True, False, False, True, True, True)
        if m == "agentic":
            return cls(m, False, False, False, False, True, True, True, True, True, True, True)
        if m == "dev":
            return cls(m, False, False, False, False, True, True, True, True, True, True, True)
        return cls("minimal", False, False, False, False, False, False, False, False, False, True, True)


def resolve_runtime_mode(explicit_mode: str | None) -> str:
    if explicit_mode:
        return str(explicit_mode).strip().lower()
    env_mode = str(os.getenv("ALICE_RUNTIME_MODE", "") or "").strip().lower()
    if env_mode:
        return env_mode
    return "minimal"

