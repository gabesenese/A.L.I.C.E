"""Generates compact what-if scenarios for decision support."""

from __future__ import annotations

from typing import Dict, List


class HypotheticalScenarioGenerator:
    def generate(self, prompt: str, max_scenarios: int = 3) -> List[Dict[str, str]]:
        text = str(prompt or "").strip()
        if not text:
            return []
        scenarios = [
            {
                "name": "Best case",
                "assumption": "Primary dependencies hold and execution stays within expected bounds.",
                "impact": "Fast completion with minimal correction overhead.",
            },
            {
                "name": "Most likely",
                "assumption": "One moderate issue appears and is fixed within one iteration.",
                "impact": "Slight delay, but objective remains achievable today.",
            },
            {
                "name": "Risk case",
                "assumption": "Two constraints conflict or an external dependency fails.",
                "impact": "Need fallback plan and staged rollout.",
            },
        ]
        return scenarios[: max(1, int(max_scenarios or 3))]
