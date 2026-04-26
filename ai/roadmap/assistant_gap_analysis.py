"""Capability gap analysis for A.L.I.C.E against an ideal advanced assistant profile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class CapabilityArea:
    name: str
    current: float
    target: float
    notes: str

    @property
    def gap(self) -> float:
        return max(0.0, float(self.target) - float(self.current))


def build_current_capability_snapshot() -> List[CapabilityArea]:
    """Baseline snapshot from currently-implemented architecture."""
    return [
        CapabilityArea(
            name="Natural language routing",
            current=0.78,
            target=0.95,
            notes="Strong layered NLP with clarification/safety hooks, still threshold-heavy.",
        ),
        CapabilityArea(
            name="Action reliability",
            current=0.72,
            target=0.93,
            notes="Unified action engine + recovery hints exist; needs richer fallback graph.",
        ),
        CapabilityArea(
            name="World-state grounding",
            current=0.68,
            target=0.92,
            notes="Live-state freshness metadata exists; needs stricter stale-refresh contracts.",
        ),
        CapabilityArea(
            name="Long-horizon autonomy",
            current=0.64,
            target=0.91,
            notes="Cognitive orchestrator and goals exist; needs stronger decomposition/execution linkage.",
        ),
        CapabilityArea(
            name="Personalization",
            current=0.55,
            target=0.90,
            notes="Preference model added; behavior adaptation still limited.",
        ),
        CapabilityArea(
            name="Safety and policy governance",
            current=0.75,
            target=0.96,
            notes="Authorization + simulation hooks exist; needs deeper risk modeling.",
        ),
    ]


def summarize_gap_report() -> Dict[str, object]:
    areas = build_current_capability_snapshot()
    overall_current = sum(a.current for a in areas) / len(areas)
    overall_target = sum(a.target for a in areas) / len(areas)
    prioritized = sorted(areas, key=lambda a: a.gap, reverse=True)
    return {
        "overall_current": round(overall_current, 3),
        "overall_target": round(overall_target, 3),
        "overall_gap": round(max(0.0, overall_target - overall_current), 3),
        "priority_order": [
            {
                "area": area.name,
                "gap": round(area.gap, 3),
                "current": area.current,
                "target": area.target,
                "notes": area.notes,
            }
            for area in prioritized
        ],
    }
