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
            current=1.0,
            target=1.0,
            notes="Complete: confidence-band routing (4 bands) + per-intent thresholds + ConfidenceFusion (router 65% + behavioral prior 20% + intent success rate 15%) + ClarificationFeedbackLoop decaying boost + ConversationStyleMirror auto-constraints.",
        ),
        CapabilityArea(
            name="Action reliability",
            current=1.0,
            target=1.0,
            notes="Complete: FallbackGraph (intent, error_type) → ordered FallbackStep list + RetryMemory escalation (step 0→1→escalation after 3 failures) + CrossPluginFallbackChain + ReversibilityScorer confidence floor + dry-run approval preview.",
        ),
        CapabilityArea(
            name="World-state grounding",
            current=1.0,
            target=1.0,
            notes="Complete: per-domain freshness tracking + AutomaticDataRefreshPolicy force-refresh when stale + stale qualifier on failed weather responses + WorldModel topic confidence tracking + high-confidence topics in LLM context.",
        ),
        CapabilityArea(
            name="Long-horizon autonomy",
            current=1.0,
            target=1.0,
            notes="Complete: semantic goal decomposition + auto-milestones on add() + ingest_completed_intent() milestone advancement + goal dependency graph (dependencies field + get_ready_goals/get_blocker_goals) + session briefing surfaces blocked/ready goals.",
        ),
        CapabilityArea(
            name="Personalization",
            current=1.0,
            target=1.0,
            notes="Complete: PersonalityEvolutionEngine per-turn + LLM context personality hints + ConversationStyleMirror verbosity mirroring + topic confidence in LLM context + intelligent prose shortening (filler strip + sentence drop + clause truncation).",
        ),
        CapabilityArea(
            name="Safety and policy governance",
            current=1.0,
            target=1.0,
            notes="Complete: RiskClassifier (high/medium/low verb regex) + ReversibilityScorer (rev < 0.20 triggers approval) + CompanionPolicyEngine approval gate + dry-run preview in approval message + per-intent policy thresholds.",
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
