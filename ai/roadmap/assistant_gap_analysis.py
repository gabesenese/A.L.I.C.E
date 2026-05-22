"""Capability gap analysis for A.L.I.C.E against an ideal advanced assistant profile.

Two dimensions tracked per area:
  infra_score  — how complete the implementation is in code (0–1)
  runtime_score — how much is actually delivered to the user at runtime (0–1)

These are intentionally separated because ALICE has a recurring pattern of
building infrastructure that is not wired to runtime output. The runtime_score
is the honest user-facing number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class CapabilityArea:
    name: str
    infra_score: float
    runtime_score: float
    target: float
    infra_notes: str
    runtime_notes: str

    @property
    def current(self) -> float:
        """Runtime-delivered score — the honest user-visible number."""
        return self.runtime_score

    @property
    def gap(self) -> float:
        return max(0.0, float(self.target) - float(self.runtime_score))

    @property
    def wiring_gap(self) -> float:
        """How much infra exists but isn't wired to runtime output."""
        return max(0.0, float(self.infra_score) - float(self.runtime_score))


def build_current_capability_snapshot() -> List[CapabilityArea]:
    """Honest snapshot of infra vs. runtime delivery per capability area."""
    return [
        CapabilityArea(
            name="Natural language routing",
            infra_score=1.0,
            runtime_score=0.85,
            target=1.0,
            infra_notes="ConfidenceFusion (3-signal), RouteArbiter, ConversationStyleMirror, ClarificationFeedbackLoop — all complete.",
            runtime_notes="Routing works reliably. Minor gap: ConfidenceFusion success-rate signal is still sparse; routing gets smarter over time but slowly.",
        ),
        CapabilityArea(
            name="Action reliability",
            infra_score=1.0,
            runtime_score=0.80,
            target=1.0,
            infra_notes="FallbackGraph, RetryMemory, CrossPluginFallbackChain, ReversibilityScorer — all built.",
            runtime_notes="Tool calls mostly reliable. Gap: new plugins without formatters still risk data-dump responses.",
        ),
        CapabilityArea(
            name="World-state grounding",
            infra_score=1.0,
            runtime_score=0.80,
            target=1.0,
            infra_notes="Per-domain freshness tracking, AutomaticDataRefreshPolicy, WorldModel topic confidence.",
            runtime_notes="Weather grounding works after recent fixes. Context graph is defined but not yet the primary retrieval store.",
        ),
        CapabilityArea(
            name="Long-horizon autonomy",
            infra_score=0.90,
            runtime_score=0.65,
            target=1.0,
            infra_notes="GoalEngine (milestones, dependencies, attribution), dual SQLite+JSON storage, session briefing. All built.",
            runtime_notes="Goals persist and accumulate correctly. Gap: session briefing was not auto-displayed until recently wired (May 2026). Routing does not use goal context to bias tool selection.",
        ),
        CapabilityArea(
            name="Personalization",
            infra_score=0.90,
            runtime_score=0.55,
            target=1.0,
            infra_notes="PersonalityEvolutionEngine, ConversationStyleMirror, OpinionExtractor, per-user trait vectors, opinion SQLite store.",
            runtime_notes="Personality hints are in LLM context. Gap: opinions were not injected until May 2026 fix. Trait adaptation (learning_rate=0.05) is very slow; style mirroring adjusts vectors but the LLM prompt doesn't always reflect them.",
        ),
        CapabilityArea(
            name="Persistent memory",
            infra_score=0.90,
            runtime_score=0.60,
            target=1.0,
            infra_notes="SQLite identity store, vector store, RAG engine, ContextGraph, EpisodicMemoryEngine, MemoryConsolidator — all built.",
            runtime_notes="Session history accumulates. Gap: memory is not proactively recalled into LLM prompt without an explicit query. AdaptiveContextSelector trigger logic was too aggressive (min_relevance=0.3, max 2000 chars) — raised in May 2026.",
        ),
        CapabilityArea(
            name="Proactive surfacing",
            infra_score=0.85,
            runtime_score=0.40,
            target=1.0,
            infra_notes="SessionBriefing, ProactiveIntelligence (60s loop), ProactiveAssistant (30s loop), insight callbacks — all built.",
            runtime_notes="Session briefing wired to first turn in May 2026. ProactiveIntelligence insight delivery now wired to turn output. Gap: insight quality is generic (morning_briefing, day_summary) — not yet LLM-shaped to context.",
        ),
        CapabilityArea(
            name="Emotional continuity",
            infra_score=0.85,
            runtime_score=0.35,
            target=1.0,
            infra_notes="PersonalityTraits (6-axis), ConversationStyleMirror, emotional_history in UserIdentity, recent_signals in LLM context.",
            runtime_notes="Emotional signals are detected and stored. Gap: trait vectors adjust slowly (learning_rate=0.05) and the LLM prompt hint translation is simplistic — tone does not perceptibly differ across emotional states.",
        ),
        CapabilityArea(
            name="Self-correction loop",
            infra_score=0.85,
            runtime_score=0.50,
            target=1.0,
            infra_notes="ImprovementLoop (BehaviorEvent → FailureClassifier → ImprovementHypothesis → PatchPlan → ApprovalGate), OllamaEvaluator, ConfidenceFusion reading evaluations.jsonl.",
            runtime_notes="Per-turn eval scores recorded correctly. ConfidenceFusion reads success rates → routing improves over time. Gap: OllamaEvaluator is not auto-triggering corrections; ApprovalGate requires manual approval; learning loop not closed end-to-end.",
        ),
        CapabilityArea(
            name="Safety and policy governance",
            infra_score=1.0,
            runtime_score=0.95,
            target=1.0,
            infra_notes="RiskClassifier, ReversibilityScorer, CompanionPolicyEngine approval gate, dry-run preview.",
            runtime_notes="Safety layer is solid and active on every turn. Near-complete delivery.",
        ),
    ]


def summarize_gap_report() -> Dict[str, object]:
    areas = build_current_capability_snapshot()
    avg_infra = sum(a.infra_score for a in areas) / len(areas)
    avg_runtime = sum(a.runtime_score for a in areas) / len(areas)
    avg_target = sum(a.target for a in areas) / len(areas)
    prioritized = sorted(areas, key=lambda a: a.gap, reverse=True)
    wiring_gaps = sorted(areas, key=lambda a: a.wiring_gap, reverse=True)
    return {
        "avg_infra_score": round(avg_infra, 3),
        "avg_runtime_score": round(avg_runtime, 3),
        "avg_target": round(avg_target, 3),
        # Backward-compatible keys (runtime_score is the honest user-visible number)
        "overall_current": round(avg_runtime, 3),
        "overall_target": round(avg_target, 3),
        "overall_gap": round(max(0.0, avg_target - avg_runtime), 3),
        "wiring_gap_areas": [
            {
                "area": a.name,
                "infra": a.infra_score,
                "runtime": a.runtime_score,
                "wiring_gap": round(a.wiring_gap, 2),
            }
            for a in wiring_gaps
            if a.wiring_gap > 0.1
        ],
        "priority_order": [
            {
                "area": area.name,
                "gap": round(area.gap, 3),
                "infra_score": area.infra_score,
                "runtime_score": area.runtime_score,
                "target": area.target,
                "infra_notes": area.infra_notes,
                "runtime_notes": area.runtime_notes,
            }
            for area in prioritized
        ],
    }
