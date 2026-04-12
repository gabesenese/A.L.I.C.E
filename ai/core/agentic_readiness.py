"""Actionable planning utilities for making A.L.I.C.E more agentic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class AgenticFocusItem:
    area: str
    priority: int
    rationale: str
    kpi: str
    target: float


def build_agentic_focus_plan(metrics: Dict[str, Any]) -> List[AgenticFocusItem]:
    """
    Generate prioritized work items from runtime metrics.

    Expected metric keys (all optional):
    - clarification_precision
    - wrong_tool_rate
    - recovery_success_rate
    - stale_state_rate
    - long_horizon_completion_rate
    - personalization_satisfaction
    """
    clarification_precision = float(metrics.get("clarification_precision", 0.0) or 0.0)
    wrong_tool_rate = float(metrics.get("wrong_tool_rate", 1.0) or 1.0)
    recovery_success_rate = float(metrics.get("recovery_success_rate", 0.0) or 0.0)
    stale_state_rate = float(metrics.get("stale_state_rate", 1.0) or 1.0)
    long_horizon_completion_rate = float(
        metrics.get("long_horizon_completion_rate", 0.0) or 0.0
    )
    personalization_satisfaction = float(
        metrics.get("personalization_satisfaction", 0.0) or 0.0
    )

    items: List[AgenticFocusItem] = []

    if wrong_tool_rate > 0.03:
        items.append(
            AgenticFocusItem(
                area="routing_reliability",
                priority=1,
                rationale="Wrong-tool execution is above target and hurts agent trust.",
                kpi="wrong_tool_rate",
                target=0.03,
            )
        )

    if recovery_success_rate < 0.70:
        items.append(
            AgenticFocusItem(
                area="recovery_graph_depth",
                priority=2,
                rationale="Recovery paths need more deterministic branching.",
                kpi="recovery_success_rate",
                target=0.70,
            )
        )

    if stale_state_rate > 0.02:
        items.append(
            AgenticFocusItem(
                area="world_state_freshness",
                priority=3,
                rationale="Stale-world responses degrade perceived autonomy quality.",
                kpi="stale_state_rate",
                target=0.02,
            )
        )

    if long_horizon_completion_rate < 0.65:
        items.append(
            AgenticFocusItem(
                area="goal_continuity",
                priority=4,
                rationale="Long-running goals are not completing frequently enough.",
                kpi="long_horizon_completion_rate",
                target=0.65,
            )
        )

    if personalization_satisfaction < 0.75:
        items.append(
            AgenticFocusItem(
                area="personalization",
                priority=5,
                rationale="Preference model exists but adaptation quality is still low.",
                kpi="personalization_satisfaction",
                target=0.75,
            )
        )

    if clarification_precision < 0.85:
        items.append(
            AgenticFocusItem(
                area="clarification_precision",
                priority=6,
                rationale="Clarification quality should be improved to reduce friction.",
                kpi="clarification_precision",
                target=0.85,
            )
        )

    items.sort(key=lambda item: item.priority)
    return items
