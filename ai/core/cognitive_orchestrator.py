"""
Cognitive Orchestrator.

Unifies five high-leverage capabilities required for a realistic cognitive system:
1) Continuous cognition loop (runs in background even without user input)
2) Event-driven cognition (reacts to environment and turns)
3) Long-horizon goal tracking (project-level memory beyond single turns)
4) Self-improvement planning (meta-cognitive adaptation from failure patterns)
5) Simulation reasoning (evaluate options before acting)
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

from ai.core.goal_tracker import GoalTracker, get_goal_tracker
from ai.core.reflection_engine import ReflectionEngine, get_reflection_engine
from ai.core.response_quality_tracker import ResponseQualityTracker, get_response_quality_tracker
from ai.infrastructure.event_bus import EventBus, EventPriority, EventType, get_event_bus


@dataclass
class ProjectGoal:
    """Long-horizon project goal tracked across many turns and idle cycles."""

    goal_id: str
    description: str
    horizon_days: int = 30
    milestones: List[str] = field(default_factory=list)
    completed_milestones: List[str] = field(default_factory=list)
    progress: float = 0.0
    last_updated: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "horizon_days": int(self.horizon_days),
            "milestones": list(self.milestones),
            "completed_milestones": list(self.completed_milestones),
            "progress": round(float(self.progress), 3),
            "last_updated": float(self.last_updated),
            "metadata": dict(self.metadata),
        }


@dataclass
class SimulationScore:
    """Scored simulation record for an action candidate."""

    action_id: str
    score: float
    expected_gain: float
    risk: float
    cost: float
    confidence: float
    reversible: bool
    rationale: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "score": round(float(self.score), 4),
            "expected_gain": round(float(self.expected_gain), 4),
            "risk": round(float(self.risk), 4),
            "cost": round(float(self.cost), 4),
            "confidence": round(float(self.confidence), 4),
            "reversible": bool(self.reversible),
            "rationale": list(self.rationale),
        }


class CognitiveOrchestrator:
    """Background cognition coordinator for continuous, event-driven intelligence."""

    def __init__(
        self,
        *,
        event_bus: Optional[EventBus] = None,
        goal_tracker: Optional[GoalTracker] = None,
        reflection_engine: Optional[ReflectionEngine] = None,
        response_quality_tracker: Optional[ResponseQualityTracker] = None,
        tick_interval_seconds: float = 30.0,
    ) -> None:
        self.event_bus = event_bus or get_event_bus()
        self.goal_tracker = goal_tracker or get_goal_tracker()
        self.reflection_engine = reflection_engine or get_reflection_engine()
        self.response_quality_tracker = (
            response_quality_tracker or get_response_quality_tracker()
        )

        self.tick_interval_seconds = max(0.05, float(tick_interval_seconds or 30.0))

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._turn_trace: Deque[Dict[str, Any]] = deque(maxlen=120)
        self._project_goals: Dict[str, ProjectGoal] = {}
        self._latest_improvement_plan: List[Dict[str, Any]] = []
        self._last_cycle_report: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            name="CognitiveOrchestrator",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None

    # ------------------------------------------------------------------
    # Turn + event ingestion
    # ------------------------------------------------------------------

    def observe_turn(
        self,
        *,
        user_input: str,
        intent: str,
        response: str,
        gate_accepted: bool = True,
        route: str = "llm",
        decision_scores: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Ingest one completed turn into cognition state and adaptation loops."""
        decision_scores = decision_scores or {"llm": 0.5, "tools": 0.5}

        quality = self.response_quality_tracker.track_turn(
            user_input=user_input,
            response=response,
            intent=intent,
            gate_accepted=gate_accepted,
            reflection_score=0.0,
        )

        reflection = self.reflection_engine.reflect(
            user_input=user_input,
            intent=intent,
            response=response,
            route=route,
            gate_accepted=gate_accepted,
            decision_scores=decision_scores,
            prior_confidence=0.7,
            quality_metrics=quality.as_dict(),
            failure_type=quality.failure_type,
        )

        turn_record = {
            "timestamp": datetime.now().isoformat(),
            "intent": intent,
            "route": route,
            "gate_accepted": bool(gate_accepted),
            "quality": quality.as_dict(),
            "reflection": reflection.as_dict(),
        }
        with self._lock:
            self._turn_trace.append(turn_record)

        # Event-driven reaction path
        self.event_bus.emit(
            EventType.USER_QUERY,
            {
                "intent": intent,
                "quality": quality.as_dict(),
                "failure_type": quality.failure_type,
            },
            priority=EventPriority.NORMAL,
            source="cognitive_orchestrator",
        )

        if quality.failure_type != "none":
            self.event_bus.emit(
                EventType.SYSTEM_WARNING,
                {
                    "warning": "quality_regression",
                    "failure_type": quality.failure_type,
                    "intent": intent,
                },
                priority=EventPriority.HIGH,
                source="cognitive_orchestrator",
            )

        self._sync_goal_tracker_status()
        plan = self.create_self_improvement_plan()

        return {
            "quality": quality.as_dict(),
            "reflection": reflection.as_dict(),
            "improvement_plan": plan,
        }

    # ------------------------------------------------------------------
    # Long-horizon goals
    # ------------------------------------------------------------------

    def register_project_goal(
        self,
        *,
        goal_id: str,
        description: str,
        horizon_days: int = 30,
        milestones: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._lock:
            self._project_goals[goal_id] = ProjectGoal(
                goal_id=goal_id,
                description=description,
                horizon_days=max(1, int(horizon_days or 30)),
                milestones=list(milestones or []),
                metadata=dict(metadata or {}),
            )

    def mark_milestone_completed(self, *, goal_id: str, milestone: str) -> bool:
        with self._lock:
            goal = self._project_goals.get(goal_id)
            if goal is None:
                return False
            if milestone not in goal.completed_milestones:
                goal.completed_milestones.append(milestone)
            total = max(1, len(goal.milestones))
            goal.progress = min(1.0, len(goal.completed_milestones) / total)
            goal.last_updated = time.time()
            return True

    # ------------------------------------------------------------------
    # Meta-cognitive self-improvement
    # ------------------------------------------------------------------

    def create_self_improvement_plan(self) -> List[Dict[str, Any]]:
        """Build a tactical self-improvement plan from rolling quality signals."""
        summary = self.response_quality_tracker.get_quality_summary()
        failures = summary.get("failure_counts", {}) or {}
        plan: List[Dict[str, Any]] = []

        weak = int(failures.get("weak_knowledge", 0))
        overgen = int(failures.get("overgeneralization", 0))
        routing = int(failures.get("routing_mistake", 0))
        drift = int(failures.get("topic_drift", 0))
        assumption = int(failures.get("assumption_without_evidence", 0))
        redundant = int(failures.get("redundant_suggestion", 0))

        if weak + overgen >= 3:
            plan.append(
                {
                    "action_id": "improve_grounding",
                    "priority": "high",
                    "reason": "Repeated weak-knowledge / overgeneralization failures",
                }
            )

        if routing >= 2:
            plan.append(
                {
                    "action_id": "tighten_tool_veto",
                    "priority": "high",
                    "reason": "Multiple routing mistakes indicate tool-selection instability",
                }
            )

        if drift >= 2:
            plan.append(
                {
                    "action_id": "increase_clarification_bias",
                    "priority": "medium",
                    "reason": "Topic drift suggests clarification threshold should rise",
                }
            )

        if assumption + redundant >= 2:
            plan.append(
                {
                    "action_id": "enforce_evidence_before_suggestion",
                    "priority": "high",
                    "reason": "Assumption/redundancy failures require premise-check guardrails",
                }
            )

        if float(summary.get("gate_accept_rate", 1.0) or 1.0) < 0.55:
            plan.append(
                {
                    "action_id": "review_gate_policy",
                    "priority": "medium",
                    "reason": "Low gate acceptance indicates plan/response mismatch",
                }
            )

        with self._lock:
            self._latest_improvement_plan = plan

        if plan:
            self.event_bus.emit_custom(
                "cognition.self_improvement_plan",
                {
                    "plan": plan,
                    "failure_counts": failures,
                    "turns_tracked": summary.get("turns_tracked", 0),
                },
                priority=EventPriority.HIGH,
            )

        return plan

    # ------------------------------------------------------------------
    # Simulation reasoning
    # ------------------------------------------------------------------

    def simulate_before_action(
        self,
        candidates: List[Dict[str, Any]],
        *,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Score options before acting and return the safest high-value candidate."""
        context = context or {}
        scored: List[SimulationScore] = []

        for cand in candidates or []:
            action_id = str(cand.get("action_id") or cand.get("id") or "unknown")
            gain = float(cand.get("expected_gain", 0.0) or 0.0)
            risk = float(cand.get("risk", 0.0) or 0.0)
            cost = float(cand.get("cost", 0.0) or 0.0)
            confidence = float(cand.get("confidence", 0.5) or 0.5)
            reversible = bool(cand.get("reversible", False))

            # Weighted utility with explicit risk/cost penalties.
            score = (0.48 * gain) + (0.22 * confidence) - (0.22 * risk) - (0.08 * cost)
            if reversible:
                score += 0.07

            rationale: List[str] = []
            if risk >= 0.65:
                rationale.append("high_risk_penalty")
            if cost >= 0.65:
                rationale.append("high_cost_penalty")
            if gain >= 0.65:
                rationale.append("high_expected_gain")
            if reversible:
                rationale.append("reversible_bonus")
            if confidence < 0.40:
                rationale.append("low_confidence_penalty")

            scored.append(
                SimulationScore(
                    action_id=action_id,
                    score=score,
                    expected_gain=gain,
                    risk=risk,
                    cost=cost,
                    confidence=confidence,
                    reversible=reversible,
                    rationale=rationale,
                )
            )

        ranked = sorted(scored, key=lambda item: item.score, reverse=True)
        best = ranked[0].as_dict() if ranked else {}
        payload = {
            "best_action": best,
            "ranked": [item.as_dict() for item in ranked],
            "context": context,
        }

        if ranked:
            self.event_bus.emit_custom(
                "cognition.simulation_decision",
                {
                    "best_action": best,
                    "candidate_count": len(ranked),
                },
                priority=EventPriority.NORMAL,
            )

        return payload

    # ------------------------------------------------------------------
    # Continuous loop internals
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while self._running:
            try:
                cycle = self._run_cycle()
                with self._lock:
                    self._last_cycle_report = cycle
            except Exception:
                # Keep loop resilient and non-fatal.
                pass
            time.sleep(self.tick_interval_seconds)

    def _run_cycle(self) -> Dict[str, Any]:
        now_ts = time.time()

        # Continuous cognition heartbeat.
        self.event_bus.emit(
            EventType.SYSTEM_IDLE,
            {
                "source": "cognitive_orchestrator",
                "timestamp": now_ts,
            },
            priority=EventPriority.LOW,
            source="cognitive_orchestrator",
        )

        stale_goals = self._review_long_horizon_goals(now_ts)
        plan = self.create_self_improvement_plan()

        return {
            "timestamp": now_ts,
            "stale_goal_count": len(stale_goals),
            "improvement_actions": len(plan),
        }

    def _review_long_horizon_goals(self, now_ts: float) -> List[Dict[str, Any]]:
        stale: List[Dict[str, Any]] = []
        with self._lock:
            goals = list(self._project_goals.values())

        for goal in goals:
            stale_after_seconds = min(72.0, float(goal.horizon_days)) * 3600.0
            age = now_ts - float(goal.last_updated)
            if age > stale_after_seconds and goal.progress < 1.0:
                stale_info = {
                    "goal_id": goal.goal_id,
                    "description": goal.description,
                    "hours_since_update": round(age / 3600.0, 2),
                    "progress": round(goal.progress, 3),
                }
                stale.append(stale_info)
                self.event_bus.emit(
                    EventType.SYSTEM_WARNING,
                    {
                        "warning": "stale_long_horizon_goal",
                        **stale_info,
                    },
                    priority=EventPriority.NORMAL,
                    source="cognitive_orchestrator",
                )
        return stale

    def _sync_goal_tracker_status(self) -> None:
        status = self.goal_tracker.get_status() if self.goal_tracker else None
        if not status:
            return

        goal_desc = str(status.get("goal_description") or "").strip()
        if not goal_desc:
            return

        with self._lock:
            goal = self._project_goals.get("conversation_goal")
            if goal is None:
                self._project_goals["conversation_goal"] = ProjectGoal(
                    goal_id="conversation_goal",
                    description=goal_desc,
                    horizon_days=14,
                    milestones=[sg.get("desc", "") for sg in status.get("subgoals", []) if sg.get("desc")],
                    progress=float(status.get("progress_score", 0.0) or 0.0),
                )
            else:
                goal.description = goal_desc
                goal.progress = float(status.get("progress_score", goal.progress) or goal.progress)
                goal.last_updated = time.time()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_cognitive_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            goals = [goal.as_dict() for goal in self._project_goals.values()]
            latest_plan = list(self._latest_improvement_plan)
            cycle = dict(self._last_cycle_report)
            turn_count = len(self._turn_trace)

        return {
            "running": bool(self._running),
            "tick_interval_seconds": float(self.tick_interval_seconds),
            "tracked_turns": int(turn_count),
            "project_goals": goals,
            "latest_improvement_plan": latest_plan,
            "last_cycle_report": cycle,
        }


_cognitive_orchestrator: Optional[CognitiveOrchestrator] = None


def get_cognitive_orchestrator(
    *,
    event_bus: Optional[EventBus] = None,
    goal_tracker: Optional[GoalTracker] = None,
    reflection_engine: Optional[ReflectionEngine] = None,
    response_quality_tracker: Optional[ResponseQualityTracker] = None,
    tick_interval_seconds: float = 30.0,
) -> CognitiveOrchestrator:
    global _cognitive_orchestrator
    if _cognitive_orchestrator is None:
        _cognitive_orchestrator = CognitiveOrchestrator(
            event_bus=event_bus,
            goal_tracker=goal_tracker,
            reflection_engine=reflection_engine,
            response_quality_tracker=response_quality_tracker,
            tick_interval_seconds=tick_interval_seconds,
        )
    return _cognitive_orchestrator
