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
from ai.core.response_quality_tracker import (
    ResponseQualityTracker,
    get_response_quality_tracker,
)
from ai.infrastructure.event_bus import (
    EventBus,
    EventPriority,
    EventType,
    get_event_bus,
)

GOAL_CREATED = "created"
GOAL_ACTIVE = "active"
GOAL_PROGRESSING = "progressing"
GOAL_COMPLETED = "completed"
GOAL_ABANDONED = "abandoned"
GOAL_BLOCKED = "blocked"
GOAL_DRIFTED = "drifted"

IMPROVEMENT_PENDING = "pending"
IMPROVEMENT_IN_PROGRESS = "in_progress"
IMPROVEMENT_DONE = "done"
IMPROVEMENT_DEFERRED = "deferred"


@dataclass
class ProjectGoal:
    """Long-horizon project goal tracked across many turns and idle cycles."""

    goal_id: str
    description: str
    horizon_days: int = 30
    milestones: List[str] = field(default_factory=list)
    completed_milestones: List[str] = field(default_factory=list)
    state: str = GOAL_CREATED
    priority_score: float = 0.5
    progress: float = 0.0
    last_updated: float = field(default_factory=time.time)
    related_conversations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "horizon_days": int(self.horizon_days),
            "milestones": list(self.milestones),
            "completed_milestones": list(self.completed_milestones),
            "state": str(self.state),
            "priority_score": round(float(self.priority_score), 3),
            "progress": round(float(self.progress), 3),
            "last_updated": float(self.last_updated),
            "related_conversations": list(self.related_conversations[-20:]),
            "metadata": dict(self.metadata),
        }


@dataclass
class FailureRecord:
    """Aggregated failure pattern record for cognitive monitoring."""

    failure_type: str
    frequency: int = 0
    last_occurrence: float = field(default_factory=time.time)
    suspected_cause: str = "unknown"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "type": self.failure_type,
            "frequency": int(self.frequency),
            "last_occurrence": float(self.last_occurrence),
            "suspected_cause": self.suspected_cause,
        }


@dataclass
class ImprovementTask:
    """Internal improvement backlog task for adaptive self-optimization."""

    task_id: str
    description: str
    source: str
    priority: float = 0.5
    status: str = IMPROVEMENT_PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "source": self.source,
            "priority": round(float(self.priority), 3),
            "status": self.status,
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
        }


@dataclass
class CognitiveState:
    """Lightweight internal cognition state used by background loop decisions."""

    active_goal: str = ""
    recent_problem: str = ""
    current_priority: str = ""
    attention_focus: str = ""
    improvement_targets: List[str] = field(default_factory=list)
    last_importance_score: float = 0.0
    reasoning_triggered_at: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "active_goal": self.active_goal,
            "recent_problem": self.recent_problem,
            "current_priority": self.current_priority,
            "attention_focus": self.attention_focus,
            "improvement_targets": list(self.improvement_targets),
            "last_importance_score": round(float(self.last_importance_score), 3),
            "reasoning_triggered_at": float(self.reasoning_triggered_at),
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
        executive_controller: Optional[Any] = None,
        conversation_state_tracker: Optional[Any] = None,
        response_planner: Optional[Any] = None,
        tick_interval_seconds: float = 5.0,
        reasoning_importance_threshold: float = 0.74,
    ) -> None:
        self.event_bus = event_bus or get_event_bus()
        self.goal_tracker = goal_tracker or get_goal_tracker()
        self.reflection_engine = reflection_engine or get_reflection_engine()
        self.response_quality_tracker = (
            response_quality_tracker or get_response_quality_tracker()
        )
        self.executive_controller = executive_controller
        self.conversation_state_tracker = conversation_state_tracker
        self.response_planner = response_planner

        self.tick_interval_seconds = max(0.05, float(tick_interval_seconds or 5.0))
        self.reasoning_importance_threshold = max(
            0.0, min(1.0, float(reasoning_importance_threshold or 0.74))
        )

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._turn_trace: Deque[Dict[str, Any]] = deque(maxlen=120)
        self._project_goals: Dict[str, ProjectGoal] = {}
        self._failure_log: Dict[str, FailureRecord] = {}
        self._improvement_queue: List[ImprovementTask] = []
        self._cognitive_state = CognitiveState()
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

        self._record_failure(quality.failure_type)

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
        self._update_goal_lifecycle_on_turn(user_input=user_input, intent=intent)
        plan = self.create_self_improvement_plan()

        return {
            "quality": quality.as_dict(),
            "reflection": reflection.as_dict(),
            "improvement_plan": plan,
        }

    def ingest_user_feedback(
        self,
        *,
        user_input: str,
        previous_intent: str,
        corrected_intent: str = "",
        severity: float = 0.85,
    ) -> None:
        """Capture direct user correction and schedule high-priority adaptation."""
        self._record_failure("routing_mistake")
        self._enqueue_improvement_task(
            description=(
                f"Incorporate user correction: '{previous_intent}'"
                + (f" -> '{corrected_intent}'" if corrected_intent else "")
            ),
            source="user_feedback",
            priority=max(0.70, min(1.0, float(severity or 0.85))),
        )
        self.event_bus.emit_custom(
            "cognition.user_feedback",
            {
                "user_input": str(user_input or "")[:300],
                "previous_intent": previous_intent,
                "corrected_intent": corrected_intent,
                "severity": float(severity or 0.85),
            },
            priority=EventPriority.HIGH,
        )

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
                state=GOAL_ACTIVE,
                priority_score=float(
                    (metadata or {}).get("priority_score", 0.6) or 0.6
                ),
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
            goal.state = GOAL_COMPLETED if goal.progress >= 1.0 else GOAL_PROGRESSING
            goal.metadata["stagnation_turns"] = 0
            goal.metadata["drift_turns"] = 0
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
        for item in plan:
            self._enqueue_improvement_task(
                description=str(
                    item.get("reason")
                    or item.get("action_id")
                    or "improve system behavior"
                ),
                source="reflection",
                priority=(
                    0.9 if str(item.get("priority") or "").lower() == "high" else 0.65
                ),
            )

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
    # Failure monitoring + improvement queue
    # ------------------------------------------------------------------

    def _record_failure(self, failure_type: str) -> None:
        kind = str(failure_type or "none").strip().lower()
        if kind in ("", "none"):
            return

        suspected = {
            "routing_mistake": "tool selection instability",
            "topic_drift": "weak context carryover",
            "weak_knowledge": "insufficient grounding",
            "overgeneralization": "low precision response policy",
            "assumption_without_evidence": "premise validation missing",
            "redundant_suggestion": "capability awareness drift",
        }.get(kind, "unclassified regression")

        with self._lock:
            rec = self._failure_log.get(kind)
            if rec is None:
                rec = FailureRecord(
                    failure_type=kind, frequency=0, suspected_cause=suspected
                )
                self._failure_log[kind] = rec
            rec.frequency += 1
            rec.last_occurrence = time.time()
            rec.suspected_cause = suspected
            self._cognitive_state.recent_problem = kind

    def _enqueue_improvement_task(
        self,
        *,
        description: str,
        source: str,
        priority: float,
    ) -> None:
        desc = str(description or "").strip()
        if not desc:
            return
        with self._lock:
            for task in self._improvement_queue:
                if task.description == desc and task.status in (
                    IMPROVEMENT_PENDING,
                    IMPROVEMENT_IN_PROGRESS,
                ):
                    task.priority = max(float(task.priority), float(priority))
                    task.updated_at = time.time()
                    return
            task_id = f"impr_{int(time.time() * 1000)}_{len(self._improvement_queue)}"
            self._improvement_queue.append(
                ImprovementTask(
                    task_id=task_id,
                    description=desc,
                    source=str(source or "reflection"),
                    priority=max(0.0, min(1.0, float(priority))),
                )
            )

    def _reprioritize_improvement_queue(self) -> None:
        now_ts = time.time()
        with self._lock:
            failures = dict(self._failure_log)
            for task in self._improvement_queue:
                age_hours = max(0.0, (now_ts - task.created_at) / 3600.0)
                urgency_bonus = min(0.15, age_hours / 72.0)
                failure_bonus = 0.0
                desc = task.description.lower()
                if "routing" in desc:
                    failure_bonus += min(
                        0.20,
                        failures.get("routing_mistake", FailureRecord("x")).frequency
                        * 0.05,
                    )
                if "plausibility" in desc or "intent" in desc:
                    failure_bonus += min(
                        0.20,
                        failures.get("weak_knowledge", FailureRecord("x")).frequency
                        * 0.04,
                    )
                if "planning" in desc:
                    failure_bonus += min(
                        0.20,
                        failures.get("overgeneralization", FailureRecord("x")).frequency
                        * 0.04,
                    )
                task.priority = max(
                    0.0, min(1.0, float(task.priority) + urgency_bonus + failure_bonus)
                )
                task.updated_at = now_ts
            self._improvement_queue.sort(key=lambda t: t.priority, reverse=True)

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

        goal_report = self._monitor_active_goals(now_ts)
        failure_report = self._check_recent_failures(now_ts)
        priority_report = self._update_priorities(now_ts)
        schedule_report = self._schedule_improvements_if_needed(now_ts)
        trigger_report = self._evaluate_controlled_reasoning_triggers(now_ts)

        return {
            "timestamp": now_ts,
            "stale_goal_count": int(goal_report.get("stale_goal_count", 0)),
            "improvement_actions": int(schedule_report.get("improvements_enqueued", 0)),
            "failure_patterns": int(failure_report.get("patterns_detected", 0)),
            "priority_score": float(priority_report.get("importance_score", 0.0)),
            "reasoning_triggered": bool(trigger_report.get("triggered", False)),
        }

    def _monitor_active_goals(self, now_ts: float) -> Dict[str, Any]:
        stale_goals = self._review_long_horizon_goals(now_ts)
        active_goal = ""
        with self._lock:
            for goal in self._project_goals.values():
                if goal.progress >= 1.0:
                    goal.state = GOAL_COMPLETED
                elif goal.progress > 0.0:
                    if goal.state not in (GOAL_BLOCKED, GOAL_DRIFTED):
                        goal.state = GOAL_PROGRESSING
                elif goal.state not in (
                    GOAL_ACTIVE,
                    GOAL_CREATED,
                    GOAL_BLOCKED,
                    GOAL_DRIFTED,
                ):
                    goal.state = GOAL_ACTIVE
            ranked = sorted(
                self._project_goals.values(),
                key=lambda g: (float(g.priority_score), float(g.progress)),
                reverse=True,
            )
            if ranked:
                active_goal = ranked[0].description
                self._cognitive_state.active_goal = active_goal
        return {"stale_goal_count": len(stale_goals), "active_goal": active_goal}

    def _check_recent_failures(self, now_ts: float) -> Dict[str, Any]:
        cutoff = now_ts - 3600.0
        patterns = 0
        flagged: List[str] = []
        with self._lock:
            for record in self._failure_log.values():
                if record.last_occurrence >= cutoff and record.frequency >= 2:
                    patterns += 1
                    flagged.append(record.failure_type)
        for kind in flagged:
            self._enqueue_improvement_task(
                description=f"Stabilize recurring failure pattern: {kind}",
                source="failure_monitor",
                priority=0.85,
            )
        return {"patterns_detected": patterns, "flagged_failures": flagged}

    def _score_attention_importance(self, now_ts: float) -> float:
        with self._lock:
            goals = list(self._project_goals.values())
            failures = list(self._failure_log.values())

        goal_relevance = 0.0
        if goals:
            goal_relevance = (
                max(
                    min(1.0, float(g.priority_score) * (1.0 - float(g.progress)))
                    for g in goals
                    if g.state not in (GOAL_COMPLETED, GOAL_ABANDONED)
                )
                if any(g.state not in (GOAL_COMPLETED, GOAL_ABANDONED) for g in goals)
                else 0.0
            )

        recency = 0.0
        if self._turn_trace:
            recency = min(1.0, len(self._turn_trace) / 50.0)

        error_weight = 0.0
        for rec in failures:
            age_minutes = max(0.1, (now_ts - rec.last_occurrence) / 60.0)
            decay = max(0.15, 1.0 / age_minutes)
            error_weight += min(0.25, rec.frequency * 0.03 * decay)
        error_weight = max(0.0, min(1.0, error_weight))

        user_focus = 0.0
        if self.conversation_state_tracker is not None:
            try:
                state = self.conversation_state_tracker.get_state_summary()
                depth = int(state.get("depth_level", 0) or 0)
                user_focus = min(1.0, depth / 5.0)
                self._cognitive_state.attention_focus = str(
                    state.get("conversation_topic") or state.get("user_goal") or ""
                )
            except Exception:
                user_focus = 0.0

        complexity = min(1.0, max(0.0, (goal_relevance + user_focus) / 2.0))
        score = (
            (0.30 * goal_relevance)
            + (0.20 * recency)
            + (0.30 * error_weight)
            + (0.10 * user_focus)
            + (0.10 * complexity)
        )
        return max(0.0, min(1.0, score))

    def _update_priorities(self, now_ts: float) -> Dict[str, Any]:
        score = self._score_attention_importance(now_ts)
        with self._lock:
            self._cognitive_state.last_importance_score = score
            if self._improvement_queue:
                self._cognitive_state.current_priority = self._improvement_queue[
                    0
                ].description
            elif self._cognitive_state.active_goal:
                self._cognitive_state.current_priority = (
                    self._cognitive_state.active_goal
                )
        return {"importance_score": score}

    def _schedule_improvements_if_needed(self, now_ts: float) -> Dict[str, Any]:
        self.create_self_improvement_plan()
        self._reprioritize_improvement_queue()
        with self._lock:
            top_targets = [t.description for t in self._improvement_queue[:3]]
            self._cognitive_state.improvement_targets = top_targets
            queue_size = len(self._improvement_queue)
        return {"improvements_enqueued": queue_size, "top_targets": top_targets}

    def _evaluate_controlled_reasoning_triggers(self, now_ts: float) -> Dict[str, Any]:
        with self._lock:
            importance = float(self._cognitive_state.last_importance_score)
            recent_problem = self._cognitive_state.recent_problem
            last_trigger = float(self._cognitive_state.reasoning_triggered_at)
            active_goal = self._cognitive_state.active_goal
            queue_len = len(self._improvement_queue)

        repeated_failure = bool(recent_problem) and any(
            rec.frequency >= 3 for rec in self._failure_log.values()
        )
        goal_blocked = bool(active_goal) and queue_len >= 2 and importance >= 0.55
        major_context_shift = False
        if self.conversation_state_tracker is not None:
            try:
                state = self.conversation_state_tracker.get_state_summary()
                major_context_shift = (
                    len(state.get("intent_chain", [])[-3:]) >= 3
                    and len(set(state.get("intent_chain", [])[-3:])) >= 3
                )
            except Exception:
                major_context_shift = False

        cooldown_passed = (now_ts - last_trigger) >= 45.0
        triggers = {
            "high_importance_score": importance >= self.reasoning_importance_threshold,
            "repeated_failure_pattern": repeated_failure,
            "goal_blockage_detected": goal_blocked,
            "major_context_shift": major_context_shift,
        }
        triggered = cooldown_passed and any(triggers.values())

        if triggered:
            with self._lock:
                self._cognitive_state.reasoning_triggered_at = now_ts
            self.event_bus.emit_custom(
                "cognition.reasoning_trigger",
                {
                    "importance": importance,
                    "triggers": triggers,
                    "active_goal": active_goal,
                    "recent_problem": recent_problem,
                },
                priority=EventPriority.HIGH,
            )
            self._integrate_with_runtime_systems(triggers=triggers)

        return {"triggered": triggered, "triggers": triggers}

    def _integrate_with_runtime_systems(self, *, triggers: Dict[str, bool]) -> None:
        """Push lightweight updates into runtime subsystems without calling the LLM."""
        if self.executive_controller is not None and triggers.get(
            "repeated_failure_pattern"
        ):
            try:
                self.executive_controller.apply_reflection(
                    {
                        "routing_adjustments": {
                            "clarify": 0.05,
                            "llm": -0.03,
                            "tools": 0.03,
                        }
                    }
                )
            except Exception:
                pass

        if self.response_planner is not None and triggers.get("high_importance_score"):
            self.event_bus.emit_custom(
                "cognition.planner_hint",
                {
                    "hint": "increase_structure_depth",
                    "reason": "high_importance_score",
                },
                priority=EventPriority.NORMAL,
            )

        if self.conversation_state_tracker is not None and triggers.get(
            "major_context_shift"
        ):
            self.event_bus.emit_custom(
                "cognition.conversation_shift",
                {
                    "hint": "reconfirm_goal_and_scope",
                },
                priority=EventPriority.NORMAL,
            )

    def _review_long_horizon_goals(self, now_ts: float) -> List[Dict[str, Any]]:
        stale: List[Dict[str, Any]] = []
        with self._lock:
            goals = list(self._project_goals.values())

        for goal in goals:
            stale_after_seconds = min(72.0, float(goal.horizon_days)) * 3600.0
            age = now_ts - float(goal.last_updated)
            if (
                goal.progress < 0.2
                and age > max(3600.0, stale_after_seconds * 0.5)
                and goal.state not in (GOAL_COMPLETED, GOAL_ABANDONED)
            ):
                goal.state = GOAL_BLOCKED
            if age > stale_after_seconds and goal.progress < 1.0:
                goal.state = (
                    GOAL_ABANDONED if age > (stale_after_seconds * 2.0) else goal.state
                )
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
                    milestones=[
                        sg.get("desc", "")
                        for sg in status.get("subgoals", [])
                        if sg.get("desc")
                    ],
                    state=GOAL_ACTIVE,
                    priority_score=0.7,
                    progress=float(status.get("progress_score", 0.0) or 0.0),
                )
            else:
                goal.description = goal_desc
                goal.progress = float(
                    status.get("progress_score", goal.progress) or goal.progress
                )
                goal.state = (
                    GOAL_COMPLETED if goal.progress >= 1.0 else GOAL_PROGRESSING
                )
                goal.last_updated = time.time()

    def _update_goal_lifecycle_on_turn(self, *, user_input: str, intent: str) -> None:
        turn_ref = f"{datetime.now().isoformat()}::{intent}"
        with self._lock:
            for goal in self._project_goals.values():
                if goal.state in (GOAL_COMPLETED, GOAL_ABANDONED):
                    continue
                text = f"{user_input} {intent}".lower()
                overlap = self._goal_relevance_overlap(goal.description.lower(), text)
                drift_turns = int(goal.metadata.get("drift_turns", 0) or 0)
                stagnation_turns = int(goal.metadata.get("stagnation_turns", 0) or 0)
                prev_progress = float(goal.progress)
                if overlap > 0.18:
                    goal.state = (
                        GOAL_PROGRESSING if goal.progress > 0.0 else GOAL_ACTIVE
                    )
                    goal.priority_score = min(
                        1.0, max(goal.priority_score, 0.55 + overlap)
                    )
                    goal.last_updated = time.time()
                    goal.related_conversations.append(turn_ref)
                    goal.metadata["drift_turns"] = 0
                    if goal.progress <= prev_progress + 1e-6:
                        stagnation_turns += 1
                    else:
                        stagnation_turns = 0
                    goal.metadata["stagnation_turns"] = stagnation_turns
                    if goal.progress < 1.0 and stagnation_turns >= 3:
                        goal.state = GOAL_BLOCKED
                elif overlap < 0.08 and goal.state in (
                    GOAL_ACTIVE,
                    GOAL_PROGRESSING,
                    GOAL_BLOCKED,
                ):
                    drift_turns += 1
                    goal.metadata["drift_turns"] = drift_turns
                    if drift_turns >= 2:
                        goal.state = GOAL_DRIFTED

    def get_runtime_guidance(self) -> Dict[str, Any]:
        """Return compact between-turn guidance derived from cognition metrics."""
        with self._lock:
            importance = float(self._cognitive_state.last_importance_score)
            active_goal = str(self._cognitive_state.active_goal or "")
            recent_problem = str(self._cognitive_state.recent_problem or "")
            queue_len = len(self._improvement_queue)

        route_bias = "balanced"
        if recent_problem in ("routing_mistake", "topic_drift"):
            route_bias = "clarify_first"
        elif active_goal and importance >= 0.65:
            route_bias = "goal_first"

        tool_budget = 1
        if route_bias == "clarify_first":
            tool_budget = 0
        elif importance >= 0.68 or queue_len >= 3:
            tool_budget = 2

        thinking_depth = 1
        if importance >= 0.50:
            thinking_depth += 1
        if active_goal:
            thinking_depth += 1

        planner_hint = "increase_structure_depth" if thinking_depth >= 3 else "default"

        return {
            "route_bias": route_bias,
            "tool_budget": int(max(0, min(3, tool_budget))),
            "thinking_depth": int(max(1, min(4, thinking_depth))),
            "planner_hint": planner_hint,
        }

    def _goal_relevance_overlap(self, goal_text: str, utterance_text: str) -> float:
        goal_terms = {t for t in goal_text.split() if len(t) > 2}
        utt_terms = {t for t in utterance_text.split() if len(t) > 2}
        if not goal_terms:
            return 0.0
        return len(goal_terms.intersection(utt_terms)) / float(len(goal_terms))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_cognitive_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            goals = [goal.as_dict() for goal in self._project_goals.values()]
            failures = [record.as_dict() for record in self._failure_log.values()]
            queue = [task.as_dict() for task in self._improvement_queue]
            working_state = self._cognitive_state.as_dict()
            latest_plan = list(self._latest_improvement_plan)
            cycle = dict(self._last_cycle_report)
            turn_count = len(self._turn_trace)

        return {
            "running": bool(self._running),
            "tick_interval_seconds": float(self.tick_interval_seconds),
            "reasoning_importance_threshold": float(
                self.reasoning_importance_threshold
            ),
            "tracked_turns": int(turn_count),
            "project_goals": goals,
            "failure_log": failures,
            "improvement_queue": queue,
            "cognitive_state": working_state,
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
    executive_controller: Optional[Any] = None,
    conversation_state_tracker: Optional[Any] = None,
    response_planner: Optional[Any] = None,
    tick_interval_seconds: float = 5.0,
    reasoning_importance_threshold: float = 0.74,
) -> CognitiveOrchestrator:
    global _cognitive_orchestrator
    if _cognitive_orchestrator is None:
        _cognitive_orchestrator = CognitiveOrchestrator(
            event_bus=event_bus,
            goal_tracker=goal_tracker,
            reflection_engine=reflection_engine,
            response_quality_tracker=response_quality_tracker,
            executive_controller=executive_controller,
            conversation_state_tracker=conversation_state_tracker,
            response_planner=response_planner,
            tick_interval_seconds=tick_interval_seconds,
            reasoning_importance_threshold=reasoning_importance_threshold,
        )
    return _cognitive_orchestrator
