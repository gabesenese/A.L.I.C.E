"""
Executive control layer for high-level action decisions.

This module keeps a structured internal reasoning state and outputs a compact
decision signal for routing behavior. It is intentionally not chain-of-thought.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, List


class TaskType(str, Enum):
    CONVERSATION = "conversation"
    DIRECT_TOOL_ACTION = "direct tool action"
    MULTI_STEP_TASK = "multi-step task"
    AUTONOMOUS_FOLLOW_THROUGH = "autonomous follow-through"
    CLARIFICATION_REQUIRED = "clarification-required"
    BLOCKED_ESCALATED = "blocked/escalated"


class RouteChoice(str, Enum):
    TOOL = "tool"
    LLM = "llm"
    CLARIFY = "clarify"
    BLOCKED = "blocked"


class TerminalAction(str, Enum):
    PROCEED = "proceed"
    CLARIFY = "clarify"
    BLOCKED = "blocked"


class NextActionType(str, Enum):
    RESPOND = "respond"
    EXECUTE_TOOL = "execute_tool"
    CONTINUE_GOAL = "continue_goal"
    ASK_CLARIFICATION = "ask_clarification"
    RETRY_TOOL = "retry_tool"
    REPLAN = "replan"
    ESCALATE = "escalate"


class NextActionOwner(str, Enum):
    EXECUTIVE = "executive"
    ACTION_ENGINE = "action_engine"
    VERIFIER = "verifier"
    RESPONSE = "response_layer"


class PostExecutionPhase(str, Enum):
    EXECUTED = "executed"
    VERIFIED = "verified"
    RETRY = "retry"
    REPLANNED = "replanned"
    COMPLETED = "completed"
    ESCALATED = "escalated"


@dataclass
class ReasoningState:
    user_intent: str
    source_text: str
    topic: str
    confidence: float
    intent_plausibility: float
    conversation_goal: str
    user_goal: str
    depth_level: int
    goal_status: str = ""
    goal_next_action: str = ""
    goal_blockers: List[str] = field(default_factory=list)
    goal_active_count: int = 0
    planner_hint: str = ""
    planner_depth: int = 1
    route_bias: str = "balanced"
    tool_budget: int = 1
    pending_followup_slot: bool = False
    pending_followup_slot_name: str = ""
    pending_followup_slot_state: Dict[str, Any] = field(default_factory=dict)
    plan: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "user_intent": self.user_intent,
            "source_text": self.source_text,
            "topic": self.topic,
            "confidence": self.confidence,
            "intent_plausibility": self.intent_plausibility,
            "conversation_goal": self.conversation_goal,
            "user_goal": self.user_goal,
            "goal_status": self.goal_status,
            "goal_next_action": self.goal_next_action,
            "goal_blockers": list(self.goal_blockers),
            "goal_active_count": self.goal_active_count,
            "depth_level": self.depth_level,
            "planner_hint": self.planner_hint,
            "planner_depth": self.planner_depth,
            "route_bias": self.route_bias,
            "tool_budget": self.tool_budget,
            "pending_followup_slot": self.pending_followup_slot,
            "pending_followup_slot_name": self.pending_followup_slot_name,
            "pending_followup_slot_state": dict(self.pending_followup_slot_state or {}),
            "plan": list(self.plan),
        }


@dataclass
class ExecutiveDecision:
    action: (
        str  # use_plugin | use_llm | ask_clarification | ignore | answer_direct | defer
    )
    reason: str
    store_memory: bool = True
    clarification_question: str = ""


@dataclass
class TurnContinuation:
    """Machine-usable continuation payload derived from next_action."""

    action_type: str = NextActionType.RESPOND.value
    payload: Dict[str, Any] = field(default_factory=dict)
    retry_target: str = ""
    blocking_reason: str = ""
    owner: str = NextActionOwner.EXECUTIVE.value

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action_type": str(self.action_type or NextActionType.RESPOND.value),
            "payload": dict(self.payload or {}),
            "retry_target": str(self.retry_target or ""),
            "blocking_reason": str(self.blocking_reason or ""),
            "owner": str(self.owner or NextActionOwner.EXECUTIVE.value),
        }


@dataclass
class TurnExecutionContract:
    """Canonical per-turn executive contract shared across runtime layers."""

    task_type: str
    goal: str
    constraints: List[str] = field(default_factory=list)
    chosen_route: str = ""
    success_criteria: List[str] = field(default_factory=list)
    next_action: str = ""
    continuation: TurnContinuation = field(default_factory=TurnContinuation)

    def as_dict(self) -> Dict[str, Any]:
        continuation = self.continuation.as_dict() if self.continuation else {}
        return {
            "task_type": str(self.task_type or TaskType.CONVERSATION.value),
            "goal": str(self.goal or ""),
            "constraints": [
                str(x) for x in list(self.constraints or []) if str(x).strip()
            ],
            "chosen_route": str(self.chosen_route or RouteChoice.LLM.value),
            "success_criteria": [
                str(x) for x in list(self.success_criteria or []) if str(x).strip()
            ],
            "next_action": str(self.next_action or ""),
            "next_action_type": str(
                continuation.get("action_type") or NextActionType.RESPOND.value
            ),
            "continuation_payload": dict(continuation.get("payload") or {}),
            "retry_target": str(continuation.get("retry_target") or ""),
            "blocking_reason": str(continuation.get("blocking_reason") or ""),
            "next_action_owner": str(
                continuation.get("owner") or NextActionOwner.EXECUTIVE.value
            ),
            "continuation": continuation,
        }


@dataclass
class TurnStateMachineResult:
    """Single executive state-machine output for one turn."""

    state: str
    chosen_route: str
    should_try_plugins: bool
    terminal_action: str
    contract: TurnExecutionContract

    def as_dict(self) -> Dict[str, Any]:
        return {
            "state": str(self.state or TaskType.CONVERSATION.value),
            "chosen_route": str(self.chosen_route or RouteChoice.LLM.value),
            "should_try_plugins": bool(self.should_try_plugins),
            "terminal_action": str(
                self.terminal_action or TerminalAction.PROCEED.value
            ),
            "contract": self.contract.as_dict() if self.contract else {},
        }


@dataclass
class TurnExecutionOutcome:
    """Post-execution outcome separates tool success from goal progress."""

    tool_success: bool
    goal_advanced: bool
    verification_passed: bool
    needs_retry: bool
    needs_escalation: bool
    recommended_next_action: str = ""
    verification_confidence: float = 0.0
    issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tool_success": bool(self.tool_success),
            "goal_advanced": bool(self.goal_advanced),
            "verification_passed": bool(self.verification_passed),
            "needs_retry": bool(self.needs_retry),
            "needs_escalation": bool(self.needs_escalation),
            "recommended_next_action": str(self.recommended_next_action or ""),
            "verification_confidence": max(
                0.0, min(1.0, float(self.verification_confidence or 0.0))
            ),
            "issues": [str(x) for x in list(self.issues or []) if str(x).strip()],
            "metadata": dict(self.metadata or {}),
        }


@dataclass
class PostExecutionStateMachineResult:
    """Post-execution state machine output for execute/verify/retry loops."""

    phase: str
    terminal_action: str
    should_retry: bool
    should_replan: bool
    contract: TurnExecutionContract
    outcome: TurnExecutionOutcome

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phase": str(self.phase or PostExecutionPhase.EXECUTED.value),
            "terminal_action": str(
                self.terminal_action or TerminalAction.PROCEED.value
            ),
            "should_retry": bool(self.should_retry),
            "should_replan": bool(self.should_replan),
            "contract": self.contract.as_dict() if self.contract else {},
            "outcome": self.outcome.as_dict() if self.outcome else {},
        }


class ExecutiveController:
    """Produces high-level decisions from compact state and runtime hints."""

    TOOL_DOMAINS = (
        "notes:",
        "email:",
        "calendar:",
        "file_operations:",
        "memory:",
        "reminder:",
        "system:",
        "weather:",
        "time:",
    )
    SIMPLE_SCAFFOLD_INTENTS = {
        "conversation:ack",
        "conversation:acknowledgment",
        "thanks",
        "greeting",
        "status_inquiry",
        "farewell",
    }
    SIMPLE_NATIVE_RE = re.compile(
        r"^(?:"
        r"hi|hello|hey|yo|sup|"
        r"thanks|thank\s+you|thx|"
        r"bye|goodbye|see\s+you|"
        r"how\s+are\s+you|how\s+are\s+you\s+doing|hows\s+it\s+going|"
        r"ok|okay|sure|got\s+it|understood|noted"
        r")\??$",
        re.IGNORECASE,
    )
    RISKY_CONVERSATION_RE = re.compile(
        r"\b(medical|diagnos|legal|lawsuit|financial|investment|tax|security|exploit|malware|password)\b",
        re.IGNORECASE,
    )
    SEMANTIC_STOPWORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "of",
        "for",
        "with",
        "in",
        "on",
        "at",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "we",
        "they",
        "he",
        "she",
        "them",
        "my",
        "your",
        "our",
        "their",
        "me",
        "can",
        "could",
        "would",
        "should",
        "want",
        "need",
        "help",
        "today",
        "world",
        "real",
        "learn",
        "about",
        "explain",
        "tell",
        "please",
    }

    def __init__(self) -> None:
        # Adaptive routing matrix (updated by reflection loop).
        self._routing_weights: Dict[str, float] = {
            "tools": 1.0,
            "memory": 1.0,
            "rag": 1.0,
            "llm": 1.0,
            "clarify": 1.0,
            "search": 1.0,
            "reject": 1.0,
            "defer": 1.0,
        }

    def build_state(
        self,
        user_input: str,
        intent: str,
        confidence: float,
        entities: Dict[str, Any],
        conversation_state: Dict[str, Any],
    ) -> ReasoningState:
        entities = entities or {}
        conversation_state = conversation_state or {}

        hidden_snapshot = dict(
            conversation_state.get("hidden_situation_snapshot") or {}
        )
        hidden_goal_state = dict(hidden_snapshot.get("goal_state") or {})
        priority_goal = dict(hidden_goal_state.get("priority_goal") or {})
        current_goal = dict(
            conversation_state.get("current_goal") or priority_goal or {}
        )

        raw_goal_stack: List[Dict[str, Any]] = []
        for candidate in (
            [current_goal]
            + list(conversation_state.get("active_goal_stack") or [])
            + list(conversation_state.get("active_goals") or [])
            + list(hidden_goal_state.get("active_goal_stack") or [])
        ):
            if isinstance(candidate, dict):
                raw_goal_stack.append(dict(candidate))

        if raw_goal_stack:
            deduped_goal_stack: List[Dict[str, Any]] = []
            seen = set()
            for goal in raw_goal_stack:
                goal_key = str(goal.get("goal_id") or goal.get("title") or "").strip()
                if goal_key and goal_key in seen:
                    continue
                if goal_key:
                    seen.add(goal_key)
                deduped_goal_stack.append(goal)
            priority_goal = dict(deduped_goal_stack[0] or {})
            current_goal = dict(priority_goal)

        goal_status = (
            str(
                current_goal.get("status")
                or priority_goal.get("status")
                or hidden_goal_state.get("status")
                or ""
            )
            .strip()
            .lower()
        )
        goal_next_action = str(
            current_goal.get("next_action")
            or priority_goal.get("next_action")
            or hidden_goal_state.get("next_action")
            or ""
        ).strip()

        goal_blockers: List[str] = []
        raw_blockers = (
            current_goal.get("blockers")
            or priority_goal.get("blockers")
            or hidden_goal_state.get("blockers")
            or []
        )
        if isinstance(raw_blockers, list):
            for blocker in raw_blockers:
                blocker_text = str(blocker or "").strip()
                if blocker_text and blocker_text not in goal_blockers:
                    goal_blockers.append(blocker_text)
        else:
            blocker_text = str(raw_blockers or "").strip()
            if blocker_text:
                goal_blockers.append(blocker_text)

        goal_active_count = int(hidden_goal_state.get("active_goal_count") or 0)
        if goal_active_count <= 0:
            goal_active_count = (
                len(raw_goal_stack) if raw_goal_stack else (1 if priority_goal else 0)
            )

        topic = str(
            entities.get("topic")
            or conversation_state.get("conversation_topic")
            or priority_goal.get("title")
            or ""
        ).strip()

        conversation_goal = str(
            conversation_state.get("conversation_goal")
            or hidden_goal_state.get("status")
            or ""
        ).strip()
        user_goal = str(
            conversation_state.get("user_goal")
            or entities.get("goal")
            or priority_goal.get("title")
            or ""
        ).strip()
        depth_level = int(conversation_state.get("depth_level") or 0)
        intent_plausibility = max(
            0.0,
            min(1.0, float(entities.get("_intent_plausibility", 1.0) or 1.0)),
        )
        planner_hint = str(conversation_state.get("planner_hint") or "").strip().lower()
        planner_depth = int(conversation_state.get("planner_depth") or 1)
        route_bias = (
            str(conversation_state.get("route_bias") or "balanced").strip().lower()
        )
        tool_budget = int(conversation_state.get("tool_budget") or 1)
        pending_followup_slot = bool(
            conversation_state.get("pending_followup_slot", False)
        )
        pending_followup_slot_name = str(
            conversation_state.get("pending_followup_slot_name") or ""
        ).strip()
        pending_followup_slot_state = dict(
            conversation_state.get("pending_followup_slot_state") or {}
        )

        plan = self._derive_plan(
            intent=intent, topic=topic, depth_level=depth_level, user_input=user_input
        )

        return ReasoningState(
            user_intent=intent or "unknown",
            source_text=str(user_input or ""),
            topic=topic,
            confidence=max(0.0, min(1.0, float(confidence or 0.0))),
            intent_plausibility=intent_plausibility,
            conversation_goal=conversation_goal,
            user_goal=user_goal,
            goal_status=goal_status,
            goal_next_action=goal_next_action,
            goal_blockers=goal_blockers,
            goal_active_count=max(0, goal_active_count),
            depth_level=depth_level,
            planner_hint=planner_hint,
            planner_depth=max(1, min(4, planner_depth)),
            route_bias=route_bias or "balanced",
            tool_budget=max(0, min(3, tool_budget)),
            pending_followup_slot=pending_followup_slot,
            pending_followup_slot_name=pending_followup_slot_name,
            pending_followup_slot_state=pending_followup_slot_state,
            plan=plan,
        )

    def build_turn_contract(
        self,
        *,
        state: ReasoningState,
        decision: ExecutiveDecision,
        should_try_plugins: bool,
        has_explicit_action_cue: bool,
        has_active_goal: bool,
        pre_route_blocked: bool = False,
        tool_vetoed: bool = False,
    ) -> TurnExecutionContract:
        """Build the single canonical per-turn contract object."""

        if pre_route_blocked or decision.action == "ask_clarification":
            task_type = TaskType.CLARIFICATION_REQUIRED.value
            chosen_route = RouteChoice.CLARIFY.value
        elif tool_vetoed or decision.action in {"defer", "ignore"}:
            task_type = TaskType.BLOCKED_ESCALATED.value
            chosen_route = RouteChoice.BLOCKED.value
        elif should_try_plugins and has_active_goal and not has_explicit_action_cue:
            task_type = TaskType.AUTONOMOUS_FOLLOW_THROUGH.value
            chosen_route = RouteChoice.TOOL.value
        elif should_try_plugins and has_explicit_action_cue and not has_active_goal:
            task_type = TaskType.DIRECT_TOOL_ACTION.value
            chosen_route = RouteChoice.TOOL.value
        elif should_try_plugins:
            task_type = TaskType.MULTI_STEP_TASK.value
            chosen_route = RouteChoice.TOOL.value
        elif has_active_goal:
            task_type = TaskType.MULTI_STEP_TASK.value
            chosen_route = RouteChoice.LLM.value
        else:
            task_type = TaskType.CONVERSATION.value
            chosen_route = RouteChoice.LLM.value

        goal = str(state.user_goal or state.topic or state.source_text).strip()[:200]
        constraints: List[str] = [
            str(x).strip() for x in list(state.goal_blockers or []) if str(x).strip()
        ]
        if state.route_bias and state.route_bias != "balanced":
            constraints.append(f"route_bias:{state.route_bias}")
        if int(state.tool_budget or 1) <= 0:
            constraints.append("tool_budget:0")
        if state.pending_followup_slot:
            constraints.append("pending_followup_slot")

        if task_type == TaskType.CLARIFICATION_REQUIRED.value:
            success_criteria = [
                "missing detail captured",
                "next route can be decided confidently",
            ]
            next_action = (
                decision.clarification_question.strip()
                or "ask one targeted clarifying question"
            )
        elif task_type == TaskType.BLOCKED_ESCALATED.value:
            success_criteria = [
                "blocker identified",
                "escalation path selected",
            ]
            next_action = (
                str(state.goal_next_action or "").strip()
                or "request operator/user intervention"
            )
        elif task_type == TaskType.DIRECT_TOOL_ACTION.value:
            success_criteria = [
                "tool call succeeds",
                "result returned to user clearly",
            ]
            next_action = (
                str(state.goal_next_action or "").strip()
                or "execute requested tool and verify outcome"
            )
        elif task_type == TaskType.AUTONOMOUS_FOLLOW_THROUGH.value:
            success_criteria = [
                "active goal advances",
                "state updated for continuity",
            ]
            next_action = (
                str(state.goal_next_action or "").strip()
                or "continue the active goal without extra prompting"
            )
        elif task_type == TaskType.MULTI_STEP_TASK.value:
            success_criteria = [
                "step outcome validated",
                "next step remains actionable",
            ]
            next_action = (
                str(state.goal_next_action or "").strip()
                or "continue planned execution and verify progress"
            )
        else:
            success_criteria = [
                "answer is directly relevant",
                "response remains concise and actionable",
            ]
            next_action = "respond directly and wait for user follow-up"

        continuation = self._build_continuation(
            task_type=task_type,
            decision=decision,
            next_action=next_action,
            chosen_route=chosen_route,
            tool_vetoed=tool_vetoed,
        )

        return TurnExecutionContract(
            task_type=task_type,
            goal=goal,
            constraints=constraints,
            chosen_route=chosen_route,
            success_criteria=success_criteria,
            next_action=next_action,
            continuation=continuation,
        )

    def run_turn_state_machine(
        self,
        *,
        state: ReasoningState,
        decision: ExecutiveDecision,
        has_explicit_action_cue: bool,
        has_active_goal: bool,
        pre_route_blocked: bool = False,
        tool_vetoed: bool = False,
    ) -> TurnStateMachineResult:
        """Single-owner state machine for per-turn runtime control."""

        if pre_route_blocked:
            chosen_route = RouteChoice.CLARIFY.value
            should_try_plugins = False
            terminal_action = TerminalAction.CLARIFY.value
        elif tool_vetoed or decision.action in {"defer", "ignore"}:
            chosen_route = RouteChoice.BLOCKED.value
            should_try_plugins = False
            terminal_action = TerminalAction.BLOCKED.value
        elif decision.action == "ask_clarification":
            chosen_route = RouteChoice.CLARIFY.value
            should_try_plugins = False
            terminal_action = TerminalAction.CLARIFY.value
        elif decision.action == "use_plugin":
            chosen_route = RouteChoice.TOOL.value
            should_try_plugins = True
            terminal_action = TerminalAction.PROCEED.value
        elif decision.action in {"use_llm", "answer_direct"}:
            chosen_route = RouteChoice.LLM.value
            should_try_plugins = False
            terminal_action = TerminalAction.PROCEED.value
        else:
            # Unknown actions are treated as blocked for safety.
            chosen_route = RouteChoice.BLOCKED.value
            should_try_plugins = False
            terminal_action = TerminalAction.BLOCKED.value

        contract = self.build_turn_contract(
            state=state,
            decision=decision,
            should_try_plugins=should_try_plugins,
            has_explicit_action_cue=has_explicit_action_cue,
            has_active_goal=has_active_goal,
            pre_route_blocked=pre_route_blocked,
            tool_vetoed=tool_vetoed,
        )

        return TurnStateMachineResult(
            state=contract.task_type,
            chosen_route=chosen_route,
            should_try_plugins=should_try_plugins,
            terminal_action=terminal_action,
            contract=contract,
        )

    def build_execution_outcome(
        self,
        *,
        contract: TurnExecutionContract,
        tool_success: bool,
        goal_advanced: bool,
        verification_passed: bool,
        recommended_next_action: str = "",
        retryable: bool = False,
        issues: List[str] | None = None,
        verification_confidence: float = 0.0,
        metadata: Dict[str, Any] | None = None,
    ) -> TurnExecutionOutcome:
        """Create explicit execution outcome fields for downstream control."""

        next_action = str(recommended_next_action or "").strip().lower()
        needs_retry = bool(next_action in {"retry", "clarify_then_continue"})
        if not needs_retry and retryable and not verification_passed:
            needs_retry = True

        needs_escalation = bool(next_action in {"escalate", "escalate_and_stop"})
        if (
            not needs_escalation
            and not needs_retry
            and contract.chosen_route == RouteChoice.TOOL.value
            and not verification_passed
        ):
            needs_escalation = True

        return TurnExecutionOutcome(
            tool_success=bool(tool_success),
            goal_advanced=bool(goal_advanced),
            verification_passed=bool(verification_passed),
            needs_retry=bool(needs_retry),
            needs_escalation=bool(needs_escalation),
            recommended_next_action=next_action,
            verification_confidence=max(
                0.0, min(1.0, float(verification_confidence or 0.0))
            ),
            issues=[str(x) for x in list(issues or []) if str(x).strip()],
            metadata=dict(metadata or {}),
        )

    def run_post_execution_state_machine(
        self,
        *,
        pre_execution: TurnStateMachineResult,
        outcome: TurnExecutionOutcome,
    ) -> PostExecutionStateMachineResult:
        """Transition from route selection to execution outcome lifecycle."""

        phase = PostExecutionPhase.EXECUTED.value
        terminal_action = TerminalAction.PROCEED.value
        should_retry = False
        should_replan = False
        contract = pre_execution.contract

        if outcome.needs_escalation:
            phase = PostExecutionPhase.ESCALATED.value
            terminal_action = TerminalAction.BLOCKED.value
            contract = self._contract_with_continuation(
                contract,
                action_type=NextActionType.ESCALATE.value,
                owner=NextActionOwner.EXECUTIVE.value,
                blocking_reason=(
                    ", ".join(outcome.issues)
                    if outcome.issues
                    else "verification_failure"
                ),
                next_action="request escalation or operator intervention",
            )
        elif outcome.needs_retry:
            phase = PostExecutionPhase.RETRY.value
            should_retry = True
            contract = self._contract_with_continuation(
                contract,
                action_type=NextActionType.RETRY_TOOL.value,
                owner=NextActionOwner.ACTION_ENGINE.value,
                retry_target=str(outcome.metadata.get("plugin") or ""),
                next_action="retry the failed execution step",
            )
        elif outcome.verification_passed:
            phase = PostExecutionPhase.VERIFIED.value
            if (
                pre_execution.chosen_route == RouteChoice.TOOL.value
                and outcome.goal_advanced
            ):
                phase = PostExecutionPhase.COMPLETED.value
                contract = self._contract_with_continuation(
                    contract,
                    action_type=NextActionType.CONTINUE_GOAL.value,
                    owner=NextActionOwner.EXECUTIVE.value,
                    next_action="goal advanced; continue with the next planned step",
                )
            elif (
                pre_execution.chosen_route == RouteChoice.TOOL.value
                and not outcome.goal_advanced
            ):
                phase = PostExecutionPhase.REPLANNED.value
                should_replan = True
                contract = self._contract_with_continuation(
                    contract,
                    action_type=NextActionType.REPLAN.value,
                    owner=NextActionOwner.EXECUTIVE.value,
                    next_action="replan because tool succeeded without advancing the goal",
                )
            else:
                contract = self._contract_with_continuation(
                    contract,
                    action_type=NextActionType.RESPOND.value,
                    owner=NextActionOwner.RESPONSE.value,
                    next_action="summarize the verified outcome to the user",
                )
        else:
            phase = PostExecutionPhase.REPLANNED.value
            should_replan = True
            contract = self._contract_with_continuation(
                contract,
                action_type=NextActionType.REPLAN.value,
                owner=NextActionOwner.EXECUTIVE.value,
                blocking_reason=(
                    ", ".join(outcome.issues)
                    if outcome.issues
                    else "unverified_outcome"
                ),
                next_action="replan due to unverified execution",
            )

        return PostExecutionStateMachineResult(
            phase=phase,
            terminal_action=terminal_action,
            should_retry=bool(should_retry),
            should_replan=bool(should_replan),
            contract=contract,
            outcome=outcome,
        )

    def _build_continuation(
        self,
        *,
        task_type: str,
        decision: ExecutiveDecision,
        next_action: str,
        chosen_route: str,
        tool_vetoed: bool,
    ) -> TurnContinuation:
        if task_type == TaskType.CLARIFICATION_REQUIRED.value:
            return TurnContinuation(
                action_type=NextActionType.ASK_CLARIFICATION.value,
                payload={"question": str(decision.clarification_question or "")},
                owner=NextActionOwner.EXECUTIVE.value,
            )
        if task_type == TaskType.BLOCKED_ESCALATED.value:
            return TurnContinuation(
                action_type=NextActionType.ESCALATE.value,
                blocking_reason=("tool_vetoed" if tool_vetoed else "policy_block"),
                owner=NextActionOwner.EXECUTIVE.value,
            )
        if chosen_route == RouteChoice.TOOL.value:
            return TurnContinuation(
                action_type=NextActionType.EXECUTE_TOOL.value,
                payload={"task_type": task_type},
                owner=NextActionOwner.ACTION_ENGINE.value,
            )
        return TurnContinuation(
            action_type=NextActionType.RESPOND.value,
            payload={"task_type": task_type},
            owner=NextActionOwner.RESPONSE.value,
        )

    def _contract_with_continuation(
        self,
        contract: TurnExecutionContract,
        *,
        action_type: str,
        owner: str,
        next_action: str,
        retry_target: str = "",
        blocking_reason: str = "",
    ) -> TurnExecutionContract:
        continuation = TurnContinuation(
            action_type=action_type,
            payload={
                "task_type": contract.task_type,
                "chosen_route": contract.chosen_route,
            },
            retry_target=retry_target,
            blocking_reason=blocking_reason,
            owner=owner,
        )
        return TurnExecutionContract(
            task_type=contract.task_type,
            goal=contract.goal,
            constraints=list(contract.constraints or []),
            chosen_route=contract.chosen_route,
            success_criteria=list(contract.success_criteria or []),
            next_action=next_action,
            continuation=continuation,
        )

    def should_preempt_for_plausibility(
        self,
        state: ReasoningState,
        *,
        has_explicit_action_cue: bool,
        intent_candidates: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """Pre-routing plausibility gate to stop low-confidence tool trajectories early."""
        candidates = intent_candidates or []
        conf = max(0.0, min(1.0, float(state.confidence)))
        plausibility = max(0.0, min(1.0, float(state.intent_plausibility)))
        intent = (state.user_intent or "").lower().strip()

        if self._is_rich_conceptual_request(str(state.source_text or "")):
            return {"block": False, "reason": "rich_conceptual_prompt"}

        if intent.startswith("conversation:"):
            return {"block": False, "reason": "conversation_intent"}

        if plausibility < 0.38:
            return {
                "block": True,
                "reason": "pre_route_low_plausibility",
                "question": "I may be misclassifying this. What exact outcome do you want me to produce?",
            }

        if (not has_explicit_action_cue) and conf < 0.50 and plausibility < 0.60:
            return {
                "block": True,
                "reason": "pre_route_uncertain_without_action_cue",
                "question": "Before I route this, what exact outcome do you want me to perform?",
            }

        if len(candidates) > 1:
            top = float(candidates[0].get("score", 0.0))
            second = float(candidates[1].get("score", 0.0))
            if (
                (top - second) < 0.06
                and plausibility < 0.68
                and not has_explicit_action_cue
            ):
                return {
                    "block": True,
                    "reason": "pre_route_ambiguous_candidates",
                    "question": "I see multiple likely intents. What is the one concrete result you want right now?",
                }

        return {"block": False, "reason": "allowed"}

    def decide(
        self,
        state: ReasoningState,
        *,
        is_pure_conversation: bool,
        has_explicit_action_cue: bool,
        has_active_goal: bool,
        force_plugins_for_notes: bool,
    ) -> ExecutiveDecision:
        if len(state.user_intent.strip()) == 0:
            return ExecutiveDecision(
                action="ignore",
                reason="empty_intent",
                store_memory=False,
            )

        if self._is_pending_followup_slot_answer(
            state=state,
            has_explicit_action_cue=has_explicit_action_cue,
            has_active_goal=has_active_goal,
        ):
            return ExecutiveDecision(
                action="use_llm",
                reason="pending_followup_slot_answer",
                store_memory=True,
            )

        if self._is_simple_native_conversation(
            state=state,
            has_explicit_action_cue=has_explicit_action_cue,
            has_active_goal=has_active_goal,
        ):
            return ExecutiveDecision(
                action="answer_direct",
                reason="simple_conversational_native_path",
                store_memory=True,
            )

        if self._is_clear_informational_request(state):
            return ExecutiveDecision(
                action="answer_direct",
                reason="clear_informational_request",
                store_memory=True,
            )

        if (
            self._is_answerability_direct_question(state.source_text)
            and not has_explicit_action_cue
        ):
            return ExecutiveDecision(
                action="answer_direct",
                reason="answerability_gate_direct_question",
                store_memory=True,
            )

        normalized_intent = (state.user_intent or "").lower().strip()
        rich_conceptual = self._is_rich_conceptual_request(state.source_text)
        conceptual_build = self._is_conceptual_build_architecture_prompt(
            state.source_text
        )

        if conceptual_build:
            return ExecutiveDecision(
                action="use_llm",
                reason="conceptual_build_question",
                store_memory=True,
            )

        if rich_conceptual:
            return ExecutiveDecision(
                action="use_llm",
                reason="rich_conceptual_request",
                store_memory=True,
            )

        if (
            normalized_intent
            in {
                "conversation:help",
                "conversation:question",
                "conversation:general",
            }
            and self._is_short_framework_overview_prompt(state.source_text)
            and not has_explicit_action_cue
            and not has_active_goal
        ):
            return ExecutiveDecision(
                action="use_llm",
                reason="short_framework_overview",
                store_memory=True,
            )

        if normalized_intent == "conversation:goal_statement":
            return ExecutiveDecision(
                action="answer_direct",
                reason="goal_statement_alignment",
                store_memory=True,
            )

        if normalized_intent == "greeting":
            return ExecutiveDecision(
                action="answer_direct",
                reason="greeting_native_priority",
                store_memory=True,
            )

        if (
            normalized_intent == "conversation:clarification_needed"
            and state.confidence >= 0.45
        ):
            return ExecutiveDecision(
                action="use_llm",
                reason="clarification_answer_requested",
                store_memory=True,
            )

        scores = self.score_decisions(
            state,
            is_pure_conversation=is_pure_conversation,
            has_explicit_action_cue=has_explicit_action_cue,
            has_active_goal=has_active_goal,
            force_plugins_for_notes=force_plugins_for_notes,
        )
        axes = self._decision_axes(
            state=state,
            is_pure_conversation=is_pure_conversation,
            has_explicit_action_cue=has_explicit_action_cue,
            has_active_goal=has_active_goal,
            force_plugins_for_notes=force_plugins_for_notes,
        )

        # Minimal executive kernel (4 decisions):
        # ACT, ANSWER, ASK, WAIT/DEFER
        act_score = max(
            float(scores.get("tools", 0.0)), float(scores.get("search", 0.0))
        )
        answer_score = max(
            float(scores.get("llm", 0.0)),
            float(scores.get("memory", 0.0)),
            float(scores.get("rag", 0.0)),
        )
        ask_score = float(scores.get("clarify", 0.0))
        wait_score = max(
            float(scores.get("defer", 0.0)), float(scores.get("reject", 0.0))
        )

        if (
            normalized_intent.startswith("conversation:")
            and float(state.confidence or 0.0) < 0.35
            and not has_explicit_action_cue
            and not has_active_goal
        ):
            return ExecutiveDecision(
                action="ask_clarification",
                reason="kernel_low_confidence_conversation",
                store_memory=False,
                clarification_question="I want to help accurately. What exact result do you want?",
            )

        if (
            float(axes.get("target_confidence", 0.0)) < 0.38
            and float(axes.get("enough_information", 0.0)) < 0.45
            and float(axes.get("can_act_now", 0.0)) < 0.50
        ):
            return ExecutiveDecision(
                action="ask_clarification",
                reason="kernel_low_confidence_ask",
                store_memory=False,
                clarification_question="I want to help accurately. What exact result do you want?",
            )

        if ask_score >= max(act_score, answer_score, wait_score):
            return ExecutiveDecision(
                action="ask_clarification",
                reason="score_clarify",
                store_memory=False,
                clarification_question="I want to help accurately. What exact result do you want?",
            )
        if wait_score >= max(act_score, answer_score, 0.58):
            return ExecutiveDecision(
                action="defer",
                reason="score_defer",
                store_memory=False,
            )
        if act_score >= max(answer_score, ask_score, wait_score):
            winner = (
                "search"
                if float(scores.get("search", 0.0)) > float(scores.get("tools", 0.0))
                else "tools"
            )
            return ExecutiveDecision(
                action="use_plugin", reason=f"score_{winner}", store_memory=True
            )

        return ExecutiveDecision(
            action="use_llm", reason="score_llm", store_memory=True
        )

    def _is_help_opener(self, state: ReasoningState) -> bool:
        # Help-openers should not downgrade routing decisions.
        return False

    def _is_rich_conceptual_request(self, text: str) -> bool:
        """Detect conceptual architecture prompts that are answerable without clarification."""
        low = str(text or "").lower().strip()
        if not low:
            return False

        tokens = re.findall(r"\b[a-z0-9']+\b", low)
        if len(tokens) < 6:
            return False

        has_task = any(
            re.search(pattern, low)
            for pattern in (
                r"\blet'?s\s+imagine\b",
                r"\bhow\s+would\b",
                r"\bhow\s+can\s+i\s+create\b",
                r"\bhow\s+can\s+i\s+build\b",
                r"\bhow\s+would\s+i\s+build\b",
                r"\bhow\s+would\s+someone\s+build\b",
                r"\bwould\s+.+\s+be\s+built\b",
                r"\bhow\s+.+\s+be\s+created\b",
                r"\barchitecture\b",
            )
        )
        if not has_task:
            return False

        has_build_verb = bool(
            re.search(r"\b(create|created|build|built|make|made)\b", low)
        )

        has_constraint = any(
            cue in low
            for cue in (
                "no fiction",
                "non fiction",
                "non-fiction",
                "real world",
                "real-world",
                "today's technology",
                "todays technology",
            )
        )
        if not has_constraint:
            return False

        has_subject = any(
            cue in low
            for cue in (
                "assistant",
                "fictional inventor",
                "assistant",
                "technology",
                "system",
                "architecture",
                "foundations",
                "ai",
            )
        )
        if has_subject and has_constraint and has_build_verb:
            return True

        stop = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "for",
            "with",
            "in",
            "on",
            "how",
            "would",
            "no",
            "fiction",
            "today",
            "real",
            "world",
        }
        strong_terms = [t for t in tokens if len(t) >= 5 and t not in stop]
        return len(set(strong_terms)) >= 3

    def _is_conceptual_build_architecture_prompt(self, text: str) -> bool:
        """Detect build/create architecture questions for dedicated reasoning routes."""
        low = str(text or "").lower().strip()
        if not low:
            return False
        if not self._is_rich_conceptual_request(low):
            return False

        has_build = bool(re.search(r"\b(create|created|build|built|make|made)\b", low))
        has_subject = any(
            cue in low
            for cue in (
                "ai",
                "assistant",
                "assistant",
                "system",
                "architecture",
            )
        )
        has_question_frame = bool(
            re.search(
                r"\b(how\s+can\s+i|how\s+would\s+i|how\s+would\s+someone|how\s+would|let'?s\s+imagine)\b",
                low,
            )
        )
        return has_build and has_subject and has_question_frame

    def _is_short_framework_overview_prompt(self, text: str) -> bool:
        """Detect short conceptual framework prompts that should stay on conversational LLM path."""
        low = str(text or "").strip().lower()
        if not low:
            return False

        tokens = re.findall(r"\b[a-z0-9']+\b", low)
        if len(tokens) < 2 or len(tokens) > 7:
            return False

        action_cues = {
            "create",
            "build",
            "implement",
            "delete",
            "remove",
            "run",
            "open",
            "save",
            "search",
            "list",
            "set",
            "update",
            "install",
        }
        if any(tok in action_cues for tok in tokens):
            return False

        framework_cues = {
            "agentic",
            "agent",
            "agents",
            "ai",
            "framework",
            "frameworks",
            "orchestration",
            "autonomous",
            "autonomy",
            "llm",
            "rag",
            "architecture",
        }
        return bool(framework_cues.intersection(tokens))

    def _is_pending_followup_slot_answer(
        self,
        *,
        state: ReasoningState,
        has_explicit_action_cue: bool,
        has_active_goal: bool,
    ) -> bool:
        if not bool(getattr(state, "pending_followup_slot", False)):
            return False
        if has_explicit_action_cue or has_active_goal:
            return False

        slot_state = dict(getattr(state, "pending_followup_slot_state", {}) or {})
        expected_shape = (
            str(slot_state.get("expected_answer_shape") or "").strip().lower()
        )

        text = str(state.source_text or "").strip().lower()
        if not text or "?" in text:
            return False

        tokens = re.findall(r"\b[a-z0-9']+\b", text)
        if not tokens or len(tokens) > 5:
            return False

        social_words = {
            "hi",
            "hello",
            "hey",
            "thanks",
            "thank",
            "bye",
            "goodbye",
            "ok",
            "okay",
            "sure",
        }
        if all(tok in social_words for tok in tokens):
            return False

        if expected_shape in {
            "short_topic_or_subdomain",
            "short_disambiguation_or_selection",
            "ordinal_or_short_phrase",
        }:
            if re.search(r"\b(first|second|third|1st|2nd|3rd|one|two|three)\b", text):
                return True
            _slot_tokens = re.findall(r"\b[a-z0-9']+\b", text)
            if 1 <= len(_slot_tokens) <= 5:
                return True

        if expected_shape == "single_token" and len(tokens) == 1:
            return True

        return True

    def _is_clear_informational_request(self, state: ReasoningState) -> bool:
        text = str(state.source_text or "").strip().lower()
        if not text:
            return False

        has_ask = any(
            cue in text
            for cue in (
                "explain",
                "what is",
                "what's",
                "how does",
                "how do",
                "why does",
                "why do",
                "difference between",
                "compare",
                "define",
                "give me",
            )
        )
        has_format = any(
            cue in text
            for cue in (
                "step by step",
                "brief",
                "simple",
                "for beginner",
                "for beginners",
            )
        )
        topic_tokens = [
            t
            for t in re.findall(r"\b[a-z0-9']+\b", text)
            if t not in self.SEMANTIC_STOPWORDS
        ]
        if not has_ask or len(topic_tokens) < 2:
            return False
        if has_format:
            return True
        # Self-contained definitional/comparison requests should be answer-first.
        if "?" in text:
            return True
        return any(
            cue in text
            for cue in ("difference between", "compare", "define", "what is", "what's")
        )

    def _is_open_ended_learning_goal(self, text: str, intent: str) -> bool:
        """Detect exploratory learning prompts that should be refined, not hard-rejected."""
        low = str(text or "").strip().lower()
        if not low:
            return False

        normalized_intent = str(intent or "").strip().lower()
        if not (
            normalized_intent.startswith("conversation:")
            or normalized_intent.startswith("learning:")
            or normalized_intent in {"question", "study_topic"}
        ):
            return False

        if self._is_answerability_direct_question(low):
            return False

        tokens = re.findall(r"\b[a-z0-9']+\b", low)
        if len(tokens) < 4:
            return False

        explicit_action_verbs = {
            "create",
            "build",
            "delete",
            "remove",
            "open",
            "run",
            "execute",
            "set",
            "update",
            "install",
            "write",
            "save",
            "send",
            "search",
            "list",
            "edit",
        }
        if any(tok in explicit_action_verbs for tok in tokens):
            return False

        exploratory_cues = (
            "learn more",
            "learn about",
            "i want to learn",
            "tell me about",
            "overview",
            "basics",
            "introduction",
            "understand",
        )
        if not any(cue in low for cue in exploratory_cues):
            return False

        topic_tokens = [
            tok
            for tok in tokens
            if tok not in self.SEMANTIC_STOPWORDS
            and tok
            not in {
                "learn",
                "learning",
                "more",
                "about",
                "want",
                "tell",
                "overview",
                "basics",
                "introduction",
                "understand",
            }
        ]
        return len(topic_tokens) >= 1

    def _is_answerability_direct_question(self, text: str) -> bool:
        low = str(text or "").strip().lower()
        if not low:
            return False

        tokens = re.findall(r"\b[a-z0-9']+\b", low)
        if len(tokens) < 3:
            return False

        if any(
            tok in {"something", "anything", "stuff", "whatever", "idk"}
            for tok in tokens
        ):
            return False

        if re.search(r"\b(help\s+me|do\s+this|do\s+that|build\s+me|make\s+me)\b", low):
            return False

        direct_patterns = (
            r"^\s*what\s+is\b",
            r"^\s*what's\b",
            r"^\s*what\s+are\b",
            r"^\s*what(?:'s|\s+is)?\s+the\s+difference\b",
            r"\bdifference\s+between\b",
            r"^\s*compare\b",
            r"^\s*explain\b",
            r"^\s*how\s+does\b",
            r"^\s*how\s+do\b",
            r"^\s*why\s+does\b",
            r"^\s*why\s+do\b",
            r"^\s*define\b",
        )
        has_direct_structure = any(re.search(pat, low) for pat in direct_patterns)
        is_question_like = bool(
            "?" in low
            or re.match(r"^\s*(what|which|how|why|explain|define|compare)\b", low)
            or re.search(r"\bi\s+(?:want|need|would\s+like)\s+to\s+know\b", low)
            or re.search(r"\btell\s+me\b", low)
        )
        return bool(has_direct_structure and is_question_like)

    def _is_simple_native_conversation(
        self,
        *,
        state: ReasoningState,
        has_explicit_action_cue: bool,
        has_active_goal: bool,
    ) -> bool:
        text = str(state.source_text or "").strip().lower()
        if not text:
            return False

        normalized_intent = (state.user_intent or "").lower().strip()
        if not normalized_intent.startswith(
            "conversation:"
        ) and normalized_intent not in {
            "greeting",
            "thanks",
            "farewell",
            "status_inquiry",
        }:
            return False

        if has_explicit_action_cue:
            return False
        if has_active_goal:
            return False
        if self.RISKY_CONVERSATION_RE.search(text):
            return False

        # Missing-knowledge style prompts should still go through normal decisioning.
        missing_knowledge_cues = (
            "what is",
            "why",
            "how does",
            "compare",
            "explain",
            "difference between",
        )
        if any(cue in text for cue in missing_knowledge_cues):
            return False

        if len(text.split()) > 9:
            return False

        if normalized_intent in self.SIMPLE_SCAFFOLD_INTENTS:
            return True

        # Only allow help/clarification intents through deterministic native path
        # when the actual utterance is genuinely short/simple.
        if normalized_intent == "conversation:clarification_needed":
            return bool(self.SIMPLE_NATIVE_RE.match(text)) and (
                not self._is_rich_conceptual_request(text)
            )

        if normalized_intent == "conversation:help":
            return False

        return bool(self.SIMPLE_NATIVE_RE.match(text))

    def _should_force_native_scaffold(
        self,
        *,
        state: ReasoningState,
        has_explicit_action_cue: bool,
        has_active_goal: bool,
    ) -> bool:
        normalized_intent = (state.user_intent or "").lower().strip()
        text = str(state.source_text or "").strip().lower()
        if normalized_intent in self.SIMPLE_SCAFFOLD_INTENTS:
            return not has_explicit_action_cue
        if normalized_intent == "conversation:clarification_needed":
            return (
                (not has_explicit_action_cue)
                and (not has_active_goal)
                and bool(self.SIMPLE_NATIVE_RE.match(text))
                and (not self._is_rich_conceptual_request(text))
            )
        return False

    def should_veto_tool_execution(
        self,
        *,
        user_input: str,
        intent: str,
        confidence: float,
        intent_plausibility: float,
        intent_candidates: List[Dict[str, Any]] | None,
    ) -> Dict[str, Any]:
        """Final guard before plugin execution to prevent high-cost misroutes."""
        normalized_intent = (intent or "").lower().strip()
        conf = max(0.0, min(1.0, float(confidence or 0.0)))
        plausibility = max(0.0, min(1.0, float(intent_plausibility or 0.0)))
        candidates = intent_candidates or []
        text_lower = (user_input or "").lower()

        if self._is_rich_conceptual_request(text_lower):
            return {"veto": False, "reason": "rich_conceptual_prompt"}

        if normalized_intent.startswith("conversation:"):
            return {"veto": False, "reason": "conversation_intent"}

        conversational_cues = (
            "brainstorm",
            "idea",
            "explore",
            "how might",
            "could we",
            "strategy",
        )
        if any(
            cue in text_lower for cue in conversational_cues
        ) and not normalized_intent.startswith("conversation:"):
            return {
                "veto": True,
                "reason": "conversational_input_not_actionable",
                "question": "This seems high-level. What concrete outcome should I execute?",
            }

        if plausibility < 0.46:
            return {
                "veto": True,
                "reason": "low_intent_plausibility",
                "question": "I am not confident this intent is correct. Can you clarify the action you want?",
            }

        if conf < 0.42 and plausibility < 0.62:
            return {
                "veto": True,
                "reason": "low_confidence_and_plausibility",
                "question": "I need one more detail before triggering a tool. What exact outcome do you want?",
            }

        if candidates and len(candidates) > 1:
            top = float(candidates[0].get("score", 0.0))
            second = float(candidates[1].get("score", 0.0))
            if (top - second) < 0.08 and conf < 0.70:
                return {
                    "veto": True,
                    "reason": "ambiguous_top_intents",
                    "question": "I see multiple possible intents. Which single outcome should I execute?",
                }

        return {"veto": False, "reason": "allowed"}

    def should_use_planner(
        self, state: ReasoningState, scores: Dict[str, float]
    ) -> bool:
        """Executive authority for when planning is required before response."""
        intent = (state.user_intent or "").lower()
        text = (state.source_text or "").lower()
        if intent.startswith("learning:") or intent.startswith("question:"):
            return True
        if intent in {"conversation:question", "conversation:help"} and any(
            cue in text
            for cue in (
                "explain",
                "beginner",
                "step by step",
                "foundations",
                "architecture",
                "strategy",
            )
        ):
            return True
        if any(cue in text for cue in (" then ", " after ", " next ", " and then ")):
            return True
        if state.depth_level >= 2 and state.topic:
            return True
        if scores.get("tools", 0.0) >= 0.75 and state.user_goal:
            return True
        return False

    def uncertainty_behavior(
        self, state: ReasoningState, scores: Dict[str, float]
    ) -> str:
        """Return proceed | clarify | defer | reject based on confidence and score ambiguity."""
        conf = max(0.0, min(1.0, float(state.confidence)))
        plausibility = max(0.0, min(1.0, float(state.intent_plausibility)))
        normalized_intent = (state.user_intent or "").lower().strip()
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[0][1] if ranked else 0.0
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top - second

        if normalized_intent == "conversation:clarification_needed" and conf >= 0.45:
            return "proceed"

        if plausibility < 0.32:
            return "clarify"
        if conf < 0.20 and not state.topic:
            return "reject"
        if conf < 0.35:
            return "defer"
        if plausibility < 0.55 and conf < 0.65:
            return "clarify"
        if conf < 0.60 and margin < 0.12:
            return "clarify"
        return "proceed"

    def format_candidate_uncertainty(
        self,
        intent_candidates: List[Dict[str, Any]] | None,
        *,
        limit: int = 3,
    ) -> str:
        """Return a compact top-k candidate summary for user-facing clarification."""
        candidates = intent_candidates or []
        if not candidates:
            return ""
        top = sorted(
            candidates,
            key=lambda c: float(c.get("score", c.get("confidence", 0.0)) or 0.0),
            reverse=True,
        )[: max(1, int(limit or 3))]
        parts: List[str] = []
        for cand in top:
            label = str(cand.get("intent") or "unknown")
            score = float(cand.get("score", cand.get("confidence", 0.0)) or 0.0)
            parts.append(f"{label} ({score * 100:.0f}%)")
        return "I am not fully certain. Top possibilities: " + ", ".join(parts) + "."

    def score_decisions(
        self,
        state: ReasoningState,
        *,
        is_pure_conversation: bool,
        has_explicit_action_cue: bool,
        has_active_goal: bool,
        force_plugins_for_notes: bool,
    ) -> Dict[str, float]:
        """Score decisions from five compact executive axes.

        Axes:
        - can_act_now
        - safe_to_act
        - enough_information
        - target_confidence
        - expected_mission_progress (+ user_interruption_cost)
        """
        text_intent = (state.user_intent or "").lower().strip()
        text = str(state.source_text or "").lower().strip()

        axes = self._decision_axes(
            state=state,
            is_pure_conversation=is_pure_conversation,
            has_explicit_action_cue=has_explicit_action_cue,
            has_active_goal=has_active_goal,
            force_plugins_for_notes=force_plugins_for_notes,
        )

        can_act_now = float(axes["can_act_now"])
        safe_to_act = float(axes["safe_to_act"])
        enough_information = float(axes["enough_information"])
        target_confidence = float(axes["target_confidence"])
        expected_progress = float(axes["expected_mission_progress"])
        interruption_cost = float(axes["user_interruption_cost"])

        act_score = can_act_now * safe_to_act * enough_information * target_confidence
        answer_score = (
            expected_progress
            * max(target_confidence, 0.45)
            * max(enough_information, 0.45)
        )
        if can_act_now >= 0.50:
            # When execution is clearly available, bias the kernel toward ACT over ANSWER.
            answer_score *= 0.78
        ask_score = (
            (1.0 - enough_information)
            * max(can_act_now, 0.45)
            * (1.0 - (0.45 * interruption_cost))
        )
        wait_score = (1.0 - safe_to_act) * max(can_act_now, 0.45)

        scores = {
            "tools": 0.0,
            "memory": 0.0,
            "rag": 0.0,
            "llm": 0.0,
            "clarify": 0.0,
            "search": 0.0,
            "reject": 0.0,
            "defer": 0.0,
        }

        scores["tools"] = act_score
        scores["search"] = act_score * (
            1.05
            if ("search" in text_intent or "research" in text or "look up" in text)
            else 0.45
        )
        scores["llm"] = answer_score
        scores["memory"] = (0.70 * answer_score) + (0.20 if state.topic else 0.0)
        scores["rag"] = (0.65 * answer_score) + (0.25 if state.user_goal else 0.0)
        scores["clarify"] = ask_score
        scores["defer"] = wait_score
        scores["reject"] = max(0.0, (0.40 - target_confidence) + (0.35 - safe_to_act))

        normalized_intent = text_intent
        if normalized_intent == "conversation:clarification_needed":
            scores["llm"] += 0.20
            scores["clarify"] -= 0.15
        if is_pure_conversation:
            scores["llm"] += 0.08
            scores["clarify"] *= 0.85

        if state.route_bias == "clarify_first":
            scores["clarify"] += 0.15
            scores["tools"] *= 0.85
            scores["search"] *= 0.85
        elif state.route_bias == "tool_first":
            scores["tools"] += 0.12
            scores["search"] += 0.10
        elif state.route_bias == "goal_first":
            scores["llm"] += 0.06
            scores["memory"] += 0.06

        if state.tool_budget <= 0:
            scores["tools"] *= 0.30
            scores["search"] *= 0.25
            scores["defer"] += 0.10

        if state.planner_hint == "increase_structure_depth":
            scores["llm"] += 0.05
            scores["memory"] += 0.03

        for route in list(scores.keys()):
            scores[route] *= float(self._routing_weights.get(route, 1.0))

        # Normalize to 0..1 to keep all downstream thresholds stable.
        for k, v in list(scores.items()):
            scores[k] = max(0.0, min(1.0, float(v)))
        return scores

    def _decision_axes(
        self,
        *,
        state: ReasoningState,
        is_pure_conversation: bool,
        has_explicit_action_cue: bool,
        has_active_goal: bool,
        force_plugins_for_notes: bool,
    ) -> Dict[str, float]:
        """Compute compact executive axes used by the 4-way kernel."""
        intent = (state.user_intent or "").lower().strip()
        text = str(state.source_text or "").lower().strip()
        conf = max(0.0, min(1.0, float(state.confidence)))
        plausibility = max(0.0, min(1.0, float(state.intent_plausibility)))
        goal_status = str(getattr(state, "goal_status", "") or "").strip().lower()
        goal_next_action = str(getattr(state, "goal_next_action", "") or "").strip()
        goal_blockers = [
            str(x).strip()
            for x in list(getattr(state, "goal_blockers", []) or [])
            if str(x).strip()
        ]
        goal_blocked = bool(goal_blockers) or goal_status in {
            "blocked",
            "stalled",
            "waiting",
            "on_hold",
            "needs_input",
        }
        has_goal_next_action = bool(goal_next_action)
        multiple_active_goals = int(getattr(state, "goal_active_count", 0) or 0) >= 2

        can_act_now = (
            1.0
            if (
                has_explicit_action_cue
                or has_active_goal
                or force_plugins_for_notes
                or any(intent.startswith(domain) for domain in self.TOOL_DOMAINS)
                or intent.startswith("search")
            )
            else 0.15
        )
        if has_goal_next_action:
            can_act_now = max(can_act_now, 0.72)
        elif has_active_goal:
            can_act_now = max(can_act_now, 0.58)
        if goal_blocked:
            can_act_now *= 0.72
            if not has_goal_next_action and not has_explicit_action_cue:
                can_act_now = min(can_act_now, 0.42)

        safe_to_act = 1.0
        if self.RISKY_CONVERSATION_RE.search(text):
            safe_to_act *= 0.45
        if plausibility < 0.55:
            safe_to_act *= 0.78
        if plausibility < 0.40:
            safe_to_act *= 0.72
        if conf < 0.45:
            safe_to_act *= 0.82
        if has_goal_next_action and not goal_blocked:
            safe_to_act = min(1.0, safe_to_act + 0.08)
        if goal_blocked:
            safe_to_act *= 0.74

        enough_information = 0.25 + (0.45 * conf) + (0.30 * plausibility)
        if state.topic or state.user_goal:
            enough_information += 0.12
        if has_explicit_action_cue:
            enough_information += 0.18
        if self._is_clear_informational_request(state):
            enough_information += 0.12
        if intent == "conversation:clarification_needed":
            enough_information += 0.10
        if has_goal_next_action:
            enough_information += 0.14
        if goal_blocked and not has_goal_next_action:
            enough_information -= 0.10
        if multiple_active_goals and not has_explicit_action_cue and not state.topic:
            enough_information -= 0.06

        target_confidence = (0.55 * conf) + (0.45 * plausibility)
        if goal_blocked and not has_goal_next_action:
            target_confidence -= 0.04

        if can_act_now >= 0.50:
            expected_mission_progress = (
                0.10
                + (0.55 * safe_to_act * enough_information)
                + (0.35 * target_confidence)
            )
        else:
            expected_mission_progress = (
                0.25 + (0.45 * target_confidence) + (0.30 * enough_information)
            )
        if self._is_clear_informational_request(state):
            expected_mission_progress += 0.10
        if has_goal_next_action and not goal_blocked:
            expected_mission_progress += 0.14
        if goal_blocked:
            expected_mission_progress *= 0.72
            if has_goal_next_action:
                expected_mission_progress += 0.08

        user_interruption_cost = 0.30
        if is_pure_conversation or intent.startswith("conversation:"):
            user_interruption_cost += 0.20
        if self._is_clear_informational_request(state):
            user_interruption_cost += 0.25
        if can_act_now >= 0.50 and (safe_to_act < 0.50 or enough_information < 0.50):
            user_interruption_cost -= 0.18
        if has_goal_next_action and not goal_blocked:
            user_interruption_cost += 0.08
        if goal_blocked and not has_goal_next_action:
            user_interruption_cost -= 0.12

        return {
            "can_act_now": max(0.0, min(1.0, can_act_now)),
            "safe_to_act": max(0.0, min(1.0, safe_to_act)),
            "enough_information": max(0.0, min(1.0, enough_information)),
            "target_confidence": max(0.0, min(1.0, target_confidence)),
            "expected_mission_progress": max(0.0, min(1.0, expected_mission_progress)),
            "user_interruption_cost": max(0.0, min(1.0, user_interruption_cost)),
        }

    def apply_reflection(self, reflection: Dict[str, Any]) -> None:
        """Adjust routing matrix based on reflection feedback."""
        if not isinstance(reflection, dict):
            return
        adjustments = reflection.get("routing_adjustments", {}) or {}
        for route, delta in adjustments.items():
            if route not in self._routing_weights:
                continue
            current = float(self._routing_weights.get(route, 1.0))
            self._routing_weights[route] = max(0.5, min(1.5, current + float(delta)))

    def get_routing_weights(self) -> Dict[str, float]:
        return dict(self._routing_weights)

    def derive_runtime_controls(
        self,
        state: ReasoningState,
        scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """Derive runtime controls that affect route preference, depth, and tool usage."""
        ranked = sorted((scores or {}).items(), key=lambda kv: kv[1], reverse=True)
        top_route = ranked[0][0] if ranked else "llm"
        top_score = float(ranked[0][1]) if ranked else 0.0

        thinking_depth = max(1, min(4, int(state.planner_depth or 1)))
        if state.depth_level >= 3:
            thinking_depth = max(thinking_depth, 3)
        if state.route_bias == "goal_first":
            thinking_depth = min(4, thinking_depth + 1)

        allow_tools = bool(state.tool_budget > 0 and state.intent_plausibility >= 0.45)
        if state.route_bias == "clarify_first" and state.intent_plausibility < 0.65:
            allow_tools = False

        routing_preference = "balanced"
        if top_route in ("tools", "search") and top_score >= 0.62 and allow_tools:
            routing_preference = "tool_first"
        elif top_route in ("clarify", "defer", "reject"):
            routing_preference = "clarify_first"
        elif top_route in ("llm", "memory"):
            routing_preference = "llm_first"

        max_tool_hops = (
            0 if not allow_tools else max(1, min(3, int(state.tool_budget or 1)))
        )
        return {
            "routing_preference": routing_preference,
            "thinking_depth": thinking_depth,
            "allow_tools": allow_tools,
            "max_tool_hops": max_tool_hops,
        }

    def save_weights(self, path: str) -> None:
        """Persist routing weights to disk as JSON."""
        import json
        import os

        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._routing_weights, f, indent=2)
        except Exception:
            pass

    def load_weights(self, path: str, *, decay: float = 0.05) -> None:
        """
        Load routing weights from disk.
        Applies a small decay toward 1.0 so stale biases fade after a restart.
        """
        import json

        try:
            with open(path, "r", encoding="utf-8") as f:
                stored = json.load(f)
            if not isinstance(stored, dict):
                return
            for key, value in stored.items():
                if key in self._routing_weights and isinstance(value, (int, float)):
                    loaded = float(value)
                    # Pull toward 1.0 (neutral) by the decay factor each restart
                    decayed = loaded + (1.0 - loaded) * decay
                    self._routing_weights[key] = max(0.5, min(1.5, decayed))
        except (FileNotFoundError, ValueError):
            pass
        except Exception:
            pass

    def evaluate_response(
        self,
        *,
        user_input: str,
        intent: str,
        response: str,
        route: str,
        context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Response acceptance gate before content is sent to the user."""
        context = context or {}
        resp = (response or "").strip()
        normalized_intent = (intent or "").lower().strip()
        answerable_question = self._is_answerability_direct_question(user_input) and (
            normalized_intent.startswith("conversation:")
            or normalized_intent.startswith("learning:")
            or normalized_intent in {"question", "study_topic"}
        )
        open_learning_goal = self._is_open_ended_learning_goal(
            user_input,
            normalized_intent,
        )
        if not resp:
            fallback_action = (
                "safe_reply"
                if (answerable_question or open_learning_goal)
                else "clarify"
            )
            fallback_reason = (
                "llm_failed_after_answer_directly"
                if answerable_question
                else "clarification_due_to_missing_parameter"
            )
            return {
                "accepted": False,
                "score": 0.0,
                "reason": "empty_response",
                "fallback_action": fallback_action,
                "fallback_reason": fallback_reason,
            }

        semantic_guard = self._semantic_fidelity_guard(
            user_input=user_input,
            intent=intent,
            response=resp,
        )
        if semantic_guard is not None:
            if "fallback_reason" not in semantic_guard:
                semantic_guard["fallback_reason"] = (
                    "llm_failed_after_answer_directly"
                    if answerable_question
                    else "low_confidence_answer_requiring_retry"
                )
            return semantic_guard

        user_tokens = set(self._tokens(user_input))
        resp_tokens = set(self._tokens(resp))
        overlap = 0.0
        if user_tokens:
            overlap = len(user_tokens.intersection(resp_tokens)) / max(
                len(user_tokens), 1
            )

        uncertain_markers = (
            "i'm not sure",
            "i am not sure",
            "maybe",
            "possibly",
            "i don't know",
            "not certain",
        )
        uncertain_penalty = (
            0.35 if any(m in resp.lower() for m in uncertain_markers) else 0.0
        )

        generic_markers = (
            "it depends",
            "in general",
            "there are many factors",
            "cannot be determined",
        )
        generic_penalty = (
            0.20 if any(m in resp.lower() for m in generic_markers) else 0.0
        )

        score = max(
            0.0, min(1.0, 0.55 + (0.55 * overlap) - uncertain_penalty - generic_penalty)
        )
        plan_adherence = self._response_plan_adherence(resp, context)
        goal_alignment = float((context or {}).get("goal_alignment", 1.0) or 1.0)
        goal_alignment = max(0.0, min(1.0, goal_alignment))
        plan = (context or {}).get("response_plan", {}) or {}
        required_sections = (
            plan.get("required_sections", []) if isinstance(plan, dict) else []
        )
        format_hint = (
            str(plan.get("format_hint", "")).lower() if isinstance(plan, dict) else ""
        )
        _needs_steps = (
            ("steps" in required_sections)
            or ("numbered" in format_hint)
            or (
                str(plan.get("response_type", "")).lower()
                in ("instruction", "troubleshooting")
            )
        )
        _has_steps = any(f"{i}." in resp.lower() for i in range(1, 6)) or (
            "step" in resp.lower()
        )
        if _needs_steps and not _has_steps:
            _fallback_action = (
                "revise_answer"
                if (answerable_question or open_learning_goal)
                else "safe_reply"
            )
            _fallback_reason = (
                "llm_failed_after_answer_directly"
                if answerable_question
                else "low_confidence_answer_needs_refinement"
            )
            return {
                "accepted": False,
                "score": 0.0,
                "reason": "plan_violation_missing_steps",
                "fallback_action": _fallback_action,
                "fallback_reason": _fallback_reason,
            }

        score = max(
            0.0,
            min(
                1.0,
                (0.65 * score) + (0.20 * plan_adherence) + (0.15 * goal_alignment),
            ),
        )
        threshold = 0.52 if route == "llm" else 0.48
        if route == "llm" and open_learning_goal:
            threshold = 0.46

        if score >= threshold:
            return {
                "accepted": True,
                "score": score,
                "reason": (
                    "accepted"
                    if plan_adherence >= 0.60
                    else "accepted_low_plan_adherence"
                ),
                "fallback_action": "",
            }

        fallback_action = (
            "clarify" if (overlap < 0.2 or plan_adherence < 0.45) else "safe_reply"
        )
        if answerable_question or open_learning_goal:
            fallback_action = "revise_answer"

        fallback_reason = "low_confidence_answer_requiring_retry"
        if fallback_action == "clarify":
            plan_strategy = str(plan.get("strategy") or "").strip().lower()
            if plan_strategy == "ask_guiding_question":
                fallback_reason = "clarification_due_to_goal_narrowing"
            else:
                fallback_reason = "clarification_due_to_missing_parameter"
        elif answerable_question:
            fallback_reason = "llm_failed_after_answer_directly"
        elif open_learning_goal:
            fallback_reason = "low_confidence_answer_needs_refinement"

        return {
            "accepted": False,
            "score": score,
            "reason": (
                "low_alignment" if goal_alignment >= 0.35 else "goal_misalignment"
            ),
            "fallback_action": fallback_action,
            "fallback_reason": fallback_reason,
        }

    def _semantic_anchor_tokens(self, text: str) -> List[str]:
        tokens = []
        for token in self._tokens(text):
            low = token.lower().strip()
            if len(low) < 4:
                continue
            if low in self.SEMANTIC_STOPWORDS:
                continue
            tokens.append(low)
        # Preserve stable order with uniqueness.
        seen = set()
        ordered: List[str] = []
        for tok in tokens:
            if tok in seen:
                continue
            seen.add(tok)
            ordered.append(tok)
        return ordered

    def _semantic_fidelity_guard(
        self,
        *,
        user_input: str,
        intent: str,
        response: str,
    ) -> Dict[str, Any] | None:
        """Reject responses that lose core user meaning or inject obvious nonsense."""
        low_resp = (response or "").lower()
        if "person 'an ai'" in low_resp or "general_assistance" in low_resp:
            return {
                "accepted": False,
                "score": 0.0,
                "reason": "semantic_noise_in_response",
                "fallback_action": "safe_reply",
                "fallback_reason": "low_confidence_answer_requiring_retry",
            }

        normalized_intent = (intent or "").lower().strip()
        if not (
            normalized_intent.startswith("conversation:")
            or normalized_intent.startswith("learning:")
            or normalized_intent in {"greeting", "thanks"}
        ):
            return None

        user_anchors = self._semantic_anchor_tokens(user_input)
        if len(user_anchors) < 2:
            return None

        response_tokens = set(self._tokens(response))

        user_low = (user_input or "").lower()
        practical_framework_prompt = (
            any(cue in user_low for cue in ("framework", "frameworks", "architecture"))
            and any(
                cue in user_low
                for cue in (
                    "agentic autonomy",
                    "autonomous agent",
                    "ai agent",
                    "autonomy in ai",
                )
            )
            and (
                any(
                    cue in user_low
                    for cue in (
                        "build",
                        "building",
                        "implement",
                        "implementation",
                        "existing",
                        "research",
                    )
                )
                or len(re.findall(r"\b[a-z0-9']+\b", user_low)) <= 7
            )
        )
        if practical_framework_prompt:
            theoretical_terms = {
                "dennett",
                "intentional",
                "tononi",
                "iit",
                "consciousness",
                "cognitive",
                "philosophy",
            }
            practical_terms = {
                "langgraph",
                "langchain",
                "autogpt",
                "crewai",
                "react",
                "plan",
                "execute",
                "reflect",
                "verify",
                "checkpoint",
                "rollback",
                "retry",
                "escalate",
                "tool",
                "memory",
                "orchestration",
                "state machine",
                "act-r",
                "soar",
            }
            has_theoretical_bias = bool(theoretical_terms.intersection(response_tokens))
            has_practical_signal = bool(practical_terms.intersection(response_tokens))
            if has_theoretical_bias and not has_practical_signal:
                return {
                    "accepted": False,
                    "score": 0.0,
                    "reason": "semantic_drift_theoretical_domain",
                    "fallback_action": "safe_reply",
                    "fallback_reason": "low_confidence_answer_requiring_retry",
                }

        overlap = [tok for tok in user_anchors if tok in response_tokens]
        if not overlap:
            return {
                "accepted": False,
                "score": 0.0,
                "reason": "semantic_core_missing",
                "fallback_action": "safe_reply",
                "fallback_reason": "low_confidence_answer_requiring_retry",
            }

        # Guard against programming-language drift on conceptual assistant architecture questions.
        programming_drift_terms = {
            "polymorphism",
            "interface",
            "inheritance",
            "encapsulation",
            "oop",
        }
        if programming_drift_terms.intersection(
            response_tokens
        ) and not programming_drift_terms.intersection(set(user_anchors)):
            return {
                "accepted": False,
                "score": 0.0,
                "reason": "semantic_drift_programming_domain",
                "fallback_action": "safe_reply",
                "fallback_reason": "low_confidence_answer_requiring_retry",
            }

        return None

    def _response_plan_adherence(self, response: str, context: Dict[str, Any]) -> float:
        """Return 0..1 estimate of how well the response follows the active response plan."""
        plan = (context or {}).get("response_plan", {}) or {}
        if not isinstance(plan, dict) or not plan:
            return 1.0

        resp_low = (response or "").lower()
        score = 0.0
        checks = 0.0

        resp_type = str(plan.get("response_type", "")).strip().lower()
        format_hint = str(plan.get("format_hint", "")).strip().lower()
        required_sections = plan.get("required_sections", []) or []

        # Required sections are soft checks via lexical cues.
        section_markers = {
            "answer": ("answer", "in short", "summary"),
            "explanation": ("because", "means", "works by"),
            "example": ("for example", "example", "e.g."),
            "steps": ("1.", "2.", "step", "first", "next", "then"),
            "expected_result": ("expected", "result", "you should see"),
            "root_cause": ("root cause", "caused by", "reason"),
            "fix": ("fix", "solution", "change", "update"),
            "plan": ("plan", "phase", "milestone"),
            "risks": ("risk", "trade-off", "constraint"),
            "check_understanding": (
                "does that make sense",
                "want me to",
                "would you like",
            ),
        }

        for sec in required_sections:
            checks += 1.0
            markers = section_markers.get(str(sec), ())
            if markers and any(m in resp_low for m in markers):
                score += 1.0

        # Format adherence check.
        checks += 1.0
        if "numbered" in format_hint:
            if any(f"{i}." in resp_low for i in range(1, 6)):
                score += 1.0
        elif "structured" in format_hint:
            if "\n" in response and (":" in response or "- " in response):
                score += 1.0
        else:
            score += 1.0

        # Type-level sanity check.
        checks += 1.0
        if resp_type in ("instruction", "troubleshooting"):
            if any(f"{i}." in resp_low for i in range(1, 6)) or "step" in resp_low:
                score += 1.0
        elif resp_type == "explanation":
            if "because" in resp_low or "example" in resp_low:
                score += 1.0
        elif resp_type == "debugging":
            if any(m in resp_low for m in ("error", "fix", "cause", "traceback")):
                score += 1.0
        else:
            score += 1.0

        if checks <= 0.0:
            return 1.0
        return max(0.0, min(1.0, score / checks))

    def decide_learning(
        self,
        *,
        relevance: float,
        confidence: float,
        novelty: float,
        risk: float,
    ) -> str:
        """Executive authority over learning writes.

        Returns one of: store | reject | queue_review | temporary
        """
        relevance = max(0.0, min(1.0, float(relevance)))
        confidence = max(0.0, min(1.0, float(confidence)))
        novelty = max(0.0, min(1.0, float(novelty)))
        risk = max(0.0, min(1.0, float(risk)))

        utility = (0.45 * relevance) + (0.35 * confidence) + (0.20 * novelty)
        if risk >= 0.70:
            return "reject"
        if utility >= 0.72 and risk <= 0.35:
            return "store"
        if utility >= 0.55 and risk <= 0.55:
            return "temporary"
        if utility >= 0.45:
            return "queue_review"
        return "reject"

    def format_reasoning_state(self, state: ReasoningState) -> str:
        """Render compact, non-CoT internal state for system context."""
        lines = [
            "Internal reasoning state (system-only):",
            f"- user_intent: {state.user_intent}",
            f"- topic: {state.topic or 'unknown'}",
            f"- confidence: {state.confidence:.2f}",
            f"- conversation_goal: {state.conversation_goal or 'general_assistance'}",
            f"- user_goal: {state.user_goal or 'none'}",
            f"- depth_level: {state.depth_level}",
        ]
        if state.plan:
            lines.append(f"- plan: {' | '.join(state.plan)}")
        return "\n".join(lines)

    def _derive_plan(
        self, intent: str, topic: str, depth_level: int, user_input: str
    ) -> List[str]:
        lowered_intent = (intent or "").lower()
        lowered_input = (user_input or "").lower()

        if lowered_intent.startswith("learning:") or lowered_intent.startswith(
            "question:"
        ):
            if (
                depth_level >= 3
                or "example" in lowered_input
                or "code" in lowered_input
            ):
                return [
                    "explain succinctly",
                    "give concrete example",
                    "offer deeper explanation",
                ]
            return [
                "explain simply",
                "give example",
                "offer deeper explanation",
            ]

        if topic:
            return ["answer current question", "keep topic continuity"]

        return ["answer directly", "ask clarification if ambiguity remains"]

    def _tokens(self, text: str) -> List[str]:
        import re

        return re.findall(r"[a-z0-9']+", (text or "").lower())


_executive_controller: ExecutiveController | None = None


def get_executive_controller() -> ExecutiveController:
    global _executive_controller
    if _executive_controller is None:
        _executive_controller = ExecutiveController()
    return _executive_controller
