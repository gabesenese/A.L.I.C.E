from pathlib import Path

from ai.core.autonomy_dispatcher import (
    OUTCOME_ACT_AUTOMATICALLY,
    OUTCOME_ASK_USER,
    OUTCOME_ESCALATE_AND_STOP,
    TinyAutonomyDispatcher,
)
from ai.core.bounded_autonomy_manager import AutonomyLoop, BoundedAutonomyManager
from ai.core.unified_action_engine import ActionRequest, UnifiedActionEngine


class _StubPluginManager:
    def __init__(self):
        self.rollback_calls = []

    def execute_for_intent(self, intent, query, entities, context):
        if context.get("rollback"):
            self.rollback_calls.append(
                {
                    "intent": intent,
                    "query": query,
                    "entities": dict(entities or {}),
                    "context": dict(context or {}),
                }
            )
            return {
                "success": True,
                "status": "success",
                "plugin": "Notes Plugin",
                "action": "delete",
                "data": {"rolled_back": True},
                "message": "Rollback done",
                "confidence": 0.9,
                "retryable": False,
                "side_effects": ["deleted"],
                "verification": {"accepted": True, "confidence": 0.9, "issues": []},
            }

        # Normal execution path returns partial success with side effects to trigger rollback.
        return {
            "success": False,
            "status": "partial",
            "plugin": "Notes Plugin",
            "action": "create",
            "data": {"note_id": "n42", "title": "Roadmap"},
            "message": "Created but verification uncertain",
            "confidence": 0.5,
            "retryable": False,
            "side_effects": ["created"],
            "verification": {"accepted": True, "confidence": 0.8, "issues": []},
        }


class _StubToolExecutor:
    def execute(self, *, plugin_manager, intent, query, entities, context):
        class _Outcome:
            handled = True
            verified = True
            result = plugin_manager.execute_for_intent(intent, query, entities, context)

        return _Outcome()


def test_unified_action_engine_attempts_auto_rollback_on_partial_failure():
    plugin_manager = _StubPluginManager()
    engine = UnifiedActionEngine(tool_executor=_StubToolExecutor())
    engine.bind_plugin_manager(plugin_manager)

    req = ActionRequest(
        goal="create roadmap note",
        plugin="notes",
        action="create",
        params={"title": "Roadmap", "_raw_query": "create roadmap note"},
        source_intent="notes:create",
        confidence=0.8,
        rollback_policy="auto",
        target_spec={"target": "Roadmap"},
    )

    res = engine.execute(req)

    rollback = (res.state_updates or {}).get("rollback") or {}
    assert rollback.get("attempted") is True
    assert rollback.get("success") is True
    assert plugin_manager.rollback_calls


def test_unified_action_engine_rollback_preview_only_does_not_execute():
    plugin_manager = _StubPluginManager()
    engine = UnifiedActionEngine(tool_executor=_StubToolExecutor())
    engine.bind_plugin_manager(plugin_manager)

    req = ActionRequest(
        goal="create roadmap note",
        plugin="notes",
        action="create",
        params={
            "title": "Roadmap",
            "_raw_query": "create roadmap note",
            "_rollback_preview_only": True,
        },
        source_intent="notes:create",
        confidence=0.8,
        rollback_policy="auto",
        target_spec={"target": "Roadmap"},
    )

    res = engine.execute(req)

    rollback = (res.state_updates or {}).get("rollback") or {}
    assert rollback.get("status") == "rollback_preview"
    assert plugin_manager.rollback_calls == []


def test_unified_action_engine_blocks_high_risk_rollback_without_approval():
    plugin_manager = _StubPluginManager()
    engine = UnifiedActionEngine(tool_executor=_StubToolExecutor())
    engine.bind_plugin_manager(plugin_manager)

    req = ActionRequest(
        goal="create production note",
        plugin="notes",
        action="create",
        params={"title": "Prod", "_raw_query": "create prod note", "_approval_token": True},
        source_intent="notes:create",
        confidence=0.8,
        rollback_policy="auto",
        risk_level="high",
        target_spec={"target": "Prod"},
    )

    res = engine.execute(req)

    rollback = (res.state_updates or {}).get("rollback") or {}
    assert rollback.get("status") == "rollback_confirmation_required"
    assert plugin_manager.rollback_calls == []


def test_bounded_autonomy_evaluates_trigger_events(tmp_path: Path):
    manager = BoundedAutonomyManager(storage_path=str(tmp_path / "autonomy_loops.json"))
    manager.register_loop(
        AutonomyLoop(
            name="goal_health",
            permission_level="operator",
            scope="goal monitoring",
            stop_conditions=["manual_stop"],
            confidence_threshold=0.6,
            enabled=True,
        )
    )
    manager.register_loop(
        AutonomyLoop(
            name="repo_failure_watch",
            permission_level="operator",
            scope="build/test monitoring",
            stop_conditions=["manual_stop"],
            confidence_threshold=0.7,
            enabled=True,
        )
    )

    events = manager.evaluate_triggers(
        world_state={"unresolved_ambiguity": ["target_mismatch"]},
        goal_summary={"active_goals": 2},
        journal_summary={"failed": 5, "success": 1},
        execution_state={"autonomous_running": True},
    )

    reasons = {e.reason for e in events}
    assert "active_goals_with_unresolved_ambiguity" in reasons
    assert "failure_rate_spike" in reasons


def test_ambiguous_target_during_active_goal_asks_clarification_once():
    dispatcher = TinyAutonomyDispatcher()
    ask_history = {}

    event = {
        "severity": "medium",
        "reason": "active_goals_with_unresolved_ambiguity",
        "recommended_action": "ask_clarification_then_replan",
    }
    goal_summary = {
        "goals": [
            {
                "goal_id": "goal-123",
                "title": "Ship release checklist",
            }
        ]
    }

    first = dispatcher.dispatch(
        event=event,
        goal_summary=goal_summary,
        active_goal_id="goal-123",
        ask_history=ask_history,
    )
    second = dispatcher.dispatch(
        event=event,
        goal_summary=goal_summary,
        active_goal_id="goal-123",
        ask_history=ask_history,
    )

    assert first.outcome == OUTCOME_ASK_USER
    assert first.next_goal_action == "clarify"
    assert first.affected_goal_id == "goal-123"
    assert first.should_notify_user is True
    assert second.should_notify_user is False


def test_repeated_execution_failures_pause_autonomy_and_escalate():
    dispatcher = TinyAutonomyDispatcher()

    event = {
        "severity": "high",
        "reason": "failure_rate_spike",
        "recommended_action": "pause_autonomy_and_escalate",
    }
    decision = dispatcher.dispatch(event=event, goal_summary={"goals": []}, active_goal_id="goal-ops")

    assert decision.outcome == OUTCOME_ESCALATE_AND_STOP
    assert decision.pause_autonomy is True
    assert decision.escalate is True
    assert decision.next_goal_action == "pause"


def test_pending_approval_trigger_requests_operator_decision():
    dispatcher = TinyAutonomyDispatcher()

    event = {
        "severity": "medium",
        "reason": "pending_approvals_waiting",
        "recommended_action": "request_operator_decision",
    }
    decision = dispatcher.dispatch(
        event=event,
        goal_summary={"goals": [{"goal_id": "goal-write"}]},
        active_goal_id="goal-write",
        ask_history={},
    )

    assert decision.outcome == OUTCOME_ASK_USER
    assert decision.should_notify_user is True
    assert "operator" in decision.operator_message.lower()
    assert decision.affected_goal_id == "goal-write"


def test_dispatcher_decision_dict_exposes_standard_contract_fields():
    dispatcher = TinyAutonomyDispatcher()
    decision = dispatcher.dispatch(
        event={
            "severity": "medium",
            "reason": "active_goals_with_unresolved_ambiguity",
            "recommended_action": "ask_clarification_then_replan",
            "confidence": 0.73,
        },
        goal_summary={"goals": [{"goal_id": "goal-clarify"}]},
        active_goal_id="goal-clarify",
        ask_history={},
    )

    payload = decision.to_dict()

    assert payload.get("trigger") == "active_goals_with_unresolved_ambiguity"
    assert payload.get("severity") == "medium"
    assert payload.get("affected_goal") == "goal-clarify"
    assert payload.get("recommended_action") == "ask_clarification_then_replan"
    assert payload.get("runtime_outcome") in {"ask_user", "act_automatically", "log_silently", "escalate_and_stop"}
    assert isinstance(payload.get("dedupe_key"), str)
    assert payload.get("confidence") == 0.73


def test_dispatcher_medium_dedupe_key_respects_recommended_action_variant():
    dispatcher = TinyAutonomyDispatcher()
    ask_history = {}

    first = dispatcher.dispatch(
        event={
            "severity": "medium",
            "reason": "active_goals_with_unresolved_ambiguity",
            "recommended_action": "ask_clarification_then_replan",
        },
        goal_summary={"goals": [{"goal_id": "goal-123"}]},
        active_goal_id="goal-123",
        ask_history=ask_history,
    )
    second = dispatcher.dispatch(
        event={
            "severity": "medium",
            "reason": "active_goals_with_unresolved_ambiguity",
            "recommended_action": "ask_operator_for_target",
        },
        goal_summary={"goals": [{"goal_id": "goal-123"}]},
        active_goal_id="goal-123",
        ask_history=ask_history,
    )

    assert first.should_notify_user is True
    assert second.should_notify_user is True


def test_goal_verification_clarify_trigger_asks_user():
    dispatcher = TinyAutonomyDispatcher()

    decision = dispatcher.dispatch(
        event={
            "severity": "medium",
            "reason": "goal_verification_clarify_required",
            "recommended_action": "ask_clarification_then_replan",
        },
        goal_summary={"goals": [{"goal_id": "goal-verify"}]},
        active_goal_id="goal-verify",
        ask_history={},
    )

    assert decision.outcome == OUTCOME_ASK_USER
    assert decision.should_notify_user is True
    assert decision.next_goal_action == "clarify"


def test_goal_verification_retry_trigger_continues_automatically():
    dispatcher = TinyAutonomyDispatcher()

    decision = dispatcher.dispatch(
        event={
            "severity": "medium",
            "reason": "goal_verification_retry_recommended",
            "recommended_action": "retry_with_adjusted_target",
        },
        goal_summary={"goals": [{"goal_id": "goal-retry"}]},
        active_goal_id="goal-retry",
        ask_history={},
    )

    assert decision.outcome == OUTCOME_ACT_AUTOMATICALLY
    assert decision.pause_autonomy is False
    assert decision.next_goal_action == "retry"


def test_partial_side_effect_with_rollback_policy_reports_outcome():
    plugin_manager = _StubPluginManager()
    engine = UnifiedActionEngine(tool_executor=_StubToolExecutor())
    engine.bind_plugin_manager(plugin_manager)

    req = ActionRequest(
        goal="create roadmap note",
        plugin="notes",
        action="create",
        params={"title": "Roadmap", "_raw_query": "create roadmap note"},
        source_intent="notes:create",
        confidence=0.8,
        rollback_policy="auto",
        target_spec={"target": "Roadmap"},
    )

    res = engine.execute(req)
    rollback = (res.state_updates or {}).get("rollback") or {}

    assert rollback.get("attempted") is True
    assert rollback.get("status") == "rollback_success"
    assert rollback.get("success") is True


def test_successful_autonomous_recovery_continues_goal_flow_without_user_reprompt():
    plugin_manager = _StubPluginManager()
    engine = UnifiedActionEngine(tool_executor=_StubToolExecutor())
    engine.bind_plugin_manager(plugin_manager)

    req = ActionRequest(
        goal="repair note creation after partial side-effect",
        plugin="notes",
        action="create",
        params={"title": "Roadmap", "_raw_query": "create roadmap note"},
        source_intent="notes:create",
        confidence=0.8,
        risk_level="low",
        retry_budget=0,
        rollback_policy="auto",
        target_spec={"target": "Roadmap"},
    )

    res = engine.execute(req)
    rollback = (res.state_updates or {}).get("rollback") or {}
    dispatcher = TinyAutonomyDispatcher()
    decision = dispatcher.dispatch(
        event={
            "severity": "low",
            "reason": "rollback_success",
            "recommended_action": "continue_goal_flow",
        },
        goal_summary={"goals": [{"goal_id": "goal-commute"}]},
        active_goal_id="goal-commute",
    )

    assert rollback.get("status") == "rollback_success"
    assert rollback.get("success") is True
    assert decision.outcome == OUTCOME_ACT_AUTOMATICALLY
    assert decision.next_goal_action == "continue"
    assert decision.should_notify_user is False
