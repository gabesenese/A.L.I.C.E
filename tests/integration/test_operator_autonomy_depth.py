from pathlib import Path

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
