"""Unit tests for the tool chaining engine."""

from ai.runtime.tool_chain import detect_chain, execute_chain


def test_detects_compound_calendar_email_query():
    intents = detect_chain("what's my schedule and any urgent emails?", "calendar:events")
    assert len(intents) >= 2
    assert "calendar:events" in intents
    assert "email:check" in intents


def test_detects_compound_system_goals_query():
    intents = detect_chain("system status and current goals", "system:status")
    assert len(intents) >= 2
    assert "system:status" in intents
    assert "goals:list" in intents


def test_single_domain_not_chained():
    intents = detect_chain("what's my schedule today?", "calendar:events")
    assert intents == []


def test_two_domains_without_connector_not_chained():
    intents = detect_chain("schedule emails", "calendar:events")
    assert intents == []


def test_comma_separated_triggers_chain():
    intents = detect_chain("show me my schedule, emails", "calendar:events")
    assert len(intents) >= 2


def test_execute_chain_with_failing_plugins_returns_partial():
    class FakePlugin:
        name = "Fake"
        enabled = True
        def can_handle(self, intent, entities, query):
            return True
        def execute(self, intent, query, entities, context):
            if intent == "calendar:events":
                return {"success": True, "response": "Sprint planning at 10:00"}
            return {"success": False, "response": ""}

    class FakePluginManager:
        plugins = {"Fake": FakePlugin()}

    result = execute_chain(
        intents=["calendar:events", "email:check"],
        query="schedule and emails",
        entities={},
        context={},
        plugin_manager=FakePluginManager(),
    )
    assert result is not None
    assert "Sprint planning" in result
    assert "Could not retrieve" in result  # email failed


def test_execute_chain_all_fail_returns_none():
    class FailPlugin:
        name = "Fail"
        enabled = True
        def can_handle(self, intent, entities, query):
            return True
        def execute(self, intent, query, entities, context):
            return {"success": False, "response": ""}

    class FakePluginManager:
        plugins = {"Fail": FailPlugin()}

    result = execute_chain(
        intents=["calendar:events", "email:check"],
        query="schedule and emails",
        entities={},
        context={},
        plugin_manager=FakePluginManager(),
    )
    assert result is None
