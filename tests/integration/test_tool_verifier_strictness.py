from ai.core.tool_verifier import ToolResultVerifier


def test_verifier_rejects_success_with_missing_expected_data():
    verifier = ToolResultVerifier()
    res = verifier.verify(
        intent="weather:current",
        user_input="show weather",
        plugin_result={
            "plugin": "WeatherPlugin",
            "action": "get_current",
            "success": True,
            "response": "done",
            "data": {},
        },
    )
    assert res.accepted is False
    assert res.goal_satisfied is False


def test_verifier_rejects_contradictory_success():
    verifier = ToolResultVerifier()
    res = verifier.verify(
        intent="notes:list",
        user_input="list notes",
        plugin_result={
            "plugin": "NotesPlugin",
            "action": "list_notes",
            "success": True,
            "response": "Operation failed due to timeout",
            "data": {"notes": []},
        },
    )
    assert res.accepted is False
    assert any("contradicts" in issue for issue in res.issues)


def test_verifier_accepts_strong_structured_result():
    verifier = ToolResultVerifier()
    res = verifier.verify(
        intent="notes:list",
        user_input="list my notes",
        plugin_result={
            "plugin": "NotesPlugin",
            "action": "list_notes",
            "success": True,
            "response": "Here are your notes",
            "data": {"notes": [{"title": "A"}], "count": 1},
        },
    )
    assert res.accepted is True
    assert res.goal_satisfied is True


def test_verifier_rejects_status_success_contradiction():
    verifier = ToolResultVerifier()
    res = verifier.verify(
        intent="notes:list",
        user_input="list notes",
        plugin_result={
            "plugin": "NotesPlugin",
            "action": "list_notes",
            "success": True,
            "status": "error",
            "response": "Here are your notes",
            "data": {"notes": [{"title": "A"}], "count": 1},
        },
    )
    assert res.accepted is False
    assert any("status=error" in issue for issue in res.issues)


def test_verifier_rejects_high_impact_without_structured_data():
    verifier = ToolResultVerifier()
    res = verifier.verify(
        intent="system:execute",
        user_input="run shell command",
        plugin_result={
            "plugin": "SystemControlPlugin",
            "action": "run",
            "success": True,
            "response": "done",
            "data": None,
        },
    )
    assert res.accepted is False
    assert any("high-impact action" in issue for issue in res.issues)


def test_verifier_rejects_execution_context_mismatch():
    verifier = ToolResultVerifier()
    res = verifier.verify(
        intent="notes:list",
        user_input="list notes",
        plugin_result={
            "plugin": "NotesPlugin",
            "action": "list_notes",
            "success": True,
            "response": "done",
            "data": {"notes": [{"title": "A"}], "count": 1},
        },
        execution_context={
            "expected_plugin": "WeatherPlugin",
            "expected_action": "get_current",
        },
    )
    assert res.accepted is False
    assert any("execution expectation" in issue for issue in res.issues)
