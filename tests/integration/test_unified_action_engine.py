from ai.core.unified_action_engine import ActionRequest, UnifiedActionEngine


class _StubToolExecutor:
    def __init__(self, *, handled=True, verified=True, result=None):
        self.handled = handled
        self.verified = verified
        self.result = result or {}
        self.last_call = None

    def execute(self, *, plugin_manager, intent, query, entities, context):
        self.last_call = {
            "plugin_manager": plugin_manager,
            "intent": intent,
            "query": query,
            "entities": entities,
            "context": context,
        }

        class _Outcome:
            def __init__(self, handled, verified, result):
                self.handled = handled
                self.verified = verified
                self.result = result

        return _Outcome(self.handled, self.verified, self.result)


def test_unified_action_engine_uses_source_intent_and_raw_query():
    stub = _StubToolExecutor(
        handled=True,
        verified=True,
        result={
            "success": True,
            "status": "success",
            "plugin": "notes",
            "action": "create",
            "data": {"note_id": "n1"},
            "message": "Done",
            "confidence": 0.9,
            "retryable": False,
            "side_effects": ["created"],
            "verification": {"accepted": True, "confidence": 0.9, "issues": []},
        },
    )
    engine = UnifiedActionEngine(tool_executor=stub)
    engine.bind_plugin_manager(object())

    req = ActionRequest(
        goal="create project note",
        plugin="notes",
        action="create",
        params={"title": "Roadmap", "_raw_query": "create a note titled Roadmap"},
        source_intent="notes:create",
        confidence=0.88,
        requires_confirmation=False,
    )

    res = engine.execute(req)

    assert stub.last_call is not None
    assert stub.last_call["intent"] == "notes:create"
    assert stub.last_call["query"] == "create a note titled Roadmap"
    assert res.success is True
    assert res.goal_satisfied is True
    assert res.state_updates["goal_satisfied"] is True


def test_unified_action_engine_confirmation_blocks_execution():
    stub = _StubToolExecutor()
    engine = UnifiedActionEngine(tool_executor=stub)
    engine.bind_plugin_manager(object())

    req = ActionRequest(
        goal="delete that file",
        plugin="file",
        action="delete",
        params={"target": "draft.txt"},
        source_intent="file:delete",
        confidence=0.7,
        requires_confirmation=True,
    )

    res = engine.execute(req)

    assert res.success is False
    assert res.status == "requires_confirmation"
    assert "requires explicit confirmation" in (res.error or "")
    assert stub.last_call is None
