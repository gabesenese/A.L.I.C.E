from ai.core.unified_action_engine import ActionRequest, UnifiedActionEngine


class _StubToolExecutor:
    def __init__(
        self, *, handled=True, verified=True, result=None, result_sequence=None
    ):
        self.handled = handled
        self.verified = verified
        self.result = result or {}
        self.result_sequence = list(result_sequence or [])
        self.last_call = None
        self.call_count = 0

    def execute(self, *, plugin_manager, intent, query, entities, context):
        self.call_count += 1
        self.last_call = {
            "plugin_manager": plugin_manager,
            "intent": intent,
            "query": query,
            "entities": entities,
            "context": context,
        }

        current_result = (
            self.result_sequence.pop(0) if self.result_sequence else self.result
        )

        class _Outcome:
            def __init__(self, handled, verified, result):
                self.handled = handled
                self.verified = verified
                self.result = result

        return _Outcome(self.handled, self.verified, current_result)


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


def test_unified_action_engine_blocks_high_risk_without_approval():
    stub = _StubToolExecutor()
    engine = UnifiedActionEngine(tool_executor=stub)
    engine.bind_plugin_manager(object())

    req = ActionRequest(
        goal="delete production config",
        plugin="file",
        action="delete",
        params={"target": "prod.env", "_raw_query": "delete prod env"},
        source_intent="file:delete",
        confidence=0.6,
        requires_confirmation=False,
        risk_level="high",
    )

    res = engine.execute(req)

    assert res.success is False
    assert res.status == "policy_blocked"
    assert "policy_confirmation_required" in res.ambiguity_flags
    assert stub.last_call is None


def test_unified_action_engine_retries_low_risk_retryable_failure_then_succeeds():
    stub = _StubToolExecutor(
        result_sequence=[
            {
                "success": False,
                "status": "failed",
                "plugin": "notes",
                "action": "read",
                "data": {"target": "Roadmap"},
                "message": "Temporary error",
                "confidence": 0.3,
                "retryable": True,
                "side_effects": [],
                "verification": {"accepted": True, "confidence": 0.8, "issues": []},
            },
            {
                "success": True,
                "status": "success",
                "plugin": "notes",
                "action": "read",
                "data": {"note_id": "n1", "target": "Roadmap"},
                "message": "Read complete",
                "confidence": 0.9,
                "retryable": False,
                "side_effects": ["read"],
                "verification": {"accepted": True, "confidence": 0.9, "issues": []},
            },
        ]
    )
    engine = UnifiedActionEngine(tool_executor=stub)
    engine.bind_plugin_manager(object())

    req = ActionRequest(
        goal="read roadmap note",
        plugin="notes",
        action="read",
        params={"target": "Roadmap", "_raw_query": "read roadmap note"},
        source_intent="notes:read",
        confidence=0.85,
        requires_confirmation=False,
        risk_level="low",
        retry_budget=1,
        target_spec={"target": "Roadmap"},
    )

    res = engine.execute(req)

    assert stub.call_count == 2
    assert res.success is True
    assert res.goal_satisfied is True
    assert res.retry_count == 1


def test_unified_action_engine_detects_target_mismatch():
    stub = _StubToolExecutor(
        result={
            "success": True,
            "status": "success",
            "plugin": "notes",
            "action": "read",
            "data": {"target": "DifferentNote"},
            "message": "Read complete",
            "confidence": 0.9,
            "retryable": False,
            "side_effects": ["read"],
            "verification": {"accepted": True, "confidence": 0.9, "issues": []},
        }
    )
    engine = UnifiedActionEngine(tool_executor=stub)
    engine.bind_plugin_manager(object())

    req = ActionRequest(
        goal="read roadmap note",
        plugin="notes",
        action="read",
        params={"target": "Roadmap", "_raw_query": "read roadmap note"},
        source_intent="notes:read",
        confidence=0.9,
        requires_confirmation=False,
        target_spec={"target": "Roadmap"},
    )

    res = engine.execute(req)

    assert res.success is True
    assert res.goal_satisfied is False
    assert "target_mismatch" in res.ambiguity_flags


def test_unified_action_engine_blocks_medium_risk_when_simulation_fails():
    stub = _StubToolExecutor(
        result={
            "success": True,
            "status": "success",
            "plugin": "notes",
            "action": "update",
            "data": {"note_id": "n1", "target": "Roadmap"},
            "message": "Updated",
            "confidence": 0.9,
            "retryable": False,
            "side_effects": ["updated"],
            "verification": {"accepted": True, "confidence": 0.9, "issues": []},
        }
    )
    engine = UnifiedActionEngine(tool_executor=stub)
    engine.bind_plugin_manager(object())
    engine.bind_simulation_callback(
        lambda candidates, context=None: {
            "best_action": {
                "action_id": "notes:update",
                "score": 0.0,
                "risk": 0.7,
                "confidence": 0.2,
            },
            "ranked": [],
            "context": context or {},
        }
    )

    req = ActionRequest(
        goal="update roadmap note",
        plugin="notes",
        action="update",
        params={"target": "Roadmap", "_raw_query": "update roadmap note"},
        source_intent="notes:update",
        confidence=0.4,
        requires_confirmation=False,
        risk_level="medium",
    )

    res = engine.execute(req)

    assert res.success is False
    assert res.status == "simulation_blocked"
    assert stub.call_count == 0


def test_unified_action_engine_attaches_turn_diff_on_success():
    stub = _StubToolExecutor(
        result={
            "success": True,
            "status": "success",
            "plugin": "notes",
            "action": "create",
            "data": {"note_id": "n2", "title": "Plan"},
            "message": "Created",
            "confidence": 0.95,
            "retryable": False,
            "side_effects": ["created"],
            "verification": {"accepted": True, "confidence": 0.95, "issues": []},
        }
    )
    engine = UnifiedActionEngine(tool_executor=stub)
    engine.bind_plugin_manager(object())

    req = ActionRequest(
        goal="create plan note",
        plugin="notes",
        action="create",
        params={"title": "Plan", "_raw_query": "create plan note"},
        source_intent="notes:create",
        confidence=0.9,
        requires_confirmation=False,
        risk_level="low",
    )

    res = engine.execute(req)
    turn_diff = (res.state_updates or {}).get("turn_diff", {})

    assert res.success is True
    assert isinstance(turn_diff, dict)
    assert turn_diff.get("event") == "action_execution"


def test_unified_action_engine_normalizes_goal_ref_into_target_spec():
    stub = _StubToolExecutor(
        result={
            "success": True,
            "status": "success",
            "plugin": "notes",
            "action": "create",
            "data": {"note_id": "n3"},
            "message": "Created",
            "confidence": 0.9,
            "retryable": False,
            "side_effects": ["created"],
            "verification": {"accepted": True, "confidence": 0.9, "issues": []},
        }
    )
    engine = UnifiedActionEngine(tool_executor=stub)
    engine.bind_plugin_manager(object())

    req = ActionRequest(
        goal="",
        plugin="notes",
        action="create",
        params={"title": "Kernel Plan", "_raw_query": "create note kernel plan"},
        source_intent="notes:create",
        confidence=0.8,
        goal_ref={
            "goal_id": "goal_exec_01",
            "title": "Rebuild executive kernel",
            "status": "active",
        },
    )

    res = engine.execute(req)

    assert res.success is True
    assert stub.last_call is not None
    target_spec = (stub.last_call.get("context") or {}).get("target_spec", {})
    assert target_spec.get("goal_id") == "goal_exec_01"
    assert target_spec.get("goal_status") == "active"
