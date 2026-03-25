from ai.roadmap import get_roadmap_completion_stack


def test_completion_stack_core_features(tmp_path):
    stack = get_roadmap_completion_stack()

    budget = stack.context_windows.choose_budget("debug")
    assert budget >= 20

    compressed = stack.context_windows.compress_incremental(
        [{"role": "user", "content": str(i)} for i in range(50)], budget=10
    )
    assert len(compressed) <= 11

    ok, reason = stack.route_contracts.validate(route="tool", confidence=0.2)
    assert ok is False
    assert "low_confidence" in reason

    safety_ok, _ = stack.secondary_safety.validate("delete all files", has_approval=False)
    assert safety_ok is False


def test_task_lifecycle_and_replanning():
    stack = get_roadmap_completion_stack()
    task = stack.lifecycle.tasks.get("demo")
    if task is None:
        from ai.roadmap.completion_stack import ManagedTask

        stack.lifecycle.add_task(ManagedTask(task_id="demo", domain="ops", payload={}))

    next_task = stack.lifecycle.arbitrate_next()
    assert next_task is not None

    replanned = stack.replanner.replan(["step1", "step2"], "step2", "timeout")
    assert any(s.startswith("fallback:") for s in replanned)


def test_schema_dependency_and_ab_tools(tmp_path):
    stack = get_roadmap_completion_stack()

    normalized = stack.schema.normalize({"success": True, "confidence": 0.8})
    assert normalized.status == "ok"
    assert normalized.confidence == 0.8

    req_file = tmp_path / "requirements.txt"
    req_file.write_text("requests\npytest==8.0.0\nrequests==2.0.0\n", encoding="utf-8")
    report = stack.dependency_health.analyze_requirements(str(req_file))
    assert report["ok"] is True
    assert report["duplicates"]

    ab = stack.ab_validation.compare([0.4, 0.6], [0.7, 0.8])
    assert ab["winner"] in {"A", "B"}


def test_tamper_evident_audit_log(tmp_path):
    from ai.roadmap.completion_stack import TamperEvidentAuditLog

    log = TamperEvidentAuditLog(str(tmp_path / "audit.jsonl"))
    first = log.append({"action": "start"})
    second = log.append({"action": "next"})
    assert second["prev_hash"] == first["hash"]
