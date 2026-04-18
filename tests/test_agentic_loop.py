from ai.core.agentic_loop import AgenticLoop


def test_agentic_loop_runs_all_stages_and_updates_memory():
    loop = AgenticLoop(
        perceive_fn=lambda payload: {"observation": payload.get("text", "")},
        reason_fn=lambda state: {"summary": f"seen:{state['perceived'].get('observation')}"},
        goal_fn=lambda state: {"goal": "answer_user"},
        decide_fn=lambda state: {"action": "respond", "confidence": 0.9},
        execute_fn=lambda decision: {"status": "ok", "action": decision.get("action")},
        learn_fn=lambda feedback: {"adapted": feedback["execution"].get("status") == "ok"},
    )

    report = loop.run_cycle({"text": "hello"})
    memory = loop.snapshot()

    assert report.perceived["observation"] == "hello"
    assert report.reasoning["summary"].startswith("seen:")
    assert report.goals["goal"] == "answer_user"
    assert report.decision["action"] == "respond"
    assert report.execution["status"] == "ok"
    assert report.learning["adapted"] is True
    assert memory["last_goal"] == "answer_user"
    assert memory["last_action"] == "respond"
    assert memory["cycles"] == 1


def test_agentic_loop_defaults_are_safe_when_callbacks_missing():
    loop = AgenticLoop()
    report = loop.run_cycle({"text": "status"})

    assert report.reasoning["summary"] == "no_reasoner"
    assert report.goals["goal"] == "maintain_stability"
    assert report.decision["action"] == "noop"
    assert report.execution["status"] == "skipped"
