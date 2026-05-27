from ai.core.turn_state_assembler import TurnStateAssembler


class _WorldMemoryStub:
    def __init__(self, state):
        self._state = dict(state or {})
        self.memory = [
            {"message": "event one"},
            {"message": "event two"},
            {"message": "event three"},
        ]

    def snapshot(self):
        return dict(self._state)


def test_turn_state_assembler_builds_goal_stack_and_continuation_state():
    world = _WorldMemoryStub(
        {
            "active_goals": [
                {
                    "goal_id": "g2",
                    "title": "Ship parser",
                    "status": "active",
                    "next_action": "write tests",
                    "blockers": ["failing tokenizer"],
                }
            ],
            "selected_object_reference": "conversation_flow",
            "last_action": {
                "plugin": "notes",
                "action": "update",
                "status": "success",
                "success": True,
            },
        }
    )
    assembler = TurnStateAssembler(world)

    snapshot = assembler.build(
        user_input="continue",
        intent="conversation:question",
        action_context={
            "goal": {
                "goal_id": "g1",
                "title": "Improve executive kernel",
                "status": "active",
                "next_action": "reduce routing knobs",
            },
            "pending_slot": {
                "slot_type": "topic_branch",
                "parent_request": "teach me nlp",
                "parent_intent": "conversation:question",
            },
        },
    )

    goal_state = snapshot.get("goal_state", {})
    continuation = snapshot.get("continuation_state", {})

    assert goal_state.get("active_goal_count", 0) >= 2
    assert goal_state.get("priority_goal", {}).get("goal_id") == "g1"
    assert continuation.get("mode") == "clarification_followup"
    assert continuation.get("parent_context", {}).get("parent_request") == "teach me nlp"


def test_turn_state_summary_includes_authoritative_sections():
    world = _WorldMemoryStub({"active_goals": [{"goal_id": "g1", "title": "Do thing"}]})
    assembler = TurnStateAssembler(world)

    snapshot = assembler.build(
        user_input="status",
        intent="conversation:general",
        action_context={"confidence": 0.8},
    )
    summary = assembler.to_text(snapshot)

    assert "Goal stack:" in summary
    assert "Continuation mode:" in summary
    assert "Last action:" in summary
    assert "Focus:" in summary
