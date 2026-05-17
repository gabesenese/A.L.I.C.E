from ai.contracts import RouterDecision
from ai.runtime.companion_runtime import (
    CompanionRuntimeLoop,
    IdentityModel,
    CompanionState,
    PolicyDecision,
)
from memory.world_model import WorldModel


def test_world_model_persists_required_shape_and_turn_extraction(tmp_path):
    path = tmp_path / "world_model.json"
    model = WorldModel(path)

    snapshot = model.update_from_turn(
        user_input="I want to finish the routing refactor. I should write tests.",
        response_text="That makes sense.",
        route="llm",
        intent="conversation:goal_statement",
        requires_follow_up=True,
        timestamp="2026-05-17T10:00:00+00:00",
    )

    assert snapshot["user"]["mentioned_intentions"]
    assert snapshot["environment"]["open_tasks"]
    assert snapshot["alice_state"]["active_threads"]
    assert snapshot["alice_state"]["personality"]["curiosity_weight"] == 0.7

    reloaded = WorldModel(path).snapshot()
    assert reloaded["user"]["mentioned_intentions"][0]["text"].startswith("finish")
    assert reloaded["user"]["mood_signal"] == "neutral"


def test_world_model_tracks_mood_and_resolves_oldest_open_task(tmp_path):
    model = WorldModel(tmp_path / "world_model.json")
    model.update_from_turn(
        user_input="I need to review the heartbeat tomorrow. I'm exhausted.",
        response_text="We'll keep it focused.",
        timestamp="2026-05-17T10:00:00+00:00",
    )

    before = model.snapshot()
    assert before["user"]["mood_signal"] == "stressed"
    assert len(before["environment"]["open_tasks"]) == 1

    after = model.update_from_turn(
        user_input="done",
        response_text="Recorded.",
        tool_success=True,
        timestamp="2026-05-17T11:00:00+00:00",
    )
    assert after["environment"]["open_tasks"] == []


def test_companion_runtime_updates_world_model_after_turn(tmp_path):
    model = WorldModel(tmp_path / "world_model.json")
    loop = CompanionRuntimeLoop(world_model=model)
    state = CompanionState(identity_model=IdentityModel(user_id="gabriel"))

    domains = loop.update_after_turn(
        companion_state=state,
        user_input="I want to build the proactive companion layer.",
        response_text="I'll track that thread.",
        route_decision=RouterDecision(
            route="llm",
            intent="conversation:goal_statement",
            confidence=0.9,
            decision_band="execute",
        ),
        policy=PolicyDecision(decision_type="respond", reason="test"),
        verification=None,
        requires_follow_up=True,
        follow_up_question="Want me to keep this active?",
        tool_result=None,
        action_discipline={},
    )

    snapshot = model.snapshot()
    assert snapshot["user"]["mentioned_intentions"]
    assert snapshot["environment"]["open_tasks"]
    assert domains["identity"]["world_model_updated_at"]
    assert domains["preferences"]["personality"]["directness"] == 0.6
