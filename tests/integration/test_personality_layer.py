import pytest

from ai.contracts import RouterDecision
from ai.runtime.companion_runtime import (
    CompanionRuntimeLoop,
    CompanionState,
    IdentityModel,
    PolicyDecision,
)
from brain.personality import (
    PersonalityLayer,
    apply_personality_to_system_prompt,
    personality_to_system_instructions,
)
from memory.world_model import WorldModel


def test_short_response_streak_reduces_curiosity_without_extremes(tmp_path):
    model = WorldModel(tmp_path / "world_model.json")
    layer = PersonalityLayer(world_model=model)

    layer.update_after_turn(user_input="ok")
    personality = layer.update_after_turn(user_input="yes")

    assert personality["curiosity_weight"] < 0.7
    assert personality["curiosity_weight"] >= 0.1
    assert model.get_personality_meta()["short_response_streak"] == 2


def test_warm_engagement_increases_humor_threshold(tmp_path):
    model = WorldModel(tmp_path / "world_model.json")
    layer = PersonalityLayer(world_model=model)

    personality = layer.update_after_turn(user_input="thanks, that was great")

    assert personality["humor_threshold"] > 0.5
    assert personality["humor_threshold"] <= 0.9


def test_interests_grow_after_topic_appears_three_times(tmp_path):
    model = WorldModel(tmp_path / "world_model.json")
    layer = PersonalityLayer(world_model=model)

    for _ in range(3):
        layer.update_after_turn(user_input="Let's work on heartbeat runtime")

    assert "heartbeat" in model.get_personality()["interests"]
    assert "runtime" in model.get_personality()["interests"]


def test_the_drift_layer_appends_facts_not_adjectives(tmp_path):
    """It used to emit "Current ALICE personality drift:" and then dials —
    directness, humor, concern sensitivity — appended after the persona's worked
    exchanges, where on an 8B the last positive instruction wins. What survives is
    the part that is a fact about the user rather than a tone knob aimed at Alice.
    """
    model = WorldModel(tmp_path / "world_model.json")
    model.update_personality(
        {
            "curiosity_weight": 0.2,
            "directness": 0.8,
            "humor_threshold": 0.8,
            "concern_sensitivity": 0.7,
            "interests": ["heartbeat"],
        }
    )

    prompt = apply_personality_to_system_prompt("Base prompt", world_model=model)

    assert prompt.startswith("Base prompt")
    assert "heartbeat" in prompt
    assert "personality drift" not in prompt.lower()
    for adjective in ("concise", "humor", "wit", "direct", "follow-up"):
        assert adjective not in prompt.lower(), f"{adjective!r} overrides the persona from recency position"


def test_the_dials_are_still_learned_even_though_they_no_longer_shape_prose(tmp_path):
    """Reviving them means a behavioural lever, not a longer string of adjectives,
    so the learned values have to survive the change that stopped rendering them."""
    model = WorldModel(tmp_path / "world_model.json")
    model.update_personality({"directness": 0.8, "humor_threshold": 0.8})

    stored = model.get_personality()
    assert stored["directness"] == pytest.approx(0.8)
    assert stored["humor_threshold"] == pytest.approx(0.8)


def test_companion_runtime_updates_personality_after_turn(tmp_path):
    model = WorldModel(tmp_path / "world_model.json")
    loop = CompanionRuntimeLoop(world_model=model)
    state = CompanionState(identity_model=IdentityModel(user_id="gabriel"))

    loop.update_after_turn(
        companion_state=state,
        user_input="ok",
        response_text="Recorded.",
        route_decision=RouterDecision(
            route="llm",
            intent="conversation:ack",
            confidence=0.9,
            decision_band="execute",
        ),
        policy=PolicyDecision(decision_type="respond", reason="test"),
        verification=None,
        requires_follow_up=False,
        follow_up_question="",
        tool_result=None,
        action_discipline={},
    )
    domains = loop.update_after_turn(
        companion_state=state,
        user_input="yes",
        response_text="Recorded.",
        route_decision=RouterDecision(
            route="llm",
            intent="conversation:ack",
            confidence=0.9,
            decision_band="execute",
        ),
        policy=PolicyDecision(decision_type="respond", reason="test"),
        verification=None,
        requires_follow_up=False,
        follow_up_question="",
        tool_result=None,
        action_discipline={},
    )

    assert domains["preferences"]["personality"]["curiosity_weight"] < 0.7
    assert model.get_personality_meta()["short_response_streak"] == 2


def test_a_bad_payload_adds_nothing_rather_than_raising():
    """Out-of-range dials and a non-numeric one used to be clamped and described.
    Nothing describes them now, so the only contract left is that a malformed
    personality cannot take down the turn that was building a system prompt."""
    instructions = personality_to_system_instructions(
        {
            "curiosity_weight": 10,
            "directness": -5,
            "humor_threshold": "bad",
            "concern_sensitivity": 0.6,
        }
    )

    assert instructions == ""


def test_interests_are_the_one_thing_the_layer_still_contributes():
    assert personality_to_system_instructions({"interests": ["heartbeat", "embeddings"]}) == (
        "He has been working on: heartbeat, embeddings."
    )
