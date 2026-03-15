from ai.core.reflection_engine import ReflectionEngine
from ai.core.executive_controller import ExecutiveController


def test_reflection_produces_routing_adjustments() -> None:
    engine = ReflectionEngine()

    result = engine.reflect(
        user_input="What is polymorphism?",
        intent="learning:study_topic",
        response="Polymorphism allows one interface to support multiple implementations.",
        route="llm",
        gate_accepted=True,
        decision_scores={"llm": 0.82, "tools": 0.25, "clarify": 0.12},
        prior_confidence=0.80,
    ).as_dict()

    assert result["success_score"] >= 0.5
    assert isinstance(result["routing_adjustments"], dict)


def test_executive_weights_update_from_reflection() -> None:
    controller = ExecutiveController()
    before = controller.get_routing_weights()

    controller.apply_reflection(
        {
            "routing_adjustments": {
                "llm": -0.05,
                "clarify": 0.03,
            }
        }
    )

    after = controller.get_routing_weights()
    assert after["llm"] < before["llm"]
    assert after["clarify"] > before["clarify"]
