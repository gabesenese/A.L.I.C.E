from ai.roadmap import get_roadmap_completion_stack


def test_route_contract_confidence_bands():
    stack = get_roadmap_completion_stack()

    ok1, _ = stack.route_contracts.validate(route="tool", confidence=0.8)
    ok2, reason2 = stack.route_contracts.validate(route="tool", confidence=0.1)
    ok3, reason3 = stack.route_contracts.validate(route="clarify", confidence=0.9)

    assert ok1 is True
    assert ok2 is False
    assert "low_confidence" in reason2
    assert ok3 is False
    assert "high_confidence" in reason3


def test_recovery_improvement_and_failure_clustering():
    stack = get_roadmap_completion_stack()

    clustered = stack.failure_clusterer.cluster(
        [
            {"signature": "timeout"},
            {"signature": "timeout"},
            {"signature": "parse_error"},
        ]
    )
    assert clustered["timeout"] == 2

    result = stack.improvement_engine.run(
        apply_fn=lambda: "applied",
        measure_fn=lambda: 0.2,
        rollback_fn=lambda: "rolled_back",
        min_gain=0.5,
    )
    assert result["rolled_back"] is True

    snippet = stack.regression_generator.generate_test_snippet(
        {"signature": "timeout bug", "error": "timeout"}
    )
    assert "test_generated_" in snippet
