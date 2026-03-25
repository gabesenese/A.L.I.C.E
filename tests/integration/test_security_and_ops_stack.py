from ai.roadmap import get_roadmap_completion_stack


def test_network_guard_and_rate_limiter():
    stack = get_roadmap_completion_stack()

    ok_tls, _ = stack.network_guard.validate_url("https://example.com/a")
    bad_tls, reason = stack.network_guard.validate_url("http://example.com/a")

    assert ok_tls is True
    assert bad_tls is False
    assert reason == "tls_required"

    limiter = stack.rate_limiter
    decisions = [limiter.allow(cost=1.0) for _ in range(5)]
    assert any(decisions)


def test_capability_acquisition_and_benchmark():
    stack = get_roadmap_completion_stack()

    cap = stack.capability_acquisition.register_candidate("new_skill", sandboxed=True, approved=True)
    assert cap["enabled"] is True

    bench = stack.benchmark_harness.run([
        {"accuracy": 0.8, "latency_ms": 100, "success": True},
        {"accuracy": 0.6, "latency_ms": 120, "success": False},
    ])
    assert bench["accuracy"] > 0.0
    assert bench["latency_ms"] > 0.0
