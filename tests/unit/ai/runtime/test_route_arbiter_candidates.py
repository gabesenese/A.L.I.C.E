from ai.core.routing.route_arbiter import RouteArbiter, RouteCandidate


def test_arbiter_rejects_unsafe_file_tool_and_selects_safe_local_candidate():
    arbiter = RouteArbiter()
    out = arbiter.arbitrate_candidates(
        user_input="can you read alice files?",
        active_mode="",
        candidates=[
            RouteCandidate(route="tool", intent="file_operations:read", confidence=0.95, source="tool"),
            RouteCandidate(route="local", intent="code:request", confidence=0.90, source="local"),
        ],
    )
    assert out["route"] == "local"
    assert out["intent"] == "code:request"
    trace = dict(out["trace"] or {})
    assert isinstance(trace.get("rejected_candidates"), list)

