from ai.runtime.local_actions.code_analyzer import CodeAnalyzer


def test_operator_state_classified_as_state_management():
    analyzer = CodeAnalyzer()
    text = open("ai/runtime/operator_state.py", "r", encoding="utf-8").read()
    responsibility = analyzer.responsibility("ai/runtime/operator_state.py", text)
    assert responsibility == "state management"
    stats = analyzer.stats(text)
    flags = analyzer.risk_flags(text, "ai/runtime/operator_state.py", stats)
    assert "contains direct hardcoded route/intent decisions" not in flags


def test_route_arbiter_classified_as_routing():
    analyzer = CodeAnalyzer()
    text = open("ai/core/routing/route_arbiter.py", "r", encoding="utf-8").read()
    responsibility = analyzer.responsibility("ai/core/routing/route_arbiter.py", text)
    assert responsibility == "routing"
