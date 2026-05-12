from ai.runtime.local_actions.code_analyzer import CodeAnalyzer


def test_agent_loop_file_is_not_misclassified_as_routing():
    analyzer = CodeAnalyzer()
    text = open("ai/runtime/agent_loop.py", "r", encoding="utf-8").read()
    responsibility = analyzer.responsibility("ai/runtime/agent_loop.py", text)
    assert responsibility == "agent loop"


def test_operator_state_file_is_state_management():
    analyzer = CodeAnalyzer()
    text = open("ai/runtime/operator_state.py", "r", encoding="utf-8").read()
    responsibility = analyzer.responsibility("ai/runtime/operator_state.py", text)
    assert responsibility == "state management"


def test_route_mentions_without_routing_logic_do_not_trigger_routing():
    analyzer = CodeAnalyzer()
    text = "last_route = None\nlast_intent = None\nrouting_result = {}"
    responsibility = analyzer.responsibility("ai/runtime/example.py", text)
    assert responsibility != "routing"
