from ai.runtime.local_actions.code_analyzer import CodeAnalyzer


def test_agent_loop_is_classified_as_agent_loop():
    analyzer = CodeAnalyzer()
    text = open("ai/runtime/agent_loop.py", "r", encoding="utf-8").read()
    assert analyzer.responsibility("ai/runtime/agent_loop.py", text) == "agent loop"


def test_next_step_policy_is_classified_correctly():
    analyzer = CodeAnalyzer()
    text = open("ai/runtime/next_step_policy.py", "r", encoding="utf-8").read()
    assert analyzer.responsibility("ai/runtime/next_step_policy.py", text) == "next-step policy"
