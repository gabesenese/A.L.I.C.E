"""She calls him what he asks to be called.

"call me Gabe" was stored nowhere, and the persona was built once at start-up
around a fixed default name, so she went on using the default.
"""

import pytest

from ai.identity.preferred_name import preferred_name, stated_name


@pytest.mark.parametrize(
    "text, name",
    [
        ("call me Gabe", "Gabe"),
        ("you can call me gabe please", "Gabe"),
        ("I go by Sam", "Sam"),
        ("my name is Mary Anne", "Mary Anne"),
    ],
)
def test_a_name_he_gives_is_heard(text, name):
    assert stated_name(text) == name


@pytest.mark.parametrize("text", ["call me later", "call me back when you're done", "my name is on the list"])
def test_other_uses_of_call_me_are_not_a_name(text):
    assert stated_name(text) is None


def test_the_next_turn_is_written_for_that_name():
    from ai.core.llm_engine import LLMConfig, LocalLLMEngine
    from ai.runtime.companion_runtime import CompanionRuntimeLoop

    CompanionRuntimeLoop._capture_preferred_name("call me Gabe")
    prompt = LocalLLMEngine(LLMConfig(model="test-model"))._build_system_prompt()

    assert preferred_name() == "Gabe"
    assert "You run on Gabe's machine" in prompt
