"""Memory questions that name a time are answered from that time.

"What did we talk about yesterday?" reached the memory plugin with no topic, so
search answered "No search query specified" and recall listed memories from any
day. The time is a filter, not a topic.
"""

import pytest

from ai.memory.memory_system import MemorySystem
from ai.plugins.memory_plugin import MemoryPlugin


@pytest.fixture
def plugin(tmp_path):
    plugin = MemoryPlugin(memory_system=MemorySystem(data_dir=str(tmp_path)))
    plugin.memory.store_memory(content="I'm rewriting the parser this week.", memory_type="episodic")
    return plugin


@pytest.mark.parametrize("intent", ["memory:search", "memory:recall"])
def test_a_question_about_today_is_answered_from_today(plugin, intent):
    result = plugin.execute(intent, "what did we talk about this morning?", {}, {})
    assert result["success"] is True
    assert "rewriting the parser" in result["response"]


@pytest.mark.parametrize("intent", ["memory:search", "memory:recall"])
def test_nothing_from_another_day_is_passed_off_as_that_day(plugin, intent):
    result = plugin.execute(intent, "what did we talk about yesterday?", {}, {})
    assert "rewriting the parser" not in result["response"]
    assert "yesterday" in result["response"]
