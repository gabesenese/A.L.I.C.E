"""The plugin behind "remember that…" and "what do you remember about…".

Every call it made into the memory system named a method that does not exist:
`add_episodic_memory`, `get_recent_memories`, `search_memories`. The real API is
`store_memory`, `recall_memory`, `get_all_memories`. Each AttributeError was
caught by a blanket `except Exception` and rendered as a polite failure string,
so "Remember that I prefer coffee" answered "Failed to store preference:
'MemorySystem' object has no attribute 'add_episodic_memory'" and "what do you
remember about X" answered "Failed to recall memory: …".

The plugin manager reaches this plugin on memory:store / memory:recall /
memory:search / memory:delete, and nlp_processor assigns
those intents at 0.95 confidence, so this is a user-facing path and not a
disused corner.

These tests bind the plugin to the real MemorySystem API rather than a mock of
it — a mock would have passed the whole time this was broken.
"""

import pytest

from ai.memory.memory_system import MemorySystem
from ai.plugins.memory_plugin import MemoryPlugin


@pytest.fixture
def plugin(tmp_path):
    """A plugin over a memory system with its own empty database.

    The database is isolated by the autouse isolate_memory_store fixture in
    tests/conftest.py; without it this would read and write Gabriel's real
    memories.
    """
    return MemoryPlugin(memory_system=MemorySystem(data_dir=str(tmp_path)))


# -- the methods have to exist ------------------------------------------------


@pytest.mark.parametrize("name", ["store_memory", "recall_memory", "get_all_memories"])
def test_the_api_the_plugin_calls_is_the_api_that_exists(name):
    """Pinned separately from behaviour so a rename of the memory system fails
    here, loudly, rather than turning back into a caught AttributeError."""
    assert hasattr(MemorySystem, name), f"MemorySystem has no {name}"


def test_the_plugin_names_no_method_the_memory_system_lacks():
    """The general form of the defect: three separate ghost methods, all caught."""
    import re
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "ai" / "plugins" / "memory_plugin.py"
    called = sorted(set(re.findall(r"self\.memory\.(\w+)\s*\(", source.read_text(encoding="utf-8"))))
    assert called, "no memory calls found — has the plugin been rewired?"
    missing = [name for name in called if not hasattr(MemorySystem, name)]
    assert not missing, f"plugin calls methods that do not exist: {missing}"


# -- storing ------------------------------------------------------------------


def test_remembering_something_actually_stores_it(plugin):
    result = plugin.handle_request("memory:store", {"content": "Gabriel prefers coffee"}, {})

    assert result["success"] is True, result["message"]
    assert plugin.memory.get_all_memories(), "nothing reached the store"


def test_storing_reports_the_failure_rather_than_claiming_success(plugin, monkeypatch):
    def exploding(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(plugin.memory, "store_memory", exploding)
    result = plugin.handle_request("memory:store", {"content": "something"}, {})
    assert result["success"] is False


def test_storing_nothing_is_refused(plugin):
    assert plugin.handle_request("memory:store", {}, {})["success"] is False


# -- recalling ----------------------------------------------------------------


def test_what_was_stored_can_be_recalled(plugin):
    plugin.handle_request("memory:store", {"content": "Gabriel prefers coffee to tea"}, {})

    result = plugin.handle_request("memory:recall", {"topic": "coffee"}, {})

    assert result["success"] is True, result["message"]
    assert "coffee" in result["message"].lower()


def test_recall_with_no_topic_returns_what_is_there(plugin):
    plugin.handle_request("memory:store", {"content": "the schema drift is in the notes table"}, {})

    result = plugin.handle_request("memory:recall", {}, {})

    assert result["success"] is True, result["message"]
    assert "schema drift" in result["message"]


def test_an_empty_store_says_so_plainly(plugin):
    result = plugin.handle_request("memory:recall", {}, {})
    assert result["success"] is True
    assert result["count"] == 0


# -- the honest answer when recall is down ------------------------------------


def test_an_unreachable_store_is_not_reported_as_an_empty_one(plugin):
    """docs/north_star.md rule 4. "I don't have any memories about X" when the
    memories exist but could not be loaded is a confident false answer — the
    worst of the four outcomes, because the user cannot tell. A load failure has
    to reach the sentence."""
    plugin.memory.load_failed = True

    result = plugin.handle_request("memory:recall", {"topic": "coffee"}, {})

    assert "don't have any memories about" not in result["message"].lower()
    assert result.get("recall_available") is False


def test_the_same_holds_for_a_topicless_recall(plugin):
    plugin.memory.load_failed = True
    result = plugin.handle_request("memory:recall", {}, {})
    assert "don't have any" not in result["message"].lower()
    assert result.get("recall_available") is False


def test_a_working_store_says_recall_is_available(plugin):
    result = plugin.handle_request("memory:recall", {"topic": "anything"}, {})
    assert result.get("recall_available") is True


# -- forgetting ---------------------------------------------------------------


def test_forgetting_something_actually_removes_it(plugin):
    """This handler counted the matches, answered "Cleared N memory entries
    about X", and deleted nothing — with a comment saying deletion "would need
    to be implemented". On a request to forget something, a false confirmation
    is the one outcome with no recovery: he believes it is gone and stops
    asking."""
    plugin.handle_request("memory:store", {"content": "Gabriel prefers coffee"}, {})
    assert plugin.memory.get_all_memories()

    result = plugin.handle_request("memory:delete", {"topic": "coffee"}, {})

    assert result["success"] is True, result["message"]
    assert result["deleted_count"] >= 1
    remaining = [m["content"] for m in plugin.memory.get_all_memories()]
    assert not any("coffee" in c.lower() for c in remaining), remaining


def test_forgetting_what_was_never_stored_does_not_claim_a_deletion(plugin):
    result = plugin.handle_request("memory:delete", {"topic": "quantum ferrets"}, {})
    assert result["deleted_count"] == 0
    assert "deleted" not in result["message"].lower()


def test_a_deletion_that_fails_is_not_reported_as_done(plugin, monkeypatch):
    plugin.handle_request("memory:store", {"content": "Gabriel prefers coffee"}, {})
    monkeypatch.setattr(plugin.memory, "_remove_memory_by_id", lambda _id: False)

    result = plugin.handle_request("memory:delete", {"topic": "coffee"}, {})

    assert result["success"] is False
    assert result["deleted_count"] == 0


def test_forgetting_is_refused_when_memory_cannot_be_reached(plugin):
    """Deleting against a store you cannot read risks reporting a deletion that
    did not happen, on the one operation where that is unrecoverable."""
    plugin.memory.load_failed = True
    result = plugin.handle_request("memory:delete", {"topic": "coffee"}, {})
    assert result["success"] is False
    assert result["deleted_count"] == 0


# -- recall without an embedding model ----------------------------------------


def test_recall_still_works_when_semantic_search_is_unavailable(plugin, monkeypatch):
    """The embedding model is optional, and on a machine without it every
    semantic lookup returns nothing — which reads as "you never told me that".
    A literal scan is worse than semantic search and far better than silence."""
    plugin.handle_request("memory:store", {"content": "the importer rewrite is blocked on schema drift"}, {})

    def no_model(*args, **kwargs):
        raise RuntimeError("embedding model not available")

    monkeypatch.setattr(plugin.memory, "recall_memory", no_model)

    result = plugin.handle_request("memory:recall", {"topic": "schema drift"}, {})

    assert result["success"] is True, result["message"]
    assert "schema drift" in result["message"]
