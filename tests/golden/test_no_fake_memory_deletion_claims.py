from ai.memory.memory_system import MemorySystem
from ai.plugins.memory_plugin import MemoryPlugin


def test_no_fake_deleted_claim_when_delete_fails(tmp_path):
    memory = MemorySystem(data_dir=str(tmp_path / "memory"))
    plugin = MemoryPlugin(memory_system=memory)
    plugin.pending_memory_delete = {
        "scope": "topic",
        "topic": "mom",
        "matched_memory_ids": ["missing-id-1"],
        "awaiting_confirmation": True,
    }

    result = plugin.handle_request(
        "memory:delete_topic_confirm",
        {},
        {"user_input": "yes delete them"},
    )
    low = str(result.get("response") or "").lower()
    assert result["success"] is False
    assert "deleted" not in low
    assert "won't be stored anywhere" not in low
    assert "removed from my data" not in low


def test_sensitive_topic_response_stays_brief(tmp_path):
    memory = MemorySystem(data_dir=str(tmp_path / "memory"))
    memory.store_memory("User discussed private details about their mom.", memory_type="episodic")
    plugin = MemoryPlugin(memory_system=memory)

    result = plugin.handle_request(
        "memory:delete_topic",
        {},
        {"user_input": "delete memories about my mom"},
    )
    low = str(result.get("response") or "").lower()
    assert "private details" not in low
    assert "that topic" in low or "my mom" in low
