from ai.memory.memory_system import MemorySystem
from ai.plugins.memory_plugin import MemoryPlugin


def _seed_memory(mem: MemorySystem) -> None:
    mem.store_memory(
        "User said their mom likes gardening.",
        memory_type="episodic",
        context={"source": "conversation"},
        tags=["structured:personal", "domain:relationships"],
    )
    mem.store_memory(
        "User discussed weekend plans.",
        memory_type="episodic",
        context={"source": "conversation"},
    )
    mem.store_memory(
        "User mentioned concern about mom's appointment.",
        memory_type="episodic",
        context={"source": "conversation"},
        tags=["structured:personal", "domain:relationships"],
    )


def test_broad_delete_creates_pending_scope(tmp_path):
    memory = MemorySystem(data_dir=str(tmp_path / "memory"))
    _seed_memory(memory)
    plugin = MemoryPlugin(memory_system=memory)

    result = plugin.handle_request(
        "memory:delete_conversation",
        {},
        {"user_input": "move on from this convo, and delete the memories from your data"},
    )

    assert result["success"] is True
    assert result["deletion_executed"] is False
    assert "scope" in result["response"].lower()
    assert plugin.pending_memory_delete.get("scope") == "needs_clarification"


def test_topic_reply_previews_without_deleting(tmp_path):
    memory = MemorySystem(data_dir=str(tmp_path / "memory"))
    _seed_memory(memory)
    plugin = MemoryPlugin(memory_system=memory)
    plugin.pending_memory_delete = {"scope": "needs_clarification"}

    before = memory.get_statistics()["total_memories"]
    result = plugin.handle_request(
        "memory:delete_topic",
        {},
        {"user_input": "just the topic about my mom"},
    )
    after = memory.get_statistics()["total_memories"]

    assert result["success"] is True
    assert result["deletion_executed"] is False
    assert result.get("preview", {}).get("count", 0) >= 1
    assert "confirm" in result["response"].lower()
    assert before == after


def test_confirm_delete_executes_and_verifies(tmp_path):
    memory = MemorySystem(data_dir=str(tmp_path / "memory"))
    _seed_memory(memory)
    plugin = MemoryPlugin(memory_system=memory)

    plugin.handle_request(
        "memory:delete_topic",
        {},
        {"user_input": "delete memories about my mom"},
    )
    result = plugin.handle_request(
        "memory:delete_topic_confirm",
        {},
        {"user_input": "yes delete them"},
    )

    assert result["success"] is True
    assert int(result.get("deleted_count", 0)) >= 1
    assert result.get("verification_status") in {"cleared", "partial"}
    assert "local memory store" in result["response"].lower()
    assert "cannot guarantee" in result["response"].lower()
