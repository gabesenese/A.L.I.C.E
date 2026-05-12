from ai.core.nlp_processor import NLPProcessor
from ai.memory.memory_system import MemorySystem


def test_memory_delete_intent_routing_for_broad_request():
    nlp = NLPProcessor()
    result = nlp.process("move on from this convo, and delete the memories from your data")
    assert result.intent in {
        "memory:delete_conversation",
        "memory:delete_all_conversation_memory",
    }
    assert result.intent != "conversation:clarification_needed"


def test_memory_delete_intent_routing_for_topic_reply():
    nlp = NLPProcessor()
    result = nlp.process("just the topic about my mom")
    assert result.intent == "memory:delete_topic"
    assert result.intent != "conversation:goal_statement"


def test_memory_system_delete_by_topic_verifies_clear(tmp_path):
    memory = MemorySystem(data_dir=str(tmp_path / "memory"))
    memory.store_memory("User talked about their mom and travel plans.", memory_type="episodic")
    memory.store_memory("User asked for weather in Toronto.", memory_type="episodic")
    preview = memory.preview_memory_delete("mom")
    assert int(preview.get("count", 0)) >= 1

    result = memory.delete_memories_by_topic("mom")
    assert int(result.get("deleted_count", 0)) >= 1
    assert result.get("persisted") is True
    assert result.get("verification_status") in {"cleared", "partial"}
