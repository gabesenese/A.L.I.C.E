from ai.memory.memory_rights_service import MemoryRightsService
from ai.memory.memory_system import MemorySystem


def test_preview_then_confirm_delete_topic(tmp_path):
    memory = MemorySystem(data_dir=str(tmp_path / "memory"))
    memory.store_memory("User talked about mom and family plans", memory_type="episodic")
    service = MemoryRightsService(memory)

    preview = service.preview_topic_delete("mom")
    assert preview.requires_confirmation is True
    assert preview.count >= 1

    result = service.delete_topic("mom", confirmed=True)
    assert result.deleted_count >= 1
    assert result.verification_status in {"cleared", "partial"}


def test_delete_topic_without_confirmation_is_honest(tmp_path):
    memory = MemorySystem(data_dir=str(tmp_path / "memory"))
    memory.store_memory("Topic: mom details", memory_type="episodic")
    service = MemoryRightsService(memory)

    result = service.delete_topic("mom", confirmed=False)
    assert result.deleted_count == 0
    assert result.verification_status == "confirmation_required"
