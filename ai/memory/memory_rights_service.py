from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from ai.memory.memory_system import MemorySystem


@dataclass
class MemoryDeletePreview:
    topic: str
    matched_ids: List[str]
    count: int
    snippets: List[str]
    stores_affected: List[str]
    requires_confirmation: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryDeleteResult:
    topic: str
    deleted_count: int
    deleted_ids: List[str]
    skipped_ids: List[str]
    stores_updated: List[str]
    persisted: bool
    verification_status: str
    remaining_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryRightsService:
    def __init__(self, memory_system: MemorySystem) -> None:
        self.memory = memory_system
        self._suppressed_topics: Dict[str, List[str]] = {}

    def preview_topic_delete(self, topic: str, user_id: str = "default") -> MemoryDeletePreview:
        preview = dict(self.memory.preview_memory_delete(topic))
        matches = list(preview.get("matched_memory_ids") or [])
        snippets = [str(item.get("content") or "")[:160] for item in list(preview.get("matches") or [])[:5]]
        return MemoryDeletePreview(
            topic=str(topic or ""),
            matched_ids=matches,
            count=int(preview.get("count", 0) or 0),
            snippets=snippets,
            stores_affected=list(preview.get("stores_affected") or ["episodic", "semantic", "procedural", "document", "vector"]),
            requires_confirmation=True,
        )

    def delete_topic(self, topic: str, confirmed: bool, user_id: str = "default") -> MemoryDeleteResult:
        preview = self.preview_topic_delete(topic, user_id=user_id)
        if not confirmed:
            return MemoryDeleteResult(
                topic=topic,
                deleted_count=0,
                deleted_ids=[],
                skipped_ids=list(preview.matched_ids),
                stores_updated=[],
                persisted=False,
                verification_status="confirmation_required",
                remaining_count=preview.count,
            )
        delete_result = dict(self.memory.delete_memories_by_ids(list(preview.matched_ids)))
        persisted = bool(delete_result.get("persisted", False))
        if not persisted:
            try:
                self.memory.save_memories()
                persisted = True
            except Exception:
                persisted = False
        verify = self.preview_topic_delete(topic, user_id=user_id)
        verification_status = "cleared" if verify.count == 0 else "partial"
        return MemoryDeleteResult(
            topic=topic,
            deleted_count=int(delete_result.get("deleted_count", 0) or 0),
            deleted_ids=list(delete_result.get("deleted_ids") or []),
            skipped_ids=list(delete_result.get("skipped_ids") or []),
            stores_updated=list(delete_result.get("stores_updated") or []),
            persisted=persisted,
            verification_status=verification_status,
            remaining_count=verify.count,
        )

    def suppress_topic(self, topic: str, user_id: str = "default") -> Dict[str, Any]:
        key = str(user_id or "default")
        existing = list(self._suppressed_topics.get(key) or [])
        if topic not in existing:
            existing.append(topic)
        self._suppressed_topics[key] = existing
        return {"success": True, "suppressed_topics": list(existing), "topic": topic}

    def show_memories(self, topic: str, user_id: str = "default") -> List[Dict]:
        preview = dict(self.memory.preview_memory_delete(topic))
        return list(preview.get("matches") or [])
