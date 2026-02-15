"""
Persistence Manager for A.L.I.C.E Memory System
Handles memory serialization and deserialization
"""

import pickle
import json
from pathlib import Path
from typing import List, Dict, Any
from ai.memory.memory_store import MemoryEntry
import logging

logger = logging.getLogger(__name__)


class PersistenceManager:
    """Handles memory serialization and deserialization"""

    def __init__(self, data_dir: str = "data/memory") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.memory_file = self.data_dir / "memories.pkl"
        self.document_registry_file = self.data_dir / "document_registry.json"
        self.archive_file = self.data_dir / "archived_memories.pkl"

    def save_memories(self, memories: List[MemoryEntry]) -> bool:
        """
        Persist memories to disk

        Args:
            memories: List of memory entries

        Returns:
            True if saved successfully
        """
        try:
            with open(self.memory_file, 'wb') as f:
                pickle.dump(memories, f)
            logger.info(f"Saved {len(memories)} memories to {self.memory_file}")
           return True
        except (OSError, pickle.PickleError) as e:
            logger.error(f"Failed to save memories: {e}")
            return False

    def load_memories(self) -> List[MemoryEntry]:
        """
        Load memories from disk

        Returns:
            List of memory entries
        """
        if not self.memory_file.exists():
            logger.info("No existing memory file found")
            return []

        try:
            with open(self.memory_file, 'rb') as f:
                memories = pickle.load(f)
            logger.info(f"Loaded {len(memories)} memories from {self.memory_file}")
            return memories
        except (OSError, pickle.PickleError) as e:
            logger.error(f"Failed to load memories: {e}")
            return []

    def save_document_registry(self, registry: Dict[str, Dict]) -> bool:
        """
        Save document registry

        Args:
            registry: Document registry dictionary

        Returns:
            True if saved successfully
        """
        try:
            with open(self.document_registry_file, 'w', encoding='utf-8') as f:
                json.dump(registry, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved document registry: {len(registry)} documents")
            return True
        except (OSError, json.JSONEncodeError) as e:
            logger.error(f"Failed to save document registry: {e}")
            return False

    def load_document_registry(self) -> Dict[str, Dict]:
        """
        Load document registry

        Returns:
            Document registry dictionary
        """
        if not self.document_registry_file.exists():
            return {}

        try:
            with open(self.document_registry_file, 'r', encoding='utf-8') as f:
                registry = json.load(f)
            logger.info(f"Loaded document registry: {len(registry)} documents")
            return registry
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load document registry: {e}")
            return {}

    def save_archived_memories(self, archived: List[MemoryEntry]) -> bool:
        """
        Save archived memories

        Args:
            archived: List of archived memory entries

        Returns:
            True if saved successfully
        """
        # Load existing archive
        existing_archive = []
        if self.archive_file.exists():
            try:
                with open(self.archive_file, 'r', encoding='utf-8') as f:
                    existing_archive = json.load(f)
            except (json.JSONDecodeError, OSError, FileNotFoundError) as e:
                logger.debug(f"Could not load existing archive: {e}")

        # Append new archived memories
        try:
            for memory in archived:
                existing_archive.append({
                    'id': memory.id,
                    'content': memory.content,
                    'memory_type': memory.memory_type,
                    'timestamp': memory.timestamp,
                    'importance': memory.importance,
                    'archived_on': str(datetime.now())
                })

            with open(self.archive_file, 'w', encoding='utf-8') as f:
                json.dump(existing_archive, f, indent=2, ensure_ascii=False)

            logger.info(f"Archived {len(archived)} memories")
            return True
        except (OSError, json.JSONEncodeError) as e:
            logger.error(f"Failed to archive memories: {e}")
            return False

    def load_archived_memories(self) -> List[Dict[str, Any]]:
        """
        Load archived memories

        Returns:
            List of archived memory dictionaries
        """
        if not self.archive_file.exists():
            return []

        try:
            with open(self.archive_file, 'r', encoding='utf-8') as f:
                archived = json.load(f)
            return archived
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load archived memories: {e}")
            return []


from datetime import datetime

# Singleton instance
_persistence_manager: Optional[PersistenceManager] = None


def get_persistence_manager() -> PersistenceManager:
    """Get or create the PersistenceManager singleton"""
    global _persistence_manager
    if _persistence_manager is None:
        _persistence_manager = PersistenceManager()
    return _persistence_manager
