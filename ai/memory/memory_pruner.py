"""
Memory Pruning and Archival System
Automatically manages memory lifecycle to prevent unbounded growth
"""

import os
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MemoryPruner:
    """
    Manages memory lifecycle:
    - Archives old conversations
    - Prunes low-importance memories
    - Enforces retention policies
    - Compresses archived data
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.memory_dir = self.data_dir / "memory"
        self.archive_dir = self.data_dir / "archives"
        self.config_path = self.data_dir / "memory_pruning_config.json"

        self.archive_dir.mkdir(parents=True, exist_ok=True)

        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load pruning configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            default_config = {
                "enabled": True,
                "retention_days": {
                    "episodic": 90,        # Conversations
                    "semantic": 365,       # Facts/knowledge
                    "procedural": 180,     # How-to knowledge
                    "document": None       # Never auto-delete documents
                },
                "importance_threshold": {
                    "episodic": 0.3,       # Archive if importance < 0.3
                    "semantic": 0.5,
                    "procedural": 0.4,
                    "document": None       # Never prune documents
                },
                "pruning_interval_hours": 24,
                "last_prune_time": None,
                "archive_old_memories": True,
                "compress_archives": True,
                "max_memory_size_mb": 500,  # Trigger aggressive pruning
                "keep_starred_forever": True  # User-marked important memories
            }
            self._save_config(default_config)
            return default_config

    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save pruning configuration"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    def should_run(self) -> bool:
        """Check if pruning should run based on interval"""
        if not self.config.get('enabled', True):
            return False

        last_prune = self.config.get('last_prune_time')
        if not last_prune:
            return True

        last_prune_time = datetime.fromisoformat(last_prune)
        interval_hours = self.config.get('pruning_interval_hours', 24)
        next_prune_time = last_prune_time + timedelta(hours=interval_hours)

        return datetime.now() >= next_prune_time

    def prune_memories(self, memory_system) -> Dict[str, Any]:
        """
        Main pruning method
        Returns statistics about pruning operation
        """
        if not self.should_run():
            return {
                'success': True,
                'action': 'prune_skipped',
                'reason': 'Not time to prune yet'
            }

        logger.info("[MemoryPruner] Starting memory pruning...")

        stats = {
            'archived': 0,
            'pruned': 0,
            'kept': 0,
            'errors': 0,
            'by_type': {}
        }

        try:
            # Get all memories
            all_memories = memory_system.get_all_memories()

            memories_to_archive = []
            memories_to_prune = []
            memories_to_keep = []

            for memory in all_memories:
                decision = self._evaluate_memory(memory)

                if decision == 'archive':
                    memories_to_archive.append(memory)
                    stats['archived'] += 1
                elif decision == 'prune':
                    memories_to_prune.append(memory)
                    stats['pruned'] += 1
                else:  # keep
                    memories_to_keep.append(memory)
                    stats['kept'] += 1

                memory_type = memory.get('memory_type', 'unknown')
                stats['by_type'][memory_type] = stats['by_type'].get(memory_type, 0) + 1

            # Archive memories
            if memories_to_archive and self.config.get('archive_old_memories', True):
                self._archive_memories(memories_to_archive)

            # Remove pruned memories from memory system
            for memory in memories_to_prune:
                try:
                    memory_system.remove_memory(memory['id'])
                except Exception as e:
                    logger.error(f"Failed to remove memory {memory['id']}: {e}")
                    stats['errors'] += 1

            # Update last prune time
            self.config['last_prune_time'] = datetime.now().isoformat()
            self._save_config(self.config)

            logger.info(f"[MemoryPruner] Pruning complete: archived={stats['archived']}, pruned={stats['pruned']}, kept={stats['kept']}")

            return {
                'success': True,
                'action': 'prune_completed',
                'stats': stats
            }

        except Exception as e:
            logger.error(f"[MemoryPruner] Error during pruning: {e}", exc_info=True)
            return {
                'success': False,
                'action': 'prune_error',
                'error': str(e)
            }

    def _evaluate_memory(self, memory: Dict[str, Any]) -> str:
        """
        Evaluate if a memory should be kept, archived, or pruned
        Returns: 'keep', 'archive', or 'prune'
        """
        memory_type = memory.get('memory_type', 'episodic')
        importance = memory.get('importance', 0.5)
        timestamp_str = memory.get('timestamp')
        is_starred = memory.get('starred', False)

        # Never prune starred memories
        if is_starred and self.config.get('keep_starred_forever', True):
            return 'keep'

        # Never prune documents (unless explicitly configured)
        if memory_type == 'document' and self.config['retention_days'].get('document') is None:
            return 'keep'

        # Check age
        if timestamp_str:
            try:
                memory_time = datetime.fromisoformat(timestamp_str)
                age_days = (datetime.now() - memory_time).days

                retention_days = self.config['retention_days'].get(memory_type)
                if retention_days and age_days > retention_days:
                    # Too old - archive or prune based on importance
                    importance_threshold = self.config['importance_threshold'].get(memory_type, 0.5)

                    if importance >= importance_threshold:
                        return 'archive'  # Important enough to keep in archive
                    else:
                        return 'prune'    # Not important, delete

            except (ValueError, TypeError):
                pass

        # Check importance alone
        importance_threshold = self.config['importance_threshold'].get(memory_type, 0.5)
        if importance < importance_threshold:
            return 'prune'

        return 'keep'

    def _archive_memories(self, memories: List[Dict[str, Any]]) -> None:
        """Archive memories to compressed storage"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = self.archive_dir / f"memories_archive_{timestamp}.jsonl"

        # Write memories to archive
        with open(archive_file, 'w', encoding='utf-8') as f:
            for memory in memories:
                f.write(json.dumps(memory) + '\n')

        logger.info(f"[MemoryPruner] Archived {len(memories)} memories to {archive_file}")

        # Compress if configured
        if self.config.get('compress_archives', True):
            self._compress_archive(archive_file)

    def _compress_archive(self, archive_file: Path) -> None:
        """Compress archive file using gzip"""
        try:
            import gzip

            compressed_file = archive_file.with_suffix('.jsonl.gz')

            with open(archive_file, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # Remove uncompressed version
            archive_file.unlink()

            logger.info(f"[MemoryPruner] Compressed archive to {compressed_file}")

        except Exception as e:
            logger.warning(f"Failed to compress archive: {e}")

    def get_archive_list(self) -> List[Dict[str, Any]]:
        """Get list of all archives with metadata"""
        archives = []

        for archive_file in self.archive_dir.glob("memories_archive_*"):
            try:
                file_size = archive_file.stat().st_size
                file_time = datetime.fromtimestamp(archive_file.stat().st_mtime)

                archives.append({
                    'filename': archive_file.name,
                    'path': str(archive_file),
                    'size_bytes': file_size,
                    'size_mb': round(file_size / (1024 * 1024), 2),
                    'created_time': file_time.isoformat(),
                    'compressed': archive_file.suffix == '.gz'
                })

            except Exception as e:
                logger.warning(f"Failed to read archive {archive_file}: {e}")

        # Sort by creation time (newest first)
        archives.sort(key=lambda x: x['created_time'], reverse=True)

        return archives

    def restore_from_archive(self, archive_filename: str, memory_system) -> Dict[str, Any]:
        """Restore memories from an archive file"""
        archive_path = self.archive_dir / archive_filename

        if not archive_path.exists():
            return {
                'success': False,
                'action': 'restore_archive',
                'error': f"Archive not found: {archive_filename}"
            }

        try:
            restored_count = 0

            # Handle compressed archives
            if archive_path.suffix == '.gz':
                import gzip
                open_func = gzip.open
            else:
                open_func = open

            with open_func(archive_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    memory = json.loads(line.strip())
                    # Re-add to memory system
                    memory_system.store_memory(
                        content=memory['content'],
                        memory_type=memory['memory_type'],
                        context=memory.get('context', {}),
                        importance=memory.get('importance', 0.5),
                        tags=memory.get('tags', [])
                    )
                    restored_count += 1

            logger.info(f"[MemoryPruner] Restored {restored_count} memories from {archive_filename}")

            return {
                'success': True,
                'action': 'restore_archive',
                'restored_count': restored_count,
                'archive': archive_filename
            }

        except Exception as e:
            logger.error(f"Failed to restore archive: {e}", exc_info=True)
            return {
                'success': False,
                'action': 'restore_archive',
                'error': str(e)
            }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        stats = {
            'memory_dir_size_mb': 0,
            'archive_dir_size_mb': 0,
            'total_archives': 0,
            'config': self.config
        }

        # Calculate memory directory size
        if self.memory_dir.exists():
            total_size = sum(f.stat().st_size for f in self.memory_dir.rglob('*') if f.is_file())
            stats['memory_dir_size_mb'] = round(total_size / (1024 * 1024), 2)

        # Calculate archive directory size
        if self.archive_dir.exists():
            total_size = sum(f.stat().st_size for f in self.archive_dir.rglob('*') if f.is_file())
            stats['archive_dir_size_mb'] = round(total_size / (1024 * 1024), 2)
            stats['total_archives'] = len(list(self.archive_dir.glob("memories_archive_*")))

        return stats


# Singleton factory
_memory_pruner = None

def get_memory_pruner(data_dir: str = "data") -> MemoryPruner:
    """Get singleton memory pruner instance"""
    global _memory_pruner
    if _memory_pruner is None:
        _memory_pruner = MemoryPruner(data_dir)
    return _memory_pruner
