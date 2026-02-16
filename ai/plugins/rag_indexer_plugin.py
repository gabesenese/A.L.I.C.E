"""
RAG Document Indexer Plugin
Automatically indexes specified directories for semantic search
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from ai.plugins.plugin_system import PluginInterface
from ai.memory.memory_system import MemorySystem

logger = logging.getLogger(__name__)


class RAGIndexerPlugin(PluginInterface):
    """
    Automatically indexes documents from configured directories.
    Watches for new files and updates the index incrementally.
    """

    def __init__(self, memory_system: Optional[MemorySystem] = None):
        self.memory_system = memory_system
        self.config_path = Path("config/rag_indexer_config.json")
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load indexer configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Default configuration
            default_config = {
                "indexed_directories": [],
                "file_extensions": [".txt", ".md", ".pdf", ".docx", ".json", ".csv", ".html"],
                "auto_index": False,
                "index_interval_hours": 24,
                "last_index_time": None,
                "exclude_patterns": ["__pycache__", ".git", "node_modules", ".env", "venv"]
            }
            self._save_config(default_config)
            return default_config

    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save indexer configuration"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    def name(self) -> str:
        return "rag_indexer"

    def description(self) -> str:
        return "Indexes documents from configured directories for semantic search"

    def can_handle(self, intent: str, entities: Dict[str, Any]) -> bool:
        indexer_intents = [
            'rag:index',
            'rag:add_directory',
            'rag:remove_directory',
            'rag:list_directories',
            'rag:reindex',
            'rag:index_status'
        ]
        return intent in indexer_intents

    def execute(self, intent: str, user_input: str, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RAG indexer commands"""
        try:
            if intent == 'rag:add_directory':
                return self._add_directory(entities.get('directory'))

            elif intent == 'rag:remove_directory':
                return self._remove_directory(entities.get('directory'))

            elif intent == 'rag:list_directories':
                return self._list_directories()

            elif intent == 'rag:index' or intent == 'rag:reindex':
                force_reindex = intent == 'rag:reindex'
                return self._index_all_directories(force_reindex)

            elif intent == 'rag:index_status':
                return self._get_index_status()

            else:
                return {
                    'success': False,
                    'action': 'unknown_rag_command',
                    'message': f"Unknown RAG indexer command: {intent}"
                }

        except Exception as e:
            logger.error(f"RAG indexer error: {e}", exc_info=True)
            return {
                'success': False,
                'action': 'rag_indexer_error',
                'error': str(e)
            }

    def _add_directory(self, directory: Optional[str]) -> Dict[str, Any]:
        """Add a directory to the index watch list"""
        if not directory:
            return {
                'success': False,
                'action': 'add_directory',
                'message': "No directory specified"
            }

        directory_path = Path(directory).resolve()

        if not directory_path.exists():
            return {
                'success': False,
                'action': 'add_directory',
                'message': f"Directory does not exist: {directory}"
            }

        if not directory_path.is_dir():
            return {
                'success': False,
                'action': 'add_directory',
                'message': f"Path is not a directory: {directory}"
            }

        directory_str = str(directory_path)

        if directory_str in self.config['indexed_directories']:
            return {
                'success': True,
                'action': 'add_directory',
                'message': f"Directory already indexed: {directory}"
            }

        self.config['indexed_directories'].append(directory_str)
        self._save_config(self.config)

        # Automatically index the new directory
        indexed_count = self._index_directory(directory_path)

        return {
            'success': True,
            'action': 'add_directory',
            'directory': directory_str,
            'indexed_files': indexed_count,
            'message': f"Added and indexed directory: {directory} ({indexed_count} files)"
        }

    def _remove_directory(self, directory: Optional[str]) -> Dict[str, Any]:
        """Remove a directory from the index watch list"""
        if not directory:
            return {
                'success': False,
                'action': 'remove_directory',
                'message': "No directory specified"
            }

        directory_path = Path(directory).resolve()
        directory_str = str(directory_path)

        if directory_str not in self.config['indexed_directories']:
            return {
                'success': False,
                'action': 'remove_directory',
                'message': f"Directory not in index: {directory}"
            }

        self.config['indexed_directories'].remove(directory_str)
        self._save_config(self.config)

        return {
            'success': True,
            'action': 'remove_directory',
            'directory': directory_str,
            'message': f"Removed directory from index: {directory}"
        }

    def _list_directories(self) -> Dict[str, Any]:
        """List all indexed directories"""
        directories = self.config['indexed_directories']

        return {
            'success': True,
            'action': 'list_directories',
            'directories': directories,
            'count': len(directories),
            'message': f"Indexed directories: {len(directories)}"
        }

    def _index_all_directories(self, force_reindex: bool = False) -> Dict[str, Any]:
        """Index all configured directories"""
        if not self.memory_system:
            return {
                'success': False,
                'action': 'index_all',
                'message': "Memory system not available"
            }

        total_indexed = 0
        results = []

        for directory_str in self.config['indexed_directories']:
            directory_path = Path(directory_str)
            if directory_path.exists():
                count = self._index_directory(directory_path, force_reindex)
                total_indexed += count
                results.append({
                    'directory': directory_str,
                    'files_indexed': count
                })

        self.config['last_index_time'] = datetime.now().isoformat()
        self._save_config(self.config)

        return {
            'success': True,
            'action': 'index_all',
            'total_files': total_indexed,
            'directories': results,
            'message': f"Indexed {total_indexed} files from {len(results)} directories"
        }

    def _index_directory(self, directory: Path, force_reindex: bool = False) -> int:
        """Index all eligible files in a directory"""
        indexed_count = 0

        for root, dirs, files in os.walk(directory):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(pattern in d for pattern in self.config['exclude_patterns'])]

            root_path = Path(root)

            for file in files:
                file_path = root_path / file

                # Check file extension
                if file_path.suffix.lower() not in self.config['file_extensions']:
                    continue

                # Check if file should be excluded
                if any(pattern in str(file_path) for pattern in self.config['exclude_patterns']):
                    continue

                try:
                    # Check if already indexed (unless force reindex)
                    if not force_reindex and self._is_file_indexed(file_path):
                        continue

                    # Ingest the document
                    if self.memory_system:
                        result = self.memory_system.ingest_document(str(file_path))
                        if result.get('success'):
                            indexed_count += 1
                            logger.info(f"Indexed: {file_path}")

                except Exception as e:
                    logger.warning(f"Failed to index {file_path}: {e}")

        return indexed_count

    def _is_file_indexed(self, file_path: Path) -> bool:
        """Check if a file is already indexed"""
        if not self.memory_system:
            return False

        # Get document registry
        registry = self.memory_system.document_processor.document_registry

        # Check if file hash exists in registry
        import hashlib
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            return file_hash in registry
        except Exception:
            return False

    def _get_index_status(self) -> Dict[str, Any]:
        """Get current indexing status"""
        if not self.memory_system:
            return {
                'success': False,
                'action': 'index_status',
                'message': "Memory system not available"
            }

        registry = self.memory_system.document_processor.document_registry
        total_documents = len(registry)
        total_chunks = sum(doc['chunks_created'] for doc in registry.values())

        directory_stats = []
        for directory_str in self.config['indexed_directories']:
            directory_path = Path(directory_str)
            if directory_path.exists():
                doc_count = self._count_indexed_from_directory(directory_path)
                directory_stats.append({
                    'directory': directory_str,
                    'documents': doc_count
                })

        return {
            'success': True,
            'action': 'index_status',
            'total_documents': total_documents,
            'total_chunks': total_chunks,
            'indexed_directories': len(self.config['indexed_directories']),
            'directory_stats': directory_stats,
            'last_index_time': self.config.get('last_index_time'),
            'message': f"Indexed {total_documents} documents ({total_chunks} chunks)"
        }

    def _count_indexed_from_directory(self, directory: Path) -> int:
        """Count documents indexed from a specific directory"""
        if not self.memory_system:
            return 0

        registry = self.memory_system.document_processor.document_registry
        count = 0

        directory_str = str(directory.resolve())

        for doc_info in registry.values():
            file_path = doc_info.get('file_path', '')
            if file_path.startswith(directory_str):
                count += 1

        return count


# Singleton factory
_rag_indexer_plugin = None

def get_rag_indexer_plugin(memory_system: Optional[MemorySystem] = None) -> RAGIndexerPlugin:
    """Get singleton RAG indexer plugin instance"""
    global _rag_indexer_plugin
    if _rag_indexer_plugin is None:
        _rag_indexer_plugin = RAGIndexerPlugin(memory_system)
    return _rag_indexer_plugin
