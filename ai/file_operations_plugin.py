"""
A.L.I.C.E File Operations Plugin

This plugin provides file system operations including:
- Create, read, write, and delete files
- Move, copy, rename files
- List files in directories
- Search for files by pattern

Supports commands like:
- "Create a file called test.txt"
- "Read the file config.json"
- "Delete the file old_data.csv"
- "Move notes.txt to archive folder"
- "List files in this directory"
- "Search for python files"
"""

import os
import sys
import shutil
import logging
import glob
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

# Set up logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Import the proper plugin interface
from ai.plugin_system import PluginInterface


class FileOperationsPlugin(PluginInterface):
    """Plugin for file system operations"""

    def __init__(self):
        super().__init__()
        self.name = "File Operations Plugin"
        self.version = "1.0.0"
        self.description = "File system operations: create, read, delete, move, list, search files"
        self.enabled = True
        self.capabilities = [
            "create_file", "read_file", "delete_file",
            "move_file", "copy_file", "list_files", "search_files"
        ]
        self.commands = [
            "create file", "make file", "new file",
            "read file", "open file", "show file contents",
            "delete file", "remove file",
            "move file", "rename file", "copy file",
            "list files", "show files",
            "search files", "find files"
        ]
        # Safety: Only allow operations in safe directories
        self.safe_base_dir = os.getcwd()

    def initialize(self) -> bool:
        """Initialize the plugin"""
        try:
            logger.info("Initializing File Operations Plugin")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize File Operations Plugin: {e}")
            return False

    def shutdown(self):
        """Cleanup when plugin is disabled"""
        logger.info("Shutting down File Operations Plugin")

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_commands(self) -> List[str]:
        return self.commands

    def is_enabled(self) -> bool:
        return self.enabled

    def _is_safe_path(self, filepath: str) -> bool:
        """Check if path is safe (within allowed directories)"""
        try:
            # Resolve to absolute path
            abs_path = os.path.abspath(filepath)
            # Check if it's within safe base directory
            return abs_path.startswith(self.safe_base_dir)
        except:
            return False

    def handle_request(self, intent: str, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle file operation requests

        Args:
            intent: The intent (e.g., "file:create", "file:read")
            entities: Extracted entities (filename, path, pattern, etc.)
            context: Additional context

        Returns:
            Response dictionary with result
        """
        try:
            action = intent.split(":")[-1] if ":" in intent else entities.get("action", "")

            if action == "create":
                return self._create_file(entities)
            elif action == "read":
                return self._read_file(entities)
            elif action == "delete":
                return self._delete_file(entities)
            elif action == "move":
                return self._move_file(entities)
            elif action == "copy":
                return self._copy_file(entities)
            elif action == "list":
                return self._list_files(entities)
            elif action == "search":
                return self._search_files(entities)
            else:
                return {
                    "success": False,
                    "message": f"Unknown file operation: {action}",
                    "action": action
                }

        except Exception as e:
            logger.error(f"Error handling file operation: {e}")
            return {
                "success": False,
                "message": f"File operation failed: {str(e)}",
                "error": str(e)
            }

    def _create_file(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new file"""
        filename = entities.get("filename") or entities.get("file_name") or entities.get("name")
        content = entities.get("content", "")

        if not filename:
            return {"success": False, "message": "No filename specified"}

        filepath = os.path.join(self.safe_base_dir, filename)

        if not self._is_safe_path(filepath):
            return {"success": False, "message": "Access to this path is not allowed"}

        try:
            # Create parent directories if needed
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Write the file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            return {
                "success": True,
                "message": f"Created file: {filename}",
                "filepath": filepath
            }
        except Exception as e:
            return {"success": False, "message": f"Failed to create file: {str(e)}"}

    def _read_file(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Read a file's contents"""
        filename = entities.get("filename") or entities.get("file_name") or entities.get("name")

        if not filename:
            return {"success": False, "message": "No filename specified"}

        filepath = os.path.join(self.safe_base_dir, filename)

        if not self._is_safe_path(filepath):
            return {"success": False, "message": "Access to this path is not allowed"}

        if not os.path.exists(filepath):
            return {"success": False, "message": f"File not found: {filename}"}

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            return {
                "success": True,
                "message": f"Read file: {filename}",
                "content": content,
                "filepath": filepath
            }
        except Exception as e:
            return {"success": False, "message": f"Failed to read file: {str(e)}"}

    def _delete_file(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Delete a file"""
        filename = entities.get("filename") or entities.get("file_name") or entities.get("name")

        if not filename:
            return {"success": False, "message": "No filename specified"}

        filepath = os.path.join(self.safe_base_dir, filename)

        if not self._is_safe_path(filepath):
            return {"success": False, "message": "Access to this path is not allowed"}

        if not os.path.exists(filepath):
            return {"success": False, "message": f"File not found: {filename}"}

        try:
            os.remove(filepath)
            return {
                "success": True,
                "message": f"Deleted file: {filename}",
                "filepath": filepath
            }
        except Exception as e:
            return {"success": False, "message": f"Failed to delete file: {str(e)}"}

    def _move_file(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Move or rename a file"""
        source = entities.get("source") or entities.get("filename") or entities.get("file_name")
        destination = entities.get("destination") or entities.get("dest") or entities.get("new_name")

        if not source or not destination:
            return {"success": False, "message": "Source and destination must be specified"}

        src_path = os.path.join(self.safe_base_dir, source)
        dest_path = os.path.join(self.safe_base_dir, destination)

        if not self._is_safe_path(src_path) or not self._is_safe_path(dest_path):
            return {"success": False, "message": "Access to this path is not allowed"}

        if not os.path.exists(src_path):
            return {"success": False, "message": f"Source file not found: {source}"}

        try:
            # Create destination directory if needed
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            shutil.move(src_path, dest_path)
            return {
                "success": True,
                "message": f"Moved {source} to {destination}",
                "source": src_path,
                "destination": dest_path
            }
        except Exception as e:
            return {"success": False, "message": f"Failed to move file: {str(e)}"}

    def _copy_file(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Copy a file"""
        source = entities.get("source") or entities.get("filename")
        destination = entities.get("destination") or entities.get("dest")

        if not source or not destination:
            return {"success": False, "message": "Source and destination must be specified"}

        src_path = os.path.join(self.safe_base_dir, source)
        dest_path = os.path.join(self.safe_base_dir, destination)

        if not self._is_safe_path(src_path) or not self._is_safe_path(dest_path):
            return {"success": False, "message": "Access to this path is not allowed"}

        if not os.path.exists(src_path):
            return {"success": False, "message": f"Source file not found: {source}"}

        try:
            # Create destination directory if needed
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            shutil.copy2(src_path, dest_path)
            return {
                "success": True,
                "message": f"Copied {source} to {destination}",
                "source": src_path,
                "destination": dest_path
            }
        except Exception as e:
            return {"success": False, "message": f"Failed to copy file: {str(e)}"}

    def _list_files(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """List files in a directory"""
        directory = entities.get("directory") or entities.get("path") or "."

        dirpath = os.path.join(self.safe_base_dir, directory)

        if not self._is_safe_path(dirpath):
            return {"success": False, "message": "Access to this path is not allowed"}

        if not os.path.exists(dirpath):
            return {"success": False, "message": f"Directory not found: {directory}"}

        try:
            files = []
            for item in os.listdir(dirpath):
                item_path = os.path.join(dirpath, item)
                if os.path.isfile(item_path):
                    stat = os.stat(item_path)
                    files.append({
                        "name": item,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })

            return {
                "success": True,
                "message": f"Found {len(files)} files in {directory}",
                "files": files,
                "count": len(files)
            }
        except Exception as e:
            return {"success": False, "message": f"Failed to list files: {str(e)}"}

    def _search_files(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Search for files by pattern"""
        pattern = entities.get("pattern") or entities.get("query") or "*"
        directory = entities.get("directory") or entities.get("path") or "."

        dirpath = os.path.join(self.safe_base_dir, directory)

        if not self._is_safe_path(dirpath):
            return {"success": False, "message": "Access to this path is not allowed"}

        try:
            # Build search pattern
            search_pattern = os.path.join(dirpath, f"**/*{pattern}*")

            matches = []
            for filepath in glob.glob(search_pattern, recursive=True):
                if os.path.isfile(filepath):
                    rel_path = os.path.relpath(filepath, self.safe_base_dir)
                    matches.append(rel_path)

            return {
                "success": True,
                "message": f"Found {len(matches)} files matching '{pattern}'",
                "files": matches,
                "count": len(matches)
            }
        except Exception as e:
            return {"success": False, "message": f"Failed to search files: {str(e)}"}
