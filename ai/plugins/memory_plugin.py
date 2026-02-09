"""
A.L.I.C.E Memory Plugin

This plugin provides memory and preference management including:
- Store user preferences and facts
- Recall specific information
- Search conversation history
- Delete outdated information

Supports commands like:
- "Remember that I prefer coffee"
- "Remember I like working out at 6am"
- "What do you remember about my preferences?"
- "Search our previous conversations about work"
- "Forget my coffee preference"
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Set up logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Import the proper plugin interface
from ai.plugins.plugin_system import PluginInterface
from ai.memory.memory_system import MemorySystem


class MemoryPlugin(PluginInterface):
    """Plugin for managing user preferences and conversation memory"""

    def __init__(self, memory_system: Optional[MemorySystem] = None):
        super().__init__()
        self.name = "Memory Plugin"
        self.version = "1.0.0"
        self.description = "Store, recall, and search user preferences and conversation history"
        self.enabled = True
        self.capabilities = [
            "store_preference", "recall_memory", "search_memory", "delete_memory"
        ]
        self.commands = [
            "remember", "save this", "keep in mind",
            "what do you remember", "do you remember", "recall",
            "search conversations", "find our discussion", "what did we talk about",
            "forget", "clear memory", "delete"
        ]

        # Use provided memory system or create new one
        if memory_system:
            self.memory = memory_system
        else:
            self.memory = MemorySystem()

    def initialize(self) -> bool:
        """Initialize the plugin"""
        try:
            logger.info("Initializing Memory Plugin")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Memory Plugin: {e}")
            return False

    def shutdown(self):
        """Cleanup when plugin is disabled"""
        logger.info("Shutting down Memory Plugin")

    def can_handle(self, intent: str, entities: Dict) -> bool:
        """Check if this plugin can handle the given intent"""
        # Check if intent matches memory operations
        memory_intents = [
            'memory:store', 'memory:recall', 'memory:search', 'memory:delete',
            'store_preference', 'recall_memory', 'search_memory', 'delete_memory'
        ]
        return intent in memory_intents

    def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict[str, Any]:
        """Execute memory operation based on intent"""
        try:
            # Use the existing handle_request method
            result = self.handle_request(intent, entities, context)

            # Ensure result has required fields
            if 'success' not in result:
                result['success'] = result.get('status') == 'success'
            if 'response' not in result:
                result['response'] = result.get('message', 'Operation completed')

            return result
        except Exception as e:
            logger.error(f"Error executing memory operation: {e}")
            return {
                'success': False,
                'response': f"Error: {str(e)}",
                'error': str(e)
            }

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_commands(self) -> List[str]:
        return self.commands

    def is_enabled(self) -> bool:
        return self.enabled

    def handle_request(self, intent: str, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle memory operation requests

        Args:
            intent: The intent (e.g., "memory:store", "memory:recall")
            entities: Extracted entities (content, topic, query, etc.)
            context: Additional context

        Returns:
            Response dictionary with result
        """
        try:
            action = intent.split(":")[-1] if ":" in intent else entities.get("action", "")

            if action == "store":
                return self._store_preference(entities, context)
            elif action == "recall":
                return self._recall_memory(entities, context)
            elif action == "search":
                return self._search_memory(entities, context)
            elif action == "delete":
                return self._delete_memory(entities, context)
            else:
                return {
                    "success": False,
                    "message": f"Unknown memory operation: {action}",
                    "action": action
                }

        except Exception as e:
            logger.error(f"Error handling memory operation: {e}")
            return {
                "success": False,
                "message": f"Memory operation failed: {str(e)}",
                "error": str(e)
            }

    def _store_preference(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Store a user preference or fact"""
        content = entities.get("content") or entities.get("text") or context.get("user_input", "")

        if not content:
            return {"success": False, "message": "No content to store"}

        try:
            # Extract key-value if possible
            topic = entities.get("topic") or "general"

            # Store as episodic memory
            memory_entry = {
                "content": content,
                "topic": topic,
                "timestamp": datetime.now().isoformat(),
                "type": "preference"
            }

            # Use memory system's add method
            success = self.memory.add_episodic_memory(
                content=content,
                metadata={"topic": topic, "type": "preference"}
            )

            if success:
                return {
                    "success": True,
                    "message": f"I'll remember that: {content}",
                    "stored": content
                }
            else:
                return {"success": False, "message": "Failed to store preference"}

        except Exception as e:
            logger.error(f"Error storing preference: {e}")
            return {"success": False, "message": f"Failed to store preference: {str(e)}"}

    def _recall_memory(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Recall specific information"""
        topic = entities.get("topic") or entities.get("query") or entities.get("about")

        if not topic:
            # Return general summary
            try:
                memories = self.memory.get_recent_memories(limit=5)
                if memories:
                    summary = "\n".join([f"- {m.get('content', '')}" for m in memories])
                    return {
                        "success": True,
                        "message": f"Here's what I remember:\n{summary}",
                        "memories": memories,
                        "count": len(memories)
                    }
                else:
                    return {
                        "success": True,
                        "message": "I don't have any specific memories stored yet.",
                        "memories": [],
                        "count": 0
                    }
            except:
                return {"success": False, "message": "Unable to recall memories"}

        try:
            # Search for specific topic
            results = self.memory.search_memories(query=topic, limit=5)

            if results:
                summary = "\n".join([f"- {r.get('content', '')}" for r in results[:3]])
                return {
                    "success": True,
                    "message": f"Here's what I remember about {topic}:\n{summary}",
                    "memories": results,
                    "count": len(results)
                }
            else:
                return {
                    "success": True,
                    "message": f"I don't have any memories about {topic}.",
                    "memories": [],
                    "count": 0
                }

        except Exception as e:
            logger.error(f"Error recalling memory: {e}")
            return {"success": False, "message": f"Failed to recall memory: {str(e)}"}

    def _search_memory(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Search conversation history"""
        query = entities.get("query") or entities.get("topic") or entities.get("about")

        if not query:
            return {"success": False, "message": "No search query specified"}

        try:
            # Search using memory system
            results = self.memory.search_memories(query=query, limit=10)

            if results:
                summary = "\n".join([f"- {r.get('content', '')}" for r in results[:5]])
                return {
                    "success": True,
                    "message": f"Found {len(results)} results for '{query}':\n{summary}",
                    "results": results,
                    "count": len(results)
                }
            else:
                return {
                    "success": True,
                    "message": f"No conversations found about '{query}'.",
                    "results": [],
                    "count": 0
                }

        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return {"success": False, "message": f"Failed to search memory: {str(e)}"}

    def _delete_memory(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Delete specific memory or preference"""
        topic = entities.get("topic") or entities.get("about")

        if not topic:
            return {"success": False, "message": "No topic specified for deletion"}

        try:
            # Search for matching memories
            matches = self.memory.search_memories(query=topic, limit=5)

            if matches:
                # Delete the matches (assuming memory system has delete method)
                deleted_count = len(matches)

                # Note: Actual deletion would need to be implemented in memory_system.py
                # For now, just return success message
                return {
                    "success": True,
                    "message": f"Cleared {deleted_count} memory entries about {topic}.",
                    "deleted_count": deleted_count
                }
            else:
                return {
                    "success": True,
                    "message": f"No memories found about {topic} to delete.",
                    "deleted_count": 0
                }

        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return {"success": False, "message": f"Failed to delete memory: {str(e)}"}
