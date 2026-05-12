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

import logging
import re
from typing import Dict, List, Optional, Any
from datetime import datetime

# Set up logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Import the proper plugin interface
from ai.plugins.plugin_system import PluginInterface
from ai.memory.memory_system import MemorySystem
from ai.memory.memory_rights_service import MemoryRightsService


class MemoryPlugin(PluginInterface):
    """Plugin for managing user preferences and conversation memory"""

    def __init__(self, memory_system: Optional[MemorySystem] = None):
        super().__init__()
        self.name = "Memory Plugin"
        self.version = "1.0.0"
        self.description = (
            "Store, recall, and search user preferences and conversation history"
        )
        self.enabled = True
        self.capabilities = [
            "store_preference",
            "recall_memory",
            "search_memory",
            "delete_memory",
        ]
        self.commands = [
            "remember",
            "save this",
            "keep in mind",
            "what do you remember",
            "do you remember",
            "recall",
            "search conversations",
            "find our discussion",
            "what did we talk about",
            "forget",
            "clear memory",
            "delete",
        ]

        # Use provided memory system or create new one
        if memory_system:
            self.memory = memory_system
        else:
            self.memory = MemorySystem()
        self.memory_rights = MemoryRightsService(self.memory)
        self.pending_memory_delete: Dict[str, Any] = {}

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
            "memory:store",
            "memory:recall",
            "memory:search",
            "memory:delete",
            "memory:delete_topic",
            "memory:delete_topic_confirm",
            "memory:delete_conversation",
            "memory:forget_topic",
            "memory:show_topic",
            "memory:delete_topic_preview",
            "memory:suppress_topic",
            "memory:delete_all_conversation_memory",
            "memory:delete_topic_pending",
            "store_preference",
            "recall_memory",
            "search_memory",
            "delete_memory",
        ]
        return intent in memory_intents or str(intent or "").startswith("memory:")

    def execute(
        self, intent: str, query: str, entities: Dict, context: Dict
    ) -> Dict[str, Any]:
        """Execute memory operation based on intent"""
        try:
            # Use the existing handle_request method
            merged_context = dict(context or {})
            merged_context.setdefault("user_input", str(query or ""))
            if query:
                entities = dict(entities or {})
                entities.setdefault("query", str(query))
            result = self.handle_request(intent, entities, merged_context)

            # Ensure result has required fields
            if "success" not in result:
                result["success"] = result.get("status") == "success"
            if "response" not in result:
                result["response"] = result.get("message", "Operation completed")

            return result
        except Exception as e:
            logger.error(f"Error executing memory operation: {e}")
            return {"success": False, "response": f"Error: {str(e)}", "error": str(e)}

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    def get_commands(self) -> List[str]:
        return self.commands

    def is_enabled(self) -> bool:
        return self.enabled

    def handle_request(
        self, intent: str, entities: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
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
            action = (
                intent.split(":")[-1] if ":" in intent else entities.get("action", "")
            )

            if action == "store":
                return self._store_preference(entities, context)
            elif action == "recall":
                return self._recall_memory(entities, context)
            elif action == "search":
                return self._search_memory(entities, context)
            elif action in {
                "delete",
                "delete_topic",
                "delete_topic_confirm",
                "delete_conversation",
                "forget_topic",
                "show_topic",
                "delete_topic_preview",
                "suppress_topic",
                "delete_all_conversation_memory",
                "delete_topic_pending",
            }:
                query = str(context.get("user_input") or entities.get("query") or "").strip()
                return self._delete_memory_flow(action=action, query=query, entities=entities, context=context)
            elif action == "show_topic":
                topic = self._extract_topic(str(context.get("user_input") or query or ""), entities)
                items = self.memory_rights.show_memories(topic)
                return {"success": True, "response": f"I found {len(items)} local memories for that topic.", "items": items, "count": len(items)}
            elif action == "delete_topic_preview":
                topic = self._extract_topic(str(context.get("user_input") or query or ""), entities)
                preview = self.memory_rights.preview_topic_delete(topic).to_dict()
                self.pending_memory_delete = {
                    "scope": "topic",
                    "topic": topic,
                    "matched_memory_ids": list(preview.get("matched_ids") or []),
                    "awaiting_confirmation": True,
                    "requested_at": datetime.now().isoformat(),
                }
                return {"success": True, "response": f"I found {int(preview.get('count', 0))} local memories matching that topic. Confirm deletion?", "preview": preview, "deletion_executed": False}
            elif action == "suppress_topic":
                topic = self._extract_topic(str(context.get("user_input") or query or ""), entities)
                out = self.memory_rights.suppress_topic(topic)
                return {"success": True, "response": f"Suppressed topic '{topic}' from future recall preference.", **out}
            else:
                return {
                    "success": False,
                    "message": f"Unknown memory operation: {action}",
                    "action": action,
                }

        except Exception as e:
            logger.error(f"Error handling memory operation: {e}")
            return {
                "success": False,
                "message": f"Memory operation failed: {str(e)}",
                "error": str(e),
            }

    def _store_preference(
        self, entities: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Store a user preference or fact"""
        content = (
            entities.get("content")
            or entities.get("text")
            or context.get("user_input", "")
        )

        if not content:
            return {"success": False, "message": "No content to store"}

        try:
            # Extract key-value if possible
            topic = entities.get("topic") or "general"

            # Store as episodic memory
            {
                "content": content,
                "topic": topic,
                "timestamp": datetime.now().isoformat(),
                "type": "preference",
            }

            # Use memory system's add method
            success = self.memory.add_episodic_memory(
                content=content, metadata={"topic": topic, "type": "preference"}
            )

            if success:
                return {
                    "success": True,
                    "message": f"I'll remember that: {content}",
                    "stored": content,
                }
            else:
                return {"success": False, "message": "Failed to store preference"}

        except Exception as e:
            logger.error(f"Error storing preference: {e}")
            return {
                "success": False,
                "message": f"Failed to store preference: {str(e)}",
            }

    def _recall_memory(
        self, entities: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
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
                        "count": len(memories),
                    }
                else:
                    return {
                        "success": True,
                        "message": "I don't have any specific memories stored yet.",
                        "memories": [],
                        "count": 0,
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
                    "count": len(results),
                }
            else:
                return {
                    "success": True,
                    "message": f"I don't have any memories about {topic}.",
                    "memories": [],
                    "count": 0,
                }

        except Exception as e:
            logger.error(f"Error recalling memory: {e}")
            return {"success": False, "message": f"Failed to recall memory: {str(e)}"}

    def _search_memory(
        self, entities: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
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
                    "count": len(results),
                }
            else:
                return {
                    "success": True,
                    "message": f"No conversations found about '{query}'.",
                    "results": [],
                    "count": 0,
                }

        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return {"success": False, "message": f"Failed to search memory: {str(e)}"}

    def _extract_topic(self, text: str, entities: Dict[str, Any]) -> str:
        if entities.get("topic"):
            return str(entities.get("topic") or "").strip()
        low = str(text or "").lower()
        patterns = [
            r"(?:topic|memories|memory)\s+about\s+(.+)$",
            r"(?:just\s+)?the\s+topic\s+about\s+(.+)$",
            r"forget\s+the\s+topic\s+about\s+(.+)$",
            r"delete\s+the\s+topic\s+about\s+(.+)$",
        ]
        for pattern in patterns:
            m = re.search(pattern, low)
            if m:
                return str(m.group(1) or "").strip(" .,!?:;")
        return ""

    @staticmethod
    def _is_delete_confirmation(text: str) -> bool:
        low = str(text or "").lower()
        return any(
            token in low
            for token in (
                "yes",
                "yes delete",
                "delete them",
                "delete it now",
                "confirm delete",
                "do it",
            )
        )

    def _delete_memory_flow(
        self,
        *,
        action: str,
        query: str,
        entities: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        text = str(query or context.get("user_input") or "").strip()
        topic = self._extract_topic(text, entities)
        pending = dict(self.pending_memory_delete or {})

        try:
            if action in {"delete_conversation", "delete_all_conversation_memory", "delete_topic_pending"} and not topic:
                self.pending_memory_delete = {
                    "scope": "needs_clarification",
                    "requested_at": datetime.now().isoformat(),
                    "original_user_input": text,
                }
                return {
                    "success": True,
                    "response": "I can do that, but deletion needs a scope. Do you want the whole conversation memory deleted, or only a specific topic?",
                    "requires_follow_up": True,
                    "memory_delete_pending": dict(self.pending_memory_delete),
                    "deletion_executed": False,
                }

            if pending.get("scope") == "needs_clarification" and topic:
                preview = self.memory_rights.preview_topic_delete(topic).to_dict()
                self.pending_memory_delete = {
                    "scope": "topic",
                    "topic": topic,
                    "matched_memory_ids": list(preview.get("matched_ids") or []),
                    "awaiting_confirmation": True,
                    "requested_at": datetime.now().isoformat(),
                }
                return {
                    "success": True,
                    "response": f"I found {int(preview.get('count', 0))} local memories matching that topic. Confirm deletion?",
                    "requires_follow_up": True,
                    "preview": preview,
                    "deletion_executed": False,
                }

            if pending.get("awaiting_confirmation") and self._is_delete_confirmation(text):
                topic_label = str(pending.get("topic") or "that topic").strip()
                delete_result = self.memory_rights.delete_topic(topic_label, confirmed=True).to_dict()
                remaining = int(delete_result.get("remaining_count", 0))
                status = str(delete_result.get("verification_status") or "partial")
                self.pending_memory_delete = {}
                limitation = (
                    "I deleted from Alice's local memory store and vector index. I cannot guarantee erasure of existing terminal/log files."
                )
                if delete_result.get("deleted_count", 0) <= 0:
                    return {
                        "success": False,
                        "response": "I could not complete a verified memory deletion. I did not claim deletion.",
                        "deletion_executed": False,
                        "verification_status": "failed",
                    }
                return {
                    "success": True,
                    "response": f"Deleted {int(delete_result.get('deleted_count', 0))} memories related to that topic from local memory and vector index. Verification: {status}. {limitation}",
                    "deleted_count": int(delete_result.get("deleted_count", 0)),
                    "verification_status": status,
                    "remaining_count": remaining,
                    "deletion_executed": True,
                    "persisted": bool(delete_result.get("persisted")),
                }

            if topic:
                preview = self.memory_rights.preview_topic_delete(topic).to_dict()
                self.pending_memory_delete = {
                    "scope": "topic",
                    "topic": topic,
                    "matched_memory_ids": list(preview.get("matched_ids") or []),
                    "awaiting_confirmation": True,
                    "requested_at": datetime.now().isoformat(),
                }
                return {
                    "success": True,
                    "response": f"I found {int(preview.get('count', 0))} memories that appear related to that topic. Confirm delete?",
                    "requires_follow_up": True,
                    "preview": preview,
                    "deletion_executed": False,
                }

            self.pending_memory_delete = {
                "scope": "needs_clarification",
                "requested_at": datetime.now().isoformat(),
                "original_user_input": text,
            }
            return {
                "success": True,
                "response": "I can do that, but I need to confirm scope first: whole conversation memory or a specific topic?",
                "requires_follow_up": True,
                "memory_delete_pending": dict(self.pending_memory_delete),
                "deletion_executed": False,
            }
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return {
                "success": False,
                "response": "I can suppress this topic from future recall, but actual deletion failed and I did not claim deletion.",
                "error": str(e),
                "deletion_executed": False,
            }
