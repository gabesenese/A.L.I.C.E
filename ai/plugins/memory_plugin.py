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

from ai.memory.personal_memory import PersonalMemoryStore
from ai.memory.temporal_scope import Period, items_within, requested_period

# Set up logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Import the proper plugin interface
from ai.plugins.plugin_system import PluginInterface
from ai.memory.memory_system import MemorySystem


# "remember that my sister's name is Ana" -> "my sister's name is Ana"
_STORE_COMMAND_RE = re.compile(
    r"^\s*(?:(?:hey|ok|okay)\s+alice[,!]?\s*)?(?:please\s+)?(?:(?:can|could|would)\s+you\s+)?"
    r"(?:remember|save|store|note|keep in mind|don'?t forget|do not forget)(?:\s+(?:that|this))?\s*[:,-]?\s*",
    re.IGNORECASE,
)


def _fact_from_request(text: str) -> str:
    return _STORE_COMMAND_RE.sub("", str(text or ""), count=1).strip().rstrip(".!")


class MemoryPlugin(PluginInterface):
    """Plugin for managing user preferences and conversation memory"""

    def __init__(self, memory_system: Optional[MemorySystem] = None):
        super().__init__()
        self.name = "Memory Plugin"
        self.version = "1.0.0"
        self.description = "Store, recall, and search user preferences and conversation history"
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
            "store_preference",
            "recall_memory",
            "search_memory",
            "delete_memory",
        ]
        return intent in memory_intents

    def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict[str, Any]:
        """Execute memory operation based on intent"""
        try:
            # The question itself goes along, so recall can honour a time it names
            # ("what did we talk about yesterday?").
            entities = {**dict(entities or {}), "_question": query}
            result = self.handle_request(intent, entities, context)

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
                    "action": action,
                }

        except Exception as e:
            logger.error(f"Error handling memory operation: {e}")
            return {
                "success": False,
                "message": f"Memory operation failed: {str(e)}",
                "error": str(e),
            }

    # -- the seam between a sentence and the store ---------------------------
    #
    # Every call below used to name a method the memory system does not have:
    # add_episodic_memory, get_recent_memories, search_memories. Each raised
    # AttributeError into a blanket `except Exception` and came back as a polite
    # failure string, so "remember that I prefer coffee" answered "Failed to
    # store preference: 'MemorySystem' object has no attribute
    # 'add_episodic_memory'" for as long as the plugin has existed. The real API
    # is store_memory / recall_memory / get_all_memories / _remove_memory_by_id.

    def _recall_is_available(self) -> bool:
        """Whether an empty result means "nothing stored" or "cannot see".

        docs/north_star.md rule 4: a confident false answer is the worst outcome
        because the user cannot tell. "I don't have any memories about that" when
        the memories exist and the load failed is exactly that, wearing the shape
        of an honest "I don't know".
        """
        return not getattr(self.memory, "load_failed", False)

    _UNREACHABLE = "I can't reach my memory this session — it failed to load, so I can't tell you what's in it."

    def _store_preference(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Store a user preference or fact"""
        # The router fills neither "content" nor "text", and the context carries no
        # "user_input", so every "remember that ..." used to fail with "No content
        # to store". The request itself is the fact, less the command.
        content = _fact_from_request(
            entities.get("content")
            or entities.get("text")
            or context.get("user_input", "")
            or entities.get("_question", "")
        )

        if not content:
            return {"success": False, "message": "No content to store"}

        try:
            # A structured personal fact, which is where "what do you know about
            # me?" looks. Stored as a plain episodic memory it was invisible there:
            # right after "remember that my sister's name is Ana" the answer was
            # "I do not have enough saved memory yet". Something he asked to be
            # remembered also outranks a turn that merely happened, and importance
            # is what survives consolidation.
            memory_id = PersonalMemoryStore(self.memory).store_structured_memory(
                content=content,
                domain="personal_life",
                kind="personal_fact",
                scope="long_term",
                confidence=0.95,
                source="explicit_request",
                importance=0.8,
            )
        except Exception as e:
            logger.error(f"Error storing preference: {e}")
            return {"success": False, "message": f"Failed to store preference: {e}"}

        if not memory_id or memory_id == "None":
            return {"success": False, "message": "Failed to store preference"}
        return {
            "success": True,
            "message": f"I'll remember that: {content}",
            "stored": content,
            "memory_id": memory_id,
        }

    def _find(self, topic: str, limit: int = 5) -> List[Dict[str, Any]]:
        stale = PersonalMemoryStore(self.memory).invalid_ids()
        return [m for m in self._find_any(topic, limit + len(stale)) if str(m.get("id") or "") not in stale][:limit]

    def _find_any(self, topic: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Semantic recall, falling back to a substring scan.

        Recall is embedding-backed, and the embedding model is optional — on a
        machine without it every lookup returns nothing, which reads as "you
        never told me that". A literal scan is worse than semantic search and far
        better than silence.
        """
        try:
            hits = self.memory.recall_memory(topic, top_k=limit, min_similarity=0.35)
        except Exception as e:
            logger.warning(f"Semantic recall unavailable ({e}); falling back to a literal scan")
            hits = []
        if hits:
            return hits

        needle = str(topic or "").strip().lower()
        if not needle:
            return []
        return [m for m in self.memory.get_all_memories(limit=200) if needle in str(m.get("content", "")).lower()][
            :limit
        ]

    @staticmethod
    def _summarise(memories: List[Dict[str, Any]], limit: int = 3) -> str:
        return "\n".join(f"- {m.get('content', '')}" for m in memories[:limit])

    def _recall_period(self, period: Period, topic: Optional[str] = None) -> Dict[str, Any]:
        """Memories from the stretch of time the question names.

        "What did we talk about yesterday?" was answered from any day, or read
        "yesterday" as the topic and found nothing about it. The time is a
        filter, not a topic; a real topic narrows the day further when it matches.
        """
        when = period.phrase
        stale = PersonalMemoryStore(self.memory).invalid_ids()
        dated = [
            m
            for m in items_within(self.memory.get_all_memories(limit=500), period)
            if str(m.get("id") or "") not in stale
        ]
        needle = str(topic or "").strip().lower()
        if needle and needle not in when.lower():
            dated = [m for m in dated if needle in str(m.get("content", "")).lower()] or dated
        if not dated:
            return {
                "success": True,
                "message": f"I don't have anything saved from {when}.",
                "memories": [],
                "results": [],
                "count": 0,
                "recall_available": True,
            }
        return {
            "success": True,
            "message": f"Here's what I have from {when}:\n{self._summarise(dated, limit=5)}",
            "memories": dated,
            "results": dated,
            "count": len(dated),
            "recall_available": True,
        }

    def _recall_memory(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Recall specific information"""
        available = self._recall_is_available()
        topic = entities.get("topic") or entities.get("query") or entities.get("about")
        period = requested_period(str(entities.get("_question") or ""))
        if period is not None and available:
            return self._recall_period(period, topic)

        if not available:
            return {
                "success": False,
                "message": self._UNREACHABLE,
                "memories": [],
                "count": 0,
                "recall_available": False,
            }

        try:
            if not topic:
                memories = self.memory.get_all_memories(limit=5)
                if not memories:
                    return {
                        "success": True,
                        "message": "Nothing stored yet.",
                        "memories": [],
                        "count": 0,
                        "recall_available": True,
                    }
                return {
                    "success": True,
                    "message": f"Here's what I remember:\n{self._summarise(memories, limit=5)}",
                    "memories": memories,
                    "count": len(memories),
                    "recall_available": True,
                }

            results = self._find(topic)
            if not results:
                return {
                    "success": True,
                    "message": f"I don't have any memories about {topic}.",
                    "memories": [],
                    "count": 0,
                    "recall_available": True,
                }
            return {
                "success": True,
                "message": f"Here's what I remember about {topic}:\n{self._summarise(results)}",
                "memories": results,
                "count": len(results),
                "recall_available": True,
            }
        except Exception as e:
            logger.error(f"Error recalling memory: {e}")
            return {"success": False, "message": f"Failed to recall memory: {e}", "recall_available": True}

    def _search_memory(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Search conversation history"""
        query = entities.get("query") or entities.get("topic") or entities.get("about")

        period = requested_period(str(entities.get("_question") or ""))
        if period is not None:
            if not self._recall_is_available():
                return {
                    "success": False,
                    "message": self._UNREACHABLE,
                    "results": [],
                    "count": 0,
                    "recall_available": False,
                }
            return self._recall_period(period, query)

        if not query:
            return {"success": False, "message": "No search query specified"}

        if not self._recall_is_available():
            return {
                "success": False,
                "message": self._UNREACHABLE,
                "results": [],
                "count": 0,
                "recall_available": False,
            }

        try:
            results = self._find(query, limit=10)
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return {"success": False, "message": f"Failed to search memory: {e}", "recall_available": True}

        if not results:
            return {
                "success": True,
                "message": f"Nothing about '{query}'.",
                "results": [],
                "count": 0,
                "recall_available": True,
            }
        return {
            "success": True,
            "message": f"Found {len(results)} results for '{query}':\n{self._summarise(results, limit=5)}",
            "results": results,
            "count": len(results),
            "recall_available": True,
        }

    def _delete_memory(self, entities: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Delete specific memory or preference.

        This used to count the matches, return "Cleared N memory entries about
        X", and delete nothing — with a comment saying deletion "would need to be
        implemented". On a request to forget something, a false confirmation is
        the one outcome with no recovery: he believes it is gone and stops asking.
        """
        topic = entities.get("topic") or entities.get("about")

        if not topic:
            return {"success": False, "message": "No topic specified for deletion"}

        if not self._recall_is_available():
            return {"success": False, "message": self._UNREACHABLE, "deleted_count": 0, "recall_available": False}

        try:
            matches = self._find(topic)
            if not matches:
                return {
                    "success": True,
                    "message": f"Nothing stored about {topic}.",
                    "deleted_count": 0,
                    "recall_available": True,
                }

            deleted = [m for m in matches if m.get("id") and self.memory._remove_memory_by_id(m["id"])]
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            return {"success": False, "message": f"Failed to delete memory: {e}", "recall_available": True}

        if not deleted:
            return {
                "success": False,
                "message": f"Found {len(matches)} entries about {topic} but could not remove them.",
                "deleted_count": 0,
                "recall_available": True,
            }
        noun = "entry" if len(deleted) == 1 else "entries"
        return {
            "success": True,
            "message": f"Deleted {len(deleted)} {noun} about {topic}.",
            "deleted_count": len(deleted),
            "recall_available": True,
        }
