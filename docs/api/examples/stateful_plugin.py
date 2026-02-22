"""
Stateful Plugin Example
Demonstrates state management and persistence
"""

from ai.plugins.plugin_system import PluginInterface
from typing import Dict, Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class StatefulCounterPlugin(PluginInterface):
    """
    Plugin that maintains state across sessions.

    This demonstrates:
    - State management
    - Persistence (saving/loading state)
    - Multiple stateful operations
    - State file handling
    """

    def __init__(self):
        """Initialize the plugin"""
        super().__init__()

        # Required attributes
        self.name = "StatefulCounterPlugin"
        self.version = "1.0.0"
        self.description = "Counts interactions and persists state"
        self.capabilities = ["count", "stats"]

        # State
        self.state_file = Path("data/stateful_plugin_state.json")
        self.total_count = 0
        self.session_count = 0
        self.history = []

    def initialize(self) -> bool:
        """
        Initialize plugin and load persisted state.
        """
        try:
            # Create data directory if needed
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            # Load previous state
            if self.state_file.exists():
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)

                self.total_count = state_data.get('total_count', 0)
                self.history = state_data.get('history', [])

                logger.info(f"{self.name} loaded state: {self.total_count} total interactions")
            else:
                logger.info(f"{self.name} starting fresh (no saved state)")

            return True

        except Exception as e:
            logger.error(f"{self.name} initialization failed: {e}")
            return False

    def can_handle(self, intent: str, entities: Dict, query: str = None) -> bool:
        """Check if plugin can handle the request"""
        if intent in ["count", "stats", "counter"]:
            return True

        if query:
            keywords = ["count", "how many", "stats", "statistics"]
            query_lower = query.lower()
            return any(kw in query_lower for kw in keywords)

        return False

    def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict[str, Any]:
        """
        Execute plugin functionality with state updates.
        """
        try:
            from datetime import datetime

            # Determine action
            if "stats" in query.lower() or intent == "stats":
                action_type = "get_stats"
            else:
                action_type = "increment_count"

            # Update state
            if action_type == "increment_count":
                self.total_count += 1
                self.session_count += 1

                # Add to history
                self.history.append({
                    "timestamp": datetime.now().isoformat(),
                    "query": query,
                    "count": self.total_count
                })

                # Keep only last 100 entries
                if len(self.history) > 100:
                    self.history = self.history[-100:]

                # Save state
                self._save_state()

                return {
                    "success": True,
                    "action": "increment_count",
                    "data": {
                        "total_count": self.total_count,
                        "session_count": self.session_count
                    },
                    "formulate": True
                }

            else:  # get_stats
                return {
                    "success": True,
                    "action": "get_stats",
                    "data": {
                        "total_count": self.total_count,
                        "session_count": self.session_count,
                        "history_size": len(self.history),
                        "first_use": self.history[0]["timestamp"] if self.history else None,
                        "last_use": self.history[-1]["timestamp"] if self.history else None
                    },
                    "formulate": True
                }

        except Exception as e:
            logger.error(f"{self.name} execution error: {e}")
            return {
                "success": False,
                "action": intent,
                "data": {},
                "response": f"Error: {str(e)}"
            }

    def _save_state(self) -> None:
        """
        Persist state to disk.
        """
        try:
            state_data = {
                "total_count": self.total_count,
                "history": self.history
            }

            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)

            logger.debug(f"{self.name} state saved")

        except Exception as e:
            logger.error(f"{self.name} failed to save state: {e}")

    def shutdown(self) -> None:
        """
        Cleanup and save final state.
        """
        try:
            logger.info(f"{self.name} shutting down")
            logger.info(f"  Total count: {self.total_count}")
            logger.info(f"  Session count: {self.session_count}")

            # Save final state
            self._save_state()

        except Exception as e:
            logger.error(f"{self.name} shutdown error: {e}")


# Example usage
if __name__ == "__main__":
    plugin = StatefulCounterPlugin()

    if plugin.initialize():
        print(f" Initialized. Total count: {plugin.total_count}")

        # Simulate interactions
        for i in range(3):
            result = plugin.execute(
                intent="count",
                query=f"count this {i}",
                entities={},
                context={}
            )
            print(f" Execution {i+1}: {result['data']}")

        # Get stats
        stats = plugin.execute(
            intent="stats",
            query="show stats",
            entities={},
            context={}
        )
        print(f" Stats: {stats['data']}")

        # Shutdown
        plugin.shutdown()
        print(" Shut down")

        # Verify persistence
        print("\\nVerifying persistence...")
        plugin2 = StatefulCounterPlugin()
        plugin2.initialize()
        print(f" Reloaded. Total count: {plugin2.total_count}")
        plugin2.shutdown()
