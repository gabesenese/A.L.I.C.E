"""
Simple Plugin Example
Demonstrates basic plugin implementation
"""

from ai.plugins.plugin_system import PluginInterface
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class SimpleGreetingPlugin(PluginInterface):
    """
    Simple plugin that responds to greetings.

    This demonstrates:
    - Basic plugin structure
    - Proper initialization
    - Intent matching
    - Structured responses
    - Error handling
    """

    def __init__(self):
        """Initialize the plugin"""
        super().__init__()

        # Required attributes
        self.name = "SimpleGreetingPlugin"
        self.version = "1.0.0"
        self.description = "Responds to greetings"
        self.capabilities = ["greeting", "hello"]

        # Plugin-specific state
        self.greeting_count = 0

    def initialize(self) -> bool:
        """
        Initialize the plugin.

        Returns:
            True if initialization succeeded
        """
        try:
            # Check any dependencies here
            logger.info(f"{self.name} initialized successfully")
            return True
        except Exception as e:
            logger.error(f"{self.name} initialization failed: {e}")
            return False

    def can_handle(self, intent: str, entities: Dict, query: str = None) -> bool:
        """
        Check if this plugin can handle the request.

        Args:
            intent: Classified intent
            entities: Extracted entities
            query: Raw user query (optional but recommended)

        Returns:
            True if plugin can handle this request
        """
        # Check intent
        if intent in ["greeting", "hello", "hi"]:
            return True

        # Check query keywords
        if query:
            greeting_words = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon"]
            query_lower = query.lower()

            for word in greeting_words:
                if word in query_lower:
                    return True

        return False

    def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict[str, Any]:
        """
        Execute the plugin functionality.

        Args:
            intent: Classified intent
            query: Raw user input
            entities: Extracted entities
            context: Execution context (user info, etc.)

        Returns:
            Standardized response dictionary
        """
        try:
            # Increment counter
            self.greeting_count += 1

            # Detect time of day for appropriate greeting
            from datetime import datetime
            hour = datetime.now().hour

            if hour < 12:
                time_of_day = "morning"
            elif hour < 18:
                time_of_day = "afternoon"
            else:
                time_of_day = "evening"

            # Return structured response
            # Alice will learn to phrase this naturally
            return {
                "success": True,
                "action": "respond_greeting",
                "data": {
                    "time_of_day": time_of_day,
                    "greeting_count": self.greeting_count,
                    "user_name": context.get("user_name", "there")
                },
                "formulate": True  # Let Alice learn natural phrasing
            }

        except Exception as e:
            logger.error(f"{self.name} execution error: {e}")
            return {
                "success": False,
                "action": "respond_greeting",
                "data": {},
                "response": f"I encountered an error: {str(e)}"
            }

    def shutdown(self) -> None:
        """
        Cleanup when plugin is shut down.
        """
        try:
            logger.info(f"{self.name} shutting down. Total greetings: {self.greeting_count}")
            # Save state if needed
        except Exception as e:
            logger.error(f"{self.name} shutdown error: {e}")


# How to use this plugin:
if __name__ == "__main__":
    # Example usage
    plugin = SimpleGreetingPlugin()

    # Initialize
    if plugin.initialize():
        print(" Plugin initialized")

        # Check if it can handle a greeting
        can_handle = plugin.can_handle("greeting", {}, "Hello Alice!")
        print(f" Can handle greeting: {can_handle}")

        # Execute
        result = plugin.execute(
            intent="greeting",
            query="Hello Alice!",
            entities={},
            context={"user_name": "User"}
        )

        print(f" Result: {result}")

        # Shutdown
        plugin.shutdown()
        print(" Plugin shut down")
    else:
        print(" Plugin initialization failed")
