"""
Async Plugin Example
Demonstrates asynchronous operations
"""

from ai.plugins.plugin_system import PluginInterface
from typing import Dict, Any
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class AsyncAPIPlugin(PluginInterface):
    """
    Plugin that performs async operations (e.g., API calls).

    This demonstrates:
    - Async/await patterns
    - Non-blocking operations
    - Thread pool execution
    - Timeout handling
    - Concurrent requests
    """

    def __init__(self):
        """Initialize the plugin"""
        super().__init__()

        # Required attributes
        self.name = "AsyncAPIPlugin"
        self.version = "1.0.0"
        self.description = "Makes async API calls"
        self.capabilities = ["async_api", "fetch"]

        # Async components
        self.executor = None
        self.timeout = 10.0  # 10 second timeout

    def initialize(self) -> bool:
        """
        Initialize async components.
        """
        try:
            # Create thread pool executor
            self.executor = ThreadPoolExecutor(max_workers=5)

            logger.info(f"{self.name} initialized with async support")
            return True

        except Exception as e:
            logger.error(f"{self.name} initialization failed: {e}")
            return False

    def can_handle(self, intent: str, entities: Dict, query: str = None) -> bool:
        """Check if plugin can handle the request"""
        if intent in ["fetch", "api_call", "async"]:
            return True

        if query:
            return "fetch" in query.lower() or "api" in query.lower()

        return False

    def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict[str, Any]:
        """
        Execute async operation.

        Note: This is a sync method that runs async code.
        """
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run async operation
            result = loop.run_until_complete(
                self._async_execute(intent, query, entities, context)
            )

            return result

        except asyncio.TimeoutError:
            logger.error(f"{self.name} timeout")
            return {
                "success": False,
                "action": intent,
                "data": {},
                "response": f"Operation timed out after {self.timeout} seconds"
            }

        except Exception as e:
            logger.error(f"{self.name} execution error: {e}")
            return {
                "success": False,
                "action": intent,
                "data": {},
                "response": f"Error: {str(e)}"
            }

    async def _async_execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict[str, Any]:
        """
        Actual async implementation.
        """
        try:
            # Simulate async API call with timeout
            data = await asyncio.wait_for(
                self._fetch_data_async(query),
                timeout=self.timeout
            )

            return {
                "success": True,
                "action": "async_fetch",
                "data": {
                    "result": data,
                    "query": query,
                    "async": True
                },
                "formulate": True
            }

        except asyncio.TimeoutError:
            raise  # Re-raise for outer handler

        except Exception as e:
            logger.error(f"Async execution error: {e}")
            raise

    async def _fetch_data_async(self, query: str) -> Dict[str, Any]:
        """
        Simulated async data fetch.

        In real plugin, this would be:
        - async with aiohttp.ClientSession() as session:
        -     async with session.get(url) as response:
        -         return await response.json()
        """
        # Simulate network delay
        await asyncio.sleep(0.5)

        # Simulate API response
        return {
            "query": query,
            "timestamp": asyncio.get_event_loop().time(),
            "data": f"Fetched data for: {query}"
        }

    async def _fetch_multiple_async(self, queries: list) -> list:
        """
        Fetch multiple queries concurrently.

        Demonstrates concurrent async operations.
        """
        # Create tasks for all queries
        tasks = [self._fetch_data_async(query) for query in queries]

        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return results

    def shutdown(self) -> None:
        """
        Cleanup async resources.
        """
        try:
            logger.info(f"{self.name} shutting down")

            # Shutdown executor
            if self.executor:
                self.executor.shutdown(wait=True)
                logger.info("Executor shut down")

        except Exception as e:
            logger.error(f"{self.name} shutdown error: {e}")


# Example usage
if __name__ == "__main__":
    async def demo():
        """Demonstrate async plugin usage"""
        plugin = AsyncAPIPlugin()

        if plugin.initialize():
            print(" Plugin initialized")

            # Single async call
            result = plugin.execute(
                intent="fetch",
                query="test data",
                entities={},
                context={}
            )
            print(f" Single fetch: {result['data']}")

            # Multiple concurrent calls
            queries = ["query1", "query2", "query3"]
            results = await plugin._fetch_multiple_async(queries)
            print(f" Concurrent fetches: {len(results)} results")

            # Shutdown
            plugin.shutdown()
            print(" Plugin shut down")

    # Run demo
    asyncio.run(demo())
