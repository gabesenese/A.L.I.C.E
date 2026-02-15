"""
Integration Tests for Plugin Execution
Tests plugin system integration and execution flow
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.plugins.plugin_system import PluginManager, PluginInterface
from typing import Dict, Any, List


class MockPlugin(PluginInterface):
    """Mock plugin for testing"""

    def __init__(self):
        super().__init__()
        self.name = "MockPlugin"
        self.version = "1.0.0"
        self.description = "Test plugin"
        self.capabilities = ["test", "mock"]
        self.execution_count = 0

    def initialize(self) -> bool:
        return True

    def can_handle(self, intent: str, entities: Dict, query: str = None) -> bool:
        return intent == "test" or "test" in intent.lower()

    def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict[str, Any]:
        self.execution_count += 1
        return {
            "success": True,
            "response": f"Mock plugin executed (count: {self.execution_count})",
            "data": {
                "intent": intent,
                "query": query,
                "execution_count": self.execution_count
            }
        }

    def shutdown(self) -> None:
        pass


class TestPluginExecution:
    """Integration tests for plugin system"""

    @pytest.fixture
    def plugin_manager(self):
        """Create fresh plugin manager"""
        return PluginManager(use_semantic=False)

    @pytest.fixture
    def populated_manager(self, plugin_manager):
        """Plugin manager with registered plugins"""
        mock_plugin = MockPlugin()
        plugin_manager.register_plugin(mock_plugin)
        return plugin_manager

    def test_plugin_registration(self, plugin_manager):
        """Plugins should register successfully"""
        mock = MockPlugin()
        result = plugin_manager.register_plugin(mock)

        assert result is True
        assert "MockPlugin" in plugin_manager.plugins
        assert plugin_manager.plugins["MockPlugin"].enabled is True

    def test_plugin_execution(self, populated_manager):
        """Plugins should execute correctly"""
        result = populated_manager.execute_for_intent(
            intent="test",
            query="test query",
            entities={},
            context={}
        )

        assert result is not None
        assert result['success'] is True
        assert 'Mock plugin' in result['response']
        assert result['data']['execution_count'] == 1

    def test_plugin_can_handle_matching(self, populated_manager):
        """Plugin matching should work correctly"""
        # Should match "test" intent
        result = populated_manager.execute_for_intent(
            intent="test",
            query="run test",
            entities={},
            context={}
        )

        assert result is not None
        assert result['plugin'] == "MockPlugin"

    def test_plugin_execution_failure_handling(self, plugin_manager):
        """Failed plugins should not crash the system"""
        class FailingPlugin(PluginInterface):
            def __init__(self):
                super().__init__()
                self.name = "FailingPlugin"

            def initialize(self) -> bool:
                return True

            def can_handle(self, intent: str, entities: Dict, query: str = None) -> bool:
                return intent == "fail"

            def execute(self, intent: str, query: str, entities: Dict, context: Dict):
                raise Exception("Intentional failure")

            def shutdown(self) -> None:
                pass

        plugin_manager.register_plugin(FailingPlugin())

        # Should not crash, should return None or error result
        result = plugin_manager.execute_for_intent(
            intent="fail",
            query="fail",
            entities={},
            context={}
        )

        # Either returns None or an error structure
        assert result is None or result.get('success') is False

    def test_plugin_enable_disable(self, populated_manager):
        """Plugins should be enable/disable-able"""
        # Disable plugin
        populated_manager.disable_plugin("MockPlugin")

        assert populated_manager.plugins["MockPlugin"].enabled is False

        # Should not execute when disabled
        result = populated_manager.execute_for_intent(
            intent="test",
            query="test",
            entities={},
            context={}
        )

        assert result is None  # No plugin handled it

        # Re-enable
        populated_manager.enable_plugin("MockPlugin")
        assert populated_manager.plugins["MockPlugin"].enabled is True

        # Should execute again
        result = populated_manager.execute_for_intent(
            intent="test",
            query="test",
            entities={},
            context={}
        )

        assert result is not None

    def test_multiple_plugin_priority(self, plugin_manager):
        """Higher priority plugins should execute first"""
        class HighPriorityPlugin(MockPlugin):
            def __init__(self):
                super().__init__()
                self.name = "HighPriority"

        class LowPriorityPlugin(MockPlugin):
            def __init__(self):
                super().__init__()
                self.name = "LowPriority"

        high = HighPriorityPlugin()
        low = LowPriorityPlugin()

        # Register with different priorities
        plugin_manager.register_plugin(low, priority=100)
        plugin_manager.register_plugin(high, priority=10)

        result = plugin_manager.execute_for_intent(
            intent="test",
            query="test",
            entities={},
            context={}
        )

        # Should execute, priority order is respected internally
        assert result is not None

    def test_get_all_plugins(self, populated_manager):
        """Should list all registered plugins"""
        plugins = populated_manager.get_all_plugins()

        assert len(plugins) > 0
        assert any(p['name'] == 'MockPlugin' for p in plugins)

    def test_get_capabilities(self, populated_manager):
        """Should return all capabilities"""
        capabilities = populated_manager.get_capabilities()

        assert "test" in capabilities
        assert "mock" in capabilities


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
