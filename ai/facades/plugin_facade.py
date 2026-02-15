"""
Plugin Facade for A.L.I.C.E
Extends PluginManager with capability-based execution
"""

from ai.plugins.plugin_system import PluginManager
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class PluginFacade(PluginManager):
    """Extended plugin manager with capability-based execution"""

    def execute_capability(self, capability: str, **kwargs) -> Dict[str, Any]:
        """
        Execute by capability name, routing to best plugin

        Args:
            capability: Capability identifier (e.g., 'weather', 'notes')
            **kwargs: Additional parameters including 'command' and 'context'

        Returns:
            Plugin execution result dictionary
        """
        capability_map = {
            'weather': 'WeatherPlugin',
            'time': 'TimePlugin',
            'notes': 'Notes Plugin',
            'email': 'GmailPlugin',
            'files': 'FileOperationsPlugin',
            'memory': 'MemoryPlugin',
            'documents': 'Document Plugin',
            'calendar': 'Calendar Plugin',
            'music': 'Music Control',
            'maps': 'MapsPlugin',
            'system': 'SystemControlPlugin',
        }

        plugin_name = capability_map.get(capability.lower())

        if not plugin_name or plugin_name not in self.plugins:
            return {
                "success": False,
                "message": f"Unknown capability: {capability}",
                "data": {}
            }

        plugin = self.plugins[plugin_name]

        # Execute plugin with context
        command = kwargs.get('command', '')
        context = kwargs.get('context', {})

        try:
            result = plugin.execute(
                intent=capability,
                query=command,
                entities={},
                context=context
            )
            return result
        except Exception as e:
            logger.error(f"Plugin execution failed for {capability}: {e}")
            return {
                "success": False,
                "message": f"Execution error: {str(e)}",
                "data": {}
            }

    def get_all_capabilities(self) -> Dict[str, str]:
        """
        Return map of capabilities to plugin names

        Returns:
            Dict mapping capability names to plugin descriptions
        """
        return {
            plugin.name: plugin.description if hasattr(plugin, 'description') else plugin.__class__.__doc__ or ""
            for plugin in self.plugins.values()
        }

    def list_available_plugins(self) -> List[Dict[str, Any]]:
        """
        Get detailed information about all registered plugins

        Returns:
            List of plugin info dictionaries
        """
        plugin_list = []
        for plugin in self.plugins.values():
            info = {
                'name': plugin.name,
                'enabled': plugin.enabled,
                'description': getattr(plugin, 'description', ''),
                'capabilities': getattr(plugin, 'capabilities', []),
                'version': getattr(plugin, 'version', '1.0.0')
            }
            plugin_list.append(info)

        return plugin_list

    def execute_by_intent(
        self,
        intent: str,
        query: str,
        entities: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Execute plugin based on intent classification
        (Wrapper around existing execute_for_intent for clarity)

        Args:
            intent: Classified user intent
            query: Original user query
            entities: Extracted entities
            context: Execution context

        Returns:
            Plugin execution result or None
        """
        return self.execute_for_intent(intent, query, entities, context)


# Singleton instance
_plugin_facade: Optional[PluginFacade] = None


def get_plugin_facade(plugins_dir: str = "plugins", use_semantic: bool = True) -> PluginFacade:
    """Get or create the PluginFacade singleton"""
    global _plugin_facade
    if _plugin_facade is None:
        _plugin_facade = PluginFacade(plugins_dir=plugins_dir, use_semantic=use_semantic)
    return _plugin_facade
