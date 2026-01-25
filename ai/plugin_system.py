"""
Plugin System for A.L.I.C.E
Extensible architecture for adding capabilities like Jarvis
Supports: Weather, Calendar, File Operations, System Control, Web Search, etc.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import importlib
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PluginInterface(ABC):
    """
    Base interface that all plugins must implement
    Similar to Jarvis's modular capabilities
    """
    
    def __init__(self):
        self.name = "BasePlugin"
        self.version = "1.0.0"
        self.enabled = True
        self.description = "Base plugin interface"
        self.capabilities = []
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin. Return True if successful."""
        pass
    
    @abstractmethod
    def can_handle(self, intent: str, entities: Dict) -> bool:
        """Check if this plugin can handle the given intent/entities"""
        pass
    
    @abstractmethod
    def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict[str, Any]:
        """
        Execute plugin functionality
        
        Returns:
            Dict with:
            - success: bool
            - response: str (message to user)
            - data: any (optional data)
            - actions: list (optional follow-up actions)
        """
        pass
    
    @abstractmethod
    def shutdown(self):
        """Cleanup when plugin is disabled"""
        pass
    
    def get_info(self) -> Dict[str, str]:
        """Get plugin information"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "enabled": self.enabled,
            "capabilities": self.capabilities
        }


class PluginManager:
    """
    Manages all plugins for A.L.I.C.E
    Handles loading, execution, and coordination
    """
    
    def __init__(self, plugins_dir: str = "plugins"):
        self.plugins_dir = plugins_dir
        self.plugins: Dict[str, PluginInterface] = {}
        self.plugin_order: List[str] = []  # Execution priority
        
        os.makedirs(plugins_dir, exist_ok=True)
        logger.info("[OK] Plugin Manager initialized")
    
    def register_plugin(self, plugin: PluginInterface, priority: int = 50):
        """
        Register a plugin
        
        Args:
            plugin: Plugin instance
            priority: Execution priority (lower = higher priority)
        """
        try:
            if plugin.initialize():
                self.plugins[plugin.name] = plugin
                self.plugin_order.append(plugin.name)
                # Sort by priority if needed
                logger.info(f"[OK] Plugin registered: {plugin.name} v{plugin.version}")
                return True
            else:
                logger.error(f"[ERROR] Plugin initialization failed: {plugin.name}")
                return False
        except Exception as e:
            logger.error(f"[ERROR] Error registering plugin {plugin.name}: {e}")
            return False
    
    def unregister_plugin(self, plugin_name: str):
        """Unregister and shutdown a plugin"""
        if plugin_name in self.plugins:
            try:
                self.plugins[plugin_name].shutdown()
                del self.plugins[plugin_name]
                self.plugin_order.remove(plugin_name)
                logger.info(f"✅ Plugin unregistered: {plugin_name}")
                return True
            except Exception as e:
                logger.error(f"❌ Error unregistering plugin {plugin_name}: {e}")
                return False
        return False
    
    def execute_for_intent(self, intent: str, query: str, entities: Dict, context: Dict) -> Optional[Dict]:
        """
        Find and execute appropriate plugin for the given intent
        
        Returns:
            Plugin execution result or None if no plugin can handle it
        """
        for plugin_name in self.plugin_order:
            plugin = self.plugins.get(plugin_name)
            
            if plugin and plugin.enabled and plugin.can_handle(intent, entities):
                logger.info(f"🔌 Executing plugin: {plugin_name}")
                try:
                    result = plugin.execute(intent, query, entities, context)
                    result['plugin'] = plugin_name
                    return result
                except Exception as e:
                    logger.error(f"[ERROR] Plugin execution error ({plugin_name}): {e}")
                    continue
        
        return None
    
    def get_all_plugins(self) -> List[Dict]:
        """Get information about all registered plugins"""
        return [plugin.get_info() for plugin in self.plugins.values()]
    
    def get_capabilities(self) -> List[str]:
        """Get all capabilities across all plugins"""
        capabilities = []
        for plugin in self.plugins.values():
            if plugin.enabled:
                capabilities.extend(plugin.capabilities)
        return list(set(capabilities))
    
    def enable_plugin(self, plugin_name: str):
        """Enable a plugin"""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].enabled = True
            logger.info(f"[OK] Plugin enabled: {plugin_name}")
    
    def disable_plugin(self, plugin_name: str):
        """Disable a plugin"""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].enabled = False
            logger.info(f"⏸️ Plugin disabled: {plugin_name}")


# Example Plugins

class WeatherPlugin(PluginInterface):
    """Weather information plugin"""
    
    def __init__(self):
        super().__init__()
        self.name = "WeatherPlugin"
        self.description = "Provides real-time weather information"
        self.capabilities = ["weather", "forecast", "temperature"]
    
    def initialize(self) -> bool:
        logger.info("Weather plugin initialized")
        return True
    
    def can_handle(self, intent: str, entities: Dict) -> bool:
        return intent == "weather" or "weather" in str(entities).lower()
    
    def _get_coordinates(self, location: str) -> Optional[tuple]:
        """Get lat/lon for a location using geocoding API"""
        try:
            import requests
            url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    result = data['results'][0]
                    return result['latitude'], result['longitude'], result.get('name')
        except Exception as e:
            logger.error(f"Geocoding error: {e}")
        return None
    
    def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict:
        try:
            import requests
            
            # Get location from context
            location = context.get('location')
            city = context.get('city')
            
            logger.info(f"Weather plugin - Location from context: city={city}, location={location}")
            
            # Prefer city over full location string
            if city and city != 'Unknown':
                location = city
            elif not location or location == 'Unknown':
                location = None
            
            # If location in query, extract it
            words = query.lower().split()
            if 'in' in words:
                idx = words.index('in')
                if idx + 1 < len(words):
                    location = ' '.join(words[idx+1:]).strip('?!.')
                    logger.info(f"Weather plugin - Extracted location from query: {location}")
            
            # If still no location, provide helpful response
            if not location:
                logger.warning("Weather plugin - No location available")
                return {
                    "success": True,  # Changed to True so LLM doesn't try to answer
                    "response": "I don't have your location set yet. You can either tell me the city (e.g., 'weather in Rio de Janeiro') or set your location with /location City, Country"
                }
            
            # Get coordinates for location
            coords = self._get_coordinates(location)
            if not coords:
                return {
                    "success": False,
                    "response": f"Sorry, I couldn't find '{location}'. Could you be more specific with the city name?"
                }
            
            lat, lon, location_name = coords
            
            # Get weather from Open-Meteo (free, no API key needed)
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&temperature_unit=celsius&wind_speed_unit=kmh"
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                current = data['current']
                
                # Weather code descriptions
                weather_codes = {
                    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
                    45: "foggy", 48: "foggy", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
                    61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow", 75: "heavy snow",
                    80: "rain showers", 81: "rain showers", 82: "heavy rain showers", 95: "thunderstorm"
                }
                
                temp = current['temperature_2m']
                humidity = current['relative_humidity_2m']
                wind = current['wind_speed_10m']
                condition = weather_codes.get(current['weather_code'], 'unknown')
                
                return {
                    "success": True,
                    "response": f"In {location_name}, it's currently {temp}°C with {condition}. Humidity is {humidity}% and wind speed is {wind} km/h.",
                    "data": {
                        "temperature": temp,
                        "condition": condition,
                        "humidity": humidity,
                        "wind_speed": wind,
                        "location": location_name
                    }
                }
            else:
                return {
                    "success": False,
                    "response": "Sorry, I couldn't fetch the weather data right now. Try again in a moment."
                }
                
        except Exception as e:
            logger.error(f"Weather plugin error: {e}")
            return {
                "success": False,
                "response": "I'm having trouble getting weather information right now."
            }
    
    def shutdown(self):
        logger.info("Weather plugin shutdown")


class TimePlugin(PluginInterface):
    """Time and date information plugin"""
    
    def __init__(self):
        super().__init__()
        self.name = "TimePlugin"
        self.description = "Provides time and date information"
        self.capabilities = ["time", "date", "calendar"]
    
    def initialize(self) -> bool:
        logger.info("⏰ Time plugin initialized")
        return True
    
    def can_handle(self, intent: str, entities: Dict) -> bool:
        return intent == "time" or any(word in str(entities).lower() for word in ["time", "date", "day"])
    
    def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict:
        now = datetime.now()
        
        if "date" in query.lower():
            response = f"Today is {now.strftime('%A, %B %d, %Y')}"
        else:
            response = f"The current time is {now.strftime('%I:%M %p')}"
        
        return {
            "success": True,
            "response": response,
            "data": {
                "timestamp": now.isoformat(),
                "formatted_time": now.strftime('%I:%M %p'),
                "formatted_date": now.strftime('%A, %B %d, %Y')
            }
        }
    
    def shutdown(self):
        logger.info("⏰ Time plugin shutdown")


class FileOperationsPlugin(PluginInterface):
    """File and directory operations plugin"""
    
    def __init__(self):
        super().__init__()
        self.name = "FileOperationsPlugin"
        self.description = "Handle file and directory operations"
        self.capabilities = ["file_operations", "create_file", "read_file", "list_directory"]
    
    def initialize(self) -> bool:
        logger.info("📁 File Operations plugin initialized")
        return True
    
    def can_handle(self, intent: str, entities: Dict) -> bool:
        return intent == "file_operation" or intent == "command" and ("file" in str(entities).lower() or "folder" in str(entities).lower())
    
    def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict:
        # Mock implementation - add actual file operations
        query_lower = query.lower()
        
        if "create" in query_lower:
            action = "create"
            response = "I can help you create a file. What would you like to name it?"
        elif "delete" in query_lower or "remove" in query_lower:
            action = "delete"
            response = "Which file would you like me to delete? Please provide the file path."
        elif "list" in query_lower or "show" in query_lower:
            action = "list"
            response = "I'll list the files in your current directory."
        else:
            action = "unknown"
            response = "I can help with file operations. What would you like to do?"
        
        return {
            "success": True,
            "response": response,
            "data": {
                "action": action,
                "requires_confirmation": action in ["delete", "create"]
            }
        }
    
    def shutdown(self):
        logger.info("📁 File Operations plugin shutdown")


class SystemControlPlugin(PluginInterface):
    """System control plugin (volume, brightness, etc.)"""
    
    def __init__(self):
        super().__init__()
        self.name = "SystemControlPlugin"
        self.description = "Control system settings"
        self.capabilities = ["volume_control", "brightness", "system_status"]
    
    def initialize(self) -> bool:
        logger.info("⚙️ System Control plugin initialized")
        return True
    
    def can_handle(self, intent: str, entities: Dict) -> bool:
        return intent == "system_control" or any(
            word in str(entities).lower() 
            for word in ["volume", "brightness", "shutdown", "restart"]
        )
    
    def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict:
        query_lower = query.lower()
        
        if "volume" in query_lower:
            if "up" in query_lower or "increase" in query_lower:
                action = "volume_up"
                response = "Volume increased by 10%"
            elif "down" in query_lower or "decrease" in query_lower:
                action = "volume_down"
                response = "Volume decreased by 10%"
            else:
                action = "volume_status"
                response = "Current volume is at 70%"
        elif "brightness" in query_lower:
            action = "brightness_adjust"
            response = "Brightness adjusted"
        else:
            action = "unknown"
            response = "I can control volume, brightness, and other system settings."
        
        return {
            "success": True,
            "response": response,
            "data": {"action": action}
        }
    
    def shutdown(self):
        logger.info("⚙️ System Control plugin shutdown")


class WebSearchPlugin(PluginInterface):
    """Web search plugin"""
    
    def __init__(self):
        super().__init__()
        self.name = "WebSearchPlugin"
        self.description = "Search the web for information"
        self.capabilities = ["web_search", "browse", "lookup"]
    
    def initialize(self) -> bool:
        logger.info("🌐 Web Search plugin initialized")
        return True
    
    def can_handle(self, intent: str, entities: Dict) -> bool:
        return intent == "search" or any(
            word in str(entities).lower() 
            for word in ["search", "google", "find", "lookup"]
        )
    
    def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict:
        # Extract search query
        search_terms = query.replace("search for", "").replace("google", "").strip()
        
        return {
            "success": True,
            "response": f"I'll search the web for '{search_terms}'. Opening your browser...",
            "data": {
                "search_query": search_terms,
                "action": "open_browser"
            }
        }
    
    def shutdown(self):
        logger.info("🌐 Web Search plugin shutdown")


# Example usage
if __name__ == "__main__":
    print("Testing Plugin System...\n")
    
    # Create plugin manager
    pm = PluginManager()
    
    # Register plugins
    pm.register_plugin(WeatherPlugin())
    pm.register_plugin(TimePlugin())
    pm.register_plugin(FileOperationsPlugin())
    pm.register_plugin(SystemControlPlugin())
    pm.register_plugin(WebSearchPlugin())
    
    # List all plugins
    print("📋 Registered Plugins:")
    for plugin_info in pm.get_all_plugins():
        print(f"  - {plugin_info['name']}: {plugin_info['description']}")
    
    print(f"\nTotal Capabilities: {len(pm.get_capabilities())}")
    print(f"   {', '.join(pm.get_capabilities())}")
    
    # Test plugin execution
    print("\n🧪 Testing Plugin Execution:\n")
    
    test_cases = [
        ("weather", "What's the weather like?", {}, {"location": "New York"}),
        ("time", "What time is it?", {}, {}),
        ("search", "Search for Python tutorials", {}, {}),
        ("file_operation", "Create a new file", {}, {}),
    ]
    
    for intent, query, entities, context in test_cases:
        print(f"Query: {query}")
        result = pm.execute_for_intent(intent, query, entities, context)
        if result:
            print(f"  Plugin: {result['plugin']}")
            print(f"  Response: {result['response']}")
        else:
            print(f"  No plugin could handle this request")
        print()
