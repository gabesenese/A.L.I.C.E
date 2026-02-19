"""
Plugin System for A.L.I.C.E
Extensible architecture for adding capabilities
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

# Import semantic intent classifier
try:
    from ai.intent_classifier import get_intent_classifier
    SEMANTIC_CLASSIFICATION_AVAILABLE = True
except ImportError:
    logger.warning("Semantic intent classifier not available")
    SEMANTIC_CLASSIFICATION_AVAILABLE = False


class PluginInterface(ABC):
    """
    Base interface that all plugins must implement
    Modular architecture for extensibility
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
    def shutdown(self) -> None:
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
    Uses semantic intent classification for intelligent routing
    """
    
    def __init__(self, plugins_dir: str = "plugins", use_semantic: bool = True) -> None:
        self.plugins_dir = plugins_dir
        self.plugins: Dict[str, PluginInterface] = {}
        self.plugin_order: List[str] = []  # Execution priority
        self.use_semantic = use_semantic and SEMANTIC_CLASSIFICATION_AVAILABLE
        self.intent_classifier = None
        
        # Initialize semantic classifier if available
        if self.use_semantic:
            try:
                self.intent_classifier = get_intent_classifier()
                logger.info("[OK] Semantic intent classification enabled")
            except Exception as e:
                logger.warning(f"Semantic classifier failed to load: {e}")
                self.use_semantic = False
        
        os.makedirs(plugins_dir, exist_ok=True)
        logger.info("[OK] Plugin Manager initialized")
    
    def register_plugin(self, plugin: PluginInterface, priority: int = 50) -> None:
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
    
    def unregister_plugin(self, plugin_name: str) -> bool:
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
        Uses semantic classification to understand user intent when available
        
        Returns:
            Plugin execution result or None if no plugin can handle it
        """
        # First, try semantic classification if available
        if self.use_semantic and self.intent_classifier and query:
            try:
                semantic_result = self.intent_classifier.get_plugin_action(query, threshold=0.45)
                
                if semantic_result and semantic_result['confidence'] > 0.5:
                    plugin_name = semantic_result['plugin']
                    
                    # Map plugin name to registered plugin (handle variations)
                    plugin_map = {
                        'notes': 'Notes Plugin',
                        'email': 'GmailPlugin',  # Will fallback to email intent handler in main.py
                        'music': 'Music Control',  # MusicPlugin registers as "Music Control"
                        'calendar': 'Calendar Plugin',
                        'document': 'Document Plugin',
                        'weather': 'WeatherPlugin',
                        'time': 'TimePlugin',
                    }
                    
                    actual_plugin_name = plugin_map.get(plugin_name, plugin_name)
                    plugin = self.plugins.get(actual_plugin_name)
                    
                    if plugin and plugin.enabled:
                        logger.info(f"🔌 Semantic match: {plugin_name}:{semantic_result['action']} (confidence: {semantic_result['confidence']:.2f})")
                        try:
                            # Execute plugin with semantic action hint
                            result = plugin.execute(intent, query, entities, context)
                            result['plugin'] = actual_plugin_name
                            result['semantic_match'] = semantic_result
                            return result
                        except Exception as e:
                            logger.error(f"[ERROR] Plugin execution error ({actual_plugin_name}): {e}")
                            # Fall through to traditional matching
            except Exception as e:
                logger.warning(f"Semantic classification error: {e}")
        
        # Fallback to traditional pattern-based matching
        # Smart: evaluate all plugins first, then execute best match
        candidates = []
        for plugin_name in self.plugin_order:
            plugin = self.plugins.get(plugin_name)
            
            if plugin and plugin.enabled:
                try:
                    can_handle = plugin.can_handle(intent, entities, query)
                except TypeError:
                    can_handle = plugin.can_handle(intent, entities)
                
                if can_handle:
                    # Score candidate based on intent match and plugin priority
                    score = 1.0
                    if intent and plugin_name.lower() in intent.lower():
                        score = 2.0  # Strong match
                    candidates.append((plugin_name, plugin, score))
        
        # Execute best candidate first
        if candidates:
            # Sort by score (best first)
            candidates.sort(key=lambda x: x[2], reverse=True)
            plugin_name, plugin, score = candidates[0]
            
            logger.info(f"🔌 Executing plugin: {plugin_name} (score: {score:.1f})")
            try:
                result = plugin.execute(intent, query, entities, context)
                result['plugin'] = plugin_name
                result['match_score'] = score
                return result
            except Exception as e:
                logger.error(f"[ERROR] Plugin execution error ({plugin_name}): {e}")
                # Try next candidate if first fails
                if len(candidates) > 1:
                    plugin_name, plugin, score = candidates[1]
                    logger.info(f"🔌 Trying fallback plugin: {plugin_name}")
                    try:
                        result = plugin.execute(intent, query, entities, context)
                        result['plugin'] = plugin_name
                        result['match_score'] = score
                        return result
                    except Exception as e2:
                        logger.error(f"[ERROR] Fallback plugin also failed: {e2}")
        
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
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin"""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].enabled = True
            logger.info(f"[OK] Plugin enabled: {plugin_name}")
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin"""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].enabled = False
            logger.info(f"Plugin disabled: {plugin_name}")


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
        """Check if this plugin can handle the given intent"""
        intent_lower = (intent or "").lower()

        # Only match based on intent - don't search entire entities dict
        # (entities may contain old context data from previous queries)
        return "weather" in intent_lower

    def _is_forecast_request(self, intent: str, query: str, entities: Dict) -> bool:
        intent_lower = (intent or "").lower()
        query_lower = (query or "").lower()

        if "forecast" in intent_lower:
            return True

        if any(phrase in query_lower for phrase in [
            "forecast", "this week", "next week", "weekend", "7 day", "7-day",
            "tomorrow", "next few days", "next 7 days"
        ]):
            return True

        time_range_entities = entities.get("TIME_RANGE", []) if isinstance(entities, dict) else []
        if time_range_entities:
            return True

        return False
    
    def _get_coordinates(self, location: str) -> Optional[tuple]:
        """Get lat/lon for a location using geocoding API"""
        try:
            import requests
            url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()

                # Be defensive: API might return a list or a plain string on error
                if isinstance(data, dict) and data.get('results'):
                    result = data['results'][0]
                    lat = result.get('latitude')
                    lon = result.get('longitude')
                    name = result.get('name', location)
                    # Validate coordinates are present and numeric
                    if lat is None or lon is None:
                        logger.error(f"Geocoding returned incomplete coordinates for {location!r}")
                        return None
                    return lat, lon, name
                else:
                    logger.error(f"Geocoding returned no results for {location!r}: {data!r}")
        except Exception as e:
            logger.error(f"Geocoding error for {location!r}: {e}")
        return None
    
    def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict:
        try:
            import requests

            # Ensure context is a dict (main may pass a string summary)
            if not isinstance(context, dict):
                logger.debug(f"Weather plugin received non-dict context: {type(context)}")
                context = {}

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
                    "success": False,
                    "response": None,
                    "data": {
                        "error": "no_location",
                        "message_code": "weather:no_location"
                    }
                }
            
            # Get coordinates for location
            coords = self._get_coordinates(location)
            if not coords:
                return {
                    "success": False,
                    "response": None,
                    "data": {
                        "error": "unknown_location",
                        "location": location,
                        "message_code": "weather:unknown_location"
                    }
                }
            
            lat, lon, location_name = coords
            
            # Forecast or current
            is_forecast = self._is_forecast_request(intent, query, entities)

            if is_forecast:
                return self._get_forecast(lat, lon, location_name)

            # Get weather from Open-Meteo (free, no API key needed)
            url = (
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                "&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
                "&temperature_unit=celsius&wind_speed_unit=kmh"
            )
            try:
                response = requests.get(url, timeout=8)
            except requests.exceptions.Timeout:
                logger.warning(f"Weather API timed out for {location_name}")
                return {
                    "success": False,
                    "response": None,
                    "data": {
                        "error": "timeout",
                        "location": location_name,
                        "message_code": "weather:timeout"
                    }
                }
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Weather API connection error for {location_name}: {e}")
                return {
                    "success": False,
                    "response": None,
                    "data": {
                        "error": "connection_error",
                        "location": location_name,
                        "message_code": "weather:connection_error"
                    }
                }

            if response.status_code == 200:
                data = response.json()
                current = data.get('current', {})

                # Weather code descriptions
                weather_codes = {
                    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
                    45: "foggy", 48: "foggy", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
                    61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow", 75: "heavy snow",
                    80: "rain showers", 81: "rain showers", 82: "heavy rain showers", 95: "thunderstorm"
                }

                # Use .get() with None defaults - API may omit or null any field
                temp = current.get('temperature_2m')
                humidity = current.get('relative_humidity_2m')
                wind = current.get('wind_speed_10m')
                weather_code = current.get('weather_code')
                condition = weather_codes.get(weather_code, 'unknown') if weather_code is not None else 'unknown'

                # If no usable data came back, treat as a no-data response
                if temp is None and condition == 'unknown':
                    logger.warning(f"Weather API returned no usable data for {location_name}")
                    return {
                        "success": False,
                        "response": None,
                        "data": {
                            "error": "no_data",
                            "location": location_name,
                            "message_code": "weather:no_data"
                        }
                    }

                return {
                    "success": True,
                    "response": None,
                    "data": {
                        "temperature": temp,
                        "condition": condition,
                        "humidity": humidity,
                        "wind_speed": wind,
                        "location": location_name,
                        "plugin_type": "weather",
                        "message_code": "weather:current"
                    }
                }
            else:
                logger.warning(f"Weather API returned HTTP {response.status_code} for {location_name}")
                return {
                    "success": False,
                    "response": None,
                    "data": {
                        "error": "fetch_failed",
                        "http_status": response.status_code,
                        "message_code": "weather:fetch_failed"
                    }
                }

        except Exception as e:
            logger.error(f"Weather plugin error: {e}")
            return {
                "success": False,
                "response": None,
                "data": {
                    "error": str(e),
                    "message_code": "weather:error"
                }
            }
    
    def shutdown(self):
        logger.info("Weather plugin shutdown")

    def _get_forecast(self, lat: float, lon: float, location_name: str) -> Dict:
        """Fetch a 7-day forecast from Open-Meteo."""
        try:
            import requests

            url = (
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                "&daily=temperature_2m_max,temperature_2m_min,weather_code"
                "&forecast_days=7&temperature_unit=celsius&timezone=auto"
            )
            try:
                response = requests.get(url, timeout=8)
            except requests.exceptions.Timeout:
                logger.warning(f"Forecast API timed out for {location_name}")
                return {
                    "success": False,
                    "response": None,
                    "data": {
                        "error": "timeout",
                        "location": location_name,
                        "message_code": "weather:timeout"
                    }
                }
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Forecast API connection error for {location_name}: {e}")
                return {
                    "success": False,
                    "response": None,
                    "data": {
                        "error": "connection_error",
                        "location": location_name,
                        "message_code": "weather:connection_error"
                    }
                }

            if response.status_code != 200:
                logger.warning(f"Forecast API returned HTTP {response.status_code} for {location_name}")
                return {
                    "success": False,
                    "response": None,
                    "data": {
                        "error": "fetch_failed",
                        "http_status": response.status_code,
                        "message_code": "weather:fetch_failed"
                    }
                }

            data = response.json()
            daily = data.get("daily", {})

            weather_codes = {
                0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
                45: "foggy", 48: "foggy", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
                61: "light rain", 63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow", 75: "heavy snow",
                80: "rain showers", 81: "rain showers", 82: "heavy rain showers", 95: "thunderstorm"
            }

            forecast = []
            dates = daily.get("time", [])
            maxes = daily.get("temperature_2m_max", [])
            mins = daily.get("temperature_2m_min", [])
            codes = daily.get("weather_code", [])

            for i in range(len(dates)):
                # Values may be None if station data unavailable for that day
                high = maxes[i] if i < len(maxes) else None
                low = mins[i] if i < len(mins) else None
                code = codes[i] if i < len(codes) else None
                condition = weather_codes.get(code, "unknown") if code is not None else "unknown"
                forecast.append({
                    "date": dates[i],
                    "high": high,
                    "low": low,
                    "condition": condition
                })

            if not forecast:
                return {
                    "success": False,
                    "response": None,
                    "data": {
                        "error": "no_data",
                        "location": location_name,
                        "message_code": "weather:no_data"
                    }
                }

            return {
                "success": True,
                "response": None,
                "data": {
                    "forecast": forecast,
                    "location": location_name,
                    "plugin_type": "weather",
                    "message_code": "weather:forecast"
                }
            }

        except Exception as e:
            logger.error(f"Weather forecast error: {e}")
            return {
                "success": False,
                "response": None,
                "data": {
                    "error": str(e),
                    "message_code": "weather:error"
                }
            }


class TimePlugin(PluginInterface):
    """Time and date information plugin"""
    
    def __init__(self):
        super().__init__()
        self.name = "TimePlugin"
        self.description = "Provides time and date information"
        self.capabilities = ["time", "date", "calendar"]
    
    def initialize(self) -> bool:
        logger.info(" Time plugin initialized")
        return True
    
    def can_handle(self, intent: str, entities: Dict) -> bool:
        # Only handle explicit time/date requests based on intent
        return intent.lower() in ["time", "date"]
    
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
        logger.info(" Time plugin shutdown")


class FileOperationsPlugin(PluginInterface):
    """File and directory operations plugin"""
    
    def __init__(self):
        super().__init__()
        self.name = "FileOperationsPlugin"
        self.description = "Handle file and directory operations"
        self.capabilities = ["file_operations", "create_file", "read_file", "list_directory"]
    
    def initialize(self) -> bool:
        logger.info(" File Operations plugin initialized")
        return True
    
    def can_handle(self, intent: str, entities: Dict) -> bool:
        # Only match based on intent, not context pollution
        return intent in ["file_operation", "command"]
    
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
        logger.info(" File Operations plugin shutdown")


class SystemControlPlugin(PluginInterface):
    """System control plugin (volume, brightness, etc.)"""
    
    def __init__(self):
        super().__init__()
        self.name = "SystemControlPlugin"
        self.description = "Control system settings and launch applications"
        self.capabilities = ["volume_control", "brightness", "system_status", "app_launching"]
        self.task_executor = None
    
    def initialize(self) -> bool:
        from ai.planning.task_executor import TaskExecutor
        self.task_executor = TaskExecutor(safe_mode=True)
        self._installed_apps = None  # Cache for installed apps
        logger.info(" System Control plugin initialized")
        return True
    
    def _get_installed_apps(self) -> Dict[str, str]:
        """Get list of installed applications on Windows"""
        if self._installed_apps is not None:
            return self._installed_apps
            
        self._installed_apps = {}
        
        try:
            import subprocess
            import json
            
            # Use PowerShell to get installed apps
            powershell_cmd = """
            Get-WmiObject -Class Win32_Product | 
            Select-Object Name, InstallLocation | 
            Where-Object {$_.Name -ne $null -and $_.InstallLocation -ne $null} |
            ConvertTo-Json
            """
            
            # Also get apps from Programs folder and Start Menu
            start_menu_cmd = """
            Get-ChildItem -Path "$env:ProgramData\\Microsoft\\Windows\\Start Menu\\Programs", "$env:APPDATA\\Microsoft\\Windows\\Start Menu\\Programs" -Recurse -Include "*.lnk" |
            ForEach-Object { 
                $shell = New-Object -ComObject WScript.Shell
                $shortcut = $shell.CreateShortcut($_.FullName)
                @{
                    Name = [System.IO.Path]::GetFileNameWithoutExtension($_.Name)
                    Path = $shortcut.TargetPath
                }
            } | ConvertTo-Json
            """
            
            # Try to get installed programs
            try:
                result = subprocess.run(
                    ["powershell", "-Command", start_menu_cmd], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    apps_data = json.loads(result.stdout)
                    if isinstance(apps_data, list):
                        for app in apps_data:
                            if isinstance(app, dict) and app.get('Name') and app.get('Path'):
                                name_lower = app['Name'].lower()
                                self._installed_apps[name_lower] = app['Path']
                    elif isinstance(apps_data, dict) and apps_data.get('Name'):
                        name_lower = apps_data['Name'].lower()
                        self._installed_apps[name_lower] = apps_data['Path']
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.debug(f"Failed to parse Windows registry apps: {e}")
                pass

            # Add common protocol handlers and system apps
            protocol_apps = {
                "epic games launcher": "com.epicgames.launcher://",
                "epicgames": "com.epicgames.launcher://", 
                "epic": "com.epicgames.launcher://",
                "steam": "steam://",
                "spotify": "spotify:",
                "notepad": "notepad",
                "calculator": "calc",
                "paint": "mspaint",
                "explorer": "explorer"
            }
            self._installed_apps.update(protocol_apps)
            
            logger.info(f"[APP-DETECT] Found {len(self._installed_apps)} applications")
            
        except Exception as e:
            logger.error(f"[APP-DETECT] Error getting installed apps: {e}")
            # Fallback to minimal set
            self._installed_apps = {
                "notepad": "notepad",
                "calculator": "calc",
                "paint": "mspaint"
            }
        
        return self._installed_apps
    
    def _find_app(self, app_name: str) -> Optional[str]:
        """Find the best match for an app name"""
        apps = self._get_installed_apps()
        app_name_lower = app_name.lower()
        
        # Exact match
        if app_name_lower in apps:
            return apps[app_name_lower]
        
        # Partial match
        for installed_name, path in apps.items():
            if app_name_lower in installed_name or installed_name in app_name_lower:
                return path
        
        # Check if it might be an executable name
        if not app_name_lower.endswith('.exe'):
            exe_name = f"{app_name_lower}.exe"
            if exe_name in apps:
                return apps[exe_name]
        
        return None
    
    # Require query to contain control-related words so "just going to work" isn't hijacked
    _CONTROL_WORDS = ("volume", "brightness", "shutdown", "restart", "open", "launch", "mute", "unmute")
    
    def can_handle(self, intent: str, entities: Dict, query: str = None) -> bool:
        # Check query for control words if provided
        if query is not None:
            q = query.lower()
            if not any(w in q for w in self._CONTROL_WORDS):
                return False
        # Only match based on intent
        return intent == "system_control"
    
    def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict:
        query_lower = query.lower()
        
        # Handle application launching
        if any(word in query_lower for word in ["open", "launch", "start"]) and \
           not any(word in query_lower for word in ["volume", "brightness", "file"]):
            
            # Extract app name - look for word after "open", "launch", "start"
            import re
            app_match = re.search(r'\b(?:open|launch|start)\s+(.+?)(?:\s|$)', query_lower)
            if app_match:
                app_name = app_match.group(1).strip()
                
                # Find the app dynamically
                app_path = self._find_app(app_name)
                
                if app_path:
                    # Use TaskExecutor to actually launch the app
                    if self.task_executor:
                        result = self.task_executor.open_application(app_path)
                        if result.success:
                            return {
                                "success": True,
                                "response": None,
                                "data": {
                                    "action": "app_launch",
                                    "app": app_path,
                                    "app_name": app_name,
                                    "message_code": "system:app_opened"
                                }
                            }
                        else:
                            return {
                                "success": False,
                                "response": None,
                                "data": {
                                    "action": "app_launch_failed",
                                    "app": app_path,
                                    "app_name": app_name,
                                    "error": result.error or "launch_failed",
                                    "message_code": "system:app_launch_failed"
                                }
                            }
                    else:
                        return {
                            "success": False,
                            "response": None,
                            "data": {
                                "action": "app_launch_failed",
                                "error": "executor_unavailable",
                                "message_code": "system:executor_unavailable"
                            }
                        }
                else:
                    # App not found - provide helpful message
                    return {
                        "success": False,
                        "response": None,
                        "data": {
                            "action": "app_not_found",
                            "app": app_name,
                            "message_code": "system:app_not_found"
                        }
                    }
            else:
                return {
                    "success": False,
                    "response": None,
                    "data": {
                        "action": "app_launch_help",
                        "message_code": "system:app_name_missing"
                    }
                }
        
        # Handle volume controls
        elif "volume" in query_lower:
            if "up" in query_lower or "increase" in query_lower:
                action = "volume_up"
                message_code = "system:volume_up"
            elif "down" in query_lower or "decrease" in query_lower:
                action = "volume_down"
                message_code = "system:volume_down"
            else:
                action = "volume_status"
                message_code = "system:volume_status"
        
        # Handle brightness
        elif "brightness" in query_lower:
            action = "brightness_adjust"
            message_code = "system:brightness_adjust"
        
        # Default fallback
        else:
            action = "unknown"
            message_code = "system:help"
        
        return {
            "success": True,
            "response": None,
            "data": {
                "action": action,
                "message_code": message_code
            }
        }
    
    def shutdown(self):
        logger.info(" System Control plugin shutdown")


class WebSearchPlugin(PluginInterface):
    """Web search plugin"""
    
    def __init__(self):
        super().__init__()
        self.name = "WebSearchPlugin"
        self.description = "Search the web for information"
        self.capabilities = ["web_search", "browse", "lookup"]
    
    def initialize(self) -> bool:
        logger.info(" Web Search plugin initialized")
        return True
    
    def can_handle(self, intent: str, entities: Dict) -> bool:
        # Only match based on intent to avoid context pollution
        return intent == "search"
    
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
        logger.info(" Web Search plugin shutdown")


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
