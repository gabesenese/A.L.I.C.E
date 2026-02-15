# Plugin Interface API Contract

## Overview

All Alice plugins must implement the `PluginInterface` base class, which defines a standard contract for plugin lifecycle and execution.

## Location

```python
from ai.plugins.plugin_system import PluginInterface
```

## Required Methods

### `__init__(self)`

Initialize plugin state. Called once during plugin registration.

**Contract:**
- Must not make external API calls (defer to lazy initialization)
- Must not access other plugins (no cross-plugin dependencies at init)
- Should initialize configuration from environment variables
- Must call `super().__init__()`
- Should set `self.name`, `self.version`, `self.description`, `self.capabilities`

**Example:**
```python
def __init__(self):
    super().__init__()
    self.name = "MyPlugin"
    self.version = "1.0.0"
    self.description = "My custom plugin"
    self.capabilities = ["capability1", "capability2"]
    self.config = {}  # Plugin-specific state
```

---

### `initialize(self) -> bool`

Initialize the plugin's resources. Called during registration.

**Contract:**
- Return `True` if initialization succeeded, `False` otherwise
- Plugin will NOT be registered if this returns `False`
- Safe place to check dependencies, load resources, etc.
- Should log errors if initialization fails

**Example:**
```python
def initialize(self) -> bool:
    try:
        # Load dependencies, check API keys, etc.
        if not self._check_dependencies():
            return False
        logger.info(f"{self.name} initialized successfully")
        return True
    except Exception as e:
        logger.error(f"{self.name} initialization failed: {e}")
        return False
```

---

### `can_handle(self, intent: str, entities: Dict, query: str = None) -> bool`

Check if this plugin can handle the given intent/entities.

**Contract:**
- Must return `bool` (True if plugin can handle, False otherwise)
- Should be fast (no expensive operations)
- Should not modify state
- `query` parameter is optional but recommended to use for better matching

**Example:**
```python
def can_handle(self, intent: str, entities: Dict, query: str = None) -> bool:
    # Check intent
    if intent in ["weather", "forecast"]:
        return True

    # Check entities
    if "weather" in str(entities).lower():
        return True

    # Check query keywords (recommended)
    if query and any(kw in query.lower() for kw in ["temperature", "weather", "forecast"]):
        return True

    return False
```

---

### `execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict[str, Any]`

Execute plugin functionality.

**Contract:**
- Input: intent (classified), query (raw user input), entities (extracted), context (execution context)
- Output: Standardized response dictionary (see Response Format below)
- Must be idempotent where possible
- Must handle errors gracefully and return error structure
- Should update plugin state if needed

**Response Format:**
```python
{
    "success": bool,           # Required: Operation success status
    "action": str,             # Required: Action type identifier
    "data": Dict[str, Any],    # Required: Structured data for formulation
    "response": Optional[str], # Optional: Override response formulation
    "formulate": bool          # Optional: Use ResponseFormulator (default True)
}
```

**Example:**
```python
def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict[str, Any]:
    try:
        # Perform action
        result = self._get_weather(location=context.get('city'))

        return {
            "success": True,
            "action": "get_weather",
            "data": {
                "temperature": result['temp'],
                "condition": result['condition'],
                "location": context.get('city')
            },
            "formulate": True  # Let Alice learn to phrase weather responses
        }
    except Exception as e:
        logger.error(f"Weather plugin error: {e}")
        return {
            "success": False,
            "action": "get_weather",
            "data": {},
            "response": f"Failed to get weather: {str(e)}"
        }
```

---

### `shutdown(self) -> None`

Cleanup when plugin is disabled or system shuts down.

**Contract:**
- Called when plugin is unregistered or system exits
- Should close connections, save state, release resources
- Must not raise exceptions (handle internally)

**Example:**
```python
def shutdown(self) -> None:
    try:
        if hasattr(self, 'connection'):
            self.connection.close()
        logger.info(f"{self.name} shut down successfully")
    except Exception as e:
        logger.error(f"{self.name} shutdown error: {e}")
```

---

## Optional Methods

### `get_info(self) -> Dict[str, str]`

Get plugin information (already implemented in base class, can override).

**Default implementation:**
```python
def get_info(self) -> Dict[str, str]:
    return {
        "name": self.name,
        "version": self.version,
        "description": self.description,
        "enabled": self.enabled,
        "capabilities": self.capabilities
    }
```

---

## Response Formulation

If `formulate: True` (default), ResponseFormulator will generate natural response using:
- `action`: Pattern matching for learned responses
- `data`: Data to incorporate into response
- `success`: Determines tone (positive/negative)

**Best Practice**: Always set `formulate: True` and provide structured data, allowing Alice to learn natural phrasing over time.

**Example Flow:**
1. Plugin returns: `{"success": True, "action": "create_note", "data": {"title": "meeting"}}`
2. ResponseFormulator phrases it: "I've created a note called 'meeting'."
3. Alice learns this phrasing for future `create_note` actions
4. After 3+ examples, Alice can phrase independently without LLM

---

## Error Handling

Plugins must handle errors gracefully and return structured error responses.

**Good Error Handling:**
```python
try:
    result = dangerous_operation()
    return {
        "success": True,
        "action": "operation_name",
        "data": result
    }
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    return {
        "success": False,
        "action": "operation_name",
        "data": {},
        "response": f"Operation failed: {str(e)}"
    }
```

**DO NOT:**
- Use bare `except:` blocks
- Swallow exceptions silently
- Return `None` or empty strings
- Raise unhandled exceptions (they will crash the plugin system)

---

## Type Hints

All methods should include proper type hints:

```python
from typing import Dict, Any

class MyPlugin(PluginInterface):
    def initialize(self) -> bool:
        ...

    def can_handle(self, intent: str, entities: Dict, query: str = None) -> bool:
        ...

    def execute(self, intent: str, query: str, entities: Dict, context: Dict) -> Dict[str, Any]:
        ...

    def shutdown(self) -> None:
        ...
```

---

## Complete Example

See `docs/api/examples/simple_plugin.py` for a complete, working example.

---

## Testing Your Plugin

Create unit tests for your plugin:

```python
# tests/unit/test_my_plugin.py
import pytest
from plugins.my_plugin import MyPlugin

def test_plugin_initialization():
    plugin = MyPlugin()
    assert plugin.initialize() is True

def test_can_handle():
    plugin = MyPlugin()
    plugin.initialize()
    assert plugin.can_handle("test", {}, "test query") is True

def test_execute_success():
    plugin = MyPlugin()
    plugin.initialize()
    result = plugin.execute("test", "test", {}, {})
    assert result["success"] is True
    assert "data" in result
```

---

## Registration

Register your plugin in `app/main.py`:

```python
from plugins.my_plugin import MyPlugin

# In ALICE.__init__:
self.plugins.register_plugin(MyPlugin(), priority=50)
```

Lower priority number = higher priority in execution order.
