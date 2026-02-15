# Plugin Lifecycle

## Overview

This document describes the complete lifecycle of a plugin in Alice's plugin system, from registration to shutdown.

---

## Lifecycle Stages

### 1. **Plugin Creation**

```python
from ai.plugins.plugin_system import PluginInterface

class MyPlugin(PluginInterface):
    def __init__(self):
        super().__init__()
        self.name = "MyPlugin"
        # Set other attributes
```

**What Happens:**
- Plugin object is instantiated
- `__init__()` is called
- Attributes are set (name, version, capabilities)
- NO external calls should happen here

**Best Practices:**
- Keep `__init__()` lightweight
- Set basic attributes only
- Don't open connections or load heavy resources

---

### 2. **Registration**

```python
plugin_manager.register_plugin(MyPlugin(), priority=50)
```

**What Happens:**
1. `plugin.initialize()` is called
2. If initialization succeeds (returns `True`):
   - Plugin is added to `plugins` dictionary
   - Plugin is added to `plugin_order` list
   - Plugin status is set to `enabled=True`
3. If initialization fails (returns `False`):
   - Plugin is rejected
   - Error is logged
   - Plugin is NOT added to system

**Best Practices:**
- Check dependencies in `initialize()`
- Return `False` if requirements not met
- Log clear error messages on failure

---

### 3. **Active/Enabled State**

Once registered, the plugin is active and can handle requests.

**Execution Flow:**
```
User Input
    ↓
Intent Classification
    ↓
Plugin Manager: execute_for_intent()
    ↓
For each enabled plugin:
    → can_handle(intent, entities, query)?
        ↓ YES
    → execute(intent, query, entities, context)
        ↓
    → Return result
```

**What Happens:**
- PluginManager calls `can_handle()` on each enabled plugin
- First plugin that returns `True` gets to execute
- Plugin's `execute()` method is called
- Result is returned to Alice

**Best Practices:**
- Make `can_handle()` fast (no heavy operations)
- Return clear, structured results from `execute()`
- Handle all errors gracefully

---

### 4. **Disable/Enable States**

Plugins can be dynamically disabled and re-enabled:

```python
# Disable plugin
plugin_manager.disable_plugin("MyPlugin")

# Re-enable plugin
plugin_manager.enable_plugin("MyPlugin")
```

**What Happens:**
- Disabled: `plugin.enabled = False`
- Plugin stays in system but won't execute
- `can_handle()` is NOT called for disabled plugins
- Re-enabling: `plugin.enabled = True`
- Plugin starts handling requests again

**Best Practices:**
- Don't rely on plugins always being enabled
- Handle state preservation across enable/disable cycles

---

### 5. **Unregistration**

```python
plugin_manager.unregister_plugin("MyPlugin")
```

**What Happens:**
1. `plugin.shutdown()` is called
2. Plugin is removed from `plugins` dictionary
3. Plugin is removed from `plugin_order` list
4. Plugin object may be garbage collected

**Best Practices:**
- Implement `shutdown()` to cleanup resources
- Close connections, save state
- Don't raise exceptions in `shutdown()`

---

### 6. **Shutdown**

When Alice exits or plugin system shuts down:

```python
# System shutdown
for plugin in plugins.values():
    plugin.shutdown()
```

**What Happens:**
- All plugins' `shutdown()` methods are called
- Resources are released
- Connections are closed
- State is saved

**Best Practices:**
- Always implement `shutdown()` properly
- Save important state before shutdown
- Handle shutdown errors gracefully

---

## Lifecycle Diagram

```
[ Creation ]
     ↓
[ Registration ]
     ├→ initialize() succeeds → [ Active/Enabled ]
     └→ initialize() fails → [ Rejected ]

[ Active/Enabled ]
     ↓
[ Execution Loop ]
     ├→ can_handle() → execute() → Return Result
     ├→ can_handle() → False → Skip
     └→ (repeat)

[ Disable/Enable ]
     ├→ disable_plugin() → [ Disabled ]
     └→ enable_plugin() → [ Active/Enabled ]

[ Unregistration ]
     ↓
[ shutdown() ]
     ↓
[ Destroyed ]
```

---

## State Management

### Plugin State

Plugins can maintain internal state:

```python
class StatefulPlugin(PluginInterface):
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.last_result = None

    def execute(self, intent, query, entities, context):
        self.call_count += 1
        result = self._do_work()
        self.last_result = result
        return {"success": True, "data": result}
```

**Considerations:**
- State persists across executions
- State is lost on unregistration
- Use `shutdown()` to save important state

### Persistent State

For state that should survive restarts:

```python
class PersistentPlugin(PluginInterface):
    def initialize(self):
        self.state_file = "data/my_plugin_state.json"
        self.state = self._load_state()
        return True

    def _load_state(self):
        if Path(self.state_file).exists():
            with open(selfstate_file) as f:
                return json.load(f)
        return {}

    def shutdown(self):
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f)
```

---

## Error Recovery

### Execution Errors

If a plugin throws an exception during `execute()`:

```python
result = plugin.execute(...)  # Raises Exception
```

**What Happens:**
- Exception is caught by PluginManager
- Error is logged
- Next matching plugin is tried (if available)
- Alice falls back to general conversation

**Best Practice:**
- Handle errors inside `execute()` and return error result
- Don't let exceptions escape

### Initialization Errors

If `initialize()` returns `False` or raises an exception:

**What Happens:**
- Plugin is rejected
- Error is logged
- Alice continues without this plugin

**Best Practice:**
- Return `False` for expected failures (missing API key)
- Let exceptions escape for unexpected failures (bugs)

---

## Examples

See `docs/api/examples/` for complete lifecycle examples:
- `simple_plugin.py` - Basic lifecycle
- `stateful_plugin.py` - State management
- `async_plugin.py` - Async operations

---

## Testing Lifecycle

```python
def test_plugin_lifecycle():
    plugin = MyPlugin()

    # Creation
    assert plugin.name == "MyPlugin"

    # Registration
    assert plugin.initialize() is True
    assert plugin.enabled is True

    # Execution
    assert plugin.can_handle("test", {}, "test") is True
    result = plugin.execute("test",  "test", {}, {})
    assert result["success"] is True

    # Shutdown
    plugin.shutdown()  # Should not raise
```
