# Plugin Best Practices

## Design Principles

### 1. **Single Responsibility**

Each plugin should do ONE thing well.

**Good:**
```python
class WeatherPlugin(PluginInterface):
    """Provides weather information only"""
    capabilities = ["weather", "forecast"]
```

**Bad:**
```python
class SuperPlugin(PluginInterface):
    """Does weather, news, stocks, and email"""  # Too many responsibilities
    capabilities = ["weather", "news", "stocks", "email"]
```

**Why:** Easier to maintain, test, and debug.

---

### 2. **Fail Gracefully**

Always handle errors and provide useful feedback.

**Good:**
```python
def execute(self, intent, query, entities, context):
    try:
        api_key = os.getenv('API_KEY')
        if not api_key:
            return {
                "success": False,
                "action": "api_call",
                "data": {},
                "response": "API key not configured. Please set API_KEY environment variable."
            }

        result = self.api.call(api_key)
        return {"success": True, "data": result}

    except APIError as e:
        logger.error(f"API error: {e}")
        return {
            "success": False,
            "data": {},
            "response": f"API error: {str(e)}"
        }
```

**Bad:**
```python
def execute(self, intent, query, entities, context):
    api_key = os.getenv('API_KEY')  # Might be None
    result = self.api.call(api_key)  # Crashes if API_KEY missing
    return {"success": True, "data": result}
```

---

### 3. **Use Structured Data**

Always return structured data, not just strings.

**Good:**
```python
return {
    "success": True,
    "action": "get_weather",
    "data": {
        "temperature": 22,
        "condition": "sunny",
        "location": "Kitchener",
        "units": "celsius"
    },
    "formulate": True  # Let Alice learn to phrase this
}
```

**Bad:**
```python
return {
    "success": True,
    "response": "It's 22°C and sunny in Kitchener"  # Hard-coded string
}
```

**Why:** Structured data allows Alice to learn phrasing patterns and adapt responses over time.

---

### 4. **Be Idempotent**

Same input should produce same output (when possible).

**Good:**
```python
def execute(self, intent, query, entities, context):
    # Query external API for current state
    data = self.api.get_current_data()
    return {"success": True, "data": data}
```

**Bad:**
```python
counter = 0  # Global state

def execute(self, intent, query, entities, context):
    counter += 1  # Different result each time
    return {"success": True, "data": {"count": counter}}
```

**Why:** Makes testing easier and behavior more predictable.

---

### 5. **Lazy Initialization**

Don't load heavy resources until needed.

**Good:**
```python
class MyPlugin(PluginInterface):
    def __init__(self):
        super().__init__()
        self._model = None  # Don't load yet

    def _get_model(self):
        if self._model is None:
            self._model = load_heavy_model()  # Load on first use
        return self._model

    def execute(self, intent, query, entities, context):
        model = self._get_model()  # Load here
        ...
```

**Bad:**
```python
class MyPlugin(PluginInterface):
    def __init__(self):
        super().__init__()
        self.model = load_heavy_model()  # Loads even if never used
```

**Why:** Faster startup, less memory usage if plugin isn't used.

---

## Common Patterns

### Pattern 1: Configuration from Environment

```python
class ConfigurablePlugin(PluginInterface):
    def initialize(self):
        self.api_key = os.getenv('MY_PLUGIN_API_KEY')
        self.endpoint = os.getenv('MY_PLUGIN_ENDPOINT', 'https://default.api')

        if not self.api_key:
            logger.warning(f"{self.name} API key not set")
            return False

        logger.info(f"{self.name} initialized with endpoint: {self.endpoint}")
        return True
```

### Pattern 2: Caching Results

```python
from functools import lru_cache
from datetime import datetime, timedelta

class CachingPlugin(PluginInterface):
    def __init__(self):
        super().__init__()
        self.cache = {}
        self.cache_ttl = timedelta(minutes=5)

    def execute(self, intent, query, entities, context):
        cache_key = f"{intent}:{query}"

        # Check cache
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                logger.debug(f"Cache hit for {cache_key}")
                return {"success": True, "data": cached_data}

        # Fetch fresh data
        data = self._fetch_data()

        # Update cache
        self.cache[cache_key] = (data, datetime.now())

        return {"success": True, "data": data}
```

### Pattern 3: Rate Limiting

```python
import time

class RateLimitedPlugin(PluginInterface):
    def __init__(self):
        super().__init__()
        self.last_call_time = 0
        self.min_interval = 1.0  # 1 second between calls

    def execute(self, intent, query, entities, context):
        # Check rate limit
        elapsed = time.time() - self.last_call_time
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            return {
                "success": False,
                "data": {},
                "response": f"Please wait {wait_time:.1f} seconds before trying again."
            }

        self.last_call_time = time.time()

        # Proceed with execution
        ...
```

---

## Anti-Patterns (Avoid These)

### ❌ Anti-Pattern 1: Bare Except Blocks

**Bad:**
```python
try:
    result = dangerous_operation()
except:  # Catches everything, hides bugs
    return {"success": False}
```

**Good:**
```python
try:
    result = dangerous_operation()
except (APIError, NetworkError) as e:
    logger.error(f"Expected error: {e}")
    return {"success": False, "response": str(e)}
except Exception as e:
    logger.exception(f"Unexpected error: {e}")
    raise  # Re-raise unexpected errors
```

### ❌ Anti-Pattern 2: Blocking the Main Thread

**Bad:**
```python
def execute(self, intent, query, entities, context):
    time.sleep(30)  # Blocks everything for 30 seconds
    return {"success": True}
```

**Good:**
```python
import asyncio

async def execute_async(self, intent, query, entities, context):
    await asyncio.sleep(30)  # Non-blocking
    return {"success": True}
```

### ❌ Anti-Pattern 3: Mutating Input Parameters

**Bad:**
```python
def execute(self, intent, query, entities, context):
    entities['modified'] = True  # Mutates input!
    context['user'] = 'changed'  # Affects other plugins!
    return {"success": True, "data": entities}
```

**Good:**
```python
def execute(self, intent, query, entities, context):
    # Make copies if you need to modify
    my_entities = entities.copy()
    my_entities['processed'] = True
    return {"success": True, "data": my_entities}
```

### ❌ Anti-Pattern 4: Hidden Dependencies

**Bad:**
```python
from some_obscure_library import magic_function  # Not documented

class MyPlugin(PluginInterface):
    def initialize(self):
        magic_function()  # Fails if library not installed
        return True
```

**Good:**
```python
class MyPlugin(PluginInterface):
    def initialize(self):
        try:
            import some_obscure_library
            logger.info("Optional dependency loaded")
        except ImportError:
            logger.warning("some_obscure_library not found. Install with: pip install some-obscure-library")
            # Continue with fallback behavior
        return True
```

---

## Performance Tips

### 1. **Profile Before Optimizing**

Use logging to identify slow operations:

```python
import time

def execute(self, intent, query, entities, context):
    start = time.time()

    result = self._slow_operation()

    elapsed = time.time() - start
    if elapsed > 1.0:
        logger.warning(f"{self.name} took {elapsed:.2f}s")

    return {"success": True, "data": result}
```

### 2. **Use Batch Operations**

**Bad:**
```python
for item in items:
    self.api.process(item)  # N API calls
```

**Good:**
```python
self.api.process_batch(items)  # 1 API call
```

### 3. **Limit Results**

```python
def execute(self, intent, query, entities, context):
    # Don't return unlimited data
    results = self.database.query(query, limit=100)
    return {"success": True, "data": {"results": results, "total": len(results)}}
```

---

## Security Considerations

### 1. **Validate Input**

```python
def execute(self, intent, query, entities, context):
    # Validate before using
    location = context.get('location', '')
    if not location or len(location) > 100:
        return {"success": False, "response": "Invalid location"}

    # Safe to use now
    ...
```

### 2. **Don't Log Sensitive Data**

**Bad:**
```python
logger.info(f"User input: {query}")  # Might contain passwords
logger.info(f"API response: {response}")  # Might contain PII
```

**Good:**
```python
logger.info(f"Processing query of length {len(query)}")
logger.info(f"API call succeeded: {response['status']}")
```

### 3. **Use Environment Variables for Secrets**

**Bad:**
```python
API_KEY = "sk-1234567890abcdef"  # Hard-coded secret!
```

**Good:**
```python
API_KEY = os.getenv('MY_API_KEY')  # From environment
if not API_KEY:
    logger.error("API_KEY not set!")
```

---

## Testing Best Practices

### 1. **Test Happy Path AND Error Cases**

```python
def test_plugin_execute_success():
    plugin = MyPlugin()
    result = plugin.execute("test", "test", {}, {})
    assert result["success"] is True

def test_plugin_execute_missing_data():
    plugin = MyPlugin()
    result = plugin.execute("test", "", {}, {})  # Empty query
    assert result["success"] is False
    assert "error" in result["response"].lower()

def test_plugin_execute_api_failure(mocker):
    plugin = MyPlugin()
    mocker.patch.object(plugin.api, 'call', side_effect=APIError("Network error"))
    result = plugin.execute("test", "test", {}, {})
    assert result["success"] is False
```

### 2. **Use Mocks for External Dependencies**

```python
@pytest.fixture
def mock_api(mocker):
    return mocker.patch('my_plugin.external_api.call')

def test_with_mock(mock_api):
    mock_api.return_value = {"data": "test"}
    plugin = MyPlugin()
    result = plugin.execute("test", "test", {}, {})
    assert result["data"] == {"data": "test"}
    mock_api.assert_called_once()
```

---

## Documentation

### Always Document:

1. **What the plugin does**
2. **Required configuration** (environment variables, API keys)
3. **Supported intents** and keywords
4. **Return data format**
5. **Error conditions**

**Example:**
```python
class MyPlugin(PluginInterface):
    """
    Weather information plugin using OpenWeatherMap API.

    Configuration:
        - OPENWEATHER_API_KEY: Required. Get from https://openweathermap.org/api

    Supported Intents:
        - weather: Current weather
        - forecast: 7-day forecast

    Keywords:
        - weather, temperature, forecast, rain, sunny

    Returns:
        {
            "temperature": float,
            "condition": str,
            "humidity": int,
            "location": str
        }

    Errors:
        - Missing API key: Returns error, does not initialize
        - Unknown location: Returns error with suggestion
        - API failure: Returns cached data if available
    """
```

---

## Examples

See `docs/api/examples/` for complete implementations demonstrating these patterns.
