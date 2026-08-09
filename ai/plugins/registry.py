from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)

CONSTRUCTION_TIMEOUT_SECONDS = 10


class PluginProtocol(Protocol):
    name: str

    def actions(self) -> set[str]: ...

    def score(self, text: str, tokens: list[str]) -> float: ...

    async def execute(self, action: str, params: dict) -> dict: ...


def construct_plugin(factory: Callable[[], Any], label: str) -> Any | None:
    """Build a plugin, giving up if its constructor blocks.

    A plugin constructor that waits on network or credential input would otherwise
    stall every consumer of the registry, including application startup.
    """
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="plugin-init")
    try:
        return executor.submit(factory).result(timeout=CONSTRUCTION_TIMEOUT_SECONDS)
    except FutureTimeoutError:
        logger.warning(
            "Plugin %s did not initialize within %ss and stays unavailable",
            label,
            CONSTRUCTION_TIMEOUT_SECONDS,
        )
        return None
    except Exception as exc:
        logger.warning("Plugin %s failed to initialize and stays unavailable: %s", label, exc)
        return None
    finally:
        executor.shutdown(wait=False)


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, PluginProtocol] = {}

    def register(self, plugin: PluginProtocol) -> None:
        self._plugins[plugin.name] = plugin

    def get(self, name: str) -> PluginProtocol | None:
        return self._plugins.get(name)

    def score_all(self, text: str, tokens: list[str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for name, plugin in self._plugins.items():
            try:
                scores[name] = plugin.score(text, tokens)
            except Exception as exc:
                logger.warning("Plugin %s failed to score: %s", name, exc)
        return scores

    @property
    def all_actions(self) -> dict[str, set[str]]:
        actions: dict[str, set[str]] = {}
        for name, plugin in self._plugins.items():
            try:
                actions[name] = plugin.actions()
            except Exception as exc:
                logger.warning("Plugin %s failed to report actions: %s", name, exc)
        return actions


def build_unified_registry() -> "PluginRegistry":
    """Registry containing both modern PluginProtocol plugins and legacy
    PluginInterface plugins wrapped in LegacyPluginAdapter.  Modern plugins
    take precedence, so legacy adapters are skipped if the same name is already
    registered."""
    registry = discover_plugins()

    try:
        from ai.plugins.legacy_adapter import LegacyPluginAdapter
        import ai.plugins.plugin_system as _ps

        _legacy_classes = [
            getattr(_ps, cls_name, None)
            for cls_name in (
                "WeatherPlugin",
                "TimePlugin",
                "SystemControlPlugin",
                "WebSearchPlugin",
            )
        ]
        for cls in _legacy_classes:
            if cls is None:
                continue
            instance = construct_plugin(cls, cls.__name__)
            if instance is None:
                continue
            try:
                adapter = LegacyPluginAdapter(instance)
                if registry.get(adapter.name) is None:
                    registry.register(adapter)
            except Exception:
                continue

        for mod_name in (
            "ai.plugins.calendar_plugin",
            "ai.plugins.notes_plugin",
            "ai.plugins.file_operations_plugin",
            "ai.plugins.memory_plugin",
            "ai.plugins.system_plugin",
        ):
            try:
                mod = importlib.import_module(mod_name)
                for attr in dir(mod):
                    if not attr.lower().endswith("plugin"):
                        continue
                    cls = getattr(mod, attr)
                    if not isinstance(cls, type):
                        continue
                    try:
                        from ai.plugins.plugin_system import PluginInterface

                        if not issubclass(cls, PluginInterface):
                            continue
                    except Exception:
                        continue
                    instance = construct_plugin(cls, f"{mod_name}.{attr}")
                    if instance is None:
                        continue
                    try:
                        adapter = LegacyPluginAdapter(instance)
                        if registry.get(adapter.name) is None:
                            registry.register(adapter)
                    except Exception:
                        continue
            except Exception:
                continue
    except Exception:
        pass

    return registry


def discover_plugins() -> "PluginRegistry":
    registry = PluginRegistry()
    package = importlib.import_module("ai.plugins")

    for module_info in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
        module_name = module_info.name
        if module_name.endswith(".registry") or module_name.endswith(".plugin_system"):
            continue
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module_name:
                continue
            if cls.__name__.lower().endswith("plugin"):
                instance = construct_plugin(cls, f"{module_name}.{cls.__name__}")
                if instance is None:
                    continue
                if not hasattr(instance, "name"):
                    continue
                if not hasattr(instance, "score") or not hasattr(instance, "actions"):
                    continue
                registry.register(instance)

    return registry
