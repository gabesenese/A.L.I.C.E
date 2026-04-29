from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Protocol


class PluginProtocol(Protocol):
    name: str

    def actions(self) -> set[str]: ...

    def score(self, text: str, tokens: list[str]) -> float: ...

    async def execute(self, action: str, params: dict) -> dict: ...


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, PluginProtocol] = {}

    def register(self, plugin: PluginProtocol) -> None:
        self._plugins[plugin.name] = plugin

    def get(self, name: str) -> PluginProtocol | None:
        return self._plugins.get(name)

    def score_all(self, text: str, tokens: list[str]) -> dict[str, float]:
        return {
            name: plugin.score(text, tokens) for name, plugin in self._plugins.items()
        }

    @property
    def all_actions(self) -> dict[str, set[str]]:
        return {name: plugin.actions() for name, plugin in self._plugins.items()}


def discover_plugins() -> PluginRegistry:
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
                try:
                    instance = cls()
                except Exception:
                    continue
                if not hasattr(instance, "name"):
                    continue
                if not hasattr(instance, "score") or not hasattr(instance, "actions"):
                    continue
                registry.register(instance)

    return registry
