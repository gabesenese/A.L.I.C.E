"""Bridges the old PluginInterface contract into the modern PluginProtocol so
legacy plugins can register in the unified PluginRegistry without modification."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai.plugins.plugin_system import PluginInterface


class LegacyPluginAdapter:
    """Wraps a PluginInterface instance as PluginProtocol."""

    def __init__(self, plugin: "PluginInterface") -> None:
        self._plugin = plugin
        self.name: str = str(plugin.name)

    def actions(self) -> set[str]:
        caps = getattr(self._plugin, "capabilities", None) or []
        return {str(c) for c in caps} or {"handle"}

    def score(self, text: str, tokens: list[str]) -> float:
        intent = " ".join(tokens[:3]) if tokens else text[:40]
        try:
            result = self._plugin.can_handle(intent, {})
            return 0.65 if result else 0.0
        except Exception:
            return 0.0

    async def execute(self, action: str, params: dict) -> dict:
        intent = params.get("intent", action)
        query = str(params.get("query", params.get("text", "")))
        entities = dict(params.get("entities", {}))
        context = dict(params.get("context", {}))
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None,
            lambda: self._plugin.execute(intent, query, entities, context),
        )
        if isinstance(raw, dict):
            return raw
        return {"response": str(raw), "success": True}
