"""Cross-plugin fallback chain: when a primary plugin fails, try an alternative.

Maps (intent_prefix, error_type) → ordered list of (plugin_name, action_override).
The pipeline walks the chain until one succeeds, recording which depth was needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class FallbackPlugin:
    plugin: str                 # plugin name to try
    action_override: str = ""   # if set, replace the action with this value
    note: str = ""              # why this fallback exists


# (intent_prefix, error_type) → ordered alternatives
_CHAIN: Dict[Tuple[str, str], List[FallbackPlugin]] = {
    # Weather timeouts → try a second fetch with shorter timeout
    ("weather", "timeout"): [
        FallbackPlugin("weather", note="retry_shorter_timeout"),
    ],
    ("weather", "connection_error"): [
        FallbackPlugin("weather", note="retry_connection"),
    ],
    ("weather", "fetch_failed"): [
        FallbackPlugin("weather", note="retry_fetch"),
    ],
    # Notes not found → offer search by content instead of title
    ("notes", "not_found"): [
        FallbackPlugin("notes", action_override="notes:search_by_content", note="fallback_to_content_search"),
    ],
    # Local file not found → try listing files to help user
    ("local", "target_not_found"): [
        FallbackPlugin("local", action_override="code:list_files", note="list_to_help_find"),
    ],
    # Default catch-all
    ("default", "timeout"): [
        FallbackPlugin("default", note="retry_once"),
    ],
}


class CrossPluginFallbackChain:
    """Walks the fallback chain for a failed (intent, error_type) pair.

    Usage:
        chain = get_cross_plugin_fallback_chain()
        steps = chain.get_chain(intent, error_type)
        for step in steps:
            result = try_plugin(step.plugin, step.action_override or original_action)
            if result.success:
                break
    """

    def get_chain(self, intent: str, error_type: str) -> List[FallbackPlugin]:
        prefix = str(intent or "").split(":")[0].lower()
        err = str(error_type or "").lower()

        steps = _CHAIN.get((prefix, err))
        if steps:
            return list(steps)

        steps = _CHAIN.get((prefix, "default"))
        if steps:
            return list(steps)

        return list(_CHAIN.get(("default", err), _CHAIN.get(("default", "timeout"), [])))

    def resolve_action(self, step: FallbackPlugin, original_action: str) -> str:
        return str(step.action_override or original_action)

    def annotate_result(
        self,
        result: Dict[str, Any],
        *,
        fallback_depth: int,
        plugin_used: str,
        original_plugin: str,
    ) -> Dict[str, Any]:
        """Add fallback metadata to a plugin result dict."""
        out = dict(result or {})
        out["fallback_depth"] = fallback_depth
        out["plugin_used"] = plugin_used
        out["original_plugin"] = original_plugin
        out["used_fallback"] = fallback_depth > 0
        return out


_chain: CrossPluginFallbackChain | None = None


def get_cross_plugin_fallback_chain() -> CrossPluginFallbackChain:
    global _chain
    if _chain is None:
        _chain = CrossPluginFallbackChain()
    return _chain
