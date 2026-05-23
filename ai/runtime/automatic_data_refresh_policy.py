"""Automatic data refresh policy: force a plugin fetch when live data is stale.

When a route requires fresh data (e.g. weather) and the world model reports
that domain as stale, this policy signals the pipeline to execute the plugin
*before* LLM reasoning rather than serving a stale answer.
"""

from __future__ import annotations

from typing import Any, Optional


# Intents that require fresh data → which domain to check
_INTENT_DOMAIN_MAP = {
    "weather": "weather",
    "forecast": "weather",
}

# TTL per domain in seconds
_DOMAIN_TTL: dict[str, float] = {
    "weather": 1800.0,  # 30 min
    "forecast": 3600.0,  # 1 hr
}


class AutomaticDataRefreshPolicy:
    """Decide whether a turn should force a fresh data fetch before responding."""

    def should_force_refresh(
        self,
        *,
        route: str,
        intent: str,
        world_model: Any,
    ) -> bool:
        """Return True if the pipeline should execute the plugin regardless of
        whether it would normally run (e.g. stale data + weather route)."""
        intent_prefix = str(intent or "").split(":")[0].lower()
        domain = _INTENT_DOMAIN_MAP.get(intent_prefix)
        if not domain:
            return False

        route_lower = str(route or "").lower()
        if route_lower not in {"plugin", "tool"}:
            return False  # plugin will be called anyway; or not a plugin route at all

        try:
            ttl = _DOMAIN_TTL.get(domain, 1800.0)
            return bool(world_model.is_data_stale(domain, ttl_seconds=ttl))
        except Exception:
            return False

    def temporal_qualifier(self, *, domain: str, world_model: Any) -> Optional[str]:
        """Return a human-readable age qualifier if data was recently refreshed."""
        try:
            age = world_model.data_age_seconds(domain)
            if age is None:
                return None
            mins = int(age / 60)
            if mins <= 1:
                return "just checked"
            if mins <= 10:
                return f"{mins}m ago"
            if mins <= 60:
                return f"~{mins}m ago"
            hours = int(age / 3600)
            return f"~{hours}h ago"
        except Exception:
            return None


_policy: AutomaticDataRefreshPolicy | None = None


def get_auto_refresh_policy() -> AutomaticDataRefreshPolicy:
    global _policy
    if _policy is None:
        _policy = AutomaticDataRefreshPolicy()
    return _policy
