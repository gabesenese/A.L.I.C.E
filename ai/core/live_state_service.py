"""Live-state aggregator for fresh answer context.

Provides one source of truth for runtime weather snapshots by reconciling
reasoning-memory entities with persisted world-state memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class LiveSnapshot:
    source: str
    captured_at: float
    data: Dict[str, Any]


class LiveStateService:
    """Builds fresh snapshots from multiple state stores."""

    FRESHNESS_TTL_SECONDS = {
        "weather": 1800.0,
        "forecast": 3600.0,
    }

    def freshest_weather_snapshot(
        self,
        *,
        reasoning_engine: Any = None,
        world_state_memory: Any = None,
        max_age_seconds: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        candidates = []

        current = self._from_reasoning_entity(reasoning_engine, "current_weather")
        if current:
            candidates.append(current)

        forecast = self._from_reasoning_entity(reasoning_engine, "weather_forecast")
        if forecast:
            candidates.append(forecast)

        world = self._from_world_state(world_state_memory)
        if world:
            candidates.append(world)

        if not candidates:
            return None

        freshest = max(candidates, key=lambda x: float(x.captured_at or 0.0))
        ttl = float(
            max_age_seconds
            if max_age_seconds is not None
            else self.FRESHNESS_TTL_SECONDS["weather"]
        )
        age_seconds = max(
            0.0, self._to_epoch(datetime.utcnow()) - float(freshest.captured_at or 0.0)
        )
        is_stale = age_seconds > ttl if freshest.captured_at else True
        return {
            "source": freshest.source,
            "captured_at": freshest.captured_at,
            "data": dict(freshest.data or {}),
            "age_seconds": age_seconds,
            "is_stale": is_stale,
            "ttl_seconds": ttl,
        }

    def latest_weather_forecast(
        self,
        *,
        reasoning_engine: Any = None,
        world_state_memory: Any = None,
        max_age_seconds: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        candidates = []
        forecast = self._from_reasoning_entity(reasoning_engine, "weather_forecast")
        if forecast:
            candidates.append(forecast)

        world = self._from_world_state(world_state_memory)
        if world and isinstance(world.data.get("forecast"), list):
            candidates.append(world)

        if not candidates:
            return None

        freshest = max(candidates, key=lambda x: float(x.captured_at or 0.0))
        payload = dict(freshest.data or {})
        ttl = float(
            max_age_seconds
            if max_age_seconds is not None
            else self.FRESHNESS_TTL_SECONDS["forecast"]
        )
        age_seconds = max(
            0.0, self._to_epoch(datetime.utcnow()) - float(freshest.captured_at or 0.0)
        )
        payload["age_seconds"] = age_seconds
        payload["is_stale"] = age_seconds > ttl if freshest.captured_at else True
        payload["ttl_seconds"] = ttl
        return payload

    def _from_reasoning_entity(
        self, reasoning_engine: Any, entity_id: str
    ) -> Optional[LiveSnapshot]:
        if not reasoning_engine:
            return None
        try:
            entity = reasoning_engine.get_entity(entity_id)
        except Exception:
            return None
        if not entity or not getattr(entity, "data", None):
            return None

        data = dict(entity.data or {})
        captured = self._to_epoch(data.get("captured_at"))
        if captured <= 0.0:
            created_at = getattr(entity, "created_at", None)
            captured = self._to_epoch(created_at)

        return LiveSnapshot(
            source=f"reasoning_engine:{entity_id}",
            captured_at=captured,
            data=data,
        )

    def _from_world_state(self, world_state_memory: Any) -> Optional[LiveSnapshot]:
        if not world_state_memory:
            return None
        try:
            payload = world_state_memory.get_environment_state("weather")
        except Exception:
            payload = None

        if not isinstance(payload, dict):
            return None

        data = payload.get("data") if isinstance(payload.get("data"), dict) else None
        if not isinstance(data, dict) or not data:
            return None

        captured = self._to_epoch(payload.get("captured_at"))
        if captured <= 0.0:
            captured = self._to_epoch(data.get("captured_at"))

        return LiveSnapshot(
            source="world_state_memory:weather",
            captured_at=captured,
            data=data,
        )

    @staticmethod
    def _to_epoch(value: Any) -> float:
        if value is None:
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, datetime):
            try:
                return float(value.timestamp())
            except Exception:
                return 0.0
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return 0.0
            try:
                return float(raw)
            except Exception:
                pass
            try:
                # Handle trailing Z from ISO strings.
                normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
                return float(datetime.fromisoformat(normalized).timestamp())
            except Exception:
                return 0.0
        return 0.0


_live_state_service: LiveStateService | None = None


def get_live_state_service() -> LiveStateService:
    global _live_state_service
    if _live_state_service is None:
        _live_state_service = LiveStateService()
    return _live_state_service
