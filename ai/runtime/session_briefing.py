from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional


def _recent_emotional_signals(identity: Any, hours: float = 48.0) -> List[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    signals: List[str] = []
    for entry in getattr(identity, "emotional_history", []):
        try:
            ts = datetime.fromisoformat(
                str(entry.get("noted_at") or "").replace("Z", "+00:00")
            )
            if ts >= cutoff:
                signals.append(str(entry.get("signal") or ""))
        except Exception:
            pass
    return [s for s in signals if s]


def generate_session_briefing(user_id: str = "gabriel") -> str:
    """Generate a rich session-start briefing from all available data sources.

    Reads from: identity layer, goal engine, world model (calendar events,
    unread email, open tasks). Gracefully degrades if any source is unavailable.
    """
    try:
        from ai.identity.user_identity import load_identity
        from ai.goals.goal_engine import get_goal_engine
        from memory.world_model import get_world_model

        identity = load_identity(user_id)
        engine = get_goal_engine()
        snapshot = get_world_model().snapshot()
        now = datetime.now(timezone.utc)

        parts: list[str] = []

        # Time-aware greeting
        name = (identity.name if identity.name else user_id.capitalize())
        hour = now.hour
        if hour < 12:
            salutation = "Good morning"
        elif hour < 18:
            salutation = "Good afternoon"
        else:
            salutation = "Good evening"
        parts.append(f"{salutation}, {name}.")

        # Calendar: events in the world model (populated by AmbientMonitor)
        upcoming = list(snapshot.get("environment", {}).get("upcoming_calendar") or [])
        if upcoming:
            today_lines: list[str] = []
            for ev in upcoming[:6]:
                start_raw = str(ev.get("start") or "")
                if not start_raw:
                    continue
                try:
                    start_dt = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
                    if start_dt.date() == now.date():
                        if ev.get("all_day") or ev.get("all_day") == "True":
                            time_str = "All day"
                        else:
                            time_str = start_dt.strftime("%H:%M")
                        today_lines.append(f"  • {ev.get('title', 'Event')} at {time_str}")
                except (ValueError, AttributeError):
                    continue
            if today_lines:
                parts.append("Today's schedule:\n" + "\n".join(today_lines))

        # Unread email count
        unread = int(snapshot.get("environment", {}).get("unread_email_count") or 0)
        if unread:
            parts.append(
                f"You have {unread} unread email{'s' if unread != 1 else ''}."
            )

        # Active goals — top 3 only, full description
        active = engine.active()
        if active:
            goal_lines = [f"  • {g.description}" for g in active[:3]]
            parts.append("Active goals:\n" + "\n".join(goal_lines))

        # Open tasks from world model — only those not already covered by goals
        goal_texts = {g.description.lower() for g in active}
        open_tasks = list(snapshot.get("environment", {}).get("open_tasks") or [])
        novel_tasks = [
            t for t in open_tasks
            if t.get("text") and not any(
                goal_frag in t["text"].lower()
                for goal_frag in goal_texts
            )
        ]
        if novel_tasks:
            task_lines = [f"  • {t['text']}" for t in novel_tasks[:2]]
            parts.append("Open threads:\n" + "\n".join(task_lines))

        # Emotional continuity — gentle follow-up on recent state signals
        recent_emotions = _recent_emotional_signals(identity, hours=48.0)
        if recent_emotions:
            unique = list(dict.fromkeys(recent_emotions))  # ordered dedup
            signals_str = ", ".join(unique[:3])
            parts.append(f"Last session you mentioned feeling {signals_str} — how are you doing now?")

        return "\n\n".join(parts)
    except Exception:
        return ""


def get_top_goal_for_greeting(user_id: str = "gabriel") -> str:
    """Return top active goal description for injection into greeting context."""
    try:
        from ai.goals.goal_engine import get_goal_engine

        top = get_goal_engine().top_goal()
        return top.description if top else ""
    except Exception:
        return ""
