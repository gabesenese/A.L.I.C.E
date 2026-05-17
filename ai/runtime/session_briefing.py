from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


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

        # Active goals summary
        goal_summary = engine.session_summary()
        if goal_summary:
            parts.append(f"Active goals:\n{goal_summary}")

        # Stale goals (not touched in 3+ days)
        stale = engine.stale_goals(days=3.0)
        if stale:
            names = "; ".join(g.description[:50] for g in stale[:3])
            parts.append(f"Hasn't been touched in 3+ days: {names}")

        # Open tasks from world model (capped to avoid wall of text)
        open_tasks = list(snapshot.get("environment", {}).get("open_tasks") or [])
        if open_tasks:
            task_lines = [
                f"  • {t['text'][:80]}"
                for t in open_tasks[:3]
                if t.get("text")
            ]
            if task_lines:
                parts.append("Open threads:\n" + "\n".join(task_lines))

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
