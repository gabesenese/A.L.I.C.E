from __future__ import annotations

from typing import Optional


def generate_session_briefing(user_id: str = "gabriel") -> str:
    """Generate a session-start briefing from identity + active goals."""
    try:
        from ai.identity.user_identity import load_identity
        from ai.goals.goal_engine import get_goal_engine

        identity = load_identity(user_id)
        engine = get_goal_engine()

        parts: list[str] = []
        if identity.name:
            parts.append(f"Welcome back, {identity.name}.")

        goal_summary = engine.session_summary()
        if goal_summary:
            parts.append(f"Active goals:\n{goal_summary}")

        stale = engine.stale_goals(days=3.0)
        if stale:
            names = ", ".join(g.description[:40] for g in stale[:2])
            parts.append(f"Not touched in 3+ days: {names}")

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
