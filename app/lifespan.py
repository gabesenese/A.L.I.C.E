from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app.logging_config import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    # Startup: opportunistic health probes for runtime deps.
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.get("http://localhost:11434/api/tags")
            logger.info("startup_check", component="ollama", status="ok")
    except Exception as exc:  # pragma: no cover - depends on local services
        logger.warning("startup_check", component="ollama", status="degraded", error=str(exc))

    # Companion services — all daemon threads, stop automatically on process exit.
    from ai.runtime.notifier import heartbeat_output, daemon_notify
    from brain.heartbeat import Heartbeat
    from brain.ambient_monitor import get_ambient_monitor
    from ai.runtime.companion_daemon import CompanionDaemon
    from memory.world_model import get_world_model
    from ai.goals.goal_engine import get_goal_engine

    heartbeat = Heartbeat(output=heartbeat_output)
    heartbeat.start()

    daemon = CompanionDaemon(
        world_state_memory=get_world_model(),
        goal_system=get_goal_engine(),
        notify_callback=daemon_notify,
    )
    daemon.start()

    get_ambient_monitor().start()

    logger.info("companion_services", status="started")

    yield

    heartbeat.stop()
    daemon.stop()
    get_ambient_monitor().stop()

    # Foundation 2 — close session on graceful shutdown
    try:
        from ai.identity import alice_identity as _ai

        if _ai._current_session_id:
            _ai.end_session(_ai._current_session_id)
    except Exception:
        pass

    logger.info("shutdown", status="ok")
