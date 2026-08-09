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

    # Companion services are daemon threads and stop automatically on process exit.
    # The heartbeat, ambient monitor, and task scheduler belong to the ALICE instance
    # owned by the container, so building it here keeps exactly one of each running.
    from ai.infrastructure.runtime_flags import background_services_enabled
    from ai.runtime.notifier import daemon_notify
    from ai.runtime.companion_daemon import CompanionDaemon
    from memory.world_model import get_world_model
    from ai.goals.goal_engine import get_goal_engine

    if not background_services_enabled():
        logger.info("companion_services", status="disabled")
        yield
        return

    container = getattr(app.state, "container", None)
    if container is not None:
        try:
            container.alice
        except Exception as exc:
            logger.warning("companion_services", status="degraded", error=str(exc))

    daemon = CompanionDaemon(
        world_state_memory=get_world_model(),
        goal_system=get_goal_engine(),
        notify_callback=daemon_notify,
    )
    daemon.start()

    logger.info("companion_services", status="started")

    yield

    daemon.stop()

    # Foundation 2 — close session on graceful shutdown
    try:
        from ai.identity import alice_identity as _ai

        if _ai._current_session_id:
            _ai.end_session(_ai._current_session_id)
    except Exception:
        pass

    logger.info("shutdown", status="ok")
