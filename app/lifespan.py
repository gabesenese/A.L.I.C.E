from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)


def _shutdown_alice(app: FastAPI) -> None:
    """Save memory, context and conversation state before the process goes away.

    ALICE.shutdown() is what flushes those to disk. The HTTP path never called
    it, so every API restart dropped whatever the session had learned — the CLI
    saved on exit and the API silently did not.
    """
    container = getattr(app.state, "container", None)
    if container is None:
        return
    alice = getattr(container, "alice", None)
    if alice is None or not hasattr(alice, "shutdown"):
        return
    try:
        alice.shutdown()
        logger.info("alice_shutdown", status="ok")
    except Exception as exc:
        logger.warning("alice_shutdown", status="failed", error=str(exc))


def _end_identity_session() -> None:
    try:
        from ai.identity import alice_identity as _ai

        if _ai._current_session_id:
            _ai.end_session(_ai._current_session_id)
    except Exception as exc:
        logger.debug("identity_session_close_skipped", error=str(exc))


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    settings = get_settings()

    # Startup: opportunistic health probes for runtime deps.
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
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

    daemon = None
    if background_services_enabled():
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
    else:
        logger.info("companion_services", status="disabled")

    try:
        yield
    finally:
        # Shutdown has to run on every exit path, including the one where
        # background services were never started: ALICE was still built to serve
        # requests, and her state is still unsaved.
        if daemon is not None:
            try:
                daemon.stop()
            except Exception as exc:
                logger.warning("companion_daemon_stop", status="failed", error=str(exc))
        _shutdown_alice(app)
        _end_identity_session()
        logger.info("shutdown", status="ok")
