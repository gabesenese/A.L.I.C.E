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
        logger.warning(
            "startup_check", component="ollama", status="degraded", error=str(exc)
        )

    yield

    logger.info("shutdown", status="ok")
