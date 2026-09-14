from __future__ import annotations

import time
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from starlette.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.logging_config import set_trace_id


def register_middleware(app: FastAPI) -> None:
    settings = get_settings()
    # `allow_origins=["*"]` together with `allow_credentials=True` is a
    # combination the CORS spec forbids: a browser refuses a wildcard on a
    # credentialed request, so the permissive-looking setting actually blocked
    # every cross-origin call it was meant to allow. Credentials are only
    # enabled when real origins are named.
    origins = settings.cors_origins_list
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        trace_id = request.headers.get("x-trace-id") or str(uuid4())
        set_trace_id(trace_id)
        started = time.perf_counter()
        response: Response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.headers["x-trace-id"] = trace_id
        response.headers["x-response-time-ms"] = f"{elapsed_ms:.2f}"
        return response
