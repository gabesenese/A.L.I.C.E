from __future__ import annotations

import time
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from starlette.middleware.cors import CORSMiddleware

from app.logging_config import set_trace_id


def register_middleware(app: FastAPI) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
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
