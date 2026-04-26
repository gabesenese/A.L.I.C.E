from __future__ import annotations

import contextvars
import logging
from typing import Optional

import structlog


_request_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_trace_id",
    default="",
)


def set_trace_id(trace_id: str) -> None:
    _request_trace_id.set(str(trace_id or ""))


def get_trace_id() -> str:
    return _request_trace_id.get()


def _add_trace_id(
    logger: logging.Logger,
    method_name: str,
    event_dict: dict,
) -> dict:
    trace_id = get_trace_id()
    if trace_id:
        event_dict.setdefault("trace_id", trace_id)
    return event_dict


def configure_logging(log_level: str = "INFO") -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            _add_trace_id,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, str(log_level or "INFO").upper(), logging.INFO),
        handlers=[logging.StreamHandler()],
    )


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
