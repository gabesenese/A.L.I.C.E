"""Structured logging helpers with safe fallback when structlog is unavailable."""

from __future__ import annotations

import logging
import contextvars
from typing import Any, Dict

_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "alice_trace_id", default=""
)


def set_trace_id(trace_id: str) -> None:
    _trace_id_var.set(str(trace_id or ""))


def get_trace_id() -> str:
    return str(_trace_id_var.get("") or "")


class _FallbackLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = dict(kwargs.pop("extra", {}) or {})
        trace_id = get_trace_id()
        if trace_id:
            extra.setdefault("trace_id", trace_id)
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(name: str = "alice"):
    """Get a structured logger if available, otherwise stdlib logger adapter."""
    try:
        import structlog  # type: ignore

        return structlog.get_logger(name).bind(trace_id=get_trace_id())
    except Exception:
        base = logging.getLogger(name)
        if not base.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(name)s %(levelname)s %(message)s",
            )
        return _FallbackLoggerAdapter(base, {})


def bind_context(logger: Any, **fields: Dict[str, Any]):
    """Bind context fields in both structlog and fallback modes."""
    if hasattr(logger, "bind"):
        return logger.bind(**fields)
    if isinstance(logger, logging.LoggerAdapter):
        logger.extra.update(fields)
        return logger
    return logger
