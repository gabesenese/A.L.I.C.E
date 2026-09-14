"""Application entrypoints for A.L.I.C.E.

`create_app` is exported lazily (PEP 562) so that importing the CLI runtime
(`app.main`, `app.alice`) does not drag in FastAPI. The HTTP API is optional;
the terminal companion is the default surface and must start without it.
"""

from typing import Any

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from app.bootstrap import create_app as _create_app

        return _create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
