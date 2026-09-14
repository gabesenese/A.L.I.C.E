import pytest

from app.bootstrap import create_app


def _route_paths(app) -> set:
    """Collect every path the app serves.

    Newer FastAPI wraps an included router in a proxy entry that has no `path`
    of its own and keeps the real routes on `original_router`, so reading
    `.path` off every element of `app.routes` raises. Descend through whichever
    shape this version uses.
    """
    paths = set()
    pending = list(app.routes)
    seen = set()
    while pending:
        route = pending.pop()
        if id(route) in seen:
            continue
        seen.add(id(route))
        path = getattr(route, "path", None)
        if isinstance(path, str):
            paths.add(path)
        nested = getattr(route, "routes", None)
        if nested:
            pending.extend(nested)
        inner = getattr(route, "original_router", None)
        if inner is not None:
            pending.extend(getattr(inner, "routes", ()) or ())
    return paths


@pytest.mark.asyncio
async def test_create_app_has_routes() -> None:
    app = create_app()
    paths = _route_paths(app)
    assert "/chat" in paths
    assert "/ws/chat" in paths
