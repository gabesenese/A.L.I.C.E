import pytest

from app.bootstrap import create_app


@pytest.mark.asyncio
async def test_create_app_has_routes() -> None:
    app = create_app()
    paths = {route.path for route in app.routes}
    assert "/chat" in paths
    assert "/ws/chat" in paths
