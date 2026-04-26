import pytest

from app.config import get_settings


@pytest.mark.asyncio
async def test_settings_load_defaults() -> None:
    settings = get_settings()
    assert settings.ollama_model
    assert settings.max_history > 0
