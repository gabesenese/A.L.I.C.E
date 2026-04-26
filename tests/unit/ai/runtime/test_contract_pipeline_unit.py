import pytest


@pytest.mark.asyncio
async def test_pipeline_runs_turn(pipeline) -> None:
    result = await pipeline.run_turn(user_input="hello", user_id="test")
    assert isinstance(result.response_text, str)
    assert "trace_id" in (result.metadata or {})
