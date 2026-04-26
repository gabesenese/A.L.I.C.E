"""Shared pytest fixtures for integration tests."""

import atexit
import logging

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from ai.plugins.notes_plugin import NotesManager, NotesPlugin
from ai.runtime.contract_pipeline import ContractPipeline

from app.main import app
from app.api.dependencies import get_pipeline


def pytest_configure(config):
    # torch registers dump_cache_stats() via @atexit.register at module import.
    # After pytest closes its log handlers, dump_cache_stats() tries to write
    # to a closed stream and prints "--- Logging error --- ValueError: I/O
    # operation on closed file." to the console on every test run.
    # Silencing the logger and unregistering the atexit hook eliminates both.
    logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.CRITICAL)
    try:
        from torch._subclasses import fake_tensor
        atexit.unregister(fake_tensor.dump_cache_stats)
    except Exception:
        pass


@pytest.fixture
def plugin(tmp_path):
    notes_dir = tmp_path / "notes"
    notes_plugin = NotesPlugin()
    notes_plugin.manager = NotesManager(notes_dir=str(notes_dir))
    notes_plugin.last_note_id = None
    notes_plugin.last_note_title = None
    notes_plugin.last_note_result_ids = []
    notes_plugin.learning_state_path = tmp_path / "notes_learning_state.json"
    notes_plugin.telemetry_log_path = tmp_path / "notes_plugin_telemetry.jsonl"
    notes_plugin._action_token_weights = {}
    notes_plugin._note_selection_weights = {}
    return notes_plugin


@pytest_asyncio.fixture
async def pipeline() -> ContractPipeline:
    return app.state.container.pipeline


@pytest_asyncio.fixture
async def client(pipeline: ContractPipeline):
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
