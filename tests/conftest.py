"""Shared pytest fixtures for integration tests."""

import atexit
import logging
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("ALICE_ENABLE_BACKGROUND_SERVICES", "0")

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from ai.plugins.notes_plugin import NotesManager, NotesPlugin
from ai.runtime.contract_pipeline import ContractPipeline

from app.main import app
from app.api.dependencies import get_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


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


@pytest.fixture(scope="session", autouse=True)
def restore_data_directory():
    """Return data/ to its pre-session contents when the run finishes.

    Tests write learned state, journals, and goals into data/, so each run started
    from whatever the previous run left behind. Test order is fixed, yet the set of
    failures changed between identical runs, because the suite was effectively
    iterating on its own leftover state. Restoring afterwards makes every run start
    from the same baseline, so a failure means the same thing twice.
    """
    if not DATA_DIR.exists():
        yield
        return

    snapshot_root = Path(tempfile.mkdtemp(prefix="alice-data-snapshot-"))
    snapshot = snapshot_root / "data"
    shutil.copytree(DATA_DIR, snapshot)
    original = {p.relative_to(snapshot) for p in snapshot.rglob("*") if p.is_file()}
    try:
        yield
    finally:
        for relative in original:
            target = DATA_DIR / relative
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(snapshot / relative, target)
            except OSError:
                pass
        for path in list(DATA_DIR.rglob("*")):
            if path.is_file() and path.relative_to(DATA_DIR) not in original:
                try:
                    path.unlink()
                except OSError:
                    pass
        shutil.rmtree(snapshot_root, ignore_errors=True)


@pytest.fixture(autouse=True)
def isolate_project_memory(tmp_path, monkeypatch):
    """Give every test its own project memory store.

    data/project_memory.json is keyed by user id and persists between runs, so a
    test that drives a turn leaves operator state, recommendations, and inspected
    files behind for whatever runs next. That made the suite order dependent: the
    set of failures changed between identical runs while each test passed alone.
    """
    import ai.memory.project_memory as project_memory

    monkeypatch.setattr(project_memory, "PROJECT_MEMORY_PATH", tmp_path / "project_memory.json")


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
