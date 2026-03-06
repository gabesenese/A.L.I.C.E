"""Shared pytest fixtures for integration tests."""

import atexit
import logging

import pytest

from ai.plugins.notes_plugin import NotesManager, NotesPlugin


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
