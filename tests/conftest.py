"""Shared pytest fixtures for integration tests."""

import pytest

from ai.plugins.notes_plugin import NotesManager, NotesPlugin


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
