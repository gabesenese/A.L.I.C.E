"""
Integration tests for notes follow-up title questions.
"""

import sys
from pathlib import Path

import pytest

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.plugins.notes_plugin import NotesManager, NotesPlugin


class TestNotesTitleFollowup:
    @pytest.fixture
    def plugin(self, tmp_path):
        notes_dir = tmp_path / "notes"
        plugin = NotesPlugin()
        plugin.manager = NotesManager(notes_dir=str(notes_dir))
        plugin.last_note_id = None
        plugin.last_note_title = None
        return plugin

    def test_title_followup_after_list_notes(self, plugin):
        plugin.manager.create_note(title="Untitled", content="First content")

        first = plugin.execute(intent="conversation:question", query="do i have any notes?", entities={}, context={})
        assert first.get("success") is True

        second = plugin.execute(intent="conversation:general", query="what is the title of the note", entities={}, context={})

        assert second.get("success") is True
        assert second.get("action") == "get_note_title"
        assert second.get("data", {}).get("title") == "Untitled"

    def test_title_followup_single_note_without_context(self, plugin):
        plugin.manager.create_note(title="Project Plan", content="Draft milestones")

        result = plugin.execute(intent="conversation:general", query="what is the title of the note", entities={}, context={})

        assert result.get("success") is True
        assert result.get("data", {}).get("title") == "Project Plan"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
