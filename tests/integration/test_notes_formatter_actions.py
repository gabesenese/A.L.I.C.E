"""
Formatter tests for structured notes actions.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.models.simple_formatters import NotesFormatter


def test_list_notes_action_formatting():
    payload = {
        "action": "list_notes",
        "data": {
            "count": 2,
            "shown": 2,
            "limit": 10,
            "notes": [
                {"title": "Alpha", "tags": ["work"], "updated_at": "2026-02-20T10:00:00", "preview": "First"},
                {"title": "Beta", "tags": [], "updated_at": "2026-02-20T11:00:00", "preview": "Second"},
            ],
        },
    }

    text = NotesFormatter.format(payload)
    assert "Notes (2)" in text
    assert "Showing 2 of 2 (limit 10)" in text
    assert "1. Alpha" in text
    assert "2. Beta" in text


def test_summarize_note_action_formatting():
    payload = {
        "action": "summarize_note",
        "data": {
            "note_title": "Sprint",
            "summary": {
                "overview": ["Plan scope"],
                "key_points": ["Auth flow", "Testing"],
                "action_items": ["Email vendor"],
                "dates": ["2026-03-10"],
            },
        },
    }

    text = NotesFormatter.format(payload)
    assert "Summary: Sprint" in text
    assert "Key Points:" in text
    assert "Action Items:" in text


def test_note_ambiguous_action_formatting():
    payload = {
        "action": "delete_note",
        "data": {
            "error": "note_ambiguous",
            "selection_hint": "Reply with the option number (e.g., 1 or 2)",
            "candidates": [
                {"option": 1, "title": "Grocery List", "tags": ["home"], "updated_at": "2026-02-20T10:00:00"},
                {"option": 2, "title": "Grocery List Weekend", "tags": [], "updated_at": "2026-02-20T11:00:00"},
            ],
        },
    }

    text = NotesFormatter.format(payload)
    assert "I found multiple matching notes:" in text
    assert "1. Grocery List" in text
    assert "2. Grocery List Weekend" in text
    assert "Reply with the option number" in text
