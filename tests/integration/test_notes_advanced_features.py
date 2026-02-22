"""
Regression tests for the advanced Notes plugin features batch.

Covers:
  #1  Full-text content search
  #3  Create note from conversation context
  #4  Destructive action confirmation gate
  #5  Append / partial field editing
  #6  Feedback loop (telemetry bridge wiring — smoke test)
  #7  Reminders / due-date surfacing in list payload
  #8  Formatter Strategy pattern (compact vs detailed list)
  #9  Note linking (bidirectional)
"""

import sys
import json
from pathlib import Path

import pytest
from datetime import datetime, timedelta

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.plugins.notes_plugin import NotesPlugin, NotesManager, ContentSearchResult
from ai.models.simple_formatters import (
    NotesFormatter,
    CompactNotesListStrategy,
    DetailedNotesListStrategy,
    _pick_list_strategy,
    FormatterStrategy,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def plugin(tmp_path):
    """A fresh NotesPlugin wired to a temp directory so tests are isolated."""
    p = NotesPlugin()
    p.manager = NotesManager(notes_dir=str(tmp_path))
    p.initialize()
    return p


# ---------------------------------------------------------------------------
# Feature #1 — Full-text content search
# ---------------------------------------------------------------------------

class TestContentSearch:
    def test_search_by_content_finds_match(self, plugin):
        plugin.manager.create_note(
            title="Meeting recap",
            content="We discussed the quarterly budget and decided to cut costs.",
        )
        results = plugin.manager.search_by_content("quarterly budget")
        assert len(results) == 1
        assert "quarterly budget" in results[0].matched_snippet.lower()

    def test_search_by_content_returns_ContentSearchResult(self, plugin):
        plugin.manager.create_note(title="Ideas", content="Build a rocket ship and explore space.")
        results = plugin.manager.search_by_content("rocket ship")
        assert all(isinstance(r, ContentSearchResult) for r in results)
        assert results[0].score > 0

    def test_search_by_content_scores_title_higher(self, plugin):
        plugin.manager.create_note(title="rocket plans", content="General notes.")
        plugin.manager.create_note(title="Unrelated", content="a rocket is mentioned here briefly.")
        results = plugin.manager.search_by_content("rocket")
        # Note with "rocket" in title should score higher
        assert results[0].note.title == "rocket plans"

    def test_search_notes_plugin_action_content_search(self, plugin):
        plugin.manager.create_note(
            title="Project Alpha",
            content="This project involves building a distributed cache system.",
        )
        result = plugin.execute(command="search note content for distributed cache")
        assert result["success"] is True
        assert result["action"] == "search_notes_content"
        assert result["data"]["count"] == 1

    def test_search_notes_falls_back_to_content_when_fuzzy_empty(self, plugin):
        plugin.manager.create_note(
            title="XYZ",
            content="Unique phrase: zelophehad inheritance rules.",
        )
        result = plugin.execute(command="search notes for zelophehad")
        # Fuzzy search won't find "zelophehad" in title/tags — falls back to content
        assert result["success"] is True
        assert result["action"] in ("search_notes_content", "search_notes")
        assert result["data"]["count"] >= 1

    def test_content_search_respects_archived_exclude(self, plugin):
        note = plugin.manager.create_note(title="Old note", content="hidden archived text")
        plugin.manager.archive_note(note.id)
        results = plugin.manager.search_by_content("hidden archived text", include_archived=False)
        assert len(results) == 0

    def test_content_search_includes_archived_when_flag_set(self, plugin):
        note = plugin.manager.create_note(title="Old note", content="hidden archived text")
        plugin.manager.archive_note(note.id)
        results = plugin.manager.search_by_content("hidden archived text", include_archived=True)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Feature #3 — Create note from conversation context
# ---------------------------------------------------------------------------

class TestCreateNoteFromContext:
    def test_record_conversation_turn_stores_turns(self, plugin):
        plugin.record_conversation_turn("user", "what is the capital of France?")
        plugin.record_conversation_turn("assistant", "The capital of France is Paris.")
        assert len(plugin._conversation_context) == 2

    def test_sliding_window_capped_at_ten(self, plugin):
        for i in range(15):
            plugin.record_conversation_turn("user", f"turn {i}")
        assert len(plugin._conversation_context) == 10

    def test_create_from_context_requires_context(self, plugin):
        # No conversation turns seeded → should fail gracefully
        result = plugin.execute(command="save this")
        assert result["action"] == "create_note_from_context"
        assert result["data"]["error"] == "no_context_available"

    def test_create_from_context_uses_seeded_turns(self, plugin):
        plugin.record_conversation_turn("user", "Can you summarize the meeting agenda?")
        plugin.record_conversation_turn("assistant", "Sure — we covered budget review and Q3 goals.")
        result = plugin.execute(command="save this")
        assert result["success"] is True
        assert result["action"] == "create_note_from_context"
        note = plugin.manager.get_note(result["data"]["note_id"])
        assert note is not None
        assert "budget review" in note.content or "Q3 goals" in note.content

    def test_create_from_context_accepts_custom_title(self, plugin):
        plugin.record_conversation_turn("user", "Remember to call John tomorrow.")
        plugin.record_conversation_turn("assistant", "Noted.")
        result = plugin.execute(command="remember that title call john")
        assert result["success"] is True
        note = plugin.manager.get_note(result["data"]["note_id"])
        assert note is not None


# ---------------------------------------------------------------------------
# Feature #4 — Destructive action confidence guard
# ---------------------------------------------------------------------------

class TestDestructiveConfirmationGate:
    def test_delete_all_requires_confirmation(self, plugin):
        plugin.manager.create_note(title="Note A", content="content")
        result = plugin.execute(command="delete all notes")
        assert result["action"] == "requires_confirmation"
        assert plugin.pending_confirmation is not None
        assert plugin.pending_confirmation["action"] == "delete_all_notes"

    def test_confirm_executes_delete_all(self, plugin):
        plugin.manager.create_note(title="Note A", content="content")
        plugin.execute(command="delete all notes")  # triggers gate
        result = plugin.execute(command="confirm")
        assert result["success"] is True
        assert result["action"] == "delete_notes"

    def test_cancel_aborts_delete_all(self, plugin):
        plugin.manager.create_note(title="Note A", content="content")
        plugin.execute(command="delete all notes")
        result = plugin.execute(command="cancel")
        assert result["success"] is True
        assert result["action"] == "action_cancelled"
        # Note was NOT deleted
        active = plugin.manager.get_all_notes()
        assert len(active) == 1

    def test_pending_confirmation_cleared_after_cancel(self, plugin):
        plugin.manager.create_note(title="Note B", content="stuff")
        plugin.execute(command="delete all notes")
        plugin.execute(command="cancel")
        assert plugin.pending_confirmation is None

    def test_can_handle_returns_true_for_confirm(self, plugin):
        plugin.manager.create_note(title="Note C", content="x")
        plugin.execute(command="delete all notes")
        assert plugin.can_handle(command="confirm") is True


class TestNotesRoutingGuards:
    def test_can_handle_ignores_code_file_introspection_prompt(self, plugin):
        assert plugin.can_handle(command="tell me what notes_plugin.py does") is False


# ---------------------------------------------------------------------------
# Feature #5 — Append / partial field editing
# ---------------------------------------------------------------------------

class TestAppendNote:
    def test_append_adds_text_to_existing_content(self, plugin):
        note = plugin.manager.create_note(title="Shopping list", content="- Milk\n- Eggs")
        plugin.manager.append_note_content(note.id, "- Bread")
        updated = plugin.manager.get_note(note.id)
        assert "Bread" in updated.content
        assert "Milk" in updated.content  # original content preserved

    def test_append_via_execute_command(self, plugin):
        note = plugin.manager.create_note(title="Todo list", content="- Task A")
        plugin.last_note_id = note.id
        result = plugin.execute(command="append: - Task B")
        assert result["action"] == "append_note"
        assert result["success"] is True
        updated = plugin.manager.get_note(note.id)
        assert "Task B" in updated.content

    def test_patch_note_fields_updates_priority(self, plugin):
        note = plugin.manager.create_note(title="Urgent report", content="Draft.")
        plugin.manager.patch_note_fields(note.id, {"priority": "urgent"})
        updated = plugin.manager.get_note(note.id)
        assert updated.priority == "urgent"

    def test_patch_note_fields_updates_tags(self, plugin):
        note = plugin.manager.create_note(title="Dev notes", content="spec.", tags=["work"])
        plugin.manager.patch_note_fields(note.id, {"tags": ["work", "dev", "sprint"]})
        updated = plugin.manager.get_note(note.id)
        assert "sprint" in updated.tags

    def test_patch_note_fields_ignores_unknown_keys(self, plugin):
        note = plugin.manager.create_note(title="Safe note", content="content")
        result = plugin.manager.patch_note_fields(note.id, {"unknown_field": "value"})
        assert result is not None  # No exception raised

    def test_append_returns_error_when_no_target(self, plugin):
        # No last_note_id set, no title given
        plugin.last_note_id = None
        result = plugin.execute(command="append: extra text")
        assert result["action"] == "append_note"
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Feature #7 — Reminders / due-date surfacing
# ---------------------------------------------------------------------------

class TestReminderSurfacing:
    def test_get_upcoming_reminders_returns_due_within_48h(self, plugin):
        soon = (datetime.now() + timedelta(hours=12)).isoformat()
        plugin.manager.create_note(title="Due soon", content="check this", due_date=soon)
        upcoming = plugin.manager.get_upcoming_reminders(hours=48)
        assert len(upcoming) == 1
        assert upcoming[0].title == "Due soon"

    def test_get_upcoming_reminders_excludes_far_future(self, plugin):
        far = (datetime.now() + timedelta(days=10)).isoformat()
        plugin.manager.create_note(title="Far away", content="later", due_date=far)
        upcoming = plugin.manager.get_upcoming_reminders(hours=48)
        assert len(upcoming) == 0

    def test_list_notes_payload_includes_overdue_count(self, plugin):
        overdue_dt = (datetime.now() - timedelta(days=2)).isoformat()
        plugin.manager.create_note(title="Overdue task", content="late", due_date=overdue_dt)
        result = plugin.execute(command="list notes")
        assert "overdue_count" in result["data"]
        assert result["data"]["overdue_count"] >= 1

    def test_list_notes_payload_includes_upcoming_reminders(self, plugin):
        soon = (datetime.now() + timedelta(hours=6)).isoformat()
        plugin.manager.create_note(title="Morning standup", content="team sync", due_date=soon)
        result = plugin.execute(command="list notes")
        assert "upcoming_reminders" in result["data"]
        assert len(result["data"]["upcoming_reminders"]) >= 1


# ---------------------------------------------------------------------------
# Feature #8 — Formatter Strategy pattern (adaptive list rendering)
# ---------------------------------------------------------------------------

class TestFormatterStrategy:
    def test_pick_strategy_compact_for_large_lists(self):
        strategy = _pick_list_strategy(note_count=9)
        assert isinstance(strategy, CompactNotesListStrategy)

    def test_pick_strategy_detailed_for_small_lists(self):
        strategy = _pick_list_strategy(note_count=3)
        assert isinstance(strategy, DetailedNotesListStrategy)

    def test_pick_strategy_boundary_at_8(self):
        assert isinstance(_pick_list_strategy(8), DetailedNotesListStrategy)
        assert isinstance(_pick_list_strategy(9), CompactNotesListStrategy)

    def test_compact_strategy_satisfies_protocol(self):
        s = CompactNotesListStrategy()
        assert isinstance(s, FormatterStrategy)

    def test_detailed_strategy_satisfies_protocol(self):
        s = DetailedNotesListStrategy()
        assert isinstance(s, FormatterStrategy)

    def test_compact_strategy_single_line_per_note(self):
        notes = [
            {"title": "Alpha", "tags": ["work"], "updated_at": "2026-02-20", "preview": "some preview"}
            for _ in range(10)
        ]
        s = CompactNotesListStrategy()
        lines = s.render_notes_list(notes, count=10, shown=10, header="Notes")
        # One line per note (no preview lines in compact mode)
        content_lines = [l for l in lines if l.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."))]
        assert len(content_lines) == 9  # lines 1-9

    def test_detailed_strategy_includes_preview(self):
        notes = [{"title": "Alpha", "tags": [], "updated_at": "2026-02-20", "preview": "First paragraph here."}]
        s = DetailedNotesListStrategy()
        lines = s.render_notes_list(notes, count=1, shown=1, header="Notes")
        assert any("First paragraph" in l for l in lines)

    def test_formatter_uses_strategy_for_list_action(self):
        # 10 notes → compact rendering expected
        payload = {
            "action": "list_notes",
            "data": {
                "count": 10,
                "shown": 10,
                "limit": 10,
                "notes": [
                    {"title": f"Note {i}", "tags": [], "updated_at": "2026-02-20", "preview": "body"}
                    for i in range(10)
                ],
            },
        }
        text = NotesFormatter.format(payload)
        assert "Notes (10)" in text
        # Preview text should NOT appear in compact mode (>8 notes)
        assert "body" not in text

    def test_formatter_reminder_notice_in_list(self):
        payload = {
            "action": "list_notes",
            "data": {
                "count": 1,
                "shown": 1,
                "limit": 10,
                "notes": [{"title": "Alpha", "tags": [], "updated_at": "2026-02-20", "preview": ""}],
                "overdue_count": 2,
                "upcoming_reminders": [
                    {"title": "Standup", "due_date": "2026-02-22T09:00:00", "reminder": None, "priority": "high"}
                ],
            },
        }
        text = NotesFormatter.format(payload)
        assert "overdue" in text.lower()
        assert "Standup" in text


# ---------------------------------------------------------------------------
# Feature #9 — Note linking
# ---------------------------------------------------------------------------

class TestNoteLinking:
    def test_link_notes_by_ids_is_bidirectional(self, plugin):
        a = plugin.manager.create_note(title="Concept A", content="first idea")
        b = plugin.manager.create_note(title="Concept B", content="second idea")
        result = plugin.manager.link_notes_by_ids(a.id, b.id)
        assert result is True
        assert b.id in plugin.manager.get_note(a.id).related_notes
        assert a.id in plugin.manager.get_note(b.id).related_notes

    def test_link_notes_idempotent(self, plugin):
        a = plugin.manager.create_note(title="X", content=".")
        b = plugin.manager.create_note(title="Y", content=".")
        plugin.manager.link_notes_by_ids(a.id, b.id)
        plugin.manager.link_notes_by_ids(a.id, b.id)  # second call
        assert plugin.manager.get_note(a.id).related_notes.count(b.id) == 1

    def test_link_notes_via_execute_parses_titles(self, plugin):
        plugin.manager.create_note(title="Frontend plan", content="React redesign")
        plugin.manager.create_note(title="Backend plan", content="API redesign")
        result = plugin.execute(command="link frontend plan to backend plan")
        assert result["success"] is True
        assert result["action"] == "link_notes"
        assert result["data"]["bidirectional"] is True

    def test_link_notes_returns_error_for_unknown_title(self, plugin):
        plugin.manager.create_note(title="Real note", content="exists")
        result = plugin.execute(command="link real note to nonexistent one")
        assert result["success"] is False
        assert result["action"] == "link_notes"

    def test_formatter_link_notes_action(self):
        payload = {
            "action": "link_notes",
            "data": {
                "note_a_title": "Frontend plan",
                "note_b_title": "Backend plan",
                "bidirectional": True,
            },
        }
        text = NotesFormatter.format(payload)
        assert "Frontend plan" in text
        assert "Backend plan" in text


# ---------------------------------------------------------------------------
# Feature #2 — NoteContextProvider Protocol
# ---------------------------------------------------------------------------

class TestNoteContextProvider:
    def test_plugin_satisfies_note_context_provider_protocol(self, plugin):
        from ai.plugins.notes_plugin import NoteContextProvider
        # NotesPlugin should satisfy the runtime-checkable Protocol
        assert isinstance(plugin, NoteContextProvider)

    def test_get_note_context_snippet_returns_empty_for_no_notes(self, plugin):
        snippet = plugin.get_note_context_snippet("budget")
        assert snippet == ""

    def test_get_note_context_snippet_returns_content(self, plugin):
        plugin.manager.create_note(
            title="Q3 planning",
            content="We need to reduce the quarterly budget by 20% and hire two engineers.",
        )
        snippet = plugin.get_note_context_snippet("quarterly budget")
        assert "Q3 planning" in snippet or "quarterly budget" in snippet.lower()

    def test_get_note_context_snippet_respects_max_chars(self, plugin):
        for i in range(10):
            plugin.manager.create_note(title=f"Note {i}", content=f"Content {i} " * 30)
        snippet = plugin.get_note_context_snippet("content", max_chars=120)
        assert len(snippet) <= 200  # Allow some header overhead


# ---------------------------------------------------------------------------
# Additional formatter handlers for CRUD actions
# ---------------------------------------------------------------------------

class TestNewFormatterHandlers:
    def test_format_create_note(self):
        payload = {
            "action": "create_note",
            "data": {"title": "Sprint tasks", "tags": ["work", "sprint"], "note_type": "todo"},
        }
        text = NotesFormatter.format(payload)
        assert "Sprint tasks" in text
        assert "work" in text or "sprint" in text

    def test_format_append_note(self):
        payload = {
            "action": "append_note",
            "data": {"note_title": "Shopping list", "appended_text": "- Butter", "new_length": 30},
        }
        text = NotesFormatter.format(payload)
        assert "Shopping list" in text
        assert "Butter" in text

    def test_format_delete_note(self):
        payload = {
            "action": "delete_note",
            "data": {"note_title": "Old note", "archived": True, "restorable": True},
        }
        text = NotesFormatter.format(payload)
        assert "Old note" in text
        assert "archived" in text.lower() or "restored" in text.lower()

    def test_format_pin_note(self):
        payload = {
            "action": "pin_note",
            "data": {"note_title": "Important reminder", "pinned": True},
        }
        text = NotesFormatter.format(payload)
        assert "Pinned" in text
        assert "Important reminder" in text

    def test_format_set_priority(self):
        payload = {
            "action": "set_priority",
            "data": {"note_title": "Deadline task", "priority": "urgent"},
        }
        text = NotesFormatter.format(payload)
        assert "urgent" in text
        assert "Deadline task" in text

    def test_format_show_overdue_notes_none(self):
        payload = {
            "action": "show_overdue_notes",
            "data": {"count": 0, "notes": []},
        }
        text = NotesFormatter.format(payload)
        assert "caught up" in text.lower() or "no overdue" in text.lower()

    def test_format_requires_confirmation(self):
        payload = {
            "action": "requires_confirmation",
            "data": {
                "error": "requires_confirmation",
                "pending_action": "delete_all_notes",
                "note_title": "ALL 5 active notes",
                "prompt": 'This will archive "ALL 5 active notes". Reply "confirm" to proceed or "cancel" to abort.',
            },
        }
        text = NotesFormatter.format(payload)
        assert "confirm" in text.lower()
        assert "cancel" in text.lower()

    def test_format_search_notes_content(self):
        payload = {
            "action": "search_notes_content",
            "data": {
                "query": "rocket",
                "count": 2,
                "found": True,
                "results": [
                    {"title": "Space ideas", "score": 12.5, "matched_snippet": "Build a rocket ship"},
                    {"title": "Engineering", "score": 8.0, "matched_snippet": "rocket engine design"},
                ],
            },
        }
        text = NotesFormatter.format(payload)
        assert "rocket" in text
        assert "Space ideas" in text
        assert "Engineering" in text

    def test_format_show_recent_note_changes(self):
        payload = {
            "action": "show_recent_note_changes",
            "data": {
                "count": 2,
                "changes": [
                    {"note_title": "A", "action": "update_note", "reason": ""},
                    {"note_title": "B", "action": "set_priority", "reason": "urgency"},
                ],
            },
        }
        text = NotesFormatter.format(payload)
        assert "Recent note changes" in text
        assert "set_priority" in text

    def test_format_notes_health_check(self):
        payload = {
            "action": "notes_health_check",
            "data": {"healthy": True, "checked_notes": 3, "anomaly_count": 0, "anomalies": []},
        }
        text = NotesFormatter.format(payload)
        assert "health check passed" in text.lower()


# ---------------------------------------------------------------------------
# New hardening features — versioning config, change feed, health check
# ---------------------------------------------------------------------------

class TestNotesHardeningFeatures:
    def test_versioning_respects_settings_and_adds_metadata(self, tmp_path):
        manager = NotesManager(
            notes_dir=str(tmp_path),
            settings={
                "versioning_enabled": True,
                "max_versions_per_note": 2,
            },
        )
        note = manager.create_note(title="Config test", content="v1")
        manager.update_note(note.id, content="v2")
        manager.update_note(note.id, content="v3")
        manager.update_note(note.id, content="v4")

        versions = manager.get_note_versions(note.id)
        assert len(versions) == 2
        assert all("version_action" in v for v in versions)
        assert all(v.get("version_action") == "update_note" for v in versions)

    def test_show_recent_changes_action(self, plugin):
        created = plugin.manager.create_note(title="Change feed", content="before")
        plugin.manager.update_note(created.id, content="after")

        result = plugin.execute(command="show recent changes to my notes")
        assert result["success"] is True
        assert result["action"] == "show_recent_note_changes"
        assert result["data"]["count"] >= 1
        assert "action" in result["data"]["changes"][0]

    def test_notes_health_check_action(self, plugin):
        plugin.manager.create_note(title="Health A", content="ok")
        plugin.manager.create_note(title="Health B", content="ok")

        result = plugin.execute(command="run notes health check")
        assert result["success"] is True
        assert result["action"] == "notes_health_check"
        assert result["data"]["checked_notes"] >= 2
        assert "anomalies" in result["data"]

    def test_user_templates_file_is_merged(self, tmp_path):
        templates_file = tmp_path / "templates.json"
        templates_file.write_text(
            json.dumps(
                {
                    "retro": {
                        "title_prefix": "Retro",
                        "category": "work",
                        "note_type": "meeting",
                        "template": "Wins:\nLearnings:\nActions:\n",
                        "tags": ["retro", "team"],
                    }
                }
            ),
            encoding="utf-8",
        )

        manager = NotesManager(notes_dir=str(tmp_path))
        assert "retro" in manager.list_templates()
