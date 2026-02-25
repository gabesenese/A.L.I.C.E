"""
Integration tests for notes follow-up title questions.
"""

import json


class TestNotesTitleFollowup:
    def test_title_followup_after_list_notes(self, plugin):
        plugin.manager.create_note(title="Untitled", content="First content")

        first = plugin.execute(intent="conversation:question", query="do i have any notes?", entities={}, context={})
        assert first.get("success") is True
        assert first.get("action") == "list_notes"
        assert first.get("formulate") is True
        assert first.get("data", {}).get("count") == 1

        second = plugin.execute(intent="conversation:general", query="what is the title of the note", entities={}, context={})

        assert second.get("success") is True
        assert second.get("action") == "get_note_title"
        assert second.get("data", {}).get("title") == "Untitled"

    def test_title_followup_single_note_without_context(self, plugin):
        plugin.manager.create_note(title="Project Plan", content="Draft milestones")

        result = plugin.execute(intent="conversation:general", query="what is the title of the note", entities={}, context={})

        assert result.get("success") is True
        assert result.get("data", {}).get("title") == "Project Plan"

    def test_title_followup_resolves_by_ordinal_from_last_list(self, plugin):
        plugin.manager.create_note(title="Alpha Note", content="A")
        plugin.manager.create_note(title="Beta Note", content="B")

        listed = plugin.execute(intent="conversation:question", query="show my notes", entities={}, context={})
        assert listed.get("success") is True
        assert listed.get("action") == "list_notes"

        followup = plugin.execute(
            intent="conversation:general",
            query="what is the title of the second note",
            entities={},
            context={},
        )
        assert followup.get("success") is True
        assert followup.get("action") == "get_note_title"
        assert followup.get("data", {}).get("title") in {"Alpha Note", "Beta Note"}

    def test_delete_note_disambiguation_when_multiple_match(self, plugin):
        plugin.manager.create_note(title="Grocery List", content="milk")
        plugin.manager.create_note(title="Grocery List Weekend", content="eggs")

        result = plugin.execute(intent="conversation:general", query="delete grocery list", entities={}, context={})

        assert result.get("success") is False
        assert result.get("action") == "delete_note"
        assert result.get("data", {}).get("error") == "note_ambiguous"
        assert len(result.get("data", {}).get("candidates", [])) >= 2

    def test_telemetry_and_learning_state_written(self, plugin):
        plugin.manager.create_note(title="Telemetry Note", content="observe me")

        result = plugin.execute(intent="conversation:question", query="show my notes", entities={}, context={})
        assert result.get("success") is True

        assert plugin.telemetry_log_path.exists()
        telemetry_lines = plugin.telemetry_log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(telemetry_lines) >= 1
        telemetry_entry = json.loads(telemetry_lines[-1])
        assert telemetry_entry.get("action") == "list_notes"

        assert plugin.learning_state_path.exists()
        learning_state = json.loads(plugin.learning_state_path.read_text(encoding="utf-8"))
        assert "action_token_weights" in learning_state

    def test_search_guardrail_requests_query_clarification(self, plugin):
        result = plugin.execute(intent="notes:search", query="search notes", entities={}, context={})

        assert result.get("success") is False
        assert result.get("action") == "search_notes"
        data = result.get("data", {})
        assert data.get("error") == "clarification_required"
        assert data.get("requires_clarification") is True
        assert "search" in (data.get("clarification_question") or "").lower()

    def test_telemetry_includes_nlp_metadata_and_event_id(self, plugin):
        plugin.manager.create_note(title="Telemetry Meta", content="body")
        context = {
            "nlp": {
                "intent": "notes:list",
                "intent_confidence": 0.91,
                "parsed_command": {"action": "list", "object_type": "note"},
                "plugin_scores": {"notes": 3.2, "conversation": 0.2},
            }
        }

        result = plugin.execute(intent="notes:list", query="show notes", entities={}, context=context)
        assert result.get("success") is True

        telemetry_lines = plugin.telemetry_log_path.read_text(encoding="utf-8").strip().splitlines()
        telemetry_entry = json.loads(telemetry_lines[-1])
        assert telemetry_entry.get("event_id")
        assert telemetry_entry.get("intent") == "notes:list"
        assert telemetry_entry.get("intent_confidence") >= 0.9
        assert isinstance(telemetry_entry.get("parsed_command"), dict)
        assert isinstance(telemetry_entry.get("plugin_scores"), dict)

    def test_telemetry_dedupes_identical_event_burst(self, plugin):
        plugin.manager.create_note(title="Burst", content="dedupe")
        plugin.execute(intent="notes:list", query="show notes", entities={}, context={})
        plugin.execute(intent="notes:list", query="show notes", entities={}, context={})

        telemetry_lines = plugin.telemetry_log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(telemetry_lines) == 1

    def test_get_note_content_returns_full_content_payload(self, plugin):
        plugin.manager.create_note(title="Release Plan", content="Line 1\nLine 2\n- action")

        result = plugin.execute(
            intent="conversation:general",
            query="show content of that note",
            entities={},
            context={},
        )

        assert result.get("success") is True
        assert result.get("action") == "get_note_content"
        assert "Line 1" in result.get("data", {}).get("content", "")

    def test_summarize_note_returns_structured_sections(self, plugin):
        plugin.manager.create_note(
            title="Sprint Notes",
            content="Finalize auth flow\n- email vendor\nDeadline 2026-03-10\nShip release",
        )

        result = plugin.execute(
            intent="conversation:general",
            query="summarize that note",
            entities={},
            context={},
        )

        assert result.get("success") is True
        assert result.get("action") == "summarize_note"
        summary = result.get("data", {}).get("summary", {})
        assert isinstance(summary.get("key_points", []), list)
        assert isinstance(summary.get("action_items", []), list)
        assert isinstance(summary.get("dates", []), list)

    def test_list_notes_includes_pagination_metadata(self, plugin):
        plugin.manager.create_note(title="One", content="1")
        plugin.manager.create_note(title="Two", content="2")

        result = plugin.execute(
            intent="conversation:question",
            query="show 1 notes",
            entities={},
            context={},
        )

        assert result.get("success") is True
        assert result.get("action") == "list_notes"
        data = result.get("data", {})
        assert data.get("limit") == 1
        assert data.get("shown") == 1
        assert data.get("count") >= 2
        assert data.get("has_more") is True

    def test_resolver_normalizes_title_suffix_for_content(self, plugin):
        plugin.manager.create_note(title="Release Plan", content="Scope and milestones")

        result = plugin.execute(
            intent="conversation:general",
            query="read the release plan note",
            entities={},
            context={},
        )

        assert result.get("success") is True
        assert result.get("action") == "get_note_content"
        assert result.get("data", {}).get("note_title") == "Release Plan"

    def test_content_query_what_is_in_the_note_phrase(self, plugin):
        plugin.manager.create_note(title="Test", content="This is the note body")

        result = plugin.execute(
            intent="conversation:general",
            query="what is in the test note?",
            entities={},
            context={},
        )

        assert result.get("success") is True
        assert result.get("action") == "get_note_content"
        assert result.get("data", {}).get("note_title") == "Test"
        assert "This is the note body" in result.get("data", {}).get("content", "")

    def test_content_query_inside_of_my_note_phrase(self, plugin):
        plugin.manager.create_note(title="Test", content="Inside-of phrase content")

        result = plugin.execute(
            intent="conversation:question",
            query="what is inside of my test note?",
            entities={},
            context={},
        )

        assert result.get("success") is True
        assert result.get("action") == "get_note_content"
        assert result.get("data", {}).get("note_title") == "Test"
        assert "Inside-of phrase content" in result.get("data", {}).get("content", "")

    def test_content_query_read_plural_notes_phrase(self, plugin):
        plugin.manager.create_note(title="Test", content="Plural notes phrase content")

        result = plugin.execute(
            intent="conversation:general",
            query="i want you to read the test notes for me",
            entities={},
            context={},
        )

        assert result.get("success") is True
        assert result.get("action") == "get_note_content"
        assert result.get("data", {}).get("note_title") == "Test"
        assert "Plural notes phrase content" in result.get("data", {}).get("content", "")

    def test_read_it_followup_after_list_notes(self, plugin):
        plugin.manager.create_note(title="Test Note", content="Follow-up read content")

        first = plugin.execute(
            intent="conversation:question",
            query="do i have any notes?",
            entities={},
            context={},
        )
        assert first.get("success") is True
        assert first.get("action") == "list_notes"

        followup = plugin.execute(
            intent="conversation:general",
            query="read it for me",
            entities={},
            context={},
        )

        assert followup.get("success") is True
        assert followup.get("action") == "get_note_content"
        assert followup.get("data", {}).get("note_title") == "Test Note"
        assert "Follow-up read content" in followup.get("data", {}).get("content", "")

    def test_read_the_named_note_for_me_phrase(self, plugin):
        plugin.manager.create_note(title="Test", content="Direct read content")

        result = plugin.execute(
            intent="conversation:general",
            query="read the test note for me",
            entities={},
            context={},
        )

        assert result.get("success") is True
        assert result.get("action") == "get_note_content"
        assert result.get("data", {}).get("note_title") == "Test"
        assert "Direct read content" in result.get("data", {}).get("content", "")

    def test_disambiguation_selection_by_number_executes_delete(self, plugin):
        plugin.manager.create_note(title="Grocery List", content="milk")
        plugin.manager.create_note(title="Grocery List Weekend", content="eggs")

        ambiguous = plugin.execute(
            intent="conversation:general",
            query="delete grocery list",
            entities={},
            context={},
        )

        assert ambiguous.get("success") is False
        assert ambiguous.get("data", {}).get("error") == "note_ambiguous"
        assert ambiguous.get("data", {}).get("requires_selection") is True
        assert len(ambiguous.get("data", {}).get("candidates", [])) >= 2

        followup = plugin.execute(
            intent="conversation:general",
            query="2",
            entities={},
            context={},
        )

        assert followup.get("success") is True
        assert followup.get("action") == "delete_note"
        assert followup.get("data", {}).get("archived") is True
        assert followup.get("data", {}).get("diagnostics", {}).get("resolution_path") == "disambiguation_selection"

    def test_disambiguation_selection_by_title_keyword(self, plugin):
        plugin.manager.create_note(title="Grocery List", content="milk")
        plugin.manager.create_note(title="Grocery List Weekend", content="eggs")

        ambiguous = plugin.execute(
            intent="conversation:general",
            query="delete grocery list",
            entities={},
            context={},
        )
        assert ambiguous.get("success") is False
        assert ambiguous.get("data", {}).get("error") == "note_ambiguous"

        followup = plugin.execute(
            intent="conversation:general",
            query="the weekend one",
            entities={},
            context={},
        )

        assert followup.get("success") is True
        assert followup.get("action") == "delete_note"
        assert followup.get("data", {}).get("note_title") == "Grocery List Weekend"

    def test_disambiguation_selection_by_tag_hint(self, plugin):
        plugin.manager.create_note(title="Project Tasks", content="code", tags=["work"])
        plugin.manager.create_note(title="Project Tasks Personal", content="gym", tags=["personal"])

        ambiguous = plugin.execute(
            intent="conversation:general",
            query="delete project tasks",
            entities={},
            context={},
        )
        assert ambiguous.get("success") is False
        assert ambiguous.get("data", {}).get("error") == "note_ambiguous"

        followup = plugin.execute(
            intent="conversation:general",
            query="the one tagged personal",
            entities={},
            context={},
        )

        assert followup.get("success") is True
        assert followup.get("action") == "delete_note"
        assert followup.get("data", {}).get("note_title") == "Project Tasks Personal"

