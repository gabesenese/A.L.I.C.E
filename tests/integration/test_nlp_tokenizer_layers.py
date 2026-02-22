"""Integration tests for layered tokenizer pipeline in NLPProcessor."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.core.nlp_processor import NLPProcessor


class TestLayeredTokenizer:
    def setup_method(self):
        self.nlp = NLPProcessor()
        self.nlp.reset_context()
        self.nlp.set_tokenizer_profile("default")

    def test_surface_segments_meta_and_utterance(self):
        debug = self.nlp.debug_tokenizer("/debug tokens; read the test note")
        segments = debug.get("segments", [])
        assert len(segments) >= 2
        assert segments[0]["kind"] == "meta_command"

    def test_rich_tokens_include_roles_and_flags(self):
        debug = self.nlp.debug_tokenizer("read the second note #work")
        tokens = debug.get("tokens", [])
        assert any(t.get("kind") == "ordinal" for t in tokens)
        assert any(t.get("role") == "action" and t.get("normalized") == "read" for t in tokens)
        assert any(t.get("kind") == "hashtag" for t in tokens)

    def test_parsed_command_extracts_note_read_title(self):
        debug = self.nlp.debug_tokenizer("read the project alpha note for me")
        parsed = debug.get("parsed_command", {})
        assert parsed.get("action") == "read"
        assert parsed.get("object_type") == "note"
        assert "project alpha" in (parsed.get("title_hint") or "")

    def test_process_uses_structured_parse_for_notes_intent(self):
        result = self.nlp.process("do i have a note called test")
        assert result.intent.startswith("notes:")
        assert result.parsed_command.get("object_type") == "note"
        assert "notes" in result.plugin_scores

    def test_strict_profile_reduces_fuzzy_routing(self):
        self.nlp.set_tokenizer_profile("strict")
        debug = self.nlp.debug_tokenizer("please maybe check something quickly")
        scores = debug.get("plugin_scores", {})
        assert scores.get("conversation", 0) >= 0.2
