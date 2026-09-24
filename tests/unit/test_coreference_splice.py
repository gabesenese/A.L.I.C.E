"""A reference is replaced where the resolver found it, not where its letters first appear.

Resolution used text.replace(match, title, 1), which rewrites the first occurrence
of the matched characters anywhere in the sentence. The standalone "it" the regex
had found was left alone, and the "it" inside "with" was replaced instead:
"what should I do with it?" became 'what should I do w"Groceries"h it?' before
intent classification ever saw it.
"""

from ai.core.coreference import AdvancedCoreferenceResolver, DialogueMemory


def _resolver_remembering(title: str) -> AdvancedCoreferenceResolver:
    memory = DialogueMemory()
    memory.record("NOTE_REF", title, plugin="notes")
    return AdvancedCoreferenceResolver(memory)


def test_a_pronoun_is_replaced_where_it_stands():
    resolved = _resolver_remembering("Groceries").resolve_text("what should I do with it?", {})
    assert resolved == 'what should I do with "Groceries"?'


def test_a_word_that_contains_the_pronoun_is_left_intact():
    resolved = _resolver_remembering("Groceries").resolve_text(
        "I'm thinking about rewriting the memory layer. Is it worth it?", {}
    )
    assert "rewriting the memory layer" in resolved


def test_a_domain_phrase_is_replaced_where_it_stands():
    resolved = _resolver_remembering("Groceries").resolve_text("open the notebook and read the note", {})
    assert resolved == 'open the notebook and read "Groceries"'


def test_an_old_note_is_not_the_referent_of_a_new_pronoun():
    resolver = _resolver_remembering("Groceries")
    for _ in range(8):
        resolver.memory.update_from_nlp_result("conversation:general", {})

    assert resolver.resolve_text("what do you think about that?", {}) == "what do you think about that?"


def test_a_note_from_the_turn_before_is_still_the_referent():
    resolver = _resolver_remembering("Groceries")
    resolver.memory.update_from_nlp_result("conversation:general", {})

    assert resolver.resolve_text("what should I do with it?", {}) == 'what should I do with "Groceries"?'
