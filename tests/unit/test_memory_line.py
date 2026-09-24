"""Recalled memories reach the prompt as readable lines, not key=value logs."""

from ai.runtime.boundaries.boundary_factory import _memory_line


def test_stored_turn_reads_as_what_was_said():
    line = _memory_line("user=I moved the parser into app/nlp.py today.\nassistant=Nice. The tests still pass.")

    assert line == 'They said "I moved the parser into app/nlp.py today."; you answered "Nice."'
    assert "user=" not in line and "assistant=" not in line


def test_current_turn_format_reads_the_same_way():
    line = _memory_line("User said: we ship on Friday\nAlice replied: Noted.")

    assert line == 'They said "we ship on Friday"; you answered "Noted."'


def test_periods_inside_words_do_not_cut_the_memory():
    assert _memory_line("Gabriel upgraded to Python 3.11 for the build. It went fine.") == (
        "Gabriel upgraded to Python 3.11 for the build."
    )
