"""A failed codebase check removes the unsupported sentences, not the whole answer."""

from ai.runtime.turn_orchestrator import _repair_codebase_claims


def test_sentence_naming_a_missing_directory_is_dropped_and_the_rest_kept():
    answer = (
        "Routing starts in the arbiter, which scores each intent.\n"
        "The self_learning directory contains the training workflows.\n"
        "Low scores fall through to the model."
    )

    repaired = _repair_codebase_claims(answer, {"missing_directories": ["self_learning"]})

    assert repaired == "Routing starts in the arbiter, which scores each intent.\nLow scores fall through to the model."


def test_nothing_left_means_no_repair():
    assert _repair_codebase_claims("I checked app/agents.py.", {"missing_paths": ["app/agents.py"]}) == ""
