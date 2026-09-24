from ai.runtime.operator_response_surface import (
    sanitize_operator_chatter,
    strip_meta_response_artifacts,
)
from ai.runtime.response_momentum_policy import apply_response_momentum, strip_passive_followup_sentences


def test_meta_stripper_keeps_lists_and_paragraphs():
    text = "Here is what I have saved:\n- You decided on SQLite.\n- You want tests first.\n\nThat's all of it."

    assert strip_meta_response_artifacts(text) == text


def test_meta_stripper_still_drops_rewrite_notes_and_leading_filler():
    text = "Let me think about this. The cache is cold.\nNote: I've kept the facts the same."

    assert strip_meta_response_artifacts(text) == "The cache is cold."


def test_code_blocks_keep_their_indentation():
    text = "Try this:\n```python\ndef f():\n    return 1\n```"

    assert strip_meta_response_artifacts(text) == text


def test_chatter_filter_drops_whole_sentences_without_flattening():
    text = "Two things changed.\n- The router.\n- The cache. How can I help further?"

    assert sanitize_operator_chatter(text) == "Two things changed.\n- The router.\n- The cache."


def test_followup_filter_keeps_real_offers_and_never_leaves_fragments():
    text = "Local models are closing the gap. Let me know if you want the benchmark numbers."
    assert strip_passive_followup_sentences(text, mode="educational_explain") == text

    text = "That's the tradeoff. Hope this helps, and good luck."
    assert strip_passive_followup_sentences(text, mode="educational_explain") == "That's the tradeoff."


def test_momentum_does_not_cut_phrases_out_of_sentences():
    text = (
        "I'd watch the 30B class over the next year, that's where it flips. "
        "Just let me know if you want a deeper dive on the benchmarks."
    )

    out = apply_response_momentum(
        user_input="what's a good file name for this",
        response_text=text,
        intent="conversation:educational_explain",
        route="llm",
    )

    assert "Just a deeper dive" not in out
    assert "that's where it flips." in out
