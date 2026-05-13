from pathlib import Path

from ai.runtime import learned_response_examples as lre
from ai.runtime.operator_response_surface import (
    detect_context_signal,
    render_operator_response,
)


def _set_store_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        lre,
        "LEARNED_RESPONSE_EXAMPLES_PATH",
        tmp_path / "learned_response_examples.jsonl",
    )


def test_records_valid_learned_example(monkeypatch, tmp_path):
    _set_store_path(monkeypatch, tmp_path)
    example = lre.LearnedResponseExample.create(
        surface="operator_context_ack",
        context_signals=["nap", "work_session"],
        response_text="Fresh start, we'll keep it focused.",
        energy_signal="low",
        mood_signal="neutral",
        topic="Alice",
        user_context_summary="user just woke up from a nap and wants a short Alice work session",
    )
    lre.record_learned_response_example(example)
    loaded = lre.load_learned_response_examples(surface="operator_context_ack", limit=20)
    assert any(ex.response_text == "Fresh start, we'll keep it focused." for ex in loaded)


def test_deduplicates_examples(monkeypatch, tmp_path):
    _set_store_path(monkeypatch, tmp_path)
    ex1 = lre.LearnedResponseExample.create(
        surface="operator_context_ack",
        context_signals=["nap", "work_session"],
        response_text="Fresh start, we'll keep it focused.",
    )
    ex2 = lre.LearnedResponseExample.create(
        surface="operator_context_ack",
        context_signals=["nap"],
        response_text="Fresh start, we'll keep it focused.",
    )
    lre.record_learned_response_example(ex1)
    lre.record_learned_response_example(ex2)
    loaded = lre.load_learned_response_examples(surface="operator_context_ack", limit=20)
    assert len([ex for ex in loaded if ex.response_text == "Fresh start, we'll keep it focused."]) == 1


def test_finds_similar_examples(monkeypatch, tmp_path):
    _set_store_path(monkeypatch, tmp_path)
    lre.record_learned_response_example(
        lre.LearnedResponseExample.create(
            surface="operator_context_ack",
            context_signals=["nap", "work_session"],
            response_text="Fresh start, we'll keep it focused.",
        )
    )
    lre.record_learned_response_example(
        lre.LearnedResponseExample.create(
            surface="operator_context_ack",
            context_signals=["long_day", "work_session"],
            response_text="Short pass tonight.",
        )
    )
    similar = lre.find_similar_response_examples(
        context_signals=["nap"],
        surface="operator_context_ack",
        limit=3,
    )
    assert similar
    assert similar[0].response_text == "Fresh start, we'll keep it focused."


def test_does_not_store_invalid_output(monkeypatch, tmp_path):
    _set_store_path(monkeypatch, tmp_path)

    def _bad_llm(*args, **kwargs):
        return "That must have been hard. How are you feeling?"

    context = detect_context_signal("just woke up from a nap and want to work on alice")
    assert context["has_context"] is True
    render_operator_response(
        user_input="just woke up from a nap and want to work on alice",
        base_text="",
        operator_state={},
        local_execution={},
        next_step="",
        llm_generate=_bad_llm,
    )
    loaded = lre.load_learned_response_examples(surface="operator_context_ack", limit=20)
    assert loaded == []
