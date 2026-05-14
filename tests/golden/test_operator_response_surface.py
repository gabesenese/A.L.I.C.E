from pathlib import Path

from ai.runtime import learned_response_examples as lre
from ai.runtime.operator_response_surface import (
    render_local_execution_error_response,
    render_operator_response,
    strip_meta_response_artifacts,
)


def _set_store_path(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        lre,
        "LEARNED_RESPONSE_EXAMPLES_PATH",
        tmp_path / "learned_response_examples.jsonl",
    )


def test_operator_continue_response_is_compact_and_decisive_without_llm_ack(monkeypatch, tmp_path):
    _set_store_path(monkeypatch, tmp_path)
    out = render_operator_response(
        user_input="good, it was a cold day today so i just stayed home, now i am gonna work on alice for a bit",
        base_text="I've taken a look at core files and here are many options...",
        operator_state={
            "last_recommended_action": {
                "action": "inspect_file",
                "target": "ai/runtime/operator_state.py",
                "reason": "It stores active objective, current focus, inspected files, and recommendations.",
            }
        },
        local_execution={
            "success": True,
            "inspected_file": "ai/runtime/agent_loop.py",
            "analysis": {"lines": 326, "classes": 5, "functions": 5},
        },
        next_step="inspect file ai/runtime/operator_state.py because It stores active objective, current focus, inspected files, and recommendations.",
        llm_generate=None,
    )
    low = out.lower()
    assert out.startswith("I inspected ai/runtime/agent_loop.py.")
    assert "warm and cozy" not in low
    assert "what would you like to tackle first" not in low
    assert "next best move:" in low
    assert "ai/runtime/operator_state.py" in out


def test_llm_generated_acknowledgement_used_and_recorded(monkeypatch, tmp_path):
    _set_store_path(monkeypatch, tmp_path)

    def _llm(*args, **kwargs):
        return "Fresh start. We'll keep it focused."

    out = render_operator_response(
        user_input="just woke up from a nap, now i am gonna work on Alice for a bit",
        base_text="",
        operator_state={
            "last_recommended_action": {
                "action": "inspect_file",
                "target": "ai/runtime/response_momentum_policy.py",
                "reason": "It shapes whether Alice advances with momentum or drifts into passive responses.",
            }
        },
        local_execution={
            "success": True,
            "inspected_file": "ai/runtime/agent_loop.py",
            "analysis": {"responsibility": "agent loop"},
        },
        next_step="",
        llm_generate=_llm,
    )
    assert out.startswith("Fresh start. We'll keep it focused.")
    assert "I inspected ai/runtime/agent_loop.py." in out
    assert "bounded operator loop" in out.lower()
    loaded = lre.load_learned_response_examples(surface="operator_context_ack", limit=5)
    assert any(ex.response_text == "Fresh start. We'll keep it focused." for ex in loaded)


def test_generic_task_kickoff_rejected_then_retry_used(monkeypatch, tmp_path):
    _set_store_path(monkeypatch, tmp_path)
    responses = iter(
        [
            "Let's dive into some Alice development work together now.",
            "Fresh start. We'll keep it focused.",
        ]
    )

    def _llm(*args, **kwargs):
        return next(responses)

    out = render_operator_response(
        user_input="good, just woke up from a nap and im gonna work on alice for now",
        base_text="",
        operator_state={
            "last_recommended_action": {
                "action": "inspect_file",
                "target": "ai/runtime/response_momentum_policy.py",
                "reason": "It shapes whether Alice advances with momentum or drifts into passive responses.",
            }
        },
        local_execution={
            "success": True,
            "inspected_file": "ai/runtime/agent_loop.py",
            "analysis": {"responsibility": "agent loop"},
        },
        next_step="",
        llm_generate=_llm,
    )
    assert out.startswith("Fresh start. We'll keep it focused.")
    assert "Let's dive into some Alice development work together now." not in out
    loaded = lre.load_learned_response_examples(surface="operator_context_ack", limit=5)
    assert any(ex.response_text == "Fresh start. We'll keep it focused." for ex in loaded)
    assert all(
        ex.response_text != "Let's dive into some Alice development work together now."
        for ex in loaded
    )


def test_llm_unavailable_means_no_acknowledgement(monkeypatch, tmp_path):
    _set_store_path(monkeypatch, tmp_path)
    out = render_operator_response(
        user_input="just woke up from a nap, now i am gonna work on Alice for a bit",
        base_text="",
        operator_state={
            "last_recommended_action": {
                "action": "inspect_file",
                "target": "ai/runtime/response_momentum_policy.py",
                "reason": "It shapes whether Alice advances with momentum or drifts into passive responses.",
            }
        },
        local_execution={
            "success": True,
            "inspected_file": "ai/runtime/agent_loop.py",
            "analysis": {"responsibility": "agent loop"},
        },
        next_step="",
        llm_generate=None,
    )
    assert out.startswith("I inspected ai/runtime/agent_loop.py.")
    assert "Post-nap session. We'll keep this focused." not in out


def test_invalid_acknowledgement_is_omitted_and_not_saved(monkeypatch, tmp_path):
    _set_store_path(monkeypatch, tmp_path)

    calls = {"count": 0}

    def _llm(*args, **kwargs):
        calls["count"] += 1
        return "Let's work on Alice now."

    out = render_operator_response(
        user_input="just woke up from a nap, now i am gonna work on Alice for a bit",
        base_text="",
        operator_state={},
        local_execution={
            "success": True,
            "inspected_file": "ai/runtime/agent_loop.py",
            "analysis": {"responsibility": "agent loop"},
        },
        next_step="inspect ai/runtime/response_momentum_policy.py",
        llm_generate=_llm,
    )
    assert out.startswith("I inspected ai/runtime/agent_loop.py.")
    assert "Let's work on Alice now." not in out
    assert calls["count"] == 2
    loaded = lre.load_learned_response_examples(surface="operator_context_ack", limit=5)
    assert not loaded


def test_learned_examples_are_included_in_prompt(monkeypatch, tmp_path):
    _set_store_path(monkeypatch, tmp_path)
    lre.record_learned_response_example(
        lre.LearnedResponseExample.create(
            surface="operator_context_ack",
            context_signals=["nap", "work_session"],
            response_text="Fresh start. We'll keep it focused.",
            energy_signal="low",
            mood_signal="neutral",
            topic="Alice",
            user_context_summary="user just woke up from a nap and wants a short Alice work session",
        )
    )
    capture = {"prompt": ""}

    def _llm(*args, **kwargs):
        prompt = kwargs.get("prompt") if kwargs else ""
        if not prompt and args:
            prompt = args[0]
        capture["prompt"] = str(prompt or "")
        return "Quick reset. Focused pass."

    render_operator_response(
        user_input="just woke up from a nap, now i am gonna work on Alice for a bit",
        base_text="",
        operator_state={},
        local_execution={
            "success": True,
            "inspected_file": "ai/runtime/agent_loop.py",
            "analysis": {"responsibility": "agent loop"},
        },
        next_step="",
        llm_generate=_llm,
    )
    assert "Similar accepted examples for style only" in capture["prompt"]
    assert "Fresh start. We'll keep it focused." in capture["prompt"]


def test_no_hardcoded_ack_phrases_when_llm_unavailable(monkeypatch, tmp_path):
    _set_store_path(monkeypatch, tmp_path)
    out = render_operator_response(
        user_input="it was a cold day so i stayed home and now i want to work on Alice",
        base_text="",
        operator_state={},
        local_execution={
            "success": True,
            "inspected_file": "ai/runtime/agent_loop.py",
            "analysis": {"responsibility": "agent loop"},
        },
        next_step="inspect ai/runtime/response_momentum_policy.py",
        llm_generate=None,
    )
    assert "Cold day. Good night to work on the core." not in out
    assert "Makes sense. Good time for a focused pass." not in out
    assert "Long day. We'll keep this light." not in out
    assert out.startswith("I inspected ai/runtime/agent_loop.py.")


def test_next_move_reason_grammar_is_clean():
    out = render_operator_response(
        user_input="inspect",
        base_text="",
        operator_state={
            "last_recommended_action": {
                "action": "inspect_file",
                "target": "ai/runtime/response_momentum_policy.py",
                "reason": "It shapes whether Alice advances with momentum or drifts into passive responses..",
            }
        },
        local_execution={},
        next_step="",
        llm_generate=None,
    )
    assert (
        "Next best move: inspect ai/runtime/response_momentum_policy.py because it shapes whether Alice advances with momentum or drifts into passive responses."
        in out
    )
    assert "because It" not in out
    assert ".." not in out


def test_local_execution_error_surface_is_compact_and_no_meta_leak(monkeypatch, tmp_path):
    _set_store_path(monkeypatch, tmp_path)
    out = render_local_execution_error_response(
        user_input="good, i am ready to work on alice",
        base_text=(
            "I'm focused and ready to support your work on me. "
            "Ready when you are! "
            "What would you like to start working on in our codebase today? "
            "(Note: I've rewritten the response to sound more natural, clear, and concise while keeping the same facts.)"
        ),
        operator_state={},
        local_execution={"success": False, "error": "local file target could not be resolved"},
        next_step="inspect ai/runtime/agent_loop.py because active objective exists; agent loop should drive next safe step",
    )
    low = out.lower()
    assert "i couldn't verify the local step." in low
    assert "blocker: local file target could not be resolved" in low
    assert "next best move: inspect ai/runtime/agent_loop.py".lower() in low
    assert "rewritten" not in low
    assert "same facts" not in low
    assert "ready when you are" not in low
    assert "what would you like" not in low


def test_failed_local_execution_does_not_claim_inspection(monkeypatch, tmp_path):
    _set_store_path(monkeypatch, tmp_path)
    out = render_local_execution_error_response(
        user_input="continue",
        base_text="I inspected ai/runtime/agent_loop.py.",
        operator_state={},
        local_execution={"success": False, "error": "contract_local_execution_error"},
        next_step="inspect ai/runtime/agent_loop.py",
    )
    assert "I inspected" not in out
    assert out.startswith("I couldn't verify the local step.")


def test_meta_artifact_sanitizer_removes_rewrite_notes():
    out = strip_meta_response_artifacts(
        "What would you like to start working on? (Note: I've rewritten the response to sound more natural while keeping the same facts.)"
    )
    low = out.lower()
    assert "rewritten" not in low
    assert "same facts" not in low
    assert "(note:" not in low


def test_passive_question_removed_when_next_step_exists(monkeypatch, tmp_path):
    _set_store_path(monkeypatch, tmp_path)
    out = render_operator_response(
        user_input="continue",
        base_text="What would you like to start working on?",
        operator_state={},
        local_execution={"success": True, "inspected_file": "ai/runtime/agent_loop.py", "analysis": {"responsibility": "agent loop"}},
        next_step="inspect ai/runtime/response_momentum_policy.py",
        llm_generate=None,
    )
    low = out.lower()
    assert "what would you like to start working on" not in low
    assert "next best move:" in low
