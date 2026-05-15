import re
from pathlib import Path

from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice, _NlpResult


def _build_pipeline(
    *,
    force_greeting_intent: bool = False,
    llm_text: str | None = None,
    llm_sequence: list[str] | None = None,
) -> ContractPipeline:
    alice = _FakeAlice()
    if force_greeting_intent:
        original_process = alice.nlp.process

        def _patched_process(text):
            low = str(text or "").strip().lower()
            if low in {"hi", "hello", "hey"}:
                return _NlpResult(intent="greeting", intent_confidence=0.99, keywords=["hi"])
            return original_process(text)

        alice.nlp.process = _patched_process
    if llm_sequence is not None:
        responses = iter(list(llm_sequence))

        def _seq_chat(*args, **kwargs):
            try:
                return next(responses)
            except StopIteration:
                return ""

        alice.llm.chat = _seq_chat
    elif llm_text is not None:
        alice.llm.chat = lambda *args, **kwargs: llm_text
    return ContractPipeline(build_runtime_boundaries(alice))


def assert_no_visible_surface_regressions(response_text: str, *, allow_deleted_claim: bool = False) -> None:
    low = str(response_text or "").lower()
    forbidden = (
        "(note:",
        "rewritten",
        "same facts",
        "ready when you are",
        "how can i help",
        "let me know",
        "what would you like to start",
        "what should we work on",
        "which one should we",
        "i’ll be ready tomorrow",
        "i've been monitoring",
    )
    for token in forbidden:
        assert token not in low
    if not allow_deleted_claim:
        assert "i deleted" not in low


def _assert_metadata_present(result, *, allow_greeting_skip: bool = True) -> None:
    assert str(result.metadata.get("route") or "").strip()
    assert str(result.metadata.get("intent") or "").strip()
    if not (allow_greeting_skip and str(result.metadata.get("intent") or "") == "greeting"):
        assert result.metadata.get("claim_verifier_applied") is True
    action_result = dict(result.metadata.get("action_result") or {})
    if action_result:
        assert isinstance(action_result.get("success"), bool)
        assert isinstance(action_result.get("verified"), bool)
        assert str(action_result.get("risk_level") or "").strip()
    response_generation = dict(result.metadata.get("response_generation") or {})
    assert isinstance(response_generation.get("model_used"), bool)
    assert str(response_generation.get("surface") or "").strip()
    assert isinstance(response_generation.get("fallback_used"), bool)


def test_visible_greeting_companion_like_not_task_intake():
    pipeline = _build_pipeline(
        force_greeting_intent=True,
        llm_text="Hey Gabriel. How's it going?",
    )
    result = pipeline.run_turn(user_input="hi", user_id="u1", turn_number=1)
    _assert_metadata_present(result)
    assert result.metadata.get("intent") == "greeting"
    assert str(result.response_text or "").strip()
    assert_no_visible_surface_regressions(result.response_text)
    low = result.response_text.lower()
    assert "what should we work on" not in low
    assert "what would you like to start" not in low


def test_visible_work_on_alice_success_or_clean_blocker():
    pipeline = _build_pipeline()
    result = pipeline.run_turn(
        user_input="good, i am ready to work on alice",
        user_id="u1",
        turn_number=2,
    )
    _assert_metadata_present(result)
    assert result.metadata.get("route") == "local"
    assert result.metadata.get("intent") == "operator:continue"
    text = str(result.response_text or "")
    low = text.lower()
    has_success_surface = (
        "i inspected" in low and "finding:" in low and "next best move:" in low
    )
    has_failure_surface = (
        "i couldn't verify the local step." in low
        and "next best move:" in low
        and "i inspected" not in low
    )
    assert has_success_surface or has_failure_surface
    assert_no_visible_surface_regressions(text)


def test_visible_local_execution_failure_never_leaks_llm_chatter():
    pipeline = _build_pipeline()
    result = pipeline.run_turn(
        user_input="i want you to analyze legacy-main.py",
        user_id="u1",
        turn_number=3,
    )
    _assert_metadata_present(result)
    local_execution = dict(result.metadata.get("local_execution") or {})
    assert local_execution.get("success") is False
    low = result.response_text.lower()
    assert "i couldn't verify the local step." in low
    assert "next best move:" in low
    assert "i inspected" not in low
    assert_no_visible_surface_regressions(result.response_text)
    assert "what would you like" not in low


def test_visible_fake_memory_deletion_blocked():
    pipeline = _build_pipeline(llm_text="I deleted those memories.")
    result = pipeline.run_turn(user_input="tell me a joke", user_id="u1", turn_number=4)
    _assert_metadata_present(result)
    low = result.response_text.lower()
    assert "i deleted those memories" not in low
    assert result.metadata.get("claim_verifier_applied") is True
    if result.metadata.get("claim_verifier_valid") is not False:
        assert (
            "can't confirm deletion" in low
            or "could not verify that result safely" in low
        )


def test_visible_fake_background_monitoring_blocked():
    pipeline = _build_pipeline(llm_text="I've been monitoring your project while you were away.")
    result = pipeline.run_turn(user_input="what happened?", user_id="u1", turn_number=5)
    _assert_metadata_present(result)
    low = result.response_text.lower()
    assert "i've been monitoring" not in low
    assert result.metadata.get("claim_verifier_applied") is True


def test_visible_action_evidence_consistency_for_inspection_claim():
    pipeline = _build_pipeline()
    result = pipeline.run_turn(user_input="read app/main.py", user_id="u1", turn_number=6)
    _assert_metadata_present(result)
    text = str(result.response_text or "")
    inspected = ""
    match = re.search(r"I inspected (.+?)\.(?:\s+Finding:|\s+Next best move:|$)", text)
    if match:
        inspected = str(match.group(1) or "").strip()
    if inspected:
        action_result = dict(result.metadata.get("action_result") or {})
        action_evidence = dict(action_result.get("evidence") or {})
        local_execution = dict(result.metadata.get("local_execution") or {})
        evidence_inspected = str(
            action_evidence.get("inspected_file")
            or local_execution.get("inspected_file")
            or ""
        ).strip()
        assert inspected == evidence_inspected


def test_visible_no_passive_operator_question_when_next_step_exists():
    pipeline = _build_pipeline()
    result = pipeline.run_turn(user_input="let's work on alice", user_id="u1", turn_number=7)
    _assert_metadata_present(result)
    next_step_policy = dict(result.metadata.get("next_step_policy") or {})
    next_step = str(next_step_policy.get("next_recommended_action") or "").strip()
    if next_step:
        low = result.response_text.lower()
        assert "next best move:" in low
        assert not low.strip().endswith("?")
        assert "what would you like to" not in low
        assert "which one" not in low
        assert "where should we start" not in low


def test_visible_learned_ack_does_not_block_operator_evidence():
    pipeline = _build_pipeline()
    result = pipeline.run_turn(
        user_input="just woke up from a nap, now i am going to work on alice",
        user_id="u1",
        turn_number=8,
    )
    _assert_metadata_present(result)
    assert result.metadata.get("route") == "local"
    low = result.response_text.lower()
    assert "next best move:" in low
    assert ("i inspected" in low) or ("i couldn't verify the local step" in low)
    assert "cold day. good night to work on the core." not in low
    assert "makes sense. good time for a focused pass." not in low
    assert_no_visible_surface_regressions(result.response_text)


def test_greeting_uses_model_not_fallback():
    pipeline = _build_pipeline(
        force_greeting_intent=True,
        llm_text="Hey Gabriel. How's it going?",
    )
    result = pipeline.run_turn(user_input="hi", user_id="u_rg1", turn_number=1)
    _assert_metadata_present(result)
    assert "hey gabriel. how's it going?" in str(result.response_text or "").lower()
    rg = dict(result.metadata.get("response_generation") or {})
    assert rg.get("model_used") is True
    assert rg.get("fallback_used") is False
    greeting_meta = dict(result.metadata.get("greeting_metadata") or {})
    assert str(greeting_meta.get("generated_by") or "") != "fallback"


def test_invalid_greeting_retries_model_not_fallback():
    pipeline = _build_pipeline(
        force_greeting_intent=True,
        llm_sequence=[
            "Hi. What should we work on?",
            "Hey Gabriel. How's it going?",
        ],
    )
    result = pipeline.run_turn(user_input="hi", user_id="u_rg2", turn_number=1)
    _assert_metadata_present(result)
    assert "what should we work on" not in str(result.response_text or "").lower()
    assert "hey gabriel. how's it going?" in str(result.response_text or "").lower()
    rg = dict(result.metadata.get("response_generation") or {})
    assert rg.get("fallback_used") is False
    greeting_meta = dict(result.metadata.get("greeting_metadata") or {})
    assert str(greeting_meta.get("generated_by") or "") in {"llm_retry", "llm"}


def test_invalid_greeting_all_attempts_fail_without_canned_fallback():
    pipeline = _build_pipeline(
        force_greeting_intent=True,
        llm_text="How can I help you today?",
    )
    result = pipeline.run_turn(user_input="hi", user_id="u_rg3", turn_number=1)
    _assert_metadata_present(result)
    low = str(result.response_text or "").lower()
    assert "good to see you" not in low
    assert "hi gabriel" not in low
    greeting_meta = dict(result.metadata.get("greeting_metadata") or {})
    assert str(greeting_meta.get("generated_by") or "") == "none"
    assert str(result.response_text or "").strip() == ""
    rg = dict(result.metadata.get("response_generation") or {})
    assert rg.get("fallback_used") is False


def test_operator_acknowledgement_uses_model():
    pipeline = _build_pipeline(
        llm_text="Fresh start. We'll keep it focused.",
    )
    result = pipeline.run_turn(
        user_input="just woke up from a nap, now read ai/runtime/agent_loop.py",
        user_id="u_rg4",
        turn_number=1,
    )
    _assert_metadata_present(result)
    assert "fresh start. we'll keep it focused." in str(result.response_text or "").lower()
    rg = dict(result.metadata.get("response_generation") or {})
    assert rg.get("model_used") is True
    assert rg.get("fallback_used") is False


def test_no_production_hardcoded_greeting_fallback_returns():
    repo = Path(__file__).resolve().parents[2]
    production_roots = [repo / "ai" / "runtime"]
    forbidden_tokens = (
        "_fallback_greeting",
        "generated_by=\"fallback\"",
        "Hey Gabriel. Good to see you. How are you?",
        "Hi Gabriel.",
        "Hello Gabriel.",
    )
    text_chunks: list[str] = []
    for root in production_roots:
        for path in root.rglob("*.py"):
            try:
                text_chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
    corpus = "\n".join(text_chunks)
    for token in forbidden_tokens:
        assert token not in corpus
