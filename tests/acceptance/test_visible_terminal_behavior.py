import re
from pathlib import Path

from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice, _NlpResult


def _build_pipeline(
    *,
    force_greeting_intent: bool = False,
    forced_intent: str | None = None,
    seed_operator_state: dict | None = None,
    llm_text: str | None = None,
    llm_sequence: list[str] | None = None,
) -> ContractPipeline:
    alice = _FakeAlice()
    if seed_operator_state:
        alice._operator_state = dict(seed_operator_state or {})
    if force_greeting_intent:
        original_process = alice.nlp.process

        def _patched_process(text):
            low = str(text or "").strip().lower()
            if low in {"hi", "hello", "hey"}:
                return _NlpResult(intent="greeting", intent_confidence=0.99, keywords=["hi"])
            if forced_intent:
                return _NlpResult(
                    intent=str(forced_intent),
                    intent_confidence=0.99,
                    keywords=["forced"],
                )
            return original_process(text)

        alice.nlp.process = _patched_process
    elif forced_intent:
        original_process = alice.nlp.process

        def _patched_process_forced(text):
            low = str(text or "").strip().lower()
            if force_greeting_intent and low in {"hi", "hello", "hey"}:
                return _NlpResult(intent="greeting", intent_confidence=0.99, keywords=["hi"])
            if forced_intent:
                return _NlpResult(
                    intent=str(forced_intent),
                    intent_confidence=0.99,
                    keywords=["forced"],
                )
            return original_process(text)

        alice.nlp.process = _patched_process_forced
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
    assert str(response_generation.get("model_name") or "").strip()
    assert str(response_generation.get("surface") or "").strip()
    assert isinstance(response_generation.get("validation_applied"), bool)
    assert isinstance(response_generation.get("claim_verifier_applied"), bool)
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


def test_visible_greeting_task_intake_is_blocked_without_fallback():
    pipeline = _build_pipeline(
        force_greeting_intent=True,
        llm_text="Hi. What should we work on?",
    )
    result = pipeline.run_turn(user_input="hi alice", user_id="u1", turn_number=1)
    _assert_metadata_present(result)
    low = str(result.response_text or "").lower()
    assert "what should we work on" not in low
    assert "how can i help" not in low
    assert "ready when you are" not in low
    greeting_meta = dict(result.metadata.get("greeting_metadata") or {})
    assert str(greeting_meta.get("generated_by") or "") != "fallback"


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
    text = str(result.response_text or "")
    low = text.lower()
    assert "i couldn't verify the local step." in low
    assert "blocker:" in low
    assert "next best move:" in low
    assert "i inspected" not in low
    assert "target_not_found" not in low
    assert "contract_local_execution_error" not in low
    assert "unknown_action" not in low
    assert "\n\nblocker:" in low
    assert "\n\nnext best move:" in low
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
            or "i can delete those memories" in low
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


def test_visible_educational_learning_stays_clean():
    pipeline = _build_pipeline(
        forced_intent="conversation:educational_explain",
        seed_operator_state={
            "active_objective": "Improve agentic companion operator runtime",
            "current_focus": "runtime",
            "next_recommended_action": "inspect file ai/runtime/next_step_policy.py because It decides what Alice should do after each safe step.",
        },
        llm_text=(
            "An AI companion is different from a normal chatbot because it can keep context, understand preferences, use tools, and adapt over time. "
            "If you want, I can keep tracking this thread and follow up next turn."
        ),
    )
    result = pipeline.run_turn(
        user_input="im trying to learn more about ai companion",
        user_id="u_rg5",
        turn_number=1,
    )
    _assert_metadata_present(result)
    assert result.metadata.get("intent") == "conversation:educational_explain"
    low = str(result.response_text or "").lower()
    assert "current objective" not in low
    assert "next best move" not in low
    assert "inspect file" not in low
    assert "ai/runtime" not in low
    assert "operator runtime" not in low
    assert "if you want" not in low
    assert "keep tracking this thread" not in low
    assert "follow up next turn" not in low
    assert "ai companion" in low


def test_visible_clarification_response_not_polluted_by_operator_scaffolding():
    pipeline = _build_pipeline(
        forced_intent="conversation:clarification_needed",
        seed_operator_state={
            "active_objective": "Improve agentic companion operator runtime",
            "current_focus": "runtime",
            "next_recommended_action": "inspect file ai/runtime/next_step_policy.py because It decides what Alice should do after each safe step.",
        },
        llm_text=(
            "I misunderstood that response path. Please repeat your request in one line and I will answer directly. "
            "If you want, I can keep tracking this thread and follow up next turn."
        ),
    )
    result = pipeline.run_turn(
        user_input="been great, it is friday after all and i am ready to work o alice",
        user_id="u_rg6",
        turn_number=1,
    )
    _assert_metadata_present(result)
    low = str(result.response_text or "").lower()
    assert "current objective" not in low
    assert "next best move" not in low
    assert "ai/runtime" not in low
    assert "if you want" not in low
    assert "keep tracking this thread" not in low
    assert "follow up next turn" not in low


def test_visible_codebase_access_claim_requires_local_evidence_or_honest_unverified_message():
    pipeline = _build_pipeline(
        llm_text=(
            "With that in mind, let me see if I can dig up some relevant info from our codebase. "
            "Scanning through the repo, we have a few plugins and features related to proactive suggestions."
        ),
    )
    result = pipeline.run_turn(
        user_input="you have access to alice's code base, my ai project",
        user_id="u_cb1",
        turn_number=1,
    )
    _assert_metadata_present(result)
    low = str(result.response_text or "").lower()
    local = dict(result.metadata.get("local_execution") or {})
    action_result = dict(result.metadata.get("action_result") or {})
    has_local_or_action_evidence = bool(
        local.get("success")
        or str(local.get("inspected_file") or "").strip()
        or (action_result.get("success") and action_result.get("verified"))
    )
    assert has_local_or_action_evidence or (
        "i have not verified the codebase yet" in low
    )
    assert "scanning through the repo" not in low
    assert "i looked through the codebase" not in low
    assert "we have a few plugins" not in low


def test_visible_project_improvement_request_does_not_hallucinate_file_paths():
    pipeline = _build_pipeline(
        llm_text="Take a look at self_learning/contextual_awareness.py to improve the project.",
    )
    result = pipeline.run_turn(
        user_input="give me an area i can improve",
        user_id="u_cb2",
        turn_number=1,
    )
    _assert_metadata_present(result)
    low = str(result.response_text or "").lower()
    local = dict(result.metadata.get("local_execution") or {})
    action_result = dict(result.metadata.get("action_result") or {})
    has_verified_evidence = bool(
        local.get("success")
        or str(local.get("inspected_file") or "").strip()
        or (action_result.get("success") and action_result.get("verified"))
    )
    assert "self_learning/contextual_awareness.py" not in low
    assert "i found in the codebase" not in low
    assert has_verified_evidence or ("i have not verified the codebase yet" in low)


def test_visible_proactive_companion_concept_thread_flow_stays_conceptual():
    pipeline = _build_pipeline(
        llm_sequence=[
            "An AI companion is different from a chatbot because it can keep context and adapt over time.",
            "Right, then the key difference is agency: a chatbot waits, while a companion observes state and brings useful signals.",
            "Then the core is a background loop: observe, detect change, judge importance, suggest action, and ask approval before risky actions.",
            "Exactly, not movie magic, but architecture: persistent state, task monitors, memory, tools, and a notification layer.",
            "Actual proactivity means triggers and relevance scoring before suggestions. (Note: I've kept the main points intact while making minor adjustments for tone and flow)",
        ]
    )
    turns = [
        "i want to learn more about ai companion",
        "i dont want it to be like an assistant or chatbot",
        "i want alice to be proactive, like this",
        "something like jarvis",
        "i want to be actually proactive",
    ]
    outputs: list[str] = []
    for idx, user_text in enumerate(turns, start=1):
        result = pipeline.run_turn(user_input=user_text, user_id="u_flow1", turn_number=idx)
        _assert_metadata_present(result)
        outputs.append(str(result.response_text or ""))

    merged = " ".join(outputs).lower()
    assert "(note:" not in merged
    assert "rewritten" not in merged
    assert "i have not verified the codebase yet" not in merged
    assert "next best move:" not in merged
    assert "ai/runtime" not in merged
    assert "inspect file" not in merged
    assert "background loop" in merged
    assert "detect change" in merged or "detects change" in merged
    assert "suggest action" in merged or "suggestions" in merged
    assert "approval" in merged or "risky actions" in merged


def test_visible_concept_refinement_does_not_trigger_codebase_grounding():
    pipeline = _build_pipeline(
        llm_text=(
            "Actual proactivity means Alice needs triggers: observe events, detect change, "
            "judge relevance, and surface a useful suggestion."
        )
    )
    result = pipeline.run_turn(
        user_input="i want to be actually proactive",
        user_id="u_flow2",
        turn_number=1,
    )
    _assert_metadata_present(result)
    low = str(result.response_text or "").lower()
    assert "i have not verified the codebase yet" not in low
    assert "next best move:" not in low
    assert "detect change" in low or "triggers" in low


def test_visible_implementation_request_can_bridge_to_local_or_honest_unverified():
    pipeline = _build_pipeline()
    result = pipeline.run_turn(
        user_input="how do we implement this in Alice?",
        user_id="u_flow3",
        turn_number=1,
    )
    _assert_metadata_present(result)
    low = str(result.response_text or "").lower()
    assert result.metadata.get("route") in {"local", "llm"}
    assert (
        result.metadata.get("route") == "local"
        or "i have not verified the codebase yet" in low
    )


def test_visible_meta_artifact_stripped_from_final_output():
    pipeline = _build_pipeline(
        forced_intent="conversation:educational_explain",
        llm_text="An AI companion is useful. (Note: I've kept the main points intact while making minor adjustments for tone and flow)",
    )
    result = pipeline.run_turn(
        user_input="teach me about ai companions",
        user_id="u_flow4",
        turn_number=1,
    )
    _assert_metadata_present(result)
    low = str(result.response_text or "").lower()
    assert "(note:" not in low
    assert "minor adjustments for tone and flow" not in low
    assert "an ai companion is useful." in low


def test_visible_greeting_rejects_soft_continuity_again():
    pipeline = _build_pipeline(
        force_greeting_intent=True,
        llm_sequence=[
            "Hey Gabriel! It's great to see you again. How's your day been so far?",
            "Hey Gabriel. How's your day been so far?",
        ],
    )
    result = pipeline.run_turn(user_input="hi alice", user_id="u_flow5", turn_number=1)
    _assert_metadata_present(result)
    low = str(result.response_text or "").lower()
    assert "see you again" not in low
    greeting_meta = dict(result.metadata.get("greeting_metadata") or {})
    assert str(greeting_meta.get("generated_by") or "") in {"llm_retry", "llm"}
