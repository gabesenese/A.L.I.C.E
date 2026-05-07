from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from ai.runtime.continuity_claim_guard import assess_continuity_claims
from ai.runtime.greeting_surface_policy import render_grounded_greeting


class _NlpResult:
    def __init__(self, intent: str, confidence: float):
        self.intent = intent
        self.intent_confidence = confidence
        self.keywords = []


class _GreetingNlp:
    def process(self, text: str):
        low = str(text or "").lower().strip()
        if low in {"hi", "hi alice", "hello", "hey"}:
            return _NlpResult("conversation:greeting", 0.92)
        return _NlpResult("conversation:general", 0.84)


class _Memory:
    def __init__(self):
        self._stored = []

    def search(self, query, top_k=8):
        return [
            {
                "content": "We discussed machine learning techniques and training loops.",
                "score": 0.91,
                "context": {
                    "source": "vector_recall",
                    "timestamp": "2025-01-01T00:00:00+00:00",
                },
            }
        ][:top_k]

    def store_memory(
        self, content, memory_type="episodic", context=None, importance=0.5, tags=None
    ):
        self._stored.append({"content": content, "context": context or {}})
        return "ok"


class _Llm:
    def chat(self, user_input, use_history=True):
        return "we were discussing machine learning last time"


class _Alice:
    def __init__(self):
        self.nlp = _GreetingNlp()
        self.memory = _Memory()
        self.plugins = None
        self.llm = _Llm()
        self.self_reflection = None
        self._operator_state = {}

    def _is_location_query(self, _text):
        return False


def test_a_greeting_ignores_stale_vector_memory_topic():
    alice = _Alice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        "hi alice", "u1", 1
    )

    low = result.response_text.lower()
    assert ("good to see you" in low) or ("what's on your mind" in low) or ("what are we thinking about" in low)
    assert "machine learning" not in low
    assert "last time we talked about" not in low
    assert "conversation history suggests" not in low
    assert "no active task is loaded" not in low
    assert "operator state" not in low
    assert "memory policy" not in low
    assert "broad memory" not in low
    assert "alice's development" not in low
    assert "start fresh" not in low
    sentence_count = sum(low.count(ch) for ch in ".!?")
    assert 1 <= sentence_count <= 2
    assert result.metadata["route"] == "llm"
    assert result.metadata["verification"]["accepted"] is True
    assert (
        result.metadata["greeting_metadata"]["greeting_memory_policy"]
        == "active_state_only"
    )
    assert result.metadata["greeting_metadata"]["broad_memory_suppressed"] is True


def test_b_greeting_may_use_active_operator_state_focus():
    alice = _Alice()
    alice._operator_state = {
        "active_objective": "Improve Alice into an agentic companion/operator",
        "current_focus": "routing",
    }
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        "hi alice", "u1", 1
    )
    assert "routing" not in result.response_text.lower()
    assert "current operator state" not in result.response_text.lower()
    assert "memory policy" not in result.response_text.lower()
    assert "broad memory" not in result.response_text.lower()
    assert "machine learning" not in result.response_text.lower()
    if "routing" in result.response_text.lower():
        assert result.metadata["greeting_metadata"]["active_objective_used"] is True


def test_b_no_fake_continuity_from_stale_memory():
    alice = _Alice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        "hi alice", "u1", 1
    )
    low = result.response_text.lower()
    assert "machine learning" not in low
    assert "last time" not in low
    assert "we were discussing" not in low
    assert "conversation history suggests" not in low


def test_c_active_state_plain_greeting_does_not_force_focus():
    alice = _Alice()
    alice._operator_state = {
        "active_objective": "Improve Alice into an agentic companion/operator",
        "current_focus": "routing",
    }
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        "hi alice", "u1", 1
    )
    low = result.response_text.lower()
    assert "routing" not in low
    assert "machine learning" not in low
    assert "operator state" not in low


def test_d_active_state_plus_continuation_mentions_focus():
    alice = _Alice()
    alice._operator_state = {
        "active_objective": "Improve Alice into an agentic companion/operator",
        "current_focus": "routing",
    }
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        "hi alice, where were we?", "u1", 1
    )
    low = result.response_text.lower()
    assert "routing" in low
    assert "current operator state" not in low
    assert "machine learning" not in low


def test_f_repeated_greeting_is_shorter_and_no_project_menu_repeat():
    alice = _Alice()
    pipeline = ContractPipeline(build_runtime_boundaries(alice))
    first = pipeline.run_turn("hi alice", "u1", 1)
    second = pipeline.run_turn("hi", "u1", 2)
    first_low = first.response_text.lower()
    second_low = second.response_text.lower()
    assert "alice's development" not in first_low
    assert "start fresh" not in first_low
    assert "alice's development" not in second_low
    assert "start fresh" not in second_low
    assert len(second.response_text.split()) <= len(first.response_text.split())
    assert second.response_text.strip() != first.response_text.strip()


def test_f_constrained_llm_unsafe_output_falls_back_safely():
    result = render_grounded_greeting(
        user_name="Gabriel",
        operator_state={},
        session_state={},
        user_input="hi alice",
        llm_generate=lambda *args, **kwargs: "We were discussing machine learning last time.",
    )
    low = result.text.lower()
    assert "machine learning" not in low
    assert "we were discussing" not in low
    assert result.generated_by == "fallback"


def test_g_greeting_metadata_fields_are_present():
    alice = _Alice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        "hi alice", "u1", 1
    )
    meta = dict(result.metadata.get("greeting_metadata") or {})
    assert meta.get("greeting_memory_policy") == "active_state_only"
    assert meta.get("broad_memory_suppressed") is True
    assert "greeting_style" in meta
    assert "suppressed_project_menu" in meta
    assert "generated_by" in meta


def test_h_unsupported_continuity_claim_guard_still_blocks_unsupported_text():
    guard = assess_continuity_claims(
        text="we were discussing machine learning last time",
        memory_items=[],
        operator_state={},
    )
    assert guard.unsupported_continuity_claim is True
    assert guard.recovery_applied is True
