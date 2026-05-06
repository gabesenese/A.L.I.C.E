from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from ai.runtime.continuity_claim_guard import assess_continuity_claims


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
                "context": {"source": "vector_recall", "timestamp": "2025-01-01T00:00:00+00:00"},
            }
        ][:top_k]

    def store_memory(self, content, memory_type="episodic", context=None, importance=0.5, tags=None):
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
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn("hi alice", "u1", 1)

    low = result.response_text.lower()
    assert "machine learning" not in low
    assert "last time we talked about" not in low
    assert "conversation history suggests" not in low
    assert "no active task is loaded" not in low
    assert "operator state" not in low
    assert "memory policy" not in low
    assert "broad memory" not in low
    assert result.metadata["route"] == "llm"
    assert result.metadata["verification"]["accepted"] is True
    assert result.metadata["greeting_metadata"]["greeting_memory_policy"] == "active_state_only"
    assert result.metadata["greeting_metadata"]["broad_memory_suppressed"] is True


def test_b_greeting_may_use_active_operator_state_focus():
    alice = _Alice()
    alice._operator_state = {
        "active_objective": "Improve Alice into an agentic companion/operator",
        "current_focus": "routing",
    }
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn("hi alice", "u1", 1)
    assert "routing" in result.response_text.lower()
    assert "current operator state" not in result.response_text.lower()
    assert "memory policy" not in result.response_text.lower()
    assert "broad memory" not in result.response_text.lower()
    assert "machine learning" not in result.response_text.lower()
    assert result.metadata["greeting_metadata"]["active_objective_used"] is True


def test_c_unsupported_continuity_claim_is_flagged_and_recovered():
    alice = _Alice()
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn("status check", "u1", 1)
    assert result.metadata["verification"]["accepted"] is True
    continuity = dict(result.metadata.get("continuity_claims") or {})
    assert continuity.get("unsupported_continuity_claim") is True
    claims = continuity.get("unsupported_claims") or []
    assert any("machine learning" in str(item).lower() for item in claims)
    assert "we were discussing machine learning last time" not in result.response_text.lower()

    guard = assess_continuity_claims(
        text="we were discussing machine learning last time",
        memory_items=[],
        operator_state={},
    )
    assert guard.unsupported_continuity_claim is True
    assert guard.recovery_applied is True


def test_d_same_session_project_state_can_be_referenced():
    alice = _Alice()
    pipeline = ContractPipeline(build_runtime_boundaries(alice))
    pipeline.run_turn("let's work on Alice's memory", "u1", 1)
    result = pipeline.run_turn("hi alice", "u1", 2)
    low = result.response_text.lower()
    assert "machine learning" not in low
    assert "last time we talked about" not in low
    assert "conversation history suggests" not in low
    assert "no active task is loaded" not in low
    assert "operator state" not in low
    assert "memory policy" not in low
    assert "broad memory" not in low
    assert ("alice" in low) or ("memory" in low) or ("continue" in low)
