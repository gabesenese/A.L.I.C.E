"""Integration tests for learning-data redaction at persistence boundaries."""

import json
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.learning.learning_engine import LearningEngine
from ai.learning.phrasing_learner import PhrasingLearner
from ai.core.llm_gateway import LLMGateway
from ai.core.llm_policy import LLMCallType


class _DummyLLM:
    def chat(self, prompt, use_history=False):
        return "ok"


class _AlwaysAllowPolicy:
    def can_call_llm(self, call_type, user_input):
        return True, ""

    def record_call(self, call_type, user_input, response):
        return None


def test_learning_engine_redacts_sensitive_fields_before_persist(tmp_path):
    engine = LearningEngine(data_dir=str(tmp_path / "training"))

    engine.collect_interaction(
        user_input="email me at gabe@example.com or call +1 (519) 555-1212",
        assistant_response="Use token sk-ABCDEF1234567890ZZ for auth",
        intent="conversation:general",
        entities={"api_key": "sk-ABCDEF1234567890ZZ", "email": "gabe@example.com"},
        quality_score=0.8,
    )

    lines = (tmp_path / "training" / "training_data.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])

    assert "gabe@example.com" not in payload["user_input"]
    assert "[REDACTED_EMAIL]" in payload["user_input"]
    assert "519" not in payload["user_input"]
    assert "[REDACTED_PHONE]" in payload["user_input"]
    assert "sk-ABCDEF" not in payload["assistant_response"]
    assert "[REDACTED_API_KEY]" in payload["assistant_response"]
    assert payload["entities"]["api_key"] == "[REDACTED_SECRET]"


def test_phrasing_learner_redacts_context_and_ollama_text(tmp_path):
    learner = PhrasingLearner(storage_path=str(tmp_path / "learned_phrasings.jsonl"))

    learner.record_phrasing(
        alice_thought={"type": "operation_success", "data": {"user_input": "contact me at gabe@example.com"}},
        ollama_phrasing="Sure, use Bearer abcdefghijklmnopqrstuvwxyz123456 for this.",
        context={"tone": "helpful", "user_input": "my phone is 519-555-1212", "access_token": "abc123"},
    )

    line = (tmp_path / "learned_phrasings.jsonl").read_text(encoding="utf-8").strip()
    payload = json.loads(line)

    assert "gabe@example.com" not in json.dumps(payload)
    assert "519-555-1212" not in json.dumps(payload)
    assert payload["context"]["access_token"] == "[REDACTED_SECRET]"
    assert "[REDACTED_BEARER_TOKEN]" in payload["ollama_phrasing"]


def test_phrasing_learner_normalizes_filler_and_overexplaining(tmp_path):
    learner = PhrasingLearner(storage_path=str(tmp_path / "learned_phrasings.jsonl"))

    learner.record_phrasing(
        alice_thought={"type": "operation_success", "operation": "run_tests"},
        ollama_phrasing=(
            "Of course, I would be happy to help. "
            "Sure, this is absolutely what I can do for you and maybe perhaps just maybe "
            "I can provide additional unnecessary context that keeps going and going."
        ),
        context={"tone": "helpful"},
    )

    line = (tmp_path / "learned_phrasings.jsonl").read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    phrasing = payload["ollama_phrasing"].lower()

    assert "of course" not in phrasing
    assert "happy to help" not in phrasing
    assert "maybe" not in phrasing
    assert len(payload["ollama_phrasing"]) <= 220


def test_llm_gateway_fallback_log_is_redacted(tmp_path):
    gateway = LLMGateway(llm_engine=_DummyLLM(), learning_engine=None)
    gateway.policy = _AlwaysAllowPolicy()

    # Redirect fallback log file for test isolation.
    import ai.core.llm_gateway as gateway_module
    gateway_module.LOGGED_INTERACTIONS_PATH = str(tmp_path / "logged_interactions.jsonl")

    gateway._log_llm_fallback(
        user_input="my email is gabe@example.com",
        intent="conversation:general",
        entities={"phone": "519-555-1212", "token": "secret-value"},
        context_snapshot={"authorization": "Bearer abcdefghijklmnop123456", "note": "contact gabe@example.com"},
        llm_response="Use sk-ABCDEF1234567890ZZ now",
    )

    line = (tmp_path / "logged_interactions.jsonl").read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    blob = json.dumps(payload)

    assert "gabe@example.com" not in blob
    assert "519-555-1212" not in blob
    assert "secret-value" not in blob
    assert "sk-ABCDEF" not in blob
    assert "[REDACTED_EMAIL]" in blob
    assert "[REDACTED_PHONE]" in blob
    assert "[REDACTED_SECRET]" in blob
    assert "[REDACTED_API_KEY]" in blob
