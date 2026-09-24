"""A turn Alice could not answer is not recorded as a success.

Routing confidence learns from evaluations.jsonl, where "success" meant the
verifier accepted the reply. "I can't reach my language model" is accepted, so
an outage recorded a run of successful turns for whatever each was routed to.
"""

from ai.core.llm_engine import LLMConfig, LLMUnavailableError, LocalLLMEngine
from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


def _run(monkeypatch, post):
    import ai.learning.failure_eval_converter as converter

    written = []
    monkeypatch.setattr(converter, "write_turn_eval", lambda **row: written.append(row))
    engine = LocalLLMEngine(LLMConfig(model="test-model"))
    monkeypatch.setattr(engine, "_ensure_service_probed", lambda: None)
    monkeypatch.setattr(engine, "_available_models", [])
    monkeypatch.setattr(engine, "_post_with_retry", post)
    alice = _FakeAlice()
    alice.llm = engine
    result = ContractPipeline(build_runtime_boundaries(alice)).run_turn(
        user_input="should I use sqlite or postgres?", user_id="u1", turn_number=1
    )
    return result, written


def test_an_outage_is_recorded_as_a_failure(monkeypatch):
    def down(url, payload, what=""):
        raise LLMUnavailableError("connection refused")

    result, written = _run(monkeypatch, down)

    assert "language model" in result.response_text.lower() or "ollama" in result.response_text.lower()
    assert [row["success"] for row in written] == [False]


def test_an_answer_is_still_recorded_as_a_success(monkeypatch):
    result, written = _run(monkeypatch, lambda url, payload, what="": {"message": {"content": "SQLite, for one user."}})

    assert result.response_text == "SQLite, for one user."
    assert [row["success"] for row in written] == [True]
