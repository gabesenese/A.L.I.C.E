import os

from brain.model_router import ModelRouter
from brain.task_classifier import classify_task
from complexity import score_prompt


def test_task_classifier_detects_coding_and_planning():
    assert classify_task("please debug this python traceback").task_type == "coding"
    assert classify_task("create a roadmap and plan next steps").task_type == "planning"


def test_complexity_score_increases_for_planning_requests():
    low = score_prompt("what time is it")
    high = score_prompt("please design an architecture plan and compare tradeoffs step by step")
    assert high > low


def test_model_router_selects_coding_route():
    router = ModelRouter()
    assert router.route("refactor this code and write pytest tests") == "coding"


def test_model_router_generate_uses_standard_output(monkeypatch):
    monkeypatch.setenv("ALICE_MULTI_LLM_MOCK", "1")
    router = ModelRouter()
    out = router.generate("give me a short summary", context={"intent": "conversation:question"})
    assert set(["response", "confidence", "reasoning_used", "model"]).issubset(out.keys())
    assert isinstance(out["response"], str)
    assert 0.0 <= float(out["confidence"]) <= 1.0
