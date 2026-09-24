"""A question about her limits is a conversation, not a routing report."""

import pytest

from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.contract_pipeline import ContractPipeline
from tests.integration.test_contract_pipeline import _FakeAlice


@pytest.mark.parametrize(
    "question",
    ["what do you struggle with?", "honestly, where are you failing lately?", "what are your weak spots"],
)
def test_limits_question_is_not_answered_with_the_eval_table(question):
    pipeline = ContractPipeline(build_runtime_boundaries(_FakeAlice()))

    result = pipeline.run_turn(user_input=question, user_id="u1", turn_number=3)

    assert result.metadata.get("source") != "weak_spot_report"
    assert "```" not in result.response_text
    assert "routing performance" not in result.response_text
