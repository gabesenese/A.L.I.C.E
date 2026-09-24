"""A request that chains actions goes to the tool loop, which can take several steps.

The keyword router sent "list my notes, then read the first one" to one plugin,
which listed the notes and dropped the rest. nlp_processor detected the second
step and recorded it in its parse, and nothing read it.
"""

import pytest

from ai.contracts import RouterRequest
from ai.runtime.alice_contract_factory import build_runtime_boundaries
from ai.runtime.boundaries.boundary_factory import _is_multi_step_request
from tests.integration.test_contract_pipeline import _FakeAlice


@pytest.mark.parametrize(
    "request_text",
    [
        "list my notes, then read the first one",
        "check the weather and then add a note about it",
        "find the parser file and open it",
        "please search the workspace for TODOs, after that show me the first file",
        "first list my notes, then create one called groceries",
    ],
)
def test_chained_actions_are_multi_step(request_text):
    assert _is_multi_step_request(request_text)


@pytest.mark.parametrize(
    "text",
    [
        "list my notes",
        "I tried to run it and then it crashed",
        "add salt and pepper to the shopping note",
        "what do you think about rust and go?",
        "go ahead and run it",
    ],
)
def test_everything_else_is_not(text):
    assert not _is_multi_step_request(text)


def test_a_chained_request_is_routed_to_the_model_with_its_tools():
    boundaries = build_runtime_boundaries(_FakeAlice())
    decision = boundaries.routing.route(RouterRequest(user_input="list my notes, then read the first one", turn_number=1))
    assert decision.route == "llm"
    assert decision.metadata["reason"] == "multi_step_request"
