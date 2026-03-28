from app.main import ALICE


class _PhrasingNever:
    def can_phrase_myself(self, *_args, **_kwargs):
        return False


def test_generate_natural_response_uses_strict_fallback_without_llm():
    alice = ALICE.__new__(ALICE)
    alice.strict_no_llm = True
    alice.phrasing_learner = _PhrasingNever()
    alice._alice_direct_phrase = lambda *_args, **_kwargs: None

    out = alice._generate_natural_response(
        alice_response={"type": "operation_success", "operation": "run_tests"},
        tone="helpful",
        context=None,
        user_input="run tests",
    )

    assert out == "Completed: run tests."
