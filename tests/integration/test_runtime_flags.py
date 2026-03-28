from ai.infrastructure.runtime_flags import is_enabled


def test_quarantined_flags_default_disabled():
    assert is_enabled("session_summarizer") is False
    assert is_enabled("routing_decision_logger") is False
    assert is_enabled("contract_pipeline") is False


def test_non_quarantined_flag_default_enabled():
    assert is_enabled("some_new_component") is True
