"""Whether a finished reply gets re-generated before the user sees it."""

from types import SimpleNamespace

import pytest

from ai.infrastructure.runtime_flags import publish_polish_enabled
from ai.runtime import response_authority


@pytest.fixture(autouse=True)
def _clear_flag(monkeypatch):
    monkeypatch.delenv("ALICE_ENABLE_PUBLISH_POLISH", raising=False)


@pytest.fixture
def alice():
    """An Alice whose polish pass records that it ran and mangles the text."""
    calls = []

    def polish(*, response, user_input, intent):
        calls.append(response)
        return "POLISHED"

    built = SimpleNamespace(_publish_with_fast_llm_style=polish)
    built.polish_calls = calls
    return built


def _publish(alice, text="Fixed the schema drift; the query is fine now."):
    return response_authority.finalize_conversational_surface(
        alice=alice,
        user_input="what did you do?",
        intent="conversation:general",
        response=text,
        route="llm",
        plugin_result=None,
    )


def test_the_polish_pass_is_off_by_default():
    assert publish_polish_enabled() is False


def test_by_default_the_reply_reaches_the_user_as_written(alice):
    """It ran on nearly every published response: a second generation of text
    that already carried the persona, at temperature 0.35, instructed to be
    "concise". An extra round trip per turn, collapsing varied phrasings onto
    one, with a compression instruction applied unconditionally."""
    original = "Fixed the schema drift; the query is fine now."
    assert _publish(alice, original) == original
    assert alice.polish_calls == []


def test_with_the_flag_on_the_pass_runs_again(alice, monkeypatch):
    """Kept rather than deleted so the two can be compared with
    scripts/quality_harness.py --feel rather than argued about."""
    monkeypatch.setenv("ALICE_ENABLE_PUBLISH_POLISH", "1")
    assert _publish(alice) == "POLISHED"
    assert len(alice.polish_calls) == 1


@pytest.mark.parametrize("route", ["llm_fallback", "contract_tool_response"])
def test_routes_that_already_opted_out_stay_out(alice, monkeypatch, route):
    monkeypatch.setenv("ALICE_ENABLE_PUBLISH_POLISH", "1")
    text = "Nothing to polish here."
    result = response_authority.finalize_conversational_surface(
        alice=alice,
        user_input="x",
        intent="conversation:general",
        response=text,
        route=route,
        plugin_result=None,
    )
    assert result == text
    assert alice.polish_calls == []


def test_an_explicit_opt_out_still_wins(alice, monkeypatch):
    monkeypatch.setenv("ALICE_ENABLE_PUBLISH_POLISH", "1")
    text = "Left alone."
    result = response_authority.finalize_conversational_surface(
        alice=alice,
        user_input="x",
        intent="conversation:general",
        response=text,
        route="llm",
        plugin_result=None,
        apply_publish_style=False,
    )
    assert result == text
    assert alice.polish_calls == []
