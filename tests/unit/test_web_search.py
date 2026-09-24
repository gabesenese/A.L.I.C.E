"""Live lookups, when he allows them.

There was no way to look anything up: the web plugin was an unreachable stub,
the model had no web tool, and "who won the game last night?" was answered from
memory that could be months old. North star rule 6 keeps the default path
local, so web search is opt-in with ALICE_ENABLE_WEB_SEARCH=1.
"""

from types import SimpleNamespace

import pytest

from ai.contracts import RouterRequest
from ai.core import tool_catalog, web_search
from ai.runtime.alice_contract_factory import build_runtime_boundaries
from app.main import ALICE
from tests.integration.test_contract_pipeline import _FakeAlice

PAGE = """
<div class="result results_links web-result"><div class="links_main result__body">
  <h2 class="result__title"><a rel="nofollow" class="result__a"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.nhl.com%2Fscores&amp;rut=abc">NHL Scores &amp; Results</a></h2>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.nhl.com">Leafs beat the <b>Habs</b> 4-2 last night.</a>
</div></div>
<div class="result results_links web-result"><div class="links_main result__body">
  <h2 class="result__title"><a rel="nofollow" class="result__a" href="https://example.org/recap">Game recap</a></h2>
</div></div>
"""


def test_results_are_read_from_the_page():
    assert web_search.parse_results(PAGE) == [
        {
            "title": "NHL Scores & Results",
            "url": "https://www.nhl.com/scores",
            "snippet": "Leafs beat the Habs 4-2 last night.",
        },
        {"title": "Game recap", "url": "https://example.org/recap", "snippet": ""},
    ]


def test_it_is_off_unless_he_turns_it_on(monkeypatch):
    monkeypatch.delenv("ALICE_ENABLE_WEB_SEARCH", raising=False)

    assert web_search.search_web("leafs score")["success"] is False
    assert "web_search" not in _offered()


def test_turned_on_the_model_can_search(monkeypatch):
    monkeypatch.setenv("ALICE_ENABLE_WEB_SEARCH", "1")
    import requests

    monkeypatch.setattr(requests, "post", lambda *a, **k: SimpleNamespace(text=PAGE, raise_for_status=lambda: None))

    out = web_search.search_web("leafs score last night")

    assert out["success"] is True
    assert "Leafs beat the Habs 4-2" in out["content"]
    assert "web_search" in _offered()


def test_offline_is_said_plainly(monkeypatch):
    monkeypatch.setenv("ALICE_ENABLE_WEB_SEARCH", "1")
    import requests

    def unreachable(*a, **k):
        raise requests.ConnectionError("no route to host")

    monkeypatch.setattr(requests, "post", unreachable)

    out = web_search.search_web("leafs score")

    assert out["success"] is False
    assert out["error"].startswith("Couldn't reach the web")


def _offered():
    return [s["function"]["name"] for s in tool_catalog.build_tool_schemas(max_risk=tool_catalog.RISK_READ)]


@pytest.mark.parametrize(
    "text",
    [
        "who won the game last night?",
        "what's the score?",
        "what's the price of bitcoin?",
        "what's the latest news today?",
    ],
)
def test_questions_about_what_is_true_now_need_live_sources(text):
    assert ALICE.__new__(ALICE)._is_freshness_sensitive_current_events_request(text)


@pytest.mark.parametrize(
    "text",
    ["I won the game last night!", "who won the 2018 world cup?", "what's the score threshold in the config?"],
)
def test_other_questions_do_not(text):
    assert not ALICE.__new__(ALICE)._is_freshness_sensitive_current_events_request(text)


@pytest.mark.parametrize("enabled, freshness_route", [("0", True), ("1", False)])
def test_with_search_on_current_events_go_to_the_model_and_its_tools(monkeypatch, enabled, freshness_route):
    monkeypatch.setenv("ALICE_ENABLE_WEB_SEARCH", enabled)
    decision = build_runtime_boundaries(_FakeAlice()).routing.route(
        RouterRequest(user_input="what's the latest news today?", turn_number=1)
    )
    assert (decision.intent == "freshness:current_events") is freshness_route
