"""How learned priors are allowed to move the router's confidence.

The decision bands downstream are 0.35 / 0.60 / 0.80, so what matters is not the
exact fused number but which side of a boundary it lands on.
"""

import json

import pytest

from ai.core.confidence_fusion import ConfidenceFusion


@pytest.fixture(autouse=True)
def _clear_rate_cache():
    ConfidenceFusion._rates_cache = {}
    ConfidenceFusion._rates_stamp = None
    yield
    ConfidenceFusion._rates_cache = {}
    ConfidenceFusion._rates_stamp = None


@pytest.fixture
def fusion(monkeypatch):
    """A fusion with both priors absent unless a test supplies them."""
    monkeypatch.setattr(ConfidenceFusion, "_behavioral_prior", staticmethod(lambda **_: None))
    monkeypatch.setattr(ConfidenceFusion, "_intent_success_rate_with_support", classmethod(lambda cls, **_: (None, 0)))
    return ConfidenceFusion()


def test_absent_priors_leave_router_confidence_untouched(fusion):
    """Knowing nothing is not evidence of anything, so it must not cost confidence."""
    assert fusion.fuse(router_confidence=0.92, intent="weather:current") == 0.92


def test_neutral_priors_leave_router_confidence_untouched(monkeypatch):
    monkeypatch.setattr(ConfidenceFusion, "_behavioral_prior", staticmethod(lambda **_: 0.5))
    monkeypatch.setattr(ConfidenceFusion, "_intent_success_rate_with_support", classmethod(lambda cls, **_: (0.5, 500)))
    assert ConfidenceFusion().fuse(router_confidence=0.92, intent="weather:current") == 0.92


def test_a_confident_route_keeps_its_band_when_nothing_is_known(fusion):
    """Regression: averaging uninformative priors in dropped 0.92 to 0.77,
    demoting every confidently routed turn from execute to verify."""
    assert fusion.fuse(router_confidence=0.92, intent="weather:current") >= 0.80


def test_strong_priors_push_up_but_only_across_one_boundary(monkeypatch):
    monkeypatch.setattr(ConfidenceFusion, "_behavioral_prior", staticmethod(lambda **_: 1.0))
    monkeypatch.setattr(
        ConfidenceFusion, "_intent_success_rate_with_support", classmethod(lambda cls, **_: (1.0, 10_000))
    )
    fused = ConfidenceFusion().fuse(router_confidence=0.65, intent="weather:current")
    assert fused > 0.65
    assert fused - 0.65 <= ConfidenceFusion._MAX_TOTAL_SHIFT + 1e-9


def test_weak_priors_push_down_but_stay_bounded(monkeypatch):
    monkeypatch.setattr(ConfidenceFusion, "_behavioral_prior", staticmethod(lambda **_: 0.0))
    monkeypatch.setattr(
        ConfidenceFusion, "_intent_success_rate_with_support", classmethod(lambda cls, **_: (0.0, 10_000))
    )
    fused = ConfidenceFusion().fuse(router_confidence=0.65, intent="weather:current")
    assert fused < 0.65
    assert 0.65 - fused <= ConfidenceFusion._MAX_TOTAL_SHIFT + 1e-9


def test_a_thin_sample_moves_the_score_less_than_a_thick_one():
    thin = ConfidenceFusion._shift_from(1.0, 0.10, samples=3)
    thick = ConfidenceFusion._shift_from(1.0, 0.10, samples=500)
    assert 0 < thin < thick


def test_total_shift_never_spans_two_decision_bands():
    """The narrowest band is 0.20 wide (0.60-0.80). A prior that could move a
    score further than that would let history overrule routing outright."""
    assert ConfidenceFusion._MAX_TOTAL_SHIFT < 0.20


def _write_evals(tmp_path, monkeypatch, records):
    eval_dir = tmp_path / "data" / "evaluations"
    eval_dir.mkdir(parents=True)
    (eval_dir / "evaluations.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)


def test_failure_only_backfill_is_not_counted_as_a_success_rate(tmp_path, monkeypatch):
    """routing_failures.jsonl is backfilled into the evaluation log and nothing
    backfills the successes. Counting those rows reported near-zero success for
    every intent forever, which silently demoted every turn."""
    _write_evals(
        tmp_path,
        monkeypatch,
        [
            {"action_type": "weather:current", "overall_score": 30, "source": "routing_failure_backfill"}
            for _ in range(50)
        ],
    )
    assert ConfidenceFusion._intent_success_rate(intent="weather:current") is None


def test_live_turns_do_produce_a_success_rate(tmp_path, monkeypatch):
    _write_evals(
        tmp_path,
        monkeypatch,
        [{"action_type": "weather:current", "overall_score": 85, "source": "live_turn"} for _ in range(4)]
        + [{"action_type": "weather:current", "overall_score": 20, "source": "live_turn"}],
    )
    rate, samples = ConfidenceFusion._intent_success_rate_with_support(intent="weather:current")
    assert samples == 5
    assert rate == pytest.approx(0.8)


def test_rates_are_cached_until_the_log_changes(tmp_path, monkeypatch):
    _write_evals(
        tmp_path,
        monkeypatch,
        [{"action_type": "notes:create", "overall_score": 85, "source": "live_turn"} for _ in range(3)],
    )
    first = ConfidenceFusion._load_success_rates()
    assert ConfidenceFusion._load_success_rates() is first
