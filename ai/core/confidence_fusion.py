"""Confidence fusion: combine multiple routing signals into a single calibrated score.

Sources:
  1. Router confidence (always present)
  2. Behavioral priors from UserProfileEngine (intent frequency history)
  3. Per-intent success history (from RoutingFailureLogger or evaluation log)
"""

from __future__ import annotations

from typing import Dict, Optional


class ConfidenceFusion:
    """Adjusts the router's confidence using behavioral and success-history priors.

    Priors are applied as *signed nudges around neutral*, not averaged in as
    absolute values. Averaging punished ignorance: a router score of 0.92 blended
    with two uninformative 0.5 priors came out at 0.77, so a confidently routed
    turn dropped a decision band purely because nothing was known about it. A
    prior at 0.5 now means "no information" and moves the score by exactly zero;
    only evidence that points somewhere moves it.
    """

    # Maximum magnitude each prior may shift the router's score, in confidence
    # points, before evidence weighting.
    _MAX_SHIFT = {
        "behavioral_prior": 0.12,
        "intent_success_rate": 0.10,
    }

    # Ceiling on the combined shift. The decision bands are 0.35 / 0.60 / 0.80,
    # so the narrowest is 0.20 wide; capping below that keeps priors able to
    # carry a borderline turn across one boundary but never across two. Without
    # this, a single successful turn lifted a 0.65 router score to 0.98.
    _MAX_TOTAL_SHIFT = 0.18

    # Observations needed before a prior carries its full weight. Below it the
    # shift is scaled by n/(n+k), so three-for-three counts as suggestive rather
    # than as a settled 100% success rate.
    _EVIDENCE_HALF_WEIGHT = 12

    # A prior is only informative once it is this far from neutral; below that
    # it is rounding noise and is ignored outright.
    _NEUTRAL = 0.5
    _DEADBAND = 0.02

    def fuse(
        self,
        *,
        router_confidence: float,
        intent: str,
        user_id: str = "default",
    ) -> float:
        """Return a calibrated confidence score in [0, 1].

        With no prior data available this returns the router's own confidence
        unchanged, which is the honest answer when nothing else is known.
        """
        router_c = max(0.0, min(1.0, float(router_confidence or 0.0)))

        shift = 0.0
        shift += self._shift_from(
            self._behavioral_prior(intent=intent, user_id=user_id),
            self._MAX_SHIFT["behavioral_prior"],
        )
        rate, samples = self._intent_success_rate_with_support(intent=intent)
        shift += self._shift_from(
            rate,
            self._MAX_SHIFT["intent_success_rate"],
            samples=samples,
        )

        shift = max(-self._MAX_TOTAL_SHIFT, min(self._MAX_TOTAL_SHIFT, shift))
        fused = router_c + shift
        return round(max(0.0, min(1.0, fused)), 4)

    @classmethod
    def _shift_from(cls, prior: Optional[float], max_shift: float, samples: Optional[int] = None) -> float:
        """Map a prior in [0, 1] to a signed shift in [-max_shift, +max_shift].

        When ``samples`` is given, the shift is scaled by how much evidence backs
        the prior, so a rate drawn from a handful of turns moves the score less
        than the same rate drawn from a hundred.
        """
        if prior is None:
            return 0.0
        offset = max(0.0, min(1.0, float(prior))) - cls._NEUTRAL
        if abs(offset) < cls._DEADBAND:
            return 0.0
        # offset spans [-0.5, +0.5]; scale it to the full shift range.
        shift = (offset / cls._NEUTRAL) * max_shift
        if samples is not None:
            n = max(0, int(samples))
            shift *= n / (n + cls._EVIDENCE_HALF_WEIGHT)
        return shift

    @staticmethod
    def _behavioral_prior(*, intent: str, user_id: str) -> Optional[float]:
        """Return 0-1 confidence boost from usage history + clarification feedback."""
        base: Optional[float] = None
        try:
            from ai.learning.user_profile_engine import get_profile_engine

            priors: Dict[str, float] = get_profile_engine().get_intent_priors()
            if priors:
                intent_prefix = str(intent or "").split(":")[0].lower()
                raw_val = priors.get(intent_prefix, priors.get(intent))
                if raw_val is not None:
                    # Only apply prior when we have actual usage data for this intent.
                    # Normalize raw frequency [0,1] to confidence [0.5,1.0].
                    # Raw priors are usage-frequency ratios (e.g. 0.017 for weather).
                    # Treating them as confidence scores would heavily penalize rare but
                    # valid intents. Instead we map 0→0.5 (neutral) and higher→boosted.
                    raw = float(raw_val)
                    normalized = min(1.0, 0.5 + raw * 3.0)
                    base = max(0.0, min(1.0, normalized))
        except Exception:
            pass

        # Apply clarification feedback boost on top
        try:
            from ai.optimization.clarification_feedback_loop import (
                get_clarification_feedback_loop,
            )

            boost = get_clarification_feedback_loop().get_confidence_boost(user_id, intent)
            if boost > 0:
                base = min(1.0, (base or 0.5) + boost)
        except Exception:
            pass

        return base

    # Success rates come from an append-only evaluation log. Re-reading and
    # re-parsing it on every turn put a file read and up to 200 json.loads calls
    # in the routing hot path, so the parsed rates are cached and recomputed only
    # when the file's size or mtime changes.
    # Maps intent prefix -> (success rate, number of observations behind it).
    _rates_cache: Dict[str, tuple] = {}
    _rates_stamp: Optional[tuple] = None

    # Sources that only ever record one outcome. FailureEvalConverter backfills
    # routing_failures.jsonl — a failures-only log — into the same file, and
    # nothing backfills the successes. Counting those rows as a sample gave every
    # intent a success rate near zero no matter how well it actually performed,
    # which quietly dropped every turn a decision band. They describe failures,
    # not a rate, so they are excluded from the denominator.
    _UNRATEABLE_SOURCES = frozenset({"routing_failure_backfill"})

    @classmethod
    def _intent_success_rate_with_support(cls, *, intent: str) -> tuple:
        """Return ``(success_rate, observations)`` for this intent's prefix."""
        rates = cls._load_success_rates()
        if not rates:
            return (None, 0)
        entry = rates.get(str(intent or "").split(":")[0].lower())
        if entry is None:
            return (None, 0)
        return entry

    @classmethod
    def _intent_success_rate(cls, *, intent: str) -> Optional[float]:
        """Return the 0-1 success rate recorded for this intent's prefix."""
        return cls._intent_success_rate_with_support(intent=intent)[0]

    @classmethod
    def _load_success_rates(cls) -> Dict[str, tuple]:
        try:
            import json
            from pathlib import Path

            eval_path = Path("data/evaluations/evaluations.jsonl")
            if not eval_path.exists():
                cls._rates_stamp = None
                cls._rates_cache = {}
                return cls._rates_cache

            stat = eval_path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            if stamp == cls._rates_stamp:
                return cls._rates_cache

            totals: Dict[str, int] = {}
            successes: Dict[str, int] = {}
            # Only the most recent entries describe current behavior.
            lines = eval_path.read_text(encoding="utf-8").splitlines()[-200:]
            for line in lines:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if str(rec.get("source") or "") in cls._UNRATEABLE_SOURCES:
                    continue
                prefix = str(rec.get("action_type") or "").split(":")[0].lower()
                if not prefix:
                    continue
                totals[prefix] = totals.get(prefix, 0) + 1
                try:
                    score = int(rec.get("overall_score", 0))
                except (TypeError, ValueError):
                    score = 0
                if score >= 70:
                    successes[prefix] = successes.get(prefix, 0) + 1

            # Fewer than three observations is not a rate, it is an anecdote.
            cls._rates_cache = {
                prefix: (round(successes.get(prefix, 0) / count, 4), count)
                for prefix, count in totals.items()
                if count >= 3
            }
            cls._rates_stamp = stamp
            return cls._rates_cache
        except Exception:
            return {}


_fusion: ConfidenceFusion | None = None


def get_confidence_fusion() -> ConfidenceFusion:
    global _fusion
    if _fusion is None:
        _fusion = ConfidenceFusion()
    return _fusion
