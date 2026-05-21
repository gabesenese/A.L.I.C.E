"""Confidence fusion: combine multiple routing signals into a single calibrated score.

Sources:
  1. Router confidence (always present)
  2. Behavioral priors from UserProfileEngine (intent frequency history)
  3. Per-intent success history (from RoutingFailureLogger or evaluation log)
"""

from __future__ import annotations

from typing import Dict, Optional


class ConfidenceFusion:
    """Fuses multiple confidence signals using a weighted average.

    Weights are tuned so the router confidence dominates but priors can
    push a borderline score above or below a decision threshold.
    """

    # Source weights (must sum to 1.0 or be normalized)
    _WEIGHTS = {
        "router": 0.65,
        "behavioral_prior": 0.20,
        "intent_success_rate": 0.15,
    }

    def fuse(
        self,
        *,
        router_confidence: float,
        intent: str,
        user_id: str = "default",
    ) -> float:
        """Return a fused confidence score in [0, 1].

        Falls back gracefully when behavioral or success-rate data is absent.
        """
        router_c = max(0.0, min(1.0, float(router_confidence or 0.0)))
        behavioral_c = self._behavioral_prior(intent=intent, user_id=user_id)
        success_c = self._intent_success_rate(intent=intent)

        weights = dict(self._WEIGHTS)
        total_weight = weights["router"]
        weighted = weights["router"] * router_c

        if behavioral_c is not None:
            weighted += weights["behavioral_prior"] * behavioral_c
            total_weight += weights["behavioral_prior"]

        if success_c is not None:
            weighted += weights["intent_success_rate"] * success_c
            total_weight += weights["intent_success_rate"]

        fused = weighted / total_weight if total_weight > 0 else router_c
        return round(max(0.0, min(1.0, fused)), 4)

    @staticmethod
    def _behavioral_prior(*, intent: str, user_id: str) -> Optional[float]:
        """Return 0-1 confidence boost from usage history + clarification feedback."""
        base: Optional[float] = None
        try:
            from ai.learning.user_profile_engine import get_profile_engine
            priors: Dict[str, float] = get_profile_engine().get_intent_priors()
            if priors:
                intent_prefix = str(intent or "").split(":")[0].lower()
                raw = float(priors.get(intent_prefix, priors.get(intent, 0.5)))
                base = max(0.0, min(1.0, raw))
        except Exception:
            pass

        # Apply clarification feedback boost on top
        try:
            from ai.optimization.clarification_feedback_loop import get_clarification_feedback_loop
            boost = get_clarification_feedback_loop().get_confidence_boost(user_id, intent)
            if boost > 0:
                base = min(1.0, (base or 0.5) + boost)
        except Exception:
            pass

        return base

    @staticmethod
    def _intent_success_rate(*, intent: str) -> Optional[float]:
        """Return 0-1 success rate from evaluation log for this intent prefix."""
        try:
            import json
            from pathlib import Path
            eval_path = Path("data/evaluations/evaluations.jsonl")
            if not eval_path.exists():
                return None

            intent_prefix = str(intent or "").split(":")[0].lower()
            total = 0
            successes = 0
            # Read last 200 entries for the prefix
            lines = eval_path.read_text(encoding="utf-8").splitlines()
            for line in lines[-200:]:
                try:
                    rec = json.loads(line)
                    rec_intent = str(rec.get("action_type") or "").split(":")[0].lower()
                    if rec_intent == intent_prefix:
                        total += 1
                        if int(rec.get("overall_score", 0)) >= 70:
                            successes += 1
                except Exception:
                    continue
            if total < 3:
                return None  # not enough data
            return round(successes / total, 4)
        except Exception:
            return None


_fusion: ConfidenceFusion | None = None


def get_confidence_fusion() -> ConfidenceFusion:
    global _fusion
    if _fusion is None:
        _fusion = ConfidenceFusion()
    return _fusion
