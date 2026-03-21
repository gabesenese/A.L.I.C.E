"""Ranks candidate options against explicit hard/soft constraints."""

from __future__ import annotations

from typing import Any, Dict, List


class DecisionConstraintSolver:
    def solve(
        self,
        options: List[Dict[str, Any]],
        *,
        hard_constraints: Dict[str, Any] | None = None,
        soft_weights: Dict[str, float] | None = None,
    ) -> List[Dict[str, Any]]:
        hard = dict(hard_constraints or {})
        weights = dict(soft_weights or {})

        ranked: List[Dict[str, Any]] = []
        for opt in options or []:
            valid = True
            for key, required in hard.items():
                if key in opt and opt.get(key) != required:
                    valid = False
                    break
            if not valid:
                continue

            score = 0.0
            for key, weight in weights.items():
                val = float(opt.get(key, 0.0) or 0.0)
                score += val * float(weight)
            ranked.append({**opt, "constraint_score": round(score, 4)})

        ranked.sort(key=lambda row: float(row.get("constraint_score", 0.0)), reverse=True)
        return ranked
