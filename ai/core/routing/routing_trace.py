from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RoutingTrace:
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    accepted_candidate: Dict[str, Any] = field(default_factory=dict)
    rejected_candidates: List[Dict[str, Any]] = field(default_factory=list)
    evidence_contract_results: List[Dict[str, Any]] = field(default_factory=list)
    vetoes: List[Dict[str, Any]] = field(default_factory=list)
    final_route: str = ""
    final_intent: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "candidates": list(self.candidates),
            "accepted_candidate": dict(self.accepted_candidate),
            "rejected_candidates": list(self.rejected_candidates),
            "evidence_contract_results": list(self.evidence_contract_results),
            "vetoes": list(self.vetoes),
            "final_route": self.final_route,
            "final_intent": self.final_intent,
        }
