"""Memory-specific turn handling extracted from ContractPipeline."""

from __future__ import annotations

from typing import Any, Dict, List

from ai.memory.memory_extractor import MemoryExtractor


class MemoryTurnService:
    _correction_markers = (
        "forget that",
        "that's wrong",
        "thats wrong",
        "update that memory",
        "that's not what i meant",
        "thats not what i meant",
        "remember it this way instead",
    )

    def __init__(self, extractor: MemoryExtractor | None = None) -> None:
        self.extractor = extractor or MemoryExtractor()

    def _is_correction_turn(self, user_input: str) -> bool:
        low = str(user_input or "").lower()
        return any(marker in low for marker in self._correction_markers)

    def _build_correction_payload(self, user_input: str) -> Dict[str, Any]:
        low = str(user_input or "").lower()
        op = "mark_recent_incorrect"
        if "forget that" in low:
            op = "forget_recent"
        elif "update that memory" in low or "remember it this way instead" in low:
            op = "update_recent"
        return {"memory_operation": op, "reason": str(user_input or "").strip()}

    def build_memory_plan(
        self,
        *,
        user_input: str,
        user_name: str,
        trace_id: str,
        decision_intent: str,
        decision_route: str,
        episodic_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        extracted_candidates = self.extractor.extract_from_user_turn(
            user_text=user_input,
            user_name=user_name,
            source="conversation",
        )
        structured_payloads: List[Dict[str, Any]] = []
        for candidate in extracted_candidates:
            if not bool(candidate.should_store) or not str(candidate.content or "").strip():
                continue
            structured_payloads.append(
                {
                    "content": str(candidate.content).strip(),
                    "domain": str(candidate.domain),
                    "kind": str(candidate.kind),
                    "scope": str(candidate.scope),
                    "confidence": float(candidate.confidence),
                    "source": str(candidate.source),
                    "trace_id": trace_id,
                    "importance": max(0.6, float(candidate.confidence)),
                    "intent": decision_intent,
                    "route": decision_route,
                    "fragment": str(getattr(candidate, "fragment", "") or "").strip(),
                }
            )

        ops: List[Dict[str, Any]] = []
        if self._is_correction_turn(user_input):
            ops.append(self._build_correction_payload(user_input))

        return {
            "episodic_payload": dict(episodic_payload or {}),
            "structured_payloads": structured_payloads,
            "operation_payloads": ops,
            "memory_extraction": {
                "candidate_count": len(extracted_candidates),
                "stored_count": len(structured_payloads),
                "stored_domains": sorted(
                    {str(item.get("domain") or "") for item in structured_payloads}
                ),
                "stored_kinds": sorted(
                    {str(item.get("kind") or "") for item in structured_payloads}
                ),
                "extracted_fragments": [
                    str(getattr(candidate, "fragment", "") or "").strip()
                    for candidate in extracted_candidates
                    if str(getattr(candidate, "fragment", "") or "").strip()
                ],
            },
        }

    def store_memory_plan(self, *, boundaries: Any, plan: Dict[str, Any]) -> None:
        for payload in list(plan.get("structured_payloads") or []):
            boundaries.memory.store(dict(payload or {}))
        boundaries.memory.store(dict(plan.get("episodic_payload") or {}))
        for op in list(plan.get("operation_payloads") or []):
            boundaries.memory.store(dict(op or {}))
