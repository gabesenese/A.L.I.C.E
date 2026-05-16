from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from ai.memory.alice_memory_schema import (
    ActiveConceptThread,
    ContextFrame,
    RetrievedMemory,
)
from ai.runtime.alice_turn_router import TurnRouteDecision, route_turn


class ContextRefreshService:
    def classify_subject(self, user_input: str, intent: str) -> str:
        low = str(user_input or "").lower()
        if "alice" in low or "project" in low or "codebase" in low or str(intent or "").startswith(("operator:", "code:")):
            if any(token in low for token in ("my ", "i ", "today", "family")):
                return "mixed"
            return "project"
        if any(token in low for token in ("my ", "i ", "me ", "personal")):
            return "personal"
        return "external"

    def classify_mode(
        self,
        user_input: str,
        route: str,
        intent: str,
        active_concept_thread: Optional[ActiveConceptThread],
    ) -> TurnRouteDecision:
        return route_turn(
            user_input=user_input,
            current_intent=str(intent or ""),
            current_route=str(route or ""),
            active_concept_thread=active_concept_thread,
        )

    def build_context_frame(
        self,
        *,
        user_input: str,
        route: str,
        intent: str,
        operator_state: Dict[str, Any],
        project_state: Dict[str, Any],
        memory_service: Any,
    ) -> ContextFrame:
        active_thread = None
        if memory_service is not None:
            try:
                active_thread = memory_service.get_active_concept_thread()
            except Exception:
                active_thread = None

        decision = self.classify_mode(user_input, route, intent, active_thread)
        subject = str(decision.subject or self.classify_subject(user_input, intent))
        notes: List[str] = [f"mode={decision.mode}", f"subject={subject}", f"reason={decision.reason}"]

        if decision.mode == "concept_refinement" and memory_service is not None:
            try:
                updated = memory_service.update_active_concept_thread(user_input=user_input)
                if updated is not None:
                    active_thread = updated
                    notes.append("active_concept_thread_updated")
            except Exception:
                notes.append("active_concept_thread_update_failed")

        retrieved: List[RetrievedMemory] = []
        if decision.memory_required and memory_service is not None:
            if decision.mode == "greeting":
                notes.append("greeting_minimal_context")
            else:
                query = str(active_thread.topic if active_thread else user_input)
                try:
                    if subject == "external" and "alice" not in str(user_input or "").lower():
                        retrieved = []
                        notes.append("external_query_memory_suppressed")
                    else:
                        retrieved = list(memory_service.search_memories(query=query, limit=8))
                        notes.append(f"retrieved_memory_count={len(retrieved)}")
                except Exception:
                    notes.append("memory_retrieval_failed")

        verified_memories = [item for item in retrieved if item.confidence_label == "verified"]
        hint_memories = [item for item in retrieved if item.confidence_label != "verified"]

        safe_project_state = dict(project_state or {})
        if decision.mode in {"greeting", "educational_explain"} and subject == "external":
            safe_project_state = {}
            notes.append("project_state_suppressed_for_external_educational")

        return ContextFrame(
            mode=str(decision.mode or "companion_chat"),
            subject=subject,
            user_input=str(user_input or ""),
            active_concept_thread=active_thread,
            verified_memories=verified_memories,
            hint_memories=hint_memories,
            project_state=safe_project_state,
            evidence_required=bool(decision.evidence_required),
            tool_required=bool(decision.tool_required),
            notes=notes,
        )

    @staticmethod
    def _memory_line(item: RetrievedMemory, include_hint_tag: bool = False) -> str:
        prefix = "[hint] " if include_hint_tag else ""
        topic = str(item.record.topic or "").strip()
        content = str(item.record.content or "").strip()
        basis = topic if topic else content
        if len(basis) > 160:
            basis = basis[:157] + "..."
        return f"- {prefix}{basis} (score={item.score:.2f}, confidence={item.record.confidence:.2f})"

    def build_context_block(self, frame: ContextFrame) -> str:
        lines: List[str] = [
            "<alice_context>",
            f"mode: {frame.mode}",
            f"subject: {frame.subject}",
        ]
        if frame.active_concept_thread is not None:
            lines.append(f"active_concept: {frame.active_concept_thread.topic}")
            if frame.active_concept_thread.constraints:
                lines.append("constraints:")
                for item in frame.active_concept_thread.constraints:
                    lines.append(f"- {item}")
        lines.append("verified_memories:")
        if frame.verified_memories:
            for item in frame.verified_memories[:6]:
                lines.append(self._memory_line(item))
        else:
            lines.append("- none")
        lines.append("hint_memories:")
        if frame.hint_memories:
            for item in frame.hint_memories[:6]:
                lines.append(self._memory_line(item, include_hint_tag=True))
        else:
            lines.append("- none")
        if frame.project_state:
            lines.append("project_state:")
            for key in ("active_objective", "current_focus", "next_recommended_action"):
                value = str(frame.project_state.get(key) or "").strip()
                if value:
                    lines.append(f"- {key}: {value}")
        lines.append(f"evidence_required: {str(bool(frame.evidence_required)).lower()}")
        lines.append(f"tool_required: {str(bool(frame.tool_required)).lower()}")
        lines.append("</alice_context>")
        return "\n".join(lines)

    @staticmethod
    def should_inject_context_for_model(mode: str, intent: str) -> bool:
        normalized_mode = str(mode or "").strip().lower()
        normalized_intent = str(intent or "").strip().lower()
        if normalized_mode in {"concept_refinement", "educational_explain", "companion_chat"}:
            return True
        return normalized_intent.startswith("conversation:")

    @staticmethod
    def frame_to_metadata(frame: ContextFrame) -> Dict[str, Any]:
        out = asdict(frame)
        out["verified_memories"] = [
            {
                "id": item.record.id,
                "kind": item.record.kind,
                "topic": item.record.topic,
                "score": item.score,
                "reason": item.reason,
                "confidence_label": item.confidence_label,
            }
            for item in frame.verified_memories
        ]
        out["hint_memories"] = [
            {
                "id": item.record.id,
                "kind": item.record.kind,
                "topic": item.record.topic,
                "score": item.score,
                "reason": item.reason,
                "confidence_label": item.confidence_label,
            }
            for item in frame.hint_memories
        ]
        return out
