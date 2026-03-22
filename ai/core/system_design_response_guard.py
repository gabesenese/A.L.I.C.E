"""Guidance and response shaping for AI architecture/system-design questions."""

from __future__ import annotations

from typing import Optional


class SystemDesignResponseGuard:
    def is_architecture_query(self, text: str) -> bool:
        lowered = str(text or "").lower()
        topic_hit = any(
            k in lowered
            for k in (
                "ai architecture",
                "assistant architecture",
                "machine learning foundations",
                "if this existed",
                "if an assistant existed",
                "how would you build",
                "real-world ai system",
                "fictional ai assistant",
            )
        )
        return topic_hit and any(
            k in lowered for k in ("ai", "machine learning", "ml", "assistant")
        )

    def guidance_text(self) -> str:
        return (
            "Architecture answer policy (system-only):\n"
            "- Prioritize system design and operating constraints over product-name lists.\n"
            "- Do not overemphasize RNNs for modern foundation-model stacks unless sequence-specialized use is explicit.\n"
            "- Avoid vague/non-standard library references; describe concrete capabilities instead.\n"
            "- Explicitly include orchestration, permissions/sandbox boundaries, and real-time reliability constraints.\n"
            "- Include practical limitations and phased deployment guidance."
        )

    def direct_answer(self, text: str) -> Optional[str]:
        if not self.is_architecture_query(text):
            return None

        return (
            "A realistic advanced assistant stack today would be designed as a system, not a list of libraries:\n\n"
            "1) Core intelligence\n"
            "Use multimodal foundation models (text, audio, vision) with retrieval and tool-use adapters.\n\n"
            "2) Orchestration layer\n"
            "Run an agent orchestration runtime that separates planning, tool selection, execution, and verification.\n"
            "This layer should support retries, timeouts, and fallback routes under latency budgets.\n\n"
            "3) Memory architecture\n"
            "Combine short-term conversation state, long-term episodic logs, and semantic retrieval indexes.\n"
            "Use consolidation jobs to keep memory relevant and bounded.\n\n"
            "4) Permissions and safety\n"
            "Apply capability-based permissions for every tool call with explicit approval gates for risky actions\n"
            "(device control, finance, deletion, external messaging).\n\n"
            "5) Real-time reliability\n"
            "Design for graceful degradation: cached responses, deterministic fallback logic, and action idempotency.\n"
            "Track SLOs for response time, tool success, and hallucination-related regressions.\n\n"
            "6) Explainability and audit\n"
            "Store structured decision traces, tool arguments, and outcomes for post-incident review.\n\n"
            "7) Learning loop\n"
            "Prefer offline evaluation + controlled rollouts over autonomous online self-modification."
        )
