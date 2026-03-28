"""Factory for wiring ALICE runtime components to hard boundary contracts."""

from __future__ import annotations

from typing import Any, Dict

from ai.contracts import (
    CallableMemoryAdapter,
    CallableResponseAdapter,
    CallableRoutingAdapter,
    CallableToolAdapter,
    MemoryRequest,
    MemoryResult,
    ResponseOutput,
    ResponseRequest,
    RouterDecision,
    RouterRequest,
    RuntimeBoundaries,
    ToolInvocation,
    ToolResult,
)


def build_runtime_boundaries(alice: Any) -> RuntimeBoundaries:
    """Create runtime boundaries backed by current ALICE components."""

    def _route(req: RouterRequest) -> RouterDecision:
        nlp_result = alice.nlp.process(req.user_input)
        intent = str(getattr(nlp_result, "intent", "unknown") or "unknown")
        confidence = float(getattr(nlp_result, "intent_confidence", 0.0) or 0.0)

        route = "llm"
        if ":" in intent and not intent.startswith("conversation"):
            route = "tool"
        if confidence < 0.35:
            return RouterDecision(
                route="clarify",
                intent=intent,
                confidence=confidence,
                needs_clarification=True,
                metadata={"reason": "low_confidence"},
            )

        return RouterDecision(
            route=route,
            intent=intent,
            confidence=confidence,
            needs_clarification=False,
            metadata={"keywords": list(getattr(nlp_result, "keywords", []) or [])},
        )

    def _recall(req: MemoryRequest) -> MemoryResult:
        try:
            items = []
            if getattr(alice, "memory", None):
                items = alice.memory.search(req.query, top_k=req.max_items)
            return MemoryResult(
                items=list(items or []),
                source="memory_system",
                confidence=0.8 if items else 0.3,
                metadata={"count": len(items or [])},
            )
        except Exception as exc:
            return MemoryResult(
                items=[],
                source="memory_system",
                confidence=0.0,
                metadata={"error": str(exc)},
            )

    def _store(item: Dict[str, Any]) -> None:
        if not getattr(alice, "memory", None):
            return
        text = str(item.get("content") or item.get("text") or "").strip()
        if not text:
            return
        try:
            alice.memory.store_memory(content=text, memory_type="episodic", context=item)
        except Exception:
            # Storage errors should not block response path.
            return

    def _execute(invocation: ToolInvocation) -> ToolResult:
        if not getattr(alice, "plugins", None):
            return ToolResult(
                success=False,
                tool_name=invocation.tool_name,
                action=invocation.action,
                error="plugin manager unavailable",
                confidence=0.0,
            )

        intent = invocation.params.get("intent") or invocation.action
        query = str(invocation.params.get("query") or "")
        entities = dict(invocation.params.get("entities") or {})
        context = dict(invocation.params.get("context") or {})

        result = alice.plugins.execute_for_intent(intent, query, entities, context)
        if not result:
            return ToolResult(
                success=False,
                tool_name=invocation.tool_name,
                action=invocation.action,
                error="no plugin handled invocation",
                confidence=0.0,
            )

        return ToolResult(
            success=bool(result.get("success", False)),
            tool_name=str(result.get("plugin") or invocation.tool_name),
            action=invocation.action,
            data=dict(result),
            error=str(result.get("error") or ""),
            confidence=float(result.get("confidence", 0.7) or 0.7),
            diagnostics={"route": "plugin_manager"},
        )

    def _generate(req: ResponseRequest) -> ResponseOutput:
        if req.decision.needs_clarification:
            return ResponseOutput(
                text="Can you clarify the exact outcome you want so I can route this correctly?",
                confidence=0.6,
                requires_follow_up=True,
                follow_up_question="What exact result do you want?",
                metadata={"type": "clarification"},
            )

        if req.tool_result and req.tool_result.success:
            tool_response = str(req.tool_result.data.get("response") or "").strip()
            if tool_response:
                return ResponseOutput(
                    text=tool_response,
                    confidence=float(req.tool_result.confidence or 0.7),
                    metadata={"type": "tool_response", "tool": req.tool_result.tool_name},
                )

        llm_text = ""
        if getattr(alice, "llm", None):
            try:
                llm_text = str(alice.llm.chat(req.user_input, use_history=True) or "").strip()
            except Exception:
                llm_text = ""

        if llm_text:
            return ResponseOutput(
                text=llm_text,
                confidence=max(0.45, float(req.decision.confidence or 0.45)),
                metadata={"type": "llm_response"},
            )

        return ResponseOutput(
            text="I couldn't complete that request reliably. Please rephrase the desired outcome.",
            confidence=0.2,
            requires_follow_up=True,
            follow_up_question="Can you rephrase with the exact action you want?",
            metadata={"type": "fallback"},
        )

    return RuntimeBoundaries(
        routing=CallableRoutingAdapter(route_fn=_route),
        memory=CallableMemoryAdapter(recall_fn=_recall, store_fn=_store),
        tools=CallableToolAdapter(execute_fn=_execute),
        response=CallableResponseAdapter(generate_fn=_generate),
    )
