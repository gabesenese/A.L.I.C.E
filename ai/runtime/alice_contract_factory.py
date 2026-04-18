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
    VerifierRequest,
    VerifierResult,
    CallableVerifierAdapter,
    validate_tool_invocation_payload,
    validate_tool_result_payload,
    ToolSchemaValidationError,
)


def build_runtime_boundaries(alice: Any) -> RuntimeBoundaries:
    """Create runtime boundaries backed by current ALICE components."""

    risky_refusal_markers = {
        "delete all",
        "wipe",
        "format disk",
        "shutdown system",
        "drop database",
        "exfiltrate",
        "bypass security",
    }

    def _surface_text(
        text: str,
        *,
        user_input: str,
        intent: str,
        route: str = "contract_pipeline",
    ) -> str:
        base = str(text or "").strip()
        if not base:
            base = "Please share one concrete detail so I can continue."
        if hasattr(alice, "_finalize_conversational_surface"):
            try:
                return str(
                    alice._finalize_conversational_surface(
                        user_input=user_input,
                        intent=intent,
                        response=base,
                        route=route,
                        plugin_result=None,
                        apply_publish_style=True,
                    )
                ).strip()
            except Exception:
                return base
        if hasattr(alice, "_clamp_final_response"):
            try:
                return str(
                    alice._clamp_final_response(
                        base,
                        tone="helpful",
                        response_type="general_response",
                        route=route,
                        user_input=user_input,
                    )
                ).strip()
            except Exception:
                return base
        return base

    def _route(req: RouterRequest) -> RouterDecision:
        if req.user_input.startswith("/"):
            return RouterDecision(
                route="command",
                intent="system:command",
                confidence=1.0,
                decision_band="execute",
                metadata={"reason": "slash_command"},
            )

        if hasattr(alice, "_is_location_query") and alice._is_location_query(
            req.user_input
        ):
            return RouterDecision(
                route="local",
                intent="system:location",
                confidence=1.0,
                decision_band="execute",
                metadata={"reason": "deterministic_location_query"},
            )

        resolved_input = req.user_input
        resolution_meta = {}
        if getattr(alice, "context_resolver", None):
            try:
                _pre_state = {
                    "current_topic": "",
                    "last_subject": "",
                    "last_intent": str(getattr(alice, "last_intent", "") or ""),
                    "active_goal": "",
                    "referenced_entities": [],
                    "last_entities": dict(getattr(alice, "last_entities", {}) or {}),
                }
                resolution = alice.context_resolver.resolve(req.user_input, _pre_state)
                resolved_input = str(resolution.rewritten_input or req.user_input)
                resolution_meta = {
                    "rewritten": resolved_input != req.user_input,
                    "resolved_bindings": dict(
                        getattr(resolution, "resolved_bindings", {}) or {}
                    ),
                }
                if getattr(resolution, "needs_clarification", False):
                    return RouterDecision(
                        route="clarify",
                        intent="clarification:context_resolution",
                        confidence=0.2,
                        decision_band="clarify",
                        needs_clarification=True,
                        metadata={
                            "reason": "context_ambiguity",
                            "options": list(
                                getattr(resolution, "clarification_options", []) or []
                            ),
                            "pronouns": list(
                                getattr(resolution, "unresolved_pronouns", []) or []
                            ),
                        },
                    )
            except Exception:
                resolution_meta = {
                    "rewritten": False,
                    "error": "context_resolver_failure",
                }

        nlp_result = alice.nlp.process(resolved_input)
        intent = str(getattr(nlp_result, "intent", "unknown") or "unknown")
        confidence = float(getattr(nlp_result, "intent_confidence", 0.0) or 0.0)

        route = "llm"
        if ":" in intent and not intent.startswith("conversation"):
            route = "tool"

        lower_input = req.user_input.lower()
        if confidence >= 0.80:
            band = "execute"
        elif confidence >= 0.60:
            band = "verify"
        elif confidence >= 0.35:
            band = "clarify"
        else:
            band = (
                "refuse"
                if any(marker in lower_input for marker in risky_refusal_markers)
                else "clarify"
            )

        if band == "clarify":
            return RouterDecision(
                route="clarify",
                intent=intent,
                confidence=confidence,
                decision_band="clarify",
                needs_clarification=True,
                metadata={"reason": "low_confidence"},
            )

        if band == "refuse":
            return RouterDecision(
                route="refuse",
                intent="safety:refuse",
                confidence=confidence,
                decision_band="refuse",
                needs_clarification=False,
                metadata={"reason": "unsafe_or_too_uncertain"},
            )

        return RouterDecision(
            route=route,
            intent=intent,
            confidence=confidence,
            decision_band=band,
            needs_clarification=False,
            metadata={
                "keywords": list(getattr(nlp_result, "keywords", []) or []),
                "resolved_input": resolved_input,
                **resolution_meta,
            },
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
            alice.memory.store_memory(
                content=text, memory_type="episodic", context=item
            )
        except Exception:
            # Storage errors should not block response path.
            return

    def _execute(invocation: ToolInvocation) -> ToolResult:
        try:
            validate_tool_invocation_payload(
                {
                    "tool_name": invocation.tool_name,
                    "action": invocation.action,
                    "params": invocation.params,
                }
            )
        except ToolSchemaValidationError as exc:
            return ToolResult(
                success=False,
                tool_name=str(invocation.tool_name or "unknown"),
                action=str(invocation.action or "unknown"),
                error=f"schema_error:{exc}",
                confidence=0.0,
                diagnostics={"stage": "pre_tool_validation"},
            )

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
            tool_result = ToolResult(
                success=False,
                tool_name=invocation.tool_name,
                action=invocation.action,
                error="no plugin handled invocation",
                confidence=0.0,
            )
            return tool_result

        tool_result = ToolResult(
            success=bool(result.get("success", False)),
            tool_name=str(result.get("plugin") or invocation.tool_name),
            action=invocation.action,
            data=dict(result),
            error=str(result.get("error") or ""),
            confidence=float(result.get("confidence", 0.7) or 0.7),
            diagnostics={"route": "plugin_manager"},
        )
        try:
            validate_tool_result_payload(
                {
                    "success": tool_result.success,
                    "tool_name": tool_result.tool_name,
                    "action": tool_result.action,
                    "data": tool_result.data,
                    "diagnostics": tool_result.diagnostics,
                }
            )
        except ToolSchemaValidationError as exc:
            return ToolResult(
                success=False,
                tool_name=tool_result.tool_name,
                action=tool_result.action,
                data=tool_result.data,
                error=f"schema_error:{exc}",
                confidence=0.0,
                diagnostics={"stage": "post_tool_validation"},
            )
        return tool_result

    def _generate(req: ResponseRequest) -> ResponseOutput:
        if req.decision.decision_band == "refuse" or req.decision.route == "refuse":
            refusal_text = _surface_text(
                "I can't safely perform that request. Share a narrower safe action and I will continue.",
                user_input=req.user_input,
                intent=req.decision.intent,
                route="contract_policy_refusal",
            )
            return ResponseOutput(
                text=refusal_text,
                confidence=1.0,
                requires_follow_up=True,
                follow_up_question=_surface_text(
                    "What safe and specific action do you want?",
                    user_input=req.user_input,
                    intent=req.decision.intent,
                    route="contract_policy_refusal",
                ),
                metadata={"type": "policy_refusal"},
            )

        if req.decision.intent == "system:command":
            command_text = _surface_text(
                "Commands are handled by the interface. Use /help to see available commands.",
                user_input=req.user_input,
                intent=req.decision.intent,
                route="contract_command_dispatch",
            )
            return ResponseOutput(
                text=command_text,
                confidence=1.0,
                metadata={"type": "command_dispatch"},
            )

        if req.decision.intent == "system:location" and hasattr(
            alice, "_build_location_payload"
        ):
            payload = alice._build_location_payload()
            text = alice._alice_direct_phrase("location_report", payload)
            return ResponseOutput(
                text=_surface_text(
                    str(text or "I could not resolve your location."),
                    user_input=req.user_input,
                    intent=req.decision.intent,
                    route="contract_location",
                ),
                confidence=0.99,
                metadata={"type": "deterministic_location"},
            )

        if req.decision.needs_clarification:
            clarify_text = _surface_text(
                "Share the exact outcome you want so I can route this correctly.",
                user_input=req.user_input,
                intent=req.decision.intent,
                route="contract_clarification",
            )
            return ResponseOutput(
                text=clarify_text,
                confidence=0.6,
                requires_follow_up=True,
                follow_up_question=_surface_text(
                    "What exact result do you want?",
                    user_input=req.user_input,
                    intent=req.decision.intent,
                    route="contract_clarification",
                ),
                metadata={"type": "clarification"},
            )

        if req.tool_result and req.tool_result.success:
            tool_response = str(req.tool_result.data.get("response") or "").strip()
            if tool_response:
                return ResponseOutput(
                    text=tool_response,
                    confidence=float(req.tool_result.confidence or 0.7),
                    metadata={
                        "type": "tool_response",
                        "tool": req.tool_result.tool_name,
                    },
                )

        llm_text = ""
        if getattr(alice, "llm", None):
            try:
                llm_text = str(
                    alice.llm.chat(req.user_input, use_history=True) or ""
                ).strip()
            except Exception:
                llm_text = ""

        if llm_text:
            if hasattr(alice, "_clamp_final_response"):
                try:
                    llm_text = str(
                        alice._clamp_final_response(
                            llm_text,
                            tone="helpful",
                            response_type="general_response",
                            route="contract_pipeline_llm",
                            user_input=req.user_input,
                        )
                    ).strip()
                except Exception:
                    llm_text = str(llm_text or "").strip()
            return ResponseOutput(
                text=llm_text,
                confidence=max(0.45, float(req.decision.confidence or 0.45)),
                metadata={"type": "llm_response"},
            )

        return ResponseOutput(
            text=_surface_text(
                "I could not complete that request reliably. Rephrase the desired outcome and I will retry.",
                user_input=req.user_input,
                intent=req.decision.intent,
                route="contract_fallback",
            ),
            confidence=0.2,
            requires_follow_up=True,
            follow_up_question=_surface_text(
                "Can you rephrase with the exact action you want?",
                user_input=req.user_input,
                intent=req.decision.intent,
                route="contract_fallback",
            ),
            metadata={"type": "fallback"},
        )

    def _verify(req: VerifierRequest) -> VerifierResult:
        response_text = str(req.proposed_response.text or "").strip()
        if not response_text:
            return VerifierResult(
                accepted=False,
                reason="empty_response",
                confidence=0.0,
                diagnostics={"stage": "verification"},
            )

        if req.decision.decision_band == "refuse":
            has_refusal = (
                "can't safely" in response_text.lower()
                or "cannot safely" in response_text.lower()
            )
            return VerifierResult(
                accepted=has_refusal,
                reason="refused_by_policy" if has_refusal else "refusal_missing",
                confidence=1.0 if has_refusal else 0.0,
                diagnostics={"band": req.decision.decision_band},
            )

        if (
            req.decision.decision_band == "verify"
            and req.decision.route in {"tool", "plugin"}
            and req.tool_result is None
        ):
            return VerifierResult(
                accepted=False,
                reason="verify_band_requires_tool_evidence",
                confidence=0.0,
                diagnostics={"band": req.decision.decision_band},
            )

        if (
            req.decision.route in {"tool", "plugin"}
            and req.tool_result
            and not req.tool_result.success
        ):
            return VerifierResult(
                accepted=False,
                reason="tool_failed",
                confidence=0.2,
                diagnostics={
                    "tool": req.tool_result.tool_name,
                    "error": req.tool_result.error,
                },
            )

        return VerifierResult(
            accepted=True,
            reason="verified",
            confidence=max(0.5, float(req.proposed_response.confidence or 0.5)),
            diagnostics={"route": req.decision.route, "intent": req.decision.intent},
        )

    return RuntimeBoundaries(
        routing=CallableRoutingAdapter(route_fn=_route),
        memory=CallableMemoryAdapter(recall_fn=_recall, store_fn=_store),
        tools=CallableToolAdapter(execute_fn=_execute),
        response=CallableResponseAdapter(generate_fn=_generate),
        verifier=CallableVerifierAdapter(verify_fn=_verify),
    )
