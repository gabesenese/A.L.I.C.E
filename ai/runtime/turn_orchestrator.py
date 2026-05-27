"""Explicit turn phases: route -> execute -> verify -> respond."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
import re
from typing import Any, Dict, Optional

from ai.runtime.response_authority import sanitize_internal_process_output
from ai.contracts import (
    MemoryRequest,
    ResponseRequest,
    RouterRequest,
    RuntimeBoundaries,
    ToolInvocation,
    ToolResult,
    VerifierRequest,
    VerifierResult,
    ResponseOutput,
    RouterDecision,
    MemoryResult,
)


logger = logging.getLogger(__name__)


def _verification_fallback(
    reason: str,
    diagnostics: dict,
    proposed_text: str,
    intent: str = "",
    user_id: str = "default",
) -> str:
    if reason == "tool_failed":
        tool = str(diagnostics.get("tool") or "")
        error_type = str(diagnostics.get("error") or diagnostics.get("error_type") or "")
        try:
            from ai.runtime.fallback_policy import get_fallback_graph, get_retry_memory

            rm = get_retry_memory()
            rm.record_failure(user_id, intent or tool, error_type)
            # Check for escalation message first (repeated failures)
            esc = rm.escalation_message(user_id, intent or tool, error_type)
            if esc:
                return esc
            # Use the appropriate FallbackStep based on repeat count
            fg = get_fallback_graph()
            steps = fg.get_steps(intent or tool, error_type)
            step_idx = rm.get_step_index(user_id, intent or tool, error_type)
            if steps and step_idx < len(steps):
                return steps[step_idx].message
        except Exception:
            pass
        if tool and error_type:
            return (
                f"The {tool} action ran into an issue ({error_type}). Try rephrasing or check that the target exists."
            )
        if tool:
            return f"The {tool} action didn't complete successfully. Try again or rephrase what you need."
        return "That action didn't complete successfully. Try rephrasing or providing more detail."

    if reason == "empty_response":
        return "I wasn't able to generate a response for that. Could you be more specific about what you need?"

    if reason == "unsupported_continuity_claim":
        return "I don't have enough context to answer that confidently. Could you give me a bit more detail?"

    if reason == "unverified_codebase_claim":
        return "I don't have the file details memorized. Use 'inspect <filename>' to get accurate info about a specific file."

    if reason == "unverified_weather_claim":
        return "I don't have live weather data. Try asking 'what's the weather in [city]?' to get current conditions."

    if reason == "verify_band_requires_tool_evidence":
        return "I need to run a tool to answer that, but nothing was executed. Try asking more directly."

    if reason == "refusal_missing":
        return "I can't safely do that. Try a more specific or narrower version of the request."

    if reason == "local_execution_target_not_found":
        close = diagnostics.get("close_matches") or []
        if close:
            suggestion = close[0] if isinstance(close[0], str) else str(close[0])
            return f"I couldn't find that file. Did you mean {suggestion}?"
        return "I couldn't find that file in the workspace. Check the name and try again."

    if proposed_text:
        return proposed_text

    return "I wasn't able to complete that — try rephrasing with a more specific action."


@dataclass(frozen=True)
class RoutePhaseResult:
    decision: RouterDecision
    resolved_input: str
    memory: MemoryResult
    plan: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutePhaseResult:
    tool_result: Optional[ToolResult] = None
    executed: bool = False


@dataclass(frozen=True)
class VerifyPhaseResult:
    proposed_response: ResponseOutput
    verification: Optional[VerifierResult]


@dataclass(frozen=True)
class RespondPhaseResult:
    response_text: str
    requires_follow_up: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class TurnOrchestrator:
    """Contract-driven turn orchestrator with explicit runtime phases."""

    def __init__(self, boundaries: RuntimeBoundaries) -> None:
        self.boundaries = boundaries

    def route_phase(
        self,
        *,
        user_input: str,
        user_id: str,
        turn_number: int,
    ) -> RoutePhaseResult:
        decision = self.boundaries.routing.route(RouterRequest(user_input=user_input, turn_number=turn_number))

        resolved_input = str((decision.metadata or {}).get("resolved_input") or user_input)
        memory = self.boundaries.memory.recall(
            MemoryRequest(
                query=resolved_input,
                user_id=user_id,
                max_items=8,
                metadata={
                    "intent": decision.intent,
                    "route": decision.route,
                    "turn_number": turn_number,
                },
            )
        )

        plan = {
            "route": decision.route,
            "intent": decision.intent,
            "decision_band": decision.decision_band,
            "needs_clarification": decision.needs_clarification,
            "step_count": 1 if decision.route in {"llm", "clarify", "refuse"} else 2,
        }

        return RoutePhaseResult(
            decision=decision,
            resolved_input=resolved_input,
            memory=memory,
            plan=plan,
        )

    def execute_phase(
        self,
        *,
        route_phase: RoutePhaseResult,
    ) -> ExecutePhaseResult:
        decision = route_phase.decision
        if decision.route not in {"tool", "plugin", "local"}:
            return ExecutePhaseResult(tool_result=None, executed=False)

        decision_metadata = dict(decision.metadata or {})
        operator_context = dict(decision_metadata.get("operator_context") or {})
        operator_state = dict(decision_metadata.get("operator_state") or {})
        turn_plan = dict(decision_metadata.get("turn_plan") or {})
        resolved_input = str(route_phase.resolved_input or "")
        target_file = ""
        explicit_target = str(decision_metadata.get("target_file") or "").strip()
        if explicit_target:
            target_file = explicit_target
        elif operator_context.get("inferred_target_file"):
            target_file = str(operator_context.get("inferred_target_file") or "").strip()
        else:
            match = re.search(r"([a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9]{1,8})\b", resolved_input)
            if match:
                target_file = str(match.group(1))

        tool_name = decision.intent.split(":", 1)[0] if ":" in decision.intent else decision.intent

        # Resolve location for the weather plugin from operator/session context.
        # The plugin accepts "city" or "location" keys in its context dict.
        _location_ctx: Dict[str, str] = {}
        if tool_name == "weather":
            _raw_loc = (
                str(operator_context.get("city") or "").strip()
                or str(operator_context.get("location") or "").strip()
                or str(operator_state.get("city") or "").strip()
                or str(operator_state.get("location") or "").strip()
            )
            if _raw_loc and _raw_loc.lower() not in {"unknown", "none", ""}:
                _location_ctx = {"city": _raw_loc, "location": _raw_loc}

        tool_result = self.boundaries.tools.execute(
            ToolInvocation(
                tool_name=tool_name,
                action=decision.intent,
                params={
                    "intent": decision.intent,
                    "query": route_phase.resolved_input,
                    "entities": {},
                    "context": {
                        "memory_count": len(route_phase.memory.items),
                        "route": decision.route,
                        "intent": decision.intent,
                        "decision_metadata": decision_metadata,
                        "operator_context": operator_context,
                        "resolved_input": resolved_input,
                        "target_file": target_file,
                        "turn_plan": turn_plan,
                        "operator_state": dict(operator_state or {}),
                        "active_mode": str(
                            operator_state.get("active_mode") or operator_context.get("active_mode") or ""
                        ),
                        "active_objective": str(operator_state.get("active_objective") or ""),
                        "previous_intent": str(operator_state.get("last_intent") or ""),
                        **_location_ctx,
                    },
                },
            )
        )

        # For known weather error codes, synthesise a direct clarifying response via
        # the FallbackGraph instead of handing a bare failure to the LLM.
        if tool_result is not None and not tool_result.success and tool_name == "weather":
            # error may be "weather:no_location" (full message_code) or just "no_location"
            _raw_err = str(
                (tool_result.data or {}).get("error")
                or ((tool_result.data or {}).get("data") or {}).get("error")
                or ((tool_result.data or {}).get("data") or {}).get("message_code", "")
                or tool_result.error
                or ""
            ).strip()
            # Strip plugin-name prefix so "weather:no_location" → "no_location"
            _err = re.sub(r"^weather:", "", _raw_err, flags=re.IGNORECASE)
            try:
                from ai.runtime.fallback_policy import (
                    get_fallback_graph,
                    get_retry_memory,
                )

                _fg = get_fallback_graph()
                _steps = _fg.get_steps("weather", _err)
                if _steps:
                    _rm = get_retry_memory()
                    _idx = _rm.get_step_index("default", "weather", _err)
                    _msg = _steps[min(_idx, len(_steps) - 1)].message
                    tool_result = ToolResult(
                        success=False,
                        tool_name=tool_result.tool_name,
                        action=tool_result.action,
                        data={
                            **(tool_result.data or {}),
                            "fallback_message": _msg,
                            "use_fallback_message": True,
                        },
                        error=tool_result.error,
                        confidence=tool_result.confidence,
                        diagnostics=tool_result.diagnostics,
                    )
            except Exception:
                pass

        return ExecutePhaseResult(tool_result=tool_result, executed=True)

    def verify_phase(
        self,
        *,
        user_input: str,
        route_phase: RoutePhaseResult,
        execute_phase: ExecutePhaseResult,
        trace_id: str,
    ) -> VerifyPhaseResult:
        proposed = self.boundaries.response.generate(
            ResponseRequest(
                user_input=user_input,
                decision=route_phase.decision,
                memory=route_phase.memory,
                tool_result=execute_phase.tool_result,
                metadata={"resolved_input": route_phase.resolved_input},
            )
        )

        verification = None
        if self.boundaries.verifier is not None:
            verification = self.boundaries.verifier.verify(
                VerifierRequest(
                    user_input=user_input,
                    decision=route_phase.decision,
                    memory=route_phase.memory,
                    proposed_response=proposed,
                    tool_result=execute_phase.tool_result,
                    metadata={"trace_id": trace_id},
                )
            )

        return VerifyPhaseResult(proposed_response=proposed, verification=verification)

    def respond_phase(
        self,
        *,
        verify_phase: VerifyPhaseResult,
    ) -> RespondPhaseResult:
        verification = verify_phase.verification
        proposed = verify_phase.proposed_response

        if verification is not None and not verification.accepted:
            _intent_for_fallback = str(
                verify_phase.proposed_response.metadata.get("intent", "")
                if verify_phase.proposed_response and verify_phase.proposed_response.metadata
                else ""
            )
            response_text = _verification_fallback(
                reason=str(verification.reason or ""),
                diagnostics=dict(verification.diagnostics or {}),
                proposed_text=str(proposed.text or "").strip() if proposed else "",
                intent=_intent_for_fallback,
            )
            return RespondPhaseResult(
                response_text=response_text,
                requires_follow_up=True,
                metadata={
                    "fallback": "verification_guard",
                    "reason": str(verification.reason or ""),
                },
            )

        return RespondPhaseResult(
            response_text=str(proposed.text or "").strip(),
            requires_follow_up=bool(proposed.requires_follow_up),
            metadata={
                **dict(proposed.metadata or {}),
                "type": str((proposed.metadata or {}).get("type") or "response"),
                "follow_up_question": str(proposed.follow_up_question or ""),
            },
        )


def _contract_pipeline_enabled(alice: Any) -> bool:
    raw_disable = str(os.getenv("ALICE_DISABLE_CONTRACT_PIPELINE", "")).strip().lower()
    if raw_disable in {"1", "true", "yes", "on"}:
        return False
    config = getattr(alice, "runtime_mode_config", None)
    if config is not None and hasattr(config, "enable_contract_pipeline"):
        return bool(getattr(config, "enable_contract_pipeline"))
    return True


def run_default_turn(alice: Any, user_input: str, use_voice: bool = False) -> str:
    """Default app turn entrypoint.

    The active path is the contract pipeline. Legacy inline orchestration is only
    a compatibility fallback when the pipeline is unavailable or explicitly
    disabled for diagnostics.
    """

    if not hasattr(alice, "structured_logger") and callable(getattr(alice, "_process_input_internal", None)):
        return sanitize_internal_process_output(alice, user_input=user_input, use_voice=use_voice)

    # Tool chaining — compound queries execute multiple plugins before the pipeline
    plugin_manager = getattr(alice, "plugins", None)
    if plugin_manager is not None:
        try:
            from ai.runtime.tool_chain import detect_chain, execute_chain

            chain_intents = detect_chain(query=user_input, primary_intent="")
            if chain_intents:
                entities: dict = {}
                ctx: dict = {
                    "user_id": str(getattr(alice, "user_name", "") or "User"),
                    "turn_number": int(getattr(alice, "_turn_count", 0) or 0),
                }
                chain_response = execute_chain(
                    intents=chain_intents,
                    query=user_input,
                    entities=entities,
                    context=ctx,
                    plugin_manager=plugin_manager,
                )
                if chain_response:
                    if use_voice and getattr(alice, "speech", None):
                        alice.speech.speak(chain_response, blocking=False)
                    return chain_response
        except Exception:
            pass

    pipeline = getattr(alice, "contract_pipeline", None)
    if pipeline is not None and _contract_pipeline_enabled(alice):
        try:
            result = pipeline.run_turn(
                user_input=user_input,
                user_id=str(getattr(alice, "user_name", "") or "User"),
                turn_number=int(getattr(alice, "_turn_count", 0) or 0),
            )
            if result and getattr(result, "handled", False) and getattr(result, "response_text", ""):
                meta = dict(getattr(result, "metadata", {}) or {})
                structured_logger = getattr(alice, "structured_logger", None)
                if structured_logger is not None:
                    try:
                        structured_logger.info(
                            "Canonical pipeline handled request",
                            component="pipeline",
                            trace_id=str(meta.get("trace_id") or ""),
                            route=str(meta.get("route") or ""),
                            intent=str(meta.get("intent") or ""),
                            verification_reason=str(((meta.get("verification") or {}).get("reason")) or ""),
                        )
                    except Exception:
                        pass
                response = str(result.response_text or "")
                if use_voice and getattr(alice, "speech", None):
                    alice.speech.speak(response, blocking=False)
                return response
            logger.debug("[ContractPipeline] Unhandled turn; falling back to legacy")
        except Exception as exc:
            logger.debug("[ContractPipeline] Legacy fallback due to: %s", exc)

    legacy = getattr(alice, "_process_input_legacy", None)
    if callable(legacy):
        return str(legacy(user_input, use_voice=use_voice) or "")

    internal = getattr(alice, "_process_input_internal", None)
    if callable(internal):
        return sanitize_internal_process_output(alice, user_input=user_input, use_voice=use_voice)

    raise RuntimeError("No active turn pipeline is configured.")
