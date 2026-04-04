"""
LLM Gateway - Single Entry Point for All LLM Calls

This gateway enforces:
- LLM policy checks before every call
- Simple formatters tried before LLM generation
- Rate limiting and budget tracking
- Automatic logging to learning engine
- User approval for non-essential calls

All code should call LLMGateway.request() instead of llm.chat() directly.
"""

import logging
import json
import os
from typing import Optional, Dict, Any, Callable
from datetime import datetime
from dataclasses import dataclass

from ai.core.llm_policy import get_llm_policy, LLMCallType
from ai.models.simple_formatters import FormatterRegistry
from ai.learning.data_redaction import sanitize_for_learning, redact_text

try:
    from brain.model_router import ModelRouter
except Exception:  # pragma: no cover - optional runtime feature
    ModelRouter = None

logger = logging.getLogger(__name__)


# Path to logged interactions file
LOGGED_INTERACTIONS_PATH = "data/training/logged_interactions.jsonl"


@dataclass
class LLMRequest:
    """Request to the LLM gateway"""

    prompt: str
    call_type: LLMCallType
    use_history: bool = False
    context: Optional[Dict[str, Any]] = None
    user_input: str = ""
    tool_name: Optional[str] = None
    tool_data: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Response from the LLM gateway"""

    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    used_llm: bool = False
    used_formatter: bool = False
    formatter_name: Optional[str] = None
    denied_by_policy: bool = False
    policy_reason: Optional[str] = None
    model_used: Optional[str] = None
    route_source: Optional[str] = None


class LLMGateway:
    """
    Single gateway for all LLM access

    Responsibilities:
    1. Check LLM policy before every call
    2. Try simple formatters first for tool outputs
    3. Enforce rate limits and budgets
    4. Log all calls for learning
    5. Provide fallback messages when LLM denied
    """

    def __init__(self, llm_engine, learning_engine=None):
        """
        Initialize gateway

        Args:
            llm_engine: LocalLLMEngine instance
            learning_engine: Optional learning engine for logging
        """
        self.llm = llm_engine
        self.learning_engine = learning_engine
        self.policy = get_llm_policy()
        self.formatter_registry = FormatterRegistry()
        self.model_router = None
        self.multi_llm_enabled = os.getenv(
            "ALICE_MULTI_LLM_ROUTER", "1"
        ).strip() not in {
            "0",
            "false",
            "off",
            "no",
        }
        self.strict_generation_router = os.getenv(
            "ALICE_MULTI_LLM_STRICT_GENERATION", "1"
        ).strip().lower() not in {
            "0",
            "false",
            "off",
            "no",
        }
        if self.multi_llm_enabled and ModelRouter is not None:
            try:
                self.model_router = ModelRouter()
            except Exception as exc:
                logger.debug("[LLMGateway] Multi-LLM router unavailable: %s", exc)
        self._last_route: Dict[str, Any] = {}

        # Advanced telemetry
        self.stats = {
            "total_requests": 0,
            "self_handlers": 0,
            "pattern_hits": 0,
            "tool_calls": 0,
            "rag_lookups": 0,
            "llm_calls": 0,
            "formatter_calls": 0,
            "multi_router_calls": 0,
            "policy_denials": 0,
            "by_type": {},
            "recent_requests": [],  # Last 100 requests for analysis
        }

        logger.info(
            "[LLMGateway] Initialized - All LLM calls now gated with advanced telemetry"
        )

    def request(
        self,
        prompt: str,
        call_type: LLMCallType,
        use_history: bool = False,
        context: Optional[Dict[str, Any]] = None,
        user_input: str = "",
        tool_name: Optional[str] = None,
        tool_data: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        """
        Request LLM generation (with policy enforcement)

        Args:
            prompt: LLM prompt
            call_type: Type of call (CHITCHAT, TOOL_FORMATTING, GENERATION, etc.)
            use_history: Whether to use conversation history
            context: Additional context
            user_input: Original user input
            tool_name: Name of tool if formatting tool output
            tool_data: Tool output data if formatting

        Returns:
            LLMResponse with result or denial reason
        """
        self.stats["total_requests"] += 1

        # Track by type
        type_key = call_type.value
        self.stats["by_type"][type_key] = self.stats["by_type"].get(type_key, 0) + 1

        # Step 1: Try formatter first for tool outputs
        if call_type == LLMCallType.TOOL_FORMATTING and tool_name and tool_data:
            formatter_result = self._try_formatter(tool_name, tool_data, context or {})
            if formatter_result:
                self.stats["formatter_calls"] += 1
                logger.info(f"[LLMGateway] [OK] Formatted {tool_name} without LLM")
                return LLMResponse(
                    success=True,
                    response=formatter_result,
                    used_formatter=True,
                    formatter_name=tool_name,
                )

        # Step 2: Check LLM policy
        allowed, reason = self.policy.can_call_llm(call_type, user_input)

        if not allowed:
            self.stats["policy_denials"] += 1
            logger.warning(f"[LLMGateway] [DENIED] LLM call denied: {reason}")

            # Do not return hardcoded user-facing text here.
            # Upstream layers should decide how to proceed (learned phrasing, formatter, retry, etc.).
            return LLMResponse(
                success=False,
                response=None,
                denied_by_policy=True,
                policy_reason=reason,
            )

        # Step 3: Route to appropriate LLM method based on call type
        try:
            logger.info(f"[LLMGateway] [CALL] LLM call ({call_type.value})")

            if (
                self._should_use_multi_router(call_type)
                and self.model_router is not None
            ):
                routed = self.model_router.generate(
                    request=prompt or user_input,
                    context={
                        "intent": (context or {}).get("intent", ""),
                        "call_type": call_type.value,
                    },
                )
                if (
                    routed.get("response")
                    and float(routed.get("confidence", 0.0)) > 0.0
                ):
                    response = str(routed.get("response"))
                    model_used = str(routed.get("model") or "")
                    self.stats["multi_router_calls"] += 1
                    self.stats["llm_calls"] += 1
                    self.policy.record_call(call_type, user_input, response)
                    self._last_route = {
                        "source": "multi_router",
                        "call_type": call_type.value,
                        "model": model_used,
                        "role": (
                            getattr(self.model_router, "last_route", {}) or {}
                        ).get("role", ""),
                    }
                    return LLMResponse(
                        success=True,
                        response=response,
                        used_llm=True,
                        model_used=model_used,
                        route_source="multi_router",
                    )
                if call_type == LLMCallType.GENERATION and self.strict_generation_router:
                    err = str(routed.get("response") or "Multi-router generation failed")
                    self._last_route = {
                        "source": "multi_router",
                        "call_type": call_type.value,
                        "model": str(routed.get("model") or ""),
                        "role": (getattr(self.model_router, "last_route", {}) or {}).get("role", ""),
                    }
                    return LLMResponse(
                        success=False,
                        error=err,
                        response=err,
                        denied_by_policy=True,
                        policy_reason="strict_generation_router",
                        route_source="multi_router",
                        model_used=str(routed.get("model") or ""),
                    )

            # Tool-based routing: Alice uses Ollama as a tool
            if call_type == LLMCallType.QUERY_KNOWLEDGE:
                # Alice asks Ollama for factual knowledge
                question = prompt if prompt else user_input
                response = self.llm.query_knowledge(question)

            elif call_type == LLMCallType.PARSE_INPUT:
                # Alice asks Ollama to parse complex input
                input_to_parse = (
                    context.get("input_to_parse", user_input) if context else user_input
                )
                parsed_result = self.llm.parse_complex_input(input_to_parse)
                response = json.dumps(
                    parsed_result, indent=2
                )  # Return as formatted JSON

            elif call_type == LLMCallType.PHRASE_RESPONSE:
                # Alice asks Ollama to phrase her structured thought
                alice_thought = (
                    context.get("alice_thought", prompt) if context else prompt
                )
                tone = (
                    context.get("tone", "warm and helpful")
                    if context
                    else "warm and helpful"
                )
                phrasing_context = {
                    "user_name": (
                        context.get("user_name", "the user") if context else "the user"
                    )
                }
                response = self.llm.phrase_with_tone(
                    alice_thought, tone, phrasing_context
                )

            elif call_type == LLMCallType.PHRASE_MICRO:
                source_text = str(
                    context.get("alice_thought", prompt) if context else prompt
                ).strip()
                tone = (
                    context.get("tone", "warm and helpful")
                    if context
                    else "warm and helpful"
                )
                micro_prompt = (
                    "Polish this text only. Keep meaning unchanged. "
                    "No new facts. Keep it under 26 words.\n\n"
                    f"Text: {source_text}"
                )
                response = self.llm.phrase_with_tone(
                    micro_prompt,
                    tone,
                    {"user_name": "", "allow_user_name": False},
                )

            elif call_type == LLMCallType.PHRASE_STRUCTURED:
                payload = (
                    context.get("structured_payload", prompt) if context else prompt
                )
                tone = (
                    context.get("tone", "warm and helpful")
                    if context
                    else "warm and helpful"
                )
                rewrite_prompt = (
                    "Rewrite this structured payload into one concise user-facing reply. "
                    "Use only payload facts. Do not add extra details.\n\n"
                    f"Payload: {payload}"
                )
                response = self.llm.phrase_with_tone(
                    rewrite_prompt,
                    tone,
                    {"user_name": "", "allow_user_name": False},
                )

            elif call_type == LLMCallType.AUDIT_LOGIC:
                # Alice asks Ollama to verify her reasoning
                logic_chain = (
                    context.get("logic_chain", [prompt]) if context else [prompt]
                )
                if not isinstance(logic_chain, list):
                    logic_chain = [logic_chain]
                audit_result = self.llm.audit_logic(logic_chain)
                response = json.dumps(
                    audit_result, indent=2
                )  # Return as formatted JSON

            elif call_type == LLMCallType.GENERATION:
                response = self._generation_last_resort(
                    prompt=prompt,
                    user_input=user_input,
                    use_history=use_history,
                    context=context,
                )

            else:
                # Legacy call types - use old chat() method
                response = self.llm.chat(prompt, use_history=use_history)

            # Record successful call
            self.policy.record_call(call_type, user_input, response)
            self.stats["llm_calls"] += 1

            # Log LLM fallback to JSONL file for training
            if call_type == LLMCallType.FALLBACK:
                self._log_llm_fallback(
                    user_input=user_input,
                    intent=context.get("intent", "unknown") if context else "unknown",
                    entities=context.get("entities", {}) if context else {},
                    context_snapshot=context or {},
                    llm_response=response,
                )

            # Log to learning engine if available
            if self.learning_engine and user_input:
                self.learning_engine.collect_interaction(
                    user_input=user_input,
                    assistant_response=response,
                    intent=call_type.value,
                    quality_score=0.8,  # Default quality for LLM responses
                )

            logger.info(f"[LLMGateway] [OK] LLM responded ({len(response)} chars)")
            model_used = self._active_model_name()
            self._last_route = {
                "source": "legacy_engine",
                "call_type": call_type.value,
                "model": model_used,
                "role": "",
            }
            return LLMResponse(
                success=True,
                response=response,
                used_llm=True,
                model_used=model_used,
                route_source="legacy_engine",
            )

        except Exception as e:
            logger.error(f"[LLMGateway] [ERROR] LLM error: {e}")
            return LLMResponse(
                success=False,
                error=str(e),
                response="I encountered an error processing that request.",
            )

    @staticmethod
    def _should_use_multi_router(call_type: LLMCallType) -> bool:
        """Only route user-facing generation paths through multi-LLM selector."""
        return call_type in {
            LLMCallType.CHITCHAT,
            LLMCallType.QUERY_KNOWLEDGE,
            LLMCallType.GENERATION,
        }

    def _active_model_name(self) -> str:
        cfg = getattr(self.llm, "config", None)
        if cfg is None:
            return "unknown"
        return str(
            getattr(cfg, "active_model", None) or getattr(cfg, "model", "unknown")
        )

    def _try_formatter(
        self, tool_name: str, data: Dict[str, Any], context: Dict[str, Any]
    ) -> Optional[str]:
        """
        Try to format tool output without LLM

        Args:
            tool_name: Name of the tool
            data: Tool output data
            context: Additional context

        Returns:
            Formatted string or None if no formatter available
        """
        try:
            return self.formatter_registry.format(tool_name, data, **context)
        except Exception as e:
            logger.debug(f"[LLMGateway] Formatter failed for {tool_name}: {e}")
            return None

    def _log_llm_fallback(
        self,
        user_input: str,
        intent: str,
        entities: Dict[str, Any],
        context_snapshot: Dict[str, Any],
        llm_response: str,
    ):
        """
        Log LLM fallback calls to JSONL file for pattern mining

        This data will be used for identifying patterns to learn,
        PEFT/instruction tuning, and understanding knowledge gaps.

        Args:
            user_input: User input text
            intent: Detected intent
            entities: Extracted entities
            context_snapshot: Current context
            llm_response: LLM response text
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(LOGGED_INTERACTIONS_PATH), exist_ok=True)

            # Create log entry
            safe_context = sanitize_for_learning(context_snapshot or {})
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_input": redact_text(user_input or ""),
                "intent": intent,
                "entities": sanitize_for_learning(entities or {}),
                "context": {
                    k: v
                    for k, v in safe_context.items()
                    if k not in ["llm_engine", "memory_system", "plugin_manager"]
                },
                "llm_response": redact_text(llm_response or ""),
                "call_type": "LLM_FALLBACK",
            }

            # Append to JSONL file
            with open(LOGGED_INTERACTIONS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            logger.debug(f"[LLMGateway] Logged LLM fallback: {user_input[:50]}...")

        except Exception as e:
            logger.error(f"[LLMGateway] Failed to log LLM fallback: {e}")

    def format_tool_result(
        self,
        tool_name: str,
        data: Dict[str, Any],
        user_input: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Format tool output (tries formatter first, falls back to LLM if allowed)

        Args:
            tool_name: Name of the tool
            data: Tool output data
            user_input: Original user input
            context: Additional context

        Returns:
            Formatted response string
        """
        # Check if the question requires contextual interpretation
        needs_llm_interpretation = self._needs_contextual_answer(user_input, tool_name)

        # Try formatter first (unless question clearly needs LLM interpretation)
        if not needs_llm_interpretation:
            formatted = self._try_formatter(tool_name, data, context or {})
            if formatted:
                return formatted

        # Formatter failed or question needs interpretation - try LLM if policy allows
        prompt = self._build_tool_format_prompt(tool_name, data, user_input)

        response = self.request(
            prompt=prompt,
            call_type=LLMCallType.TOOL_FORMATTING,
            use_history=False,
            user_input=user_input,
            tool_name=tool_name,
            tool_data=data,
        )

        if response.success and response.response:
            return response.response

        # LLM failed - try formatter as last resort
        if needs_llm_interpretation:
            formatted = self._try_formatter(tool_name, data, context or {})
            if formatted:
                return formatted

        # Both failed - return raw data
        logger.warning(f"[LLMGateway] Both formatter and LLM failed for {tool_name}")
        return f"Result: {str(data)[:500]}"

    def _needs_contextual_answer(self, user_input: str, tool_name: str) -> bool:
        """Check if the question requires LLM interpretation beyond simple formatting"""
        if not user_input:
            return False

        user_lower = user_input.lower()

        # Questions asking for advice/recommendations based on data
        advice_keywords = [
            "should i",
            "should we",
            "do i need",
            "do we need",
            "would you recommend",
            "is it good",
            "is it bad",
            "what should",
            "what would you",
            "recommend",
            "advise",
            "suggest",
            "think i should",
        ]

        if any(keyword in user_lower for keyword in advice_keywords):
            return True

        # Weather-specific: clothing/activity questions
        if tool_name == "weather":
            weather_advice_keywords = [
                "wear",
                "bring",
                "jacket",
                "umbrella",
                "coat",
                "layer",
                "dress",
                "clothes",
                "clothing",
                "go outside",
                "go for",
                "safe to",
                "good for",
            ]
            if any(keyword in user_lower for keyword in weather_advice_keywords):
                return True

        return False

    def _build_tool_format_prompt(
        self, tool_name: str, data: Dict[str, Any], user_input: str
    ) -> str:
        """Build LLM prompt for formatting tool output"""
        return f"""The user asked: "{user_input}"

The {tool_name} tool returned this data:
{data}

Please provide a natural, concise response to the user based on this data.
Be conversational and helpful. Do not mention the tool name or technical details."""

    def _generation_last_resort(
        self,
        *,
        prompt: str,
        user_input: str,
        use_history: bool,
        context: Optional[Dict[str, Any]],
    ) -> str:
        """Attempt structured assist paths before broad generation."""
        ctx = dict(context or {})
        base_text = str(user_input or prompt or "").strip()

        # 1) Parse assist first for ambiguous/complex language.
        try:
            parsed = self.llm.parse_complex_input(base_text)
            if isinstance(parsed, dict) and parsed:
                ctx["parsed"] = parsed
        except Exception:
            pass

        # 2) Knowledge assist for direct question-like prompts.
        if base_text.endswith("?") or any(
            q in base_text.lower() for q in ("what", "why", "how", "when", "where")
        ):
            try:
                knowledge = str(self.llm.query_knowledge(base_text) or "").strip()
                if knowledge:
                    return knowledge
            except Exception:
                pass

        # 3) Audit assist to tighten logic before full generation.
        try:
            audit = self.llm.audit_logic([base_text, json.dumps(ctx, default=str)])
            if isinstance(audit, dict) and not bool(audit.get("has_errors", False)):
                suggestion = str(audit.get("suggested_response") or "").strip()
                if suggestion:
                    return suggestion
        except Exception:
            pass

        # 4) Last resort: broad generation.
        return self.llm.chat(prompt, use_history=use_history)

    def get_statistics(self) -> Dict[str, Any]:
        """Get advanced gateway statistics with detailed routing breakdown"""
        total = self.stats["total_requests"]
        if total == 0:
            return self.stats

        stats = {
            **self.stats,
            "self_handler_percentage": round(
                100 * self.stats["self_handlers"] / total, 1
            ),
            "pattern_hit_percentage": round(
                100 * self.stats["pattern_hits"] / total, 1
            ),
            "tool_call_percentage": round(100 * self.stats["tool_calls"] / total, 1),
            "rag_lookup_percentage": round(100 * self.stats["rag_lookups"] / total, 1),
            "llm_fallback_percentage": round(100 * self.stats["llm_calls"] / total, 1),
            "formatter_percentage": round(
                100 * self.stats["formatter_calls"] / total, 1
            ),
            "multi_router_percentage": round(
                100 * self.stats["multi_router_calls"] / total, 1
            ),
            "denial_percentage": round(100 * self.stats["policy_denials"] / total, 1),
            "multi_llm_enabled": bool(
                self.multi_llm_enabled and self.model_router is not None
            ),
            "strict_generation_router": bool(self.strict_generation_router),
            "model_roles": (
                self.model_router.describe_models()
                if self.model_router is not None
                and hasattr(self.model_router, "describe_models")
                else {}
            ),
            "model_runtime_status": (
                self.model_router.runtime_status()
                if self.model_router is not None
                and hasattr(self.model_router, "runtime_status")
                else {}
            ),
            "last_route": dict(self._last_route or {}),
        }
        return stats

    def record_self_handler(self):
        """Record request handled by self-handler (code execution, commands, etc)"""
        self.stats["self_handlers"] += 1
        self.stats["total_requests"] += 1

    def record_pattern_hit(self):
        """Record that conversational engine used learned pattern"""
        self.stats["pattern_hits"] += 1
        self.stats["total_requests"] += 1

    def record_tool_call(self):
        """Record that request was handled by tool/plugin"""
        self.stats["tool_calls"] += 1
        self.stats["total_requests"] += 1

    def record_rag_lookup(self):
        """Record that request was answered via RAG/memory lookup"""
        self.stats["rag_lookups"] += 1
        self.stats["total_requests"] += 1

    def reset_statistics(self):
        """Reset gateway statistics"""
        self.stats = {
            "total_requests": 0,
            "self_handlers": 0,
            "pattern_hits": 0,
            "tool_calls": 0,
            "rag_lookups": 0,
            "llm_calls": 0,
            "formatter_calls": 0,
            "multi_router_calls": 0,
            "policy_denials": 0,
            "by_type": {},
            "recent_requests": [],
        }


# Singleton instance
_gateway_instance: Optional[LLMGateway] = None


def get_llm_gateway(llm_engine=None, learning_engine=None) -> LLMGateway:
    """Get or create singleton gateway instance"""
    global _gateway_instance

    if _gateway_instance is None:
        if llm_engine is None:
            raise ValueError("llm_engine required for first gateway initialization")
        _gateway_instance = LLMGateway(llm_engine, learning_engine)

    return _gateway_instance


def reset_gateway():
    """Reset gateway singleton (for testing)"""
    global _gateway_instance
    _gateway_instance = None
