"""Factory for wiring ALICE runtime components to hard boundary contracts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

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

    _code_path_pattern = re.compile(
        r"(?<![a-z0-9_./\\-])([a-zA-Z0-9_./\\-]+\.py)(?![a-z0-9_./\\-])",
        re.IGNORECASE,
    )
    _directory_claim_pattern = re.compile(
        r"\b([a-zA-Z0-9_./\\-]{2,})\s+directory\b",
        re.IGNORECASE,
    )
    _generic_directory_terms = {
        "all",
        "these",
        "those",
        "this",
        "that",
        "my",
        "your",
        "the",
        "internal",
        "source",
        "local",
    }
    _weather_fact_patterns = (
        re.compile(r"\b-?\d{1,2}(?:\.\d+)?\s*°\s*[cf]\b", re.IGNORECASE),
        re.compile(r"\b(?:high|low)\s+(?:of\s+)?-?\d{1,2}(?:\.\d+)?\s*°?\s*[cf]?\b", re.IGNORECASE),
        re.compile(r"\b(?:wind|breeze)[^\n]{0,24}\b\d{1,3}\s*(?:km/h|kph|mph|m/s)\b", re.IGNORECASE),
        re.compile(r"\bhumidity[^\n]{0,16}\b\d{1,3}%\b", re.IGNORECASE),
    )

    def _normalize_path(path_text: str) -> str:
        return (
            str(path_text or "")
            .strip()
            .strip("`'\"")
            .replace("\\", "/")
            .lstrip("./")
        )

    def _extract_code_path_claims(text: str) -> List[str]:
        claims: List[str] = []
        for match in _code_path_pattern.finditer(str(text or "")):
            claim = _normalize_path(match.group(1))
            if claim and claim not in claims:
                claims.append(claim)
        return claims

    def _extract_directory_claims(text: str) -> List[str]:
        claims: List[str] = []
        for match in _directory_claim_pattern.finditer(str(text or "")):
            candidate = _normalize_path(match.group(1))
            if not candidate:
                continue
            first_token = candidate.split("/", 1)[0].lower()
            if first_token in _generic_directory_terms:
                continue
            if candidate not in claims:
                claims.append(candidate)
        return claims

    def _looks_like_code_request(user_input: str) -> bool:
        detector = getattr(alice, "_is_code_access_or_listing_request", None)
        if callable(detector):
            try:
                return bool(detector(user_input))
            except Exception:
                pass

        text = str(user_input or "").lower().strip()
        if not text:
            return False

        text = text.replace("acess", "access")
        has_py_target = bool(_code_path_pattern.search(text))
        has_scope = has_py_target or any(
            phrase in text
            for phrase in (
                "your code",
                "your codebase",
                "internal code",
                "own code",
                "source code",
                "local code",
                "codebase",
                "repository",
                "repo",
                "project files",
                "source files",
            )
        )
        if not has_scope:
            return False

        has_action = bool(
            re.search(
                r"\b(?:check|inspect|access|read|view|show|list|search|find|analy[sz]e|summari[sz]e|open|look)\b",
                text,
            )
        )
        has_question_or_command = bool(
            re.search(r"\b(?:can|could|would|do|are)\s+you\b", text)
            or text.endswith("?")
            or text.startswith(
                (
                    "show",
                    "list",
                    "read",
                    "find",
                    "search",
                    "summarize",
                    "analyze",
                    "check",
                    "inspect",
                )
            )
        )
        return bool(has_action and has_question_or_command)

    def _looks_like_weather_request(user_input: str) -> bool:
        text = str(user_input or "").lower().strip()
        if not text:
            return False

        strong_signal_detector = getattr(alice, "_has_strong_weather_signal", None)
        if callable(strong_signal_detector):
            try:
                if bool(strong_signal_detector(user_input)):
                    return True
            except Exception:
                pass

        weather_scope_terms = (
            "weather",
            "forecast",
            "temperature",
            "rain",
            "snow",
            "wind",
            "humidity",
            "outside",
            "hot",
            "cold",
            "umbrella",
            "coat",
            "jacket",
        )
        if not any(re.search(r"\b" + re.escape(term) + r"\b", text) for term in weather_scope_terms):
            return False

        has_query_shape = bool(
            re.search(r"\b(?:can|could|should|will|would|is|are|do|does)\b", text)
            or text.endswith("?")
            or any(
                phrase in text
                for phrase in (
                    "what's the weather",
                    "what is the weather",
                    "weather forecast",
                    "should i wear",
                )
            )
        )
        return has_query_shape

    def _infer_weather_intent(user_input: str) -> str:
        text = str(user_input or "").lower().strip()
        if any(
            cue in text
            for cue in (
                "forecast",
                "this week",
                "next week",
                "weekend",
                "tomorrow",
                "tonight",
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            )
        ):
            return "weather:forecast"
        return "weather:current"

    def _looks_like_proactive_agent_design_statement(user_input: str) -> bool:
        text = str(user_input or "").lower().strip()
        if len(text) < 60:
            return False

        proactive_terms = (
            "always running",
            "always on",
            "always checking",
            "monitor",
            "monitoring",
            "surroundings",
            "alerts",
            "attention",
            "recommendation",
            "proactive",
            "background",
        )
        has_proactive_signal = any(term in text for term in proactive_terms)
        if not has_proactive_signal:
            return False

        assistant_scope_terms = (
            "assistant",
            "ai",
            "computer",
            "user",
        )
        has_scope = any(term in text for term in assistant_scope_terms)
        if not has_scope:
            return False

        framing_terms = (
            "i was thinking",
            "it is",
            "it only",
            "should",
            "would",
            "could",
        )
        return any(term in text for term in framing_terms)

    def _looks_like_current_events_request(user_input: str) -> bool:
        detector = getattr(alice, "_is_freshness_sensitive_current_events_request", None)
        if callable(detector):
            try:
                return bool(detector(user_input))
            except Exception:
                pass

        text = str(user_input or "").lower().strip()
        if not text:
            return False

        local_live_domains = (
            "weather",
            "forecast",
            "temperature",
            "calendar",
            "schedule",
            "email",
            "inbox",
            "reminder",
            "time",
            "date",
            "location",
            "where am i",
        )
        if any(term in text for term in local_live_domains):
            return False
        if "today's technology" in text or "todays technology" in text:
            return False

        freshness_markers = (
            "current",
            "right now",
            "now",
            "latest",
            "recent",
            "today",
            "this week",
            "these days",
            "at the moment",
            "happening",
            "going on",
        )
        world_markers = (
            "world",
            "global",
            "news",
            "current events",
            "economy",
            "markets",
            "market",
            "geopolitics",
            "politics",
            "war",
            "conflict",
            "inflation",
            "interest rates",
            "central banks",
            "crypto",
            "cryptocurrency",
            "supply chains",
        )
        broad_situation = bool(
            re.search(
                r"\b(?:what(?:'s|\s+is)|tell\s+me|update\s+me|brief\s+me)\b.*"
                r"\b(?:happening|going\s+on|situation|state\s+of)\b.*"
                r"\b(?:world|global|news|economy|markets?|geopolitics|politics)\b",
                text,
            )
        )
        return bool(
            (
                any(marker in text for marker in freshness_markers)
                and any(marker in text for marker in world_markers)
            )
            or broad_situation
        )

    def _freshness_required_payload(user_input: str) -> Dict[str, Any]:
        builder = getattr(alice, "_freshness_required_payload", None)
        if callable(builder):
            try:
                payload = builder(user_input)
                if isinstance(payload, dict) and payload:
                    return dict(payload)
            except Exception:
                pass
        return {
            "domain": "current events",
            "source_requirement": "live sources",
            "blocked_source": "model memory",
            "search_dimensions": ["topic", "region", "market"],
            "user_input": str(user_input or "").strip(),
        }

    def _formulate_freshness_guard_response(user_input: str) -> str:
        responder = getattr(alice, "_formulate_freshness_guard_response", None)
        if callable(responder):
            try:
                response = str(responder(user_input) or "").strip()
                if response:
                    return response
            except Exception:
                pass

        from ai.core.response_formulator import (
            ReasoningOutput,
            ResponseFormulator,
            UserResponse,
        )

        payload = _freshness_required_payload(user_input)
        formulator = getattr(alice, "response_formulator", None) or ResponseFormulator()
        final_payload = formulator.generate(
            intent="freshness:current_events",
            context={
                "user_input": str(user_input or "").strip(),
                "freshness_payload": payload,
                "response": "",
            },
            tool_results={
                "plugin": "FreshnessGuard",
                "action": "freshness_required",
                "data": payload,
            },
            reasoning_output=ReasoningOutput(
                internal_summary="freshness boundary enforced for current-events request",
                intent="freshness:current_events",
                plan=["require live-source grounding before factual claims"],
                confidence=0.98,
            ),
            mode="final_answer_only",
        )
        if isinstance(final_payload, UserResponse):
            return str(final_payload.message or "").strip()
        return str(final_payload or "").strip()

    def _verify_codebase_claims(response_text: str) -> Dict[str, Any]:
        text = str(response_text or "")
        low = text.lower()
        if not any(
            marker in low
            for marker in (
                "codebase",
                "source code",
                "python files",
                "directory",
                "directories",
                "local code",
                "read-only access",
            )
        ):
            return {}

        self_reflection = getattr(alice, "self_reflection", None)
        if self_reflection is None:
            return {}

        try:
            files = list(self_reflection.list_codebase() or [])
        except Exception:
            return {}

        available_paths = {
            _normalize_path(str(entry.get("path") or ""))
            for entry in files
            if str(entry.get("path") or "").strip()
        }
        if not available_paths:
            return {}

        available_paths_lower = {path.lower() for path in available_paths}
        available_names = {Path(path).name.lower() for path in available_paths}
        top_level_dirs = {
            path.split("/", 1)[0].lower() for path in available_paths_lower if path
        }

        def _path_exists(candidate: str) -> bool:
            normalized = _normalize_path(candidate).lower()
            if not normalized:
                return False
            if normalized in available_paths_lower:
                return True
            base_name = Path(normalized).name.lower()
            if base_name in available_names:
                return True
            return any(
                known.endswith(normalized) or normalized.endswith(known)
                for known in available_paths_lower
            )

        missing_paths: List[str] = []
        for claimed_path in _extract_code_path_claims(text):
            if not _path_exists(claimed_path):
                missing_paths.append(claimed_path)

        missing_directories: List[str] = []
        for claimed_dir in _extract_directory_claims(text):
            first_segment = _normalize_path(claimed_dir).split("/", 1)[0].lower()
            if first_segment and first_segment not in top_level_dirs:
                missing_directories.append(claimed_dir)

        if not missing_paths and not missing_directories:
            return {}

        return {
            "stage": "code_claim_validation",
            "missing_paths": missing_paths,
            "missing_directories": missing_directories,
            "available_file_count": len(available_paths),
        }

    def _verify_weather_claims(*, user_input: str, response_text: str) -> Dict[str, Any]:
        text = str(response_text or "").strip()
        low = text.lower()
        if not text:
            return {}

        has_weather_language = any(
            token in low
            for token in (
                "weather",
                "forecast",
                "temperature",
                "high",
                "low",
                "wind",
                "humidity",
                "rain",
                "snow",
                "cloudy",
                "sunny",
                "breeze",
            )
        )
        if not has_weather_language:
            return {}

        has_fact_pattern = any(pattern.search(text) for pattern in _weather_fact_patterns)
        if not has_fact_pattern and "forecast" not in low:
            return {}

        has_live_data_disclaimer = any(
            phrase in low
            for phrase in (
                "can't access live weather",
                "cannot access live weather",
                "don't have live weather",
                "do not have live weather",
                "couldn't verify weather",
                "could not verify weather",
                "share your location",
                "tell me your location",
            )
        )
        if has_live_data_disclaimer:
            return {}

        if not _looks_like_weather_request(user_input) and "according to my knowledge" not in low:
            return {}

        placeholder_markers = []
        if "[your location]" in low:
            placeholder_markers.append("[your location]")
        if "<location>" in low:
            placeholder_markers.append("<location>")

        return {
            "stage": "weather_claim_validation",
            "placeholder_markers": placeholder_markers,
            "contains_numeric_weather_claims": bool(has_fact_pattern),
        }

    def _compose_weather_response_from_tool_payload(
        *, tool_payload: Dict[str, Any], user_input: str
    ) -> str:
        payload = dict(tool_payload or {})
        nested = payload.get("data") if isinstance(payload.get("data"), dict) else {}
        if not nested:
            return ""

        message_code = str(nested.get("message_code") or "").lower()
        plugin_type = str(nested.get("plugin_type") or "").lower()
        is_weather_payload = (
            "weather:" in message_code
            or plugin_type == "weather"
            or ("temperature" in nested and "condition" in nested)
            or isinstance(nested.get("forecast"), list)
        )
        if not is_weather_payload:
            return ""

        if "forecast" in message_code or isinstance(nested.get("forecast"), list):
            forecast_payload = {
                "forecast": list(nested.get("forecast") or []),
                "location": str(nested.get("location") or ""),
                "user_input": str(user_input or ""),
            }
            if hasattr(alice, "_alice_direct_phrase"):
                try:
                    phrased = str(
                        alice._alice_direct_phrase("weather_forecast", forecast_payload)
                        or ""
                    ).strip()
                    if phrased:
                        return phrased
                except Exception:
                    pass

            if forecast_payload["forecast"]:
                first_day = dict(forecast_payload["forecast"][0] or {})
                cond = str(first_day.get("condition") or "conditions unavailable")
                high = first_day.get("high")
                low = first_day.get("low")
                location = forecast_payload["location"]
                loc = f" in {location}" if location else ""
                if high is not None and low is not None:
                    return f"Forecast{loc}: {cond}, {low}° to {high}°C."
                return f"Forecast{loc}: {cond}."
            return ""

        report_payload = {
            "temperature": nested.get("temperature"),
            "condition": nested.get("condition"),
            "location": nested.get("location"),
            "is_followup": False,
        }
        if hasattr(alice, "_alice_direct_phrase"):
            try:
                phrased = str(
                    alice._alice_direct_phrase("weather_report", report_payload) or ""
                ).strip()
                if phrased:
                    return phrased
            except Exception:
                pass

        temp = report_payload["temperature"]
        cond = str(report_payload["condition"] or "conditions unavailable")
        location = str(report_payload["location"] or "")
        loc = f" in {location}" if location else ""
        if temp is not None:
            return f"It's currently around {temp}°C with {cond}{loc}."
        return f"Current weather{loc}: {cond}."

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
                        apply_publish_style=(route != "contract_freshness_guard"),
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

        if _looks_like_code_request(req.user_input):
            return RouterDecision(
                route="local",
                intent="code:request",
                confidence=0.96,
                decision_band="execute",
                metadata={
                    "reason": "code_request_detected",
                    "resolved_input": req.user_input,
                },
            )

        if _looks_like_weather_request(req.user_input):
            weather_intent = _infer_weather_intent(req.user_input)
            return RouterDecision(
                route="tool",
                intent=weather_intent,
                confidence=0.94,
                decision_band="execute",
                metadata={
                    "reason": "weather_request_detected",
                    "resolved_input": req.user_input,
                },
            )

        if _looks_like_current_events_request(req.user_input):
            return RouterDecision(
                route="local",
                intent="freshness:current_events",
                confidence=0.98,
                decision_band="execute",
                metadata={
                    "reason": "freshness_sensitive_current_events",
                    "requires_live_sources": True,
                    "resolved_input": req.user_input,
                },
            )

        if _looks_like_proactive_agent_design_statement(req.user_input):
            return RouterDecision(
                route="llm",
                intent="conversation:goal_statement",
                confidence=0.88,
                decision_band="execute",
                metadata={
                    "reason": "proactive_agent_design_statement",
                    "resolved_input": req.user_input,
                },
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

        if req.decision.intent == "code:request" and hasattr(alice, "_handle_code_request"):
            code_response = ""
            try:
                code_response = str(
                    alice._handle_code_request(req.user_input, {}) or ""
                ).strip()
            except Exception:
                code_response = ""

            if code_response:
                return ResponseOutput(
                    text=_surface_text(
                        code_response,
                        user_input=req.user_input,
                        intent=req.decision.intent,
                        route="contract_code_request",
                    ),
                    confidence=0.95,
                    metadata={"type": "code_request"},
                )

            fallback = _surface_text(
                "I can inspect local source code in this workspace. Ask me to list files or inspect a specific file path.",
                user_input=req.user_input,
                intent=req.decision.intent,
                route="contract_code_request",
            )
            return ResponseOutput(
                text=fallback,
                confidence=0.5,
                requires_follow_up=True,
                follow_up_question=_surface_text(
                    "Do you want a file list or a summary of a specific file?",
                    user_input=req.user_input,
                    intent=req.decision.intent,
                    route="contract_code_request",
                ),
                metadata={"type": "code_request_fallback"},
            )

        if req.decision.intent == "freshness:current_events":
            payload = _freshness_required_payload(req.user_input)
            follow_up_question = _surface_text(
                _formulate_freshness_guard_response(
                    f"{req.user_input}\nrequested_focus=follow_up_slot"
                ),
                user_input=req.user_input,
                intent=req.decision.intent,
                route="contract_freshness_guard",
            )
            text = _surface_text(
                _formulate_freshness_guard_response(req.user_input),
                user_input=req.user_input,
                intent=req.decision.intent,
                route="contract_freshness_guard",
            )
            return ResponseOutput(
                text=text,
                confidence=0.99,
                requires_follow_up=True,
                follow_up_question=follow_up_question,
                metadata={
                    "type": "freshness_guard",
                    "requires_live_sources": True,
                    "freshness_payload": payload,
                    "follow_up_question": follow_up_question,
                },
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
            tool_payload = dict(req.tool_result.data or {})
            tool_response = str(tool_payload.get("response") or "").strip()
            if not tool_response:
                tool_response = _compose_weather_response_from_tool_payload(
                    tool_payload=tool_payload,
                    user_input=req.user_input,
                )
            if tool_response:
                return ResponseOutput(
                    text=_surface_text(
                        tool_response,
                        user_input=req.user_input,
                        intent=req.decision.intent,
                        route="contract_tool_response",
                    ),
                    confidence=float(req.tool_result.confidence or 0.7),
                    metadata={
                        "type": "tool_response",
                        "tool": req.tool_result.tool_name,
                    },
                )

            nested_data = (
                tool_payload.get("data") if isinstance(tool_payload.get("data"), dict) else {}
            )
            weather_tool_turn = (
                str(req.decision.intent or "").startswith("weather:")
                or str(req.tool_result.tool_name or "").lower().startswith("weather")
                or str((nested_data or {}).get("plugin_type") or "").lower() == "weather"
                or "weather:" in str((nested_data or {}).get("message_code") or "").lower()
            )
            if weather_tool_turn:
                return ResponseOutput(
                    text=_surface_text(
                        "I retrieved weather data but could not format it reliably. Ask me to retry and I will fetch it again.",
                        user_input=req.user_input,
                        intent=req.decision.intent,
                        route="contract_weather_tool_fallback",
                    ),
                    confidence=0.45,
                    requires_follow_up=True,
                    follow_up_question=_surface_text(
                        "Want me to retry the weather fetch now?",
                        user_input=req.user_input,
                        intent=req.decision.intent,
                        route="contract_weather_tool_fallback",
                    ),
                    metadata={"type": "weather_tool_fallback"},
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

        if req.decision.route == "llm":
            weather_claim_diagnostics = _verify_weather_claims(
                user_input=req.user_input,
                response_text=response_text,
            )
            if weather_claim_diagnostics:
                return VerifierResult(
                    accepted=False,
                    reason="unverified_weather_claim",
                    confidence=0.1,
                    diagnostics={
                        "route": req.decision.route,
                        "intent": req.decision.intent,
                        **weather_claim_diagnostics,
                    },
                )

            code_claim_diagnostics = _verify_codebase_claims(response_text)
            if code_claim_diagnostics:
                return VerifierResult(
                    accepted=False,
                    reason="unverified_codebase_claim",
                    confidence=0.1,
                    diagnostics={
                        "route": req.decision.route,
                        "intent": req.decision.intent,
                        **code_claim_diagnostics,
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
