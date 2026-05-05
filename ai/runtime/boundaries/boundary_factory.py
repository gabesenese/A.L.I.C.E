"""Factory for wiring ALICE runtime components to hard boundary contracts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List

from ai.core.routing.route_arbiter import RouteArbiter
from ai.core.routing.turn_segmenter import TurnSegmenter
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
from ai.memory.memory_answer_verifier import MemoryAnswerVerifier
from ai.memory.personal_memory import PersonalMemoryStore
from ai.runtime.local_action_executor import LocalActionExecutor
from ai.runtime.operator_state import update_operator_state


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
        "project",
    }
    _weather_fact_patterns = (
        re.compile(r"\b-?\d{1,2}(?:\.\d+)?\s*°\s*[cf]\b", re.IGNORECASE),
        re.compile(
            r"\b(?:high|low)\s+(?:of\s+)?-?\d{1,2}(?:\.\d+)?\s*°?\s*[cf]?\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:wind|breeze)[^\n]{0,24}\b\d{1,3}\s*(?:km/h|kph|mph|m/s)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\bhumidity[^\n]{0,16}\b\d{1,3}%\b", re.IGNORECASE),
    )
    _personal_history_patterns = (
        re.compile(r"\bwhat did i (?:talk|say|mention|share)\b", re.IGNORECASE),
        re.compile(r"\bmy personal life\b", re.IGNORECASE),
        re.compile(r"\babout me\b", re.IGNORECASE),
        re.compile(r"\bdo you remember\b", re.IGNORECASE),
        re.compile(r"\bwhat do you remember\b", re.IGNORECASE),
    )
    _personal_domain_keywords = {
        "finance": ("money", "finance", "budget", "debt", "income"),
        "fitness": ("fitness", "gym", "workout", "exercise"),
        "relationships": ("relationship", "partner", "wife", "husband", "friend"),
        "health": ("health", "sick", "illness", "medical", "pain", "sleep"),
        "work": ("work", "job", "career", "office", "manager"),
        "alice_project": ("alice", "project", "repo", "codebase", "feature"),
        "preferences": ("prefer", "preference", "like", "dislike"),
    }
    personal_memory = (
        PersonalMemoryStore(getattr(alice, "memory"))
        if getattr(alice, "memory", None)
        else None
    )
    memory_answer_verifier = MemoryAnswerVerifier()
    local_executor = LocalActionExecutor(alice)
    route_arbiter = RouteArbiter()

    def _normalize_path(path_text: str) -> str:
        return (
            str(path_text or "").strip().strip("`'\"").replace("\\", "/").lstrip("./")
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
                "alice's files",
                "alice files",
                "workspace files",
                "files alice has",
            )
        )
        if not has_scope and re.search(r"\balice['’]s\b.*\b(?:code|files)\b", text):
            has_scope = True
        if not has_scope and "alice has" in text and "file" in text:
            has_scope = True
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
            or re.search(r"\b(?:i want you to|i need you to|let[' ]?s|can we)\b", text)
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

    def _looks_like_code_list_request(user_input: str) -> bool:
        text = str(user_input or "").lower().strip()
        return bool(
            re.search(
                r"\b(what files can you (?:inspect|see)|what files can you inspect for me|list files|show workspace files|what files can you see|what files does alice have|show me files alice has)\b",
                text,
            )
        )

    def _has_explicit_file_target(user_input: str) -> bool:
        text = str(user_input or "").strip().lower()
        if not text:
            return False
        if re.search(r"\b[a-z0-9_./\\-]+\.[a-z0-9]{1,8}\b", text):
            return True
        if re.search(r"\b(?:[a-z]:\\|/|\.{1,2}/)[^\s]+\b", text):
            return True
        if re.search(r"\b(?:named|called)\s+[a-z0-9_./\\-]+\b", text):
            return True
        return False

    def _extract_code_target(user_input: str) -> str:
        match = re.search(r"([a-zA-Z0-9_./\\-]+\.py)\b", str(user_input or ""))
        return str(match.group(1)) if match else ""

    def _looks_like_code_analyze_request(user_input: str) -> bool:
        text = str(user_input or "").lower().strip()
        if _extract_code_target(user_input):
            return bool(re.search(r"\b(analy[sz]e|review|inspect|look at|read|what about|check)\b", text) or text.endswith(".py"))
        return False

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
        if not any(
            re.search(r"\b" + re.escape(term) + r"\b", text)
            for term in weather_scope_terms
        ):
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

    def _looks_like_weather_commentary_without_request(user_input: str) -> bool:
        text = str(user_input or "").lower().strip()
        if "weather" not in text:
            return False
        if "?" in text:
            return False
        return any(
            marker in text
            for marker in ("don't want you to check", "dont want you to check", "just saying")
        )

    def _looks_like_collaborative_reasoning_statement(user_input: str) -> bool:
        text = str(user_input or "").lower().strip()
        return ("let's think through" in text or "lets think through" in text)

    def _looks_like_project_work_session_start(user_input: str) -> bool:
        text = str(user_input or "").lower().strip()
        if not text:
            return False
        patterns = (
            r"\bready to work on (?:our )?(?:ai project|ai|alice|the project)\b",
            r"\blet[' ]?s work on alice\b",
            r"\blet[' ]?s continue working on alice\b",
            r"\bi[' ]?m ready to work on the project\b",
            r"\bready to keep building alice\b",
            r"\blet[' ]?s continue the ai project\b",
            r"\bback to working on alice\b",
            r"\blet[' ]?s get back to alice\b",
        )
        return any(re.search(pat, text) for pat in patterns)

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
        detector = getattr(
            alice, "_is_freshness_sensitive_current_events_request", None
        )
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

    def _is_personal_memory_query(user_input: str) -> bool:
        text = str(user_input or "").strip()
        if not text:
            return False
        low = text.lower()
        if "personal" in low and "life" in low:
            return True
        return any(pattern.search(text) for pattern in _personal_history_patterns)

    def _infer_personal_domain(user_input: str) -> str:
        text = str(user_input or "").lower()
        for domain, cues in _personal_domain_keywords.items():
            if any(cue in text for cue in cues):
                return domain
        return "personal_life"

    def _memory_strength(items: List[Dict[str, Any]]) -> Dict[str, float]:
        if not items:
            return {"count": 0.0, "avg_similarity": 0.0, "avg_weighted": 0.0}
        count = float(len(items))
        avg_similarity = (
            sum(float(i.get("similarity", 0.0) or 0.0) for i in items) / count
        )
        avg_weighted = (
            sum(
                float(i.get("weighted_score", i.get("score", 0.0)) or 0.0)
                for i in items
            )
            / count
        )
        return {
            "count": count,
            "avg_similarity": avg_similarity,
            "avg_weighted": avg_weighted,
        }

    def _evidence_sources(items: List[Dict[str, Any]]) -> List[str]:
        seen = set()
        sources: List[str] = []
        for row in items:
            ctx = dict(row.get("context") or {})
            source = str(ctx.get("source") or row.get("source") or "conversation").strip()
            if not source:
                continue
            key = source.lower()
            if key in seen:
                continue
            seen.add(key)
            sources.append(source)
        return sources

    def _has_sufficient_personal_evidence(items: List[Dict[str, Any]]) -> bool:
        if not items:
            return False
        strength = _memory_strength(items)
        top_similarity = float(items[0].get("similarity", 0.0) or 0.0)
        top_weighted = float(
            items[0].get("weighted_score", items[0].get("score", 0.0)) or 0.0
        )
        top_ctx = dict(items[0].get("context") or {})
        top_confidence = float(top_ctx.get("confidence", items[0].get("importance", 0.0)) or 0.0)
        return bool(
            (
                top_similarity >= 0.46
                and top_weighted >= 0.40
                and strength["avg_similarity"] >= 0.40
            )
            or (len(items) >= 1 and top_confidence >= 0.75)
        )

    def _personal_memory_fallback_response(user_input: str, intent: str) -> str:
        return _surface_text(
            "I do not have enough saved memory yet to answer that accurately.",
            user_input=user_input,
            intent=intent,
            route="contract_personal_memory_guard",
        )

    def _render_personal_memory_summary(
        user_input: str,
        intent: str,
        items: List[Dict[str, Any]],
    ) -> str:
        def _norm(text: str) -> str:
            value = str(text or "").strip().lower()
            value = re.sub(r"^\s*[-*]\s*", "", value)
            value = re.sub(r"[^a-z0-9\s]", " ", value)
            value = re.sub(r"\s+", " ", value).strip()
            return value

        seen = set()
        snippets: List[str] = []
        for row in items[:4]:
            content = str(row.get("content") or "").strip()
            if not content:
                continue
            key = _norm(content)
            if not key or key in seen:
                continue
            seen.add(key)
            snippets.append(content)
        if not snippets:
            return _personal_memory_fallback_response(user_input, intent)
        summary = "Here is what I have saved in memory:\n- " + "\n- ".join(snippets)
        return _surface_text(
            summary,
            user_input=user_input,
            intent=intent,
            route="contract_personal_memory_recall",
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

    def _verify_codebase_claims(
        response_text: str, user_input: str = ""
    ) -> Dict[str, Any]:
        text = str(response_text or "")
        low = text.lower()
        user_text = str(user_input or "").lower().strip()
        if not text:
            return {}

        explicit_paths = _extract_code_path_claims(text)
        explicit_dirs = _extract_directory_claims(text)

        # Only enforce codebase claim verification when the user asked for code access
        # or the assistant claimed concrete paths/directories.
        if not explicit_paths and not explicit_dirs:
            if not _looks_like_code_request(user_text):
                return {}

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

    def _verify_weather_claims(
        *, user_input: str, response_text: str
    ) -> Dict[str, Any]:
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

        has_fact_pattern = any(
            pattern.search(text) for pattern in _weather_fact_patterns
        )
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

        if (
            not _looks_like_weather_request(user_input)
            and "according to my knowledge" not in low
        ):
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
        operator_ctx = dict(getattr(alice, "_operator_context", {}) or {})
        state = dict(getattr(alice, "_operator_state", {}) or {})
        continuation_active = bool(
            operator_ctx.get("active_capability") == "code_inspection"
            and operator_ctx.get("awaiting_target")
        )
        low = str(req.user_input or "").lower()

        if ("what's the next step" in low or "what is the next step" in low) and state.get("active_objective"):
            return RouterDecision(
                route="local",
                intent="code:request",
                confidence=0.93,
                decision_band="execute",
                metadata={
                    "reason": "operator_next_step_query",
                    "resolved_input": req.user_input,
                    "operator_state": dict(getattr(alice, "_operator_state", {}) or {}),
                },
            )

        if TurnSegmenter.looks_like_project_mode(req.user_input):
            focus = ""
            low_input = str(req.user_input or "").lower()
            if "routing" in low_input or "route" in low_input:
                focus = "routing"
            elif "memory" in low_input:
                focus = "memory"
            elif "runtime" in low_input:
                focus = "runtime"
            elif "local code" in low_input or "code execution" in low_input:
                focus = "local code execution"
            elif "cleanup" in low_input:
                focus = "cleanup"
            elif "proactivity" in low_input:
                focus = "proactivity"
            elif "loop" in low_input:
                focus = "agent loop"
            setattr(
                alice,
                "_operator_state",
                update_operator_state(
                    state,
                    {
                        "active_mode": "alice_project_operator",
                        "active_objective": "Improve Alice into an agentic companion/operator",
                        "current_focus": focus or "alice runtime operator loop",
                        "awaiting_target": False,
                        "current_step": "set_objective",
                    },
                ),
            )

        if _looks_like_code_list_request(req.user_input):
            setattr(
                alice,
                "_operator_state",
                update_operator_state(
                    state,
                    {"active_mode": "code_inspection", "awaiting_target": True, "last_route": "local", "last_intent": "code:list_files"},
                ),
            )
            return RouterDecision(
                route="local",
                intent="code:list_files",
                confidence=0.95,
                decision_band="execute",
                metadata={"reason": "code_list_request", "resolved_input": req.user_input, "operator_state": dict(getattr(alice, "_operator_state", {}) or {})},
            )

        if _looks_like_code_analyze_request(req.user_input) or (
            continuation_active and bool(_extract_code_target(req.user_input))
        ):
            target = _extract_code_target(req.user_input)
            setattr(
                alice,
                "_operator_state",
                update_operator_state(
                    state,
                    {"active_mode": "code_inspection", "awaiting_target": False, "last_route": "local", "last_intent": "code:analyze_file", "last_inspected_file": target},
                ),
            )
            return RouterDecision(
                route="local",
                intent="code:analyze_file",
                confidence=0.95,
                decision_band="execute",
                metadata={
                    "reason": "code_analyze_request",
                    "resolved_input": req.user_input,
                    "operator_state": dict(getattr(alice, "_operator_state", {}) or {}),
                    "operator_context": {
                        "continuation_from_previous_turn": continuation_active,
                        "inferred_target_file": target,
                    },
                    "target_file": target,
                },
            )

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
            setattr(
                alice,
                "_operator_state",
                update_operator_state(
                    state,
                    {"active_mode": "code_inspection", "awaiting_target": True, "last_route": "local", "last_intent": "code:request"},
                ),
            )
            return RouterDecision(
                route="local",
                intent="code:request",
                confidence=0.96,
                decision_band="execute",
                metadata={
                    "reason": "code_request_detected",
                    "resolved_input": req.user_input,
                    "operator_state": dict(getattr(alice, "_operator_state", {}) or {}),
                },
            )

        if _looks_like_weather_commentary_without_request(req.user_input):
            return RouterDecision(
                route="llm",
                intent="conversation:general",
                confidence=0.86,
                decision_band="execute",
                metadata={"reason": "weather_commentary_without_action"},
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

        if _looks_like_collaborative_reasoning_statement(req.user_input):
            return RouterDecision(
                route="llm",
                intent="conversation:goal_statement",
                confidence=0.87,
                decision_band="execute",
                metadata={
                    "reason": "collaborative_reasoning_statement",
                    "resolved_input": req.user_input,
                },
            )

        if _looks_like_project_work_session_start(req.user_input):
            return RouterDecision(
                route="llm",
                intent="conversation:project_work_session",
                confidence=0.9,
                decision_band="execute",
                metadata={
                    "reason": "project_work_session_start",
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
        parsed_command = getattr(nlp_result, "parsed_command", {}) or {}
        modifiers = (
            parsed_command.get("modifiers", {})
            if isinstance(parsed_command, dict)
            else {}
        )

        route = "llm"
        if ":" in intent and not intent.startswith("conversation"):
            route = "tool"

        if str(intent).startswith("file_operations:") and not _has_explicit_file_target(req.user_input):
            arbitration = route_arbiter.arbitrate(
                user_input=req.user_input,
                candidate_route=route,
                candidate_intent=intent,
                confidence=confidence,
                active_mode=str(state.get("active_mode") or ""),
            )
            rerouted_intent = str(arbitration.get("intent") or "code:request")
            return RouterDecision(
                route=str(arbitration.get("route") or "local"),
                intent=rerouted_intent,
                confidence=max(confidence, 0.82),
                decision_band="execute",
                needs_clarification=False,
                metadata={
                    "reason": "file_tool_veto_no_explicit_target",
                    "routing_trace": {
                        **dict(arbitration.get("trace") or {}),
                        "file_tool_vetoed": True,
                        "reason": "no_explicit_file_target",
                        "original_intent": str(intent or ""),
                        "rerouted_to": rerouted_intent,
                    },
                    "original_intent": str(intent or ""),
                    "operator_state": dict(getattr(alice, "_operator_state", {}) or {}),
                },
            )

        if "file" in low and ("what files" in low or "files does alice have" in low):
            return RouterDecision(
                route="local",
                intent="code:list_files",
                confidence=max(confidence, 0.85),
                decision_band="execute",
                needs_clarification=False,
                metadata={
                    "reason": "workspace_files_question",
                    "resolved_input": resolved_input,
                    "operator_state": dict(getattr(alice, "_operator_state", {}) or {}),
                },
            )

        if route == "tool" and bool(modifiers.get("tool_execution_disabled")):
            gate = dict(modifiers.get("tool_eligibility_gate") or {})
            rerouted_to = str(gate.get("rerouted_to") or "").strip()
            if gate.get("file_tool_vetoed") and rerouted_to in {"code:request", "code:list_files"}:
                route = "local"
                intent = rerouted_to
            else:
                route = "llm"
                if not intent.startswith("conversation:"):
                    intent = "conversation:general"

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
                "tool_execution_disabled": bool(modifiers.get("tool_execution_disabled")),
                "tool_eligibility_gate": dict(modifiers.get("tool_eligibility_gate") or {}),
                "routing_trace": dict(modifiers.get("routing_trace") or {}),
                "operator_state": dict(getattr(alice, "_operator_state", {}) or {}),
                **resolution_meta,
            },
        )

    def _recall(req: MemoryRequest) -> MemoryResult:
        try:
            items: List[Dict[str, Any]] = []
            metadata: Dict[str, Any] = {"count": 0}
            if getattr(alice, "memory", None):
                personal_mode = bool(personal_memory and _is_personal_memory_query(req.query))
                if personal_mode:
                    domain = _infer_personal_domain(req.query)
                    day_to_day_detail = personal_memory.retrieve_structured_memory_detailed(
                        req.query,
                        domain=domain,
                        scope="day_to_day",
                        top_k=req.max_items,
                    )
                    day_to_day = list(day_to_day_detail.get("items") or [])
                    remaining = max(0, req.max_items - len(day_to_day))
                    long_term_detail = (
                        personal_memory.retrieve_structured_memory_detailed(
                            req.query,
                            domain=domain,
                            scope="long_term",
                            top_k=remaining,
                        )
                        if remaining > 0
                        else {"items": []}
                    )
                    long_term = list(long_term_detail.get("items") or [])
                    items = list(day_to_day) + list(long_term)
                    raw_retrieved_count = int(day_to_day_detail.get("raw_retrieved_count", 0)) + int(
                        long_term_detail.get("raw_retrieved_count", 0)
                    )
                    deduped_count = len(items)
                    downranked_mixed_count = int(day_to_day_detail.get("downranked_mixed_count", 0)) + int(
                        long_term_detail.get("downranked_mixed_count", 0)
                    )
                    metadata = {
                        "count": len(items or []),
                        "mode": "personal_structured",
                        "memory_recall_mode": True,
                        "requested_domain": domain,
                        "retrieved_memory_count": len(items or []),
                        "evidence_count": len(items or []),
                        "insufficient_evidence": not _has_sufficient_personal_evidence(items),
                        "evidence_sources": _evidence_sources(items),
                        "evidence": _memory_strength(items),
                        "raw_retrieved_count": raw_retrieved_count,
                        "deduped_count": deduped_count,
                        "duplicate_count_removed": max(0, raw_retrieved_count - deduped_count),
                        "downranked_mixed_count": downranked_mixed_count,
                    }
                else:
                    items = alice.memory.search(req.query, top_k=req.max_items)
                    metadata = {
                        "count": len(items or []),
                        "mode": "default_search",
                        "memory_recall_mode": False,
                        "requested_domain": "",
                        "retrieved_memory_count": len(items or []),
                        "evidence_count": len(items or []),
                        "insufficient_evidence": False,
                        "evidence_sources": _evidence_sources(items),
                    }
            return MemoryResult(
                items=list(items or []),
                source="memory_system",
                confidence=0.8 if items else 0.3,
                metadata=metadata,
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
        op = str(item.get("memory_operation") or "").strip().lower()
        if op and personal_memory:
            try:
                if op == "forget_recent":
                    personal_memory.forget_recent_memory()
                    return
                if op == "mark_recent_incorrect":
                    recent = personal_memory.find_recent_structured_memories(top_k=1)
                    if recent:
                        personal_memory.mark_memory_incorrect(
                            str(recent[0].get("id") or ""),
                            str(item.get("reason") or "user_marked_incorrect"),
                        )
                    return
                if op == "update_recent":
                    recent = personal_memory.find_recent_structured_memories(top_k=1)
                    if recent:
                        personal_memory.update_memory(
                            str(recent[0].get("id") or ""),
                            str(item.get("reason") or recent[0].get("content") or ""),
                        )
                    return
            except Exception:
                return
        text = str(item.get("content") or item.get("text") or "").strip()
        if not text:
            return
        try:
            has_structured_fields = all(
                str(item.get(key) or "").strip() for key in ("domain", "kind", "scope")
            )
            if personal_memory and has_structured_fields:
                personal_memory.store_structured_memory(
                    content=text,
                    domain=str(item.get("domain") or "general"),
                    kind=str(item.get("kind") or "conversation_event"),
                    scope=str(item.get("scope") or "day_to_day"),
                    confidence=float(item.get("confidence", 0.65) or 0.65),
                    source=str(item.get("source") or "conversation"),
                    trace_id=str(item.get("trace_id") or "") or None,
                    importance=float(item.get("importance", 0.7) or 0.7),
                )
            else:
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

        if str(invocation.action or "").startswith("code:") or str(invocation.action or "") in {
            "system:location",
            "freshness:current_events",
        }:
            context_payload = dict(invocation.params.get("context") or {})
            extracted_target = _extract_code_target(str(invocation.params.get("query") or ""))
            if extracted_target:
                context_payload["target_file"] = extracted_target
            local = local_executor.execute(
                action=str(invocation.action or ""),
                query=str(invocation.params.get("query") or ""),
                context=context_payload,
            )
            success = bool(local.get("success"))
            return ToolResult(
                success=success,
                tool_name="local_action_executor",
                action=invocation.action,
                data={
                    "response": str(local.get("response") or ""),
                    "operator_context": dict(local.get("operator_context") or {}),
                    "local_execution": dict(local.get("local_execution") or {}),
                    "close_matches": list((local.get("operator_context") or {}).get("close_matches") or []),
                },
                error=str(local.get("error") or ""),
                confidence=0.9 if success else 0.3,
                diagnostics={
                    "route": "local_executor",
                    "operator_context": dict(local.get("operator_context") or {}),
                    "local_execution": dict(local.get("local_execution") or {}),
                },
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

        success = bool(result.get("success", False))
        data = dict(result)
        data_error = (
            (data.get("data") or {}).get("error")
            if isinstance(data.get("data"), dict)
            else ""
        )
        message_code = (
            (data.get("data") or {}).get("message_code")
            if isinstance(data.get("data"), dict)
            else ""
        )
        error = str(result.get("error") or data_error or "")
        if not error and not success and message_code:
            error = str(message_code)

        tool_result = ToolResult(
            success=success,
            tool_name=str(result.get("plugin") or invocation.tool_name),
            action=invocation.action,
            data=data,
            error=error,
            confidence=float(result.get("confidence", 0.7) or 0.7),
            diagnostics={
                "route": "plugin_manager",
                "message_code": str(message_code or ""),
                "plugin_error": str(data_error or ""),
            },
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
        if req.tool_result is not None:
            try:
                diag_ctx = dict((req.tool_result.diagnostics or {}).get("operator_context") or {})
                if diag_ctx:
                    setattr(alice, "_operator_context", diag_ctx)
            except Exception:
                pass

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

        if req.decision.route == "local" and req.tool_result is not None and not req.tool_result.success:
            data = dict(req.tool_result.data or {})
            op_ctx = dict(data.get("operator_context") or {})
            local_exec = dict(data.get("local_execution") or {})
            inferred = str(op_ctx.get("inferred_target_file") or "")
            close = list(op_ctx.get("close_matches") or [])
            if inferred and close:
                msg = f"I could not find {inferred}. Close matches:\n- " + "\n- ".join(close[:5])
            elif inferred:
                wf_count = int(local_exec.get("workspace_file_count") or 0)
                if wf_count > 0:
                    msg = f"I could not find {inferred} in the current workspace ({wf_count} files indexed). I can list files first if you want."
                else:
                    msg = f"I could not find {inferred} in the current workspace."
            else:
                msg = str(req.tool_result.error or "I could not complete that local inspection request.")
            try:
                current_state = dict(getattr(alice, "_operator_state", {}) or {})
                inferred = str(op_ctx.get("inferred_target_file") or "")
                setattr(
                    alice,
                    "_operator_state",
                    update_operator_state(
                        current_state,
                        {
                            "last_failure": str(local_exec.get("error") or req.tool_result.error or "local_execution_failed"),
                            "known_blockers": [str(local_exec.get("error") or "local_execution_failed")],
                            "current_step": "resolve_blocker",
                            "last_inspected_file": inferred or str(current_state.get("last_inspected_file") or ""),
                        },
                    ),
                )
            except Exception:
                pass
            return ResponseOutput(
                text=_surface_text(
                    msg,
                    user_input=req.user_input,
                    intent=req.decision.intent,
                    route="contract_local_execution_error",
                ),
                confidence=0.45,
                metadata={
                    "type": "local_execution_error",
                    "operator_context": op_ctx,
                    "local_execution": local_exec,
                },
            )

        if req.decision.intent == "code:request" and hasattr(
            alice, "_handle_code_request"
        ):
            if req.tool_result and req.tool_result.success:
                local_payload = dict(req.tool_result.data or {})
                local_response = str(local_payload.get("response") or "").strip()
                if local_response:
                    return ResponseOutput(
                        text=_surface_text(
                            local_response,
                            user_input=req.user_input,
                            intent=req.decision.intent,
                            route="contract_local_code_request",
                        ),
                        confidence=float(req.tool_result.confidence or 0.85),
                        metadata={
                            "type": "local_code_request",
                            "operator_context": dict(local_payload.get("operator_context") or {}),
                            "local_execution": dict(local_payload.get("local_execution") or {}),
                        },
                    )
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
            local_file_q = bool(re.search(r"\bfile|files|code|workspace\b", str(req.user_input or ""), re.IGNORECASE))
            clarify_base = "Which file should I inspect?" if local_file_q else "What exact result should I produce next?"
            clarify_text = _surface_text(
                clarify_base,
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
            op_ctx = dict(tool_payload.get("operator_context") or {})
            if op_ctx:
                try:
                    setattr(alice, "_operator_context", op_ctx)
                except Exception:
                    pass
            tool_response = str(tool_payload.get("response") or "").strip()
            if not tool_response:
                tool_response = _compose_weather_response_from_tool_payload(
                    tool_payload=tool_payload,
                    user_input=req.user_input,
                )
            if tool_response:
                try:
                    if req.decision.route == "local":
                        lx = dict(tool_payload.get("local_execution") or {})
                        inspected = str(lx.get("inspected_file") or "")
                        current_state = dict(getattr(alice, "_operator_state", {}) or {})
                        setattr(
                            alice,
                            "_operator_state",
                            update_operator_state(
                                current_state,
                                {
                                    "last_success": str(req.decision.intent or "local_success"),
                                    "last_failure": "",
                                    "current_step": "observe_result",
                                    "files_inspected": [inspected] if inspected else [],
                                    "last_inspected_file": inspected or str(current_state.get("last_inspected_file") or ""),
                                },
                            ),
                        )
                except Exception:
                    pass
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
                tool_payload.get("data")
                if isinstance(tool_payload.get("data"), dict)
                else {}
            )
            weather_tool_turn = (
                str(req.decision.intent or "").startswith("weather:")
                or str(req.tool_result.tool_name or "").lower().startswith("weather")
                or str((nested_data or {}).get("plugin_type") or "").lower()
                == "weather"
                or "weather:"
                in str((nested_data or {}).get("message_code") or "").lower()
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

        if _is_personal_memory_query(req.user_input):
            memory_items = list(req.memory.items or [])
            if not _has_sufficient_personal_evidence(memory_items):
                return ResponseOutput(
                    text=_personal_memory_fallback_response(
                        req.user_input, req.decision.intent
                    ),
                    confidence=0.96,
                    metadata={
                        "type": "personal_memory_insufficient",
                        "memory_evidence": _memory_strength(memory_items),
                    },
                )
            grounded_text = _render_personal_memory_summary(
                req.user_input, req.decision.intent, memory_items
            )
            verification = memory_answer_verifier.verify_answer(
                answer_text=grounded_text,
                evidence_items=memory_items,
            )
            if not bool(verification.get("accepted")):
                return ResponseOutput(
                    text=_personal_memory_fallback_response(
                        req.user_input, req.decision.intent
                    ),
                    confidence=0.96,
                    metadata={
                        "type": "personal_memory_insufficient",
                        "memory_evidence": _memory_strength(memory_items),
                        "memory_answer_verification": verification,
                    },
                )
            return ResponseOutput(
                text=grounded_text,
                confidence=max(0.7, float(verification.get("confidence") or 0.7)),
                metadata={
                    "type": "personal_memory_grounded",
                    "memory_evidence": _memory_strength(memory_items),
                    "memory_answer_verification": verification,
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

            code_claim_diagnostics = _verify_codebase_claims(
                response_text, req.user_input
            )
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
