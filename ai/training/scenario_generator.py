"""
Scenario Generator using Ollama
Generates new training scenarios for A.L.I.C.E to reduce manual authoring.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ai.core.llm_engine import LLMConfig, LocalLLMEngine

logger = logging.getLogger(__name__)


DEFAULT_DOMAINS = [
    "email",
    "notes",
    "weather",
    "time",
    "system",
    "clarification",
    "memory",
    "file"
]

ALLOWED_INTENTS = {
    "email": ["list_emails", "read_email", "search_emails", "compose_email", "delete_email", "reply_email"],
    "notes": ["create_note", "search_notes", "list_notes", "delete_notes"],
    "weather": ["get_weather", "get_weather_forecast"],
    "time": ["get_time"],
    "system": ["system_status"],
    "clarification": ["vague_question", "vague_request", "vague_temporal_question", "schedule_action"],
    "memory": ["store_preference", "recall_memory", "search_memory"],
    "file": ["create_file", "read_file", "delete_file", "move_file"]
}

ALLOWED_ROUTES = ["CONVERSATIONAL", "TOOL", "CLARIFICATION", "RAG", "LLM_FALLBACK"]


class ScenarioGenerator:
    """Generate scenario JSON using Ollama."""

    def __init__(self, project_root: Optional[Path] = None, model: str = "llama3.1:8b"):
        self.project_root = project_root or Path(__file__).resolve().parents[1]
        self.output_file = self.project_root / "scenarios" / "auto_generated.json"
        self.llm = LocalLLMEngine(LLMConfig(model=model))

    def generate(self, domains: Optional[List[str]] = None, count_per_domain: int = 3) -> List[Dict[str, Any]]:
        """Generate scenarios for given domains."""
        domains = domains or DEFAULT_DOMAINS
        scenarios: List[Dict[str, Any]] = []

        for domain in domains:
            intents = ALLOWED_INTENTS.get(domain, [])
            if not intents:
                continue

            prompt = self._build_prompt(domain, intents, count_per_domain)
            raw = self.llm.chat(user_input=prompt, use_history=False)

            parsed = self._safe_parse(raw)
            if not parsed:
                logger.warning(f"[ScenarioGenerator] No scenarios generated for {domain}")
                continue

            scenarios.extend(parsed)

        # Persist (merge with any existing auto-generated scenarios)
        merged = self._merge_existing(scenarios)
        self._save(merged)
        return merged

    def generate_from_errors(self, max_scenarios: int = 50) -> List[Dict[str, Any]]:
        """Create scenarios from logged mistakes in auto_generated.jsonl."""
        log_path = self.project_root / "data" / "training" / "auto_generated.jsonl"
        if not log_path.exists():
            return []

        scenarios: List[Dict[str, Any]] = []
        seen = set()

        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    success_flag = entry.get("success_flag", entry.get("success", True))
                    if success_flag:
                        continue

                    user_input = entry.get("user_input", "").strip()
                    expected_intent = entry.get("expected_intent")
                    expected_route = entry.get("expected_route")
                    domain = entry.get("domain", "unknown")

                    if not user_input or not expected_intent or not expected_route:
                        continue

                    key = (user_input.lower(), expected_intent, expected_route)
                    if key in seen:
                        continue
                    seen.add(key)

                    scenarios.append({
                        "name": f"Auto: {expected_intent}",
                        "description": "Auto-generated from error logs",
                        "domain": domain,
                        "tags": ["generated", "error_fix", domain],
                        "steps": [
                            {
                                "user_input": user_input,
                                "expected_intent": expected_intent,
                                "expected_route": expected_route,
                                "expected_entities": {},
                                "notes": entry.get("error_type", "")
                            }
                        ]
                    })

                    if len(scenarios) >= max_scenarios:
                        break

        except Exception as e:
            logger.warning(f"[ScenarioGenerator] Error reading logs: {e}")

        if not scenarios:
            return []

        merged = self._merge_existing(scenarios)
        self._save(merged)
        return scenarios

    def _build_prompt(self, domain: str, intents: List[str], count: int) -> str:
        return (
            "You generate test scenarios for an assistant. "
            "Return ONLY JSON: a list of scenario objects. "
            "Each scenario has: name, description, domain, tags, steps. "
            "Each step has: user_input, expected_intent, expected_route, expected_entities, notes. "
            "Use ONLY these routes: " + ", ".join(ALLOWED_ROUTES) + ". "
            "Use ONLY these intents for domain '" + domain + "': " + ", ".join(intents) + ". "
            f"Create {count} scenarios for domain '{domain}'. "
            "Make them realistic and concise. "
            "If forecast-related, include time_range in expected_entities."
        )

    def _safe_parse(self, raw: Any) -> List[Dict[str, Any]]:
        if not raw:
            return []
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            return [raw]
        if isinstance(raw, str):
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return [data]
            except json.JSONDecodeError:
                return []
        return []

    def _merge_existing(self, new_scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        existing = []
        if self.output_file.exists():
            try:
                with open(self.output_file, "r", encoding="utf-8", errors="ignore") as f:
                    payload = json.load(f)
                    existing = payload.get("scenarios", []) if isinstance(payload, dict) else payload
            except Exception:
                existing = []

        combined = existing + new_scenarios

        # Deduplicate by user_input + expected_intent + expected_route
        seen = set()
        deduped = []
        for s in combined:
            try:
                step = (s.get("steps") or [{}])[0]
                key = (
                    (step.get("user_input") or "").lower(),
                    step.get("expected_intent"),
                    step.get("expected_route")
                )
            except Exception:
                key = None

            if key and key in seen:
                continue
            if key:
                seen.add(key)
            deduped.append(s)

        return deduped

    def _save(self, scenarios: List[Dict[str, Any]]) -> None:
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now().isoformat(),
            "scenarios": scenarios
        }
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


def generate_scenarios(project_root: Optional[Path] = None, domains: Optional[List[str]] = None, count_per_domain: int = 3) -> List[Dict[str, Any]]:
    """Convenience function for scenario generation."""
    generator = ScenarioGenerator(project_root=project_root)
    return generator.generate(domains=domains, count_per_domain=count_per_domain)


def generate_scenarios_from_errors(project_root: Optional[Path] = None, max_scenarios: int = 50) -> List[Dict[str, Any]]:
    """Generate scenarios from error logs."""
    generator = ScenarioGenerator(project_root=project_root)
    return generator.generate_from_errors(max_scenarios=max_scenarios)
