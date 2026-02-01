"""
Scenario Generator using Ollama
Generates new training scenarios for A.L.I.C.E to reduce manual authoring.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ai.llm_engine import LLMConfig, LocalLLMEngine

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

        # Persist
        self._save(scenarios)
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
