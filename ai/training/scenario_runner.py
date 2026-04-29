"""
Compatibility scenario runner for legacy training paths.

This module preserves the older ai.training.scenario_runner API while
delegating execution to the canonical simulator in scenarios/sim.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any, Dict, List, Optional, Tuple

from scenarios.sim.scenarios import (
    ALL_SCENARIOS,
    ExpectedRoute,
    Scenario,
    ScenarioResult as CanonicalScenarioResult,
    ScenarioStep,
)

if TYPE_CHECKING:
    from scenarios.sim.run_scenarios import ScenarioRunner as CanonicalScenarioRunner

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    """Legacy result shape kept for backward compatibility."""

    scenario_name: str
    turn_index: int
    user_input: str
    expected_intent: str
    detected_intent: Optional[str]
    intent_match: bool
    expected_tool: Optional[str]
    used_tool: Optional[str]
    tool_match: bool
    response_generated: bool
    response_type: str
    success: bool
    timestamp: str = ""


class ScenarioRunner:
    """Legacy adapter that delegates scenario execution to canonical runner."""

    def __init__(
        self,
        nlp_processor=None,
        router=None,
        output_dir: str = "data/training",
        llm_policy: str = "minimal",
        llm_model: str = "llama3.1:8b",
        use_teacher: bool = True,
    ):
        # nlp_processor/router are accepted for compatibility but unused.
        self._deprecated_nlp_processor = nlp_processor
        self._deprecated_router = router

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._delegate = self._create_delegate(
            llm_policy=llm_policy,
            llm_model=llm_model,
            use_teacher=use_teacher,
        )

        self.results: List[ScenarioResult] = []
        self.training_data: List[Dict[str, Any]] = []

    def run_scenario(
        self, scenario_name: str, scenario: Dict[str, Any]
    ) -> Tuple[List[ScenarioResult], bool]:
        """Run one legacy-style scenario dict through canonical execution."""
        canonical = self._to_canonical_scenario(scenario_name, scenario)
        delegate_results = self._delegate.run_scenario(canonical)
        expected_tools = self._expected_tools_from_legacy(scenario)

        legacy_results: List[ScenarioResult] = []
        for idx, result in enumerate(delegate_results):
            expected_tool = expected_tools[idx] if idx < len(expected_tools) else None
            legacy_results.append(
                self._to_legacy_result(
                    scenario_name=scenario_name,
                    turn_index=idx,
                    result=result,
                    expected_tool=expected_tool,
                )
            )

        self.results.extend(legacy_results)
        self.training_data.extend(
            self._build_training_entries(scenario_name, delegate_results)
        )

        all_success = all(r.success for r in legacy_results)
        return legacy_results, all_success

    def _create_delegate(
        self,
        llm_policy: str,
        llm_model: str,
        use_teacher: bool,
    ) -> "CanonicalScenarioRunner":
        """Create canonical runner lazily to avoid import-time side effects."""
        from scenarios.sim.run_scenarios import (
            ScenarioRunner as CanonicalScenarioRunner,
        )

        return CanonicalScenarioRunner(
            llm_policy=llm_policy,
            llm_model=llm_model,
            use_teacher=use_teacher,
        )

    def run_all_scenarios(self, scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Run all provided legacy scenarios and return summary stats."""
        stats = {
            "total_scenarios": len(scenarios),
            "successful_scenarios": 0,
            "failed_scenarios": 0,
            "total_turns": 0,
            "successful_turns": 0,
            "failed_turns": 0,
            "scenario_results": {},
        }

        for scenario_name, scenario in scenarios.items():
            scenario_results, success = self.run_scenario(scenario_name, scenario)

            successful_turns = sum(1 for r in scenario_results if r.success)
            failed_turns = len(scenario_results) - successful_turns

            stats["scenario_results"][scenario_name] = {
                "success": success,
                "turns": len(scenario_results),
                "successful_turns": successful_turns,
                "failed_turns": failed_turns,
            }

            stats["total_turns"] += len(scenario_results)
            stats["successful_turns"] += successful_turns
            stats["failed_turns"] += failed_turns

            if success:
                stats["successful_scenarios"] += 1
            else:
                stats["failed_scenarios"] += 1

        return stats

    def generate_training_data(self) -> List[Dict[str, Any]]:
        """Return collected training data generated by scenario execution."""
        return list(self.training_data)

    def save_training_data(self, output_file: Optional[str] = None) -> str:
        """Save collected training data in JSONL format."""
        output_path = (
            Path(output_file)
            if output_file
            else self.output_dir / "synthetic_from_scenarios.jsonl"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for entry in self.training_data:
                f.write(json.dumps(entry) + "\n")

        logger.info(
            "Saved %s training rows to %s", len(self.training_data), output_path
        )
        return str(output_path)

    def save_scenario_results(self, output_file: Optional[str] = None) -> str:
        """Save legacy-compatible scenario result rows as JSONL."""
        output_path = (
            Path(output_file)
            if output_file
            else self.output_dir / "scenario_results.jsonl"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for result in self.results:
                f.write(json.dumps(self._legacy_result_to_dict(result)) + "\n")

        logger.info("Saved %s scenario results to %s", len(self.results), output_path)
        return str(output_path)

    def get_statistics(self) -> Dict[str, Any]:
        """Compute legacy statistics for completed scenario turns."""
        total_turns = len(self.results)
        successful_turns = sum(1 for r in self.results if r.success)
        intent_matches = sum(1 for r in self.results if r.intent_match)
        tool_matches = sum(1 for r in self.results if r.tool_match)

        by_scenario: Dict[str, Dict[str, int]] = {}
        for result in self.results:
            bucket = by_scenario.setdefault(
                result.scenario_name, {"success": 0, "total": 0}
            )
            bucket["total"] += 1
            if result.success:
                bucket["success"] += 1

        return {
            "total_turns": total_turns,
            "successful_turns": successful_turns,
            "success_rate": (successful_turns / total_turns * 100)
            if total_turns
            else 0.0,
            "intent_accuracy": (intent_matches / total_turns * 100)
            if total_turns
            else 0.0,
            "tool_accuracy": (tool_matches / total_turns * 100) if total_turns else 0.0,
            "by_scenario": by_scenario,
        }

    def _to_canonical_scenario(
        self, scenario_name: str, scenario: Dict[str, Any]
    ) -> Scenario:
        """Convert legacy scenario dict to canonical Scenario dataclass."""
        steps: List[ScenarioStep] = []

        for turn in scenario.get("turns", []):
            steps.append(
                ScenarioStep(
                    user_input=turn.get("user_input", ""),
                    expected_intent=turn.get("expected_intent", ""),
                    expected_route=self._route_from_turn(turn),
                    domain=turn.get("domain") or scenario.get("domain"),
                    expected_entities=turn.get("expected_entities", {}),
                    notes=turn.get("notes", ""),
                )
            )

        return Scenario(
            name=scenario_name,
            description=scenario.get("description", "Legacy scenario"),
            domain=scenario.get("domain", "legacy"),
            steps=steps,
            tags=scenario.get("tags", ["legacy"]),
        )

    def _route_from_turn(self, turn: Dict[str, Any]) -> ExpectedRoute:
        """Derive expected route for canonical step from legacy turn fields."""
        explicit = str(turn.get("expected_route", "")).strip()
        if explicit:
            for candidate in ExpectedRoute:
                if explicit == candidate.name or explicit == candidate.value:
                    return candidate

        if turn.get("expected_tool"):
            return ExpectedRoute.TOOL
        if turn.get("should_clarify"):
            return ExpectedRoute.CLARIFICATION
        if turn.get("should_require_confirmation"):
            return ExpectedRoute.CONVERSATIONAL
        return ExpectedRoute.CONVERSATIONAL

    def _expected_tools_from_legacy(
        self, scenario: Dict[str, Any]
    ) -> List[Optional[str]]:
        """Extract expected_tool per turn from legacy scenario dict."""
        return [turn.get("expected_tool") for turn in scenario.get("turns", [])]

    def _to_legacy_result(
        self,
        scenario_name: str,
        turn_index: int,
        result: CanonicalScenarioResult,
        expected_tool: Optional[str],
    ) -> ScenarioResult:
        """Convert canonical per-step result to legacy result dataclass."""
        used_tool: Optional[str] = None
        if result.actual_route == ExpectedRoute.TOOL.value:
            used_tool = "tool_route"

        tool_match = expected_tool is None or used_tool == expected_tool

        return ScenarioResult(
            scenario_name=scenario_name,
            turn_index=turn_index,
            user_input=result.step.user_input,
            expected_intent=result.step.expected_intent,
            detected_intent=result.actual_intent,
            intent_match=result.intent_match,
            expected_tool=expected_tool,
            used_tool=used_tool,
            tool_match=tool_match,
            response_generated=bool(result.actual_response),
            response_type="teacher_guided" if result.teacher_response else "direct",
            success=result.route_match and result.intent_match,
            timestamp=datetime.now().isoformat(),
        )

    def _build_training_entries(
        self, scenario_name: str, results: List[CanonicalScenarioResult]
    ) -> List[Dict[str, Any]]:
        """Build legacy-style training rows from canonical step results."""
        entries: List[Dict[str, Any]] = []
        for idx, result in enumerate(results):
            entries.append(
                {
                    "user_input": result.step.user_input,
                    "intent": result.actual_intent or result.step.expected_intent,
                    "entities": result.step.expected_entities,
                    "response": result.actual_response,
                    "tool": result.actual_route
                    if result.actual_route == ExpectedRoute.TOOL.value
                    else None,
                    "scenario": scenario_name,
                    "turn": idx,
                    "success": result.route_match and result.intent_match,
                    "quality_score": 0.9
                    if (result.route_match and result.intent_match)
                    else 0.5,
                    "category": "synthetic_from_scenarios",
                    "timestamp": datetime.now().isoformat(),
                }
            )
        return entries

    def _legacy_result_to_dict(self, result: ScenarioResult) -> Dict[str, Any]:
        """Convert legacy result dataclass to JSON-serializable dict."""
        return {
            "scenario_name": result.scenario_name,
            "turn_index": result.turn_index,
            "user_input": result.user_input,
            "expected_intent": result.expected_intent,
            "detected_intent": result.detected_intent,
            "intent_match": result.intent_match,
            "expected_tool": result.expected_tool,
            "used_tool": result.used_tool,
            "tool_match": result.tool_match,
            "response_generated": result.response_generated,
            "response_type": result.response_type,
            "success": result.success,
            "timestamp": result.timestamp,
        }


def main() -> None:
    """Legacy CLI: run all canonical scenarios through compatibility adapter."""
    legacy_map: Dict[str, Dict[str, Any]] = {}
    for scenario in ALL_SCENARIOS:
        legacy_map[scenario.name] = {
            "description": scenario.description,
            "domain": scenario.domain,
            "tags": scenario.tags,
            "turns": [
                {
                    "user_input": step.user_input,
                    "expected_intent": step.expected_intent,
                    "expected_route": step.expected_route.value,
                    "domain": step.domain,
                    "expected_entities": step.expected_entities,
                    "notes": step.notes,
                }
                for step in scenario.steps
            ],
        }

    runner = ScenarioRunner()
    stats = runner.run_all_scenarios(legacy_map)

    print("\n" + "=" * 70)
    print("SCENARIO RUNNER STATISTICS")
    print("=" * 70)
    print(f"Total scenarios: {stats['total_scenarios']}")
    print(f"Successful scenarios: {stats['successful_scenarios']}")
    print(f"Failed scenarios: {stats['failed_scenarios']}")
    print(f"Total turns: {stats['total_turns']}")
    print(f"Successful turns: {stats['successful_turns']}")

    runner.save_scenario_results()
    runner.save_training_data()


if __name__ == "__main__":
    main()
