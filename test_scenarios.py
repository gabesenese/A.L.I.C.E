"""
A.L.I.C.E. Automated Scenario Testing
======================================
Run conversation scenarios to find errors and validate improvements.

Usage:
    python test_scenarios.py                    # Run all scenarios
    python test_scenarios.py --suite nlp        # Run specific suite
    python test_scenarios.py --report           # Generate detailed report
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from app.main import ALICE
from ai.infrastructure.metrics_collector import MetricsCollector

logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)


@dataclass
class TestScenario:
    """Single test scenario with expected behavior."""
    
    id: str
    suite: str  # nlp, notes, calendar, email, music, conversation
    description: str
    inputs: List[str]  # Sequence of user inputs
    expected_intent: Optional[str] = None
    expected_plugin: Optional[str] = None
    should_not_clarify: bool = False  # Shouldn't trigger clarification
    should_succeed: bool = True
    min_confidence: float = 0.5
    tags: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of running a test scenario."""
    
    scenario_id: str
    passed: bool
    duration_ms: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    actual_intent: Optional[str] = None
    actual_confidence: float = 0.0
    response: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScenarioRunner:
    """Runs test scenarios and collects results."""
    
    def __init__(self, scenarios_file: Optional[Path] = None):
        self.scenarios_file = scenarios_file or Path("data/test_scenarios.json")
        self.scenarios: List[TestScenario] = []
        self.results: List[TestResult] = []
        self.alice: Optional[ALICE] = None
        self.metrics = MetricsCollector()
        
    def load_scenarios(self) -> int:
        """Load test scenarios from JSON file."""
        if not self.scenarios_file.exists():
            logger.warning(f"Scenarios file not found: {self.scenarios_file}")
            self._create_default_scenarios()
            return len(self.scenarios)
        
        try:
            with open(self.scenarios_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for scenario_data in data.get("scenarios", []):
                scenario = TestScenario(**scenario_data)
                self.scenarios.append(scenario)
            
            logger.info(f"Loaded {len(self.scenarios)} test scenarios")
            return len(self.scenarios)
        except Exception as e:
            logger.error(f"Failed to load scenarios: {e}")
            return 0
    
    def _create_default_scenarios(self):
        """Create default test scenarios and save to file."""
        default_scenarios = [
            # NLP Suite - Intent Classification
            TestScenario(
                id="nlp_001",
                suite="nlp",
                description="List all notes query",
                inputs=["do i have any notes?"],
                expected_intent="notes:query_exist",
                should_not_clarify=True,
                tags=["nlp", "notes", "list-all"]
            ),
            TestScenario(
                id="nlp_002",
                suite="nlp",
                description="Check notes count",
                inputs=["how many notes do i have?"],
                expected_intent="notes:query_exist",
                should_not_clarify=True,
                tags=["nlp", "notes", "count"]
            ),
            TestScenario(
                id="nlp_003",
                suite="nlp",
                description="Search notes with query",
                inputs=["find my work notes"],
                expected_intent="notes:search",
                min_confidence=0.7,
                tags=["nlp", "notes", "search"]
            ),
            TestScenario(
                id="nlp_004",
                suite="nlp",
                description="Create note minimal",
                inputs=["create a note"],
                expected_intent="notes:create",
                should_not_clarify=False,  # Should ask for content
                tags=["nlp", "notes", "create"]
            ),
            TestScenario(
                id="nlp_005",
                suite="nlp",
                description="Create note with content",
                inputs=["create note about meeting tomorrow"],
                expected_intent="notes:create",
                min_confidence=0.7,
                tags=["nlp", "notes", "create"]
            ),
            
            # Entity Normalization Suite
            TestScenario(
                id="entity_001",
                suite="entity",
                description="Tag abbreviation normalization",
                inputs=["create note tagged wrk"],
                expected_intent="notes:create",
                tags=["entity", "normalization", "tags"]
            ),
            TestScenario(
                id="entity_002",
                suite="entity",
                description="Multiple tag abbreviations",
                inputs=["find notes tagged wrk and mtg"],
                expected_intent="notes:search",
                tags=["entity", "normalization", "tags"]
            ),
            TestScenario(
                id="entity_003",
                suite="entity",
                description="Datetime normalization tomorrow",
                inputs=["create note for tomorrow"],
                expected_intent="notes:create",
                tags=["entity", "normalization", "datetime"]
            ),
            
            # Coreference Suite
            TestScenario(
                id="coref_001",
                suite="coreference",
                description="Pronoun reference after search",
                inputs=[
                    "find my work notes",
                    "delete the first one"
                ],
                expected_intent="notes:delete",
                tags=["coreference", "ordinal"]
            ),
            TestScenario(
                id="coref_002",
                suite="coreference",
                description="Generic pronoun it",
                inputs=[
                    "create note meeting agenda",
                    "delete it"
                ],
                expected_intent="notes:delete",
                tags=["coreference", "pronoun"]
            ),
            
            # Conversation Suite
            TestScenario(
                id="conv_001",
                suite="conversation",
                description="Greeting interaction",
                inputs=["hi"],
                expected_intent="greeting",
                should_not_clarify=True,
                tags=["conversation", "greeting"]
            ),
            TestScenario(
                id="conv_002",
                suite="conversation",
                description="General question",
                inputs=["what can you do?"],
                expected_intent="conversation:help",
                should_not_clarify=True,
                tags=["conversation", "help"]
            ),
            
            # Multi-turn Conversation
            TestScenario(
                id="multi_001",
                suite="multi-turn",
                description="Follow-up context inheritance",
                inputs=[
                    "what's the weather like?",
                    "what about tomorrow?"
                ],
                expected_intent="weather:forecast",
                tags=["multi-turn", "weather", "follow-up"]
            ),
            
            # Edge Cases
            TestScenario(
                id="edge_001",
                suite="edge-cases",
                description="Empty input",
                inputs=[""],
                should_succeed=False,
                tags=["edge-cases", "empty"]
            ),
            TestScenario(
                id="edge_002",
                suite="edge-cases",
                description="Very long input",
                inputs=["create a note about " + "something " * 100],
                expected_intent="notes:create",
                tags=["edge-cases", "long-input"]
            ),
            TestScenario(
                id="edge_003",
                suite="edge-cases",
                description="Special characters",
                inputs=["create note with title: @#$%^&*()"],
                expected_intent="notes:create",
                tags=["edge-cases", "special-chars"]
            ),
        ]
        
        self.scenarios = default_scenarios
        
        # Save to file
        try:
            self.scenarios_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "scenarios": [asdict(s) for s in default_scenarios]
            }
            with open(self.scenarios_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Created default scenarios: {self.scenarios_file}")
        except Exception as e:
            logger.error(f"Failed to save scenarios: {e}")
    
    def run_scenario(self, scenario: TestScenario) -> TestResult:
        """Run a single test scenario."""
        start_time = time.time()
        result = TestResult(
            scenario_id=scenario.id,
            passed=False,
            duration_ms=0.0
        )
        
        try:
            # Initialize A.L.I.C.E. if needed
            if not self.alice:
                self.alice = ALICE(
                    voice_enabled=False,
                    user_name="TestUser",
                    debug=False
                )
            
            # Run each input in sequence
            last_response = ""
            last_intent = None
            last_confidence = 0.0
            
            for user_input in scenario.inputs:
                if not user_input:  # Handle empty input edge case
                    result.errors.append("Empty input provided")
                    continue
                
                try:
                    response = self.alice.process_input(user_input)
                    last_response = response
                    
                    # Extract intent and confidence from NLP result
                    if hasattr(self.alice, 'nlp') and hasattr(self.alice.nlp, 'context'):
                        last_intent = self.alice.nlp.context.last_intent
                    
                    # Check for clarification when it shouldn't happen
                    if scenario.should_not_clarify:
                        clarification_markers = [
                            "clarify", "need more information", 
                            "which one did you mean", "can you provide more details"
                        ]
                        if any(marker in response.lower() for marker in clarification_markers):
                            result.errors.append(
                                f"Unexpected clarification prompt: '{response[:100]}'"
                            )
                    
                except Exception as e:
                    result.errors.append(f"Processing error: {str(e)}")
                    logger.exception(f"Error processing input: {user_input}")
            
            # Store results
            result.actual_intent = last_intent
            result.response = last_response
            result.duration_ms = (time.time() - start_time) * 1000
            
            # Validate expectations
            if scenario.expected_intent and last_intent != scenario.expected_intent:
                result.errors.append(
                    f"Intent mismatch: expected '{scenario.expected_intent}', "
                    f"got '{last_intent}'"
                )
            
            # Check if scenario passed
            result.passed = (
                len(result.errors) == 0 and 
                (not scenario.should_succeed or scenario.should_succeed)
            )
            
        except Exception as e:
            result.errors.append(f"Scenario execution failed: {str(e)}")
            result.duration_ms = (time.time() - start_time) * 1000
            logger.exception(f"Scenario {scenario.id} failed")
        
        return result
    
    def run_suite(self, suite_name: Optional[str] = None) -> Dict[str, Any]:
        """Run all scenarios in a suite (or all scenarios)."""
        scenarios_to_run = self.scenarios
        if suite_name:
            scenarios_to_run = [s for s in self.scenarios if s.suite == suite_name]
        
        print(f"\n{'='*80}")
        print(f"Running {len(scenarios_to_run)} scenarios...")
        print(f"{'='*80}\n")
        
        self.results = []
        passed = 0
        failed = 0
        
        for i, scenario in enumerate(scenarios_to_run, 1):
            print(f"[{i}/{len(scenarios_to_run)}] {scenario.id}: {scenario.description}...", end=" ")
            
            result = self.run_scenario(scenario)
            self.results.append(result)
            
            if result.passed:
                print("✓ PASS")
                passed += 1
            else:
                print("✗ FAIL")
                failed += 1
                for error in result.errors:
                    print(f"    → {error}")
        
        print(f"\n{'='*80}")
        print(f"Results: {passed} passed, {failed} failed ({passed/(passed+failed)*100:.1f}% pass rate)")
        print(f"{'='*80}\n")
        
        return {
            "total": len(scenarios_to_run),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / (passed + failed) if (passed + failed) > 0 else 0.0,
            "results": self.results
        }
    
    def generate_report(self, output_file: Optional[Path] = None) -> str:
        """Generate detailed test report."""
        if not self.results:
            return "No test results available. Run scenarios first."
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("A.L.I.C.E. SCENARIO TEST REPORT")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Summary
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        pass_rate = passed / len(self.results) * 100 if self.results else 0
        
        report_lines.append("SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Scenarios:  {len(self.results)}")
        report_lines.append(f"Passed:           {passed} ({pass_rate:.1f}%)")
        report_lines.append(f"Failed:           {failed}")
        report_lines.append(f"Avg Duration:     {sum(r.duration_ms for r in self.results) / len(self.results):.2f}ms")
        report_lines.append("")
        
        # Failed scenarios detail
        if failed > 0:
            report_lines.append("FAILED SCENARIOS")
            report_lines.append("-" * 80)
            for result in self.results:
                if not result.passed:
                    report_lines.append(f"\n{result.scenario_id}")
                    report_lines.append(f"  Intent: {result.actual_intent or 'N/A'}")
                    report_lines.append(f"  Duration: {result.duration_ms:.2f}ms")
                    report_lines.append("  Errors:")
                    for error in result.errors:
                        report_lines.append(f"    - {error}")
                    if result.response:
                        report_lines.append(f"  Response: {result.response[:200]}")
            report_lines.append("")
        
        # Suite breakdown
        report_lines.append("SUITE BREAKDOWN")
        report_lines.append("-" * 80)
        suites = {}
        for result in self.results:
            scenario = next((s for s in self.scenarios if s.id == result.scenario_id), None)
            if scenario:
                suite = scenario.suite
                if suite not in suites:
                    suites[suite] = {"passed": 0, "failed": 0}
                if result.passed:
                    suites[suite]["passed"] += 1
                else:
                    suites[suite]["failed"] += 1
        
        for suite, stats in sorted(suites.items()):
            total = stats["passed"] + stats["failed"]
            rate = stats["passed"] / total * 100 if total > 0 else 0
            report_lines.append(f"  {suite:<20} {stats['passed']}/{total} ({rate:.1f}%)")
        
        report = "\n".join(report_lines)
        
        # Optionally save to file
        if output_file:
            try:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(report)
                    # Also save JSON results
                    f.write("\n\n" + "="*80 + "\n")
                    f.write("JSON RESULTS\n")
                    f.write("="*80 + "\n")
                    json.dump([asdict(r) for r in self.results], f, indent=2)
                print(f"\nReport saved to: {output_file}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
        
        return report


def main():
    """Main entry point for scenario testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run A.L.I.C.E. scenario tests")
    parser.add_argument("--suite", type=str, help="Run specific test suite")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    parser.add_argument("--output", type=str, help="Output file for report")
    parser.add_argument("--scenarios", type=str, help="Path to scenarios JSON file")
    
    args = parser.parse_args()
    
    # Initialize runner
    scenarios_path = Path(args.scenarios) if args.scenarios else None
    runner = ScenarioRunner(scenarios_path)
    
    # Load scenarios
    count = runner.load_scenarios()
    if count == 0:
        print("No scenarios loaded. Exiting.")
        return 1
    
    # Run tests
    results = runner.run_suite(args.suite)
    
    # Generate report if requested
    if args.report or args.output:
        output_path = Path(args.output) if args.output else Path("scenario_test_report.txt")
        report = runner.generate_report(output_path)
        if not args.output:
            print("\n" + report)
    
    # Return exit code based on results
    return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
