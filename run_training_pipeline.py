#!/usr/bin/env python3
"""Run all scenarios then feed results into the automated training pipeline."""
import sys
import logging
logging.disable(logging.WARNING)
sys.path.insert(0, '.')

from test_scenarios import ScenarioRunner
from scripts.automation.automated_training import AutomatedTrainingPipeline

print("Running all scenarios...")
runner = ScenarioRunner()
runner.load_scenarios()
suite_results = runner.run_suite()

# Convert TestResult objects to pipeline dict format
scenario_map = {s.id: s for s in runner.scenarios}
pipeline_input = []
for r in suite_results["results"]:
    sc = scenario_map.get(r.scenario_id)
    pipeline_input.append({
        "user_input":      (sc.inputs[-1] if sc and sc.inputs else r.scenario_id),
        "actual_response": r.response or "",
        "actual_intent":   r.actual_intent or "",
        "expected_intent": (sc.expected_intent or "") if sc else "",
        "confidence":      0.8 if r.passed else 0.3,
        "intent_match":    r.passed,
        "route_match":     r.passed,
        "entities":        {},
    })

print(f"\nScenarios done: {suite_results['passed']} passed, {suite_results['failed']} failed")
print("\nRunning automated training pipeline (--force-apply)...")
pipeline = AutomatedTrainingPipeline()
results = pipeline.run_full_pipeline(pipeline_input, force_apply=True)
report = pipeline.generate_report(results)
print(report)
