#!/usr/bin/env python3
"""
Run Scenarios + Automated Training Pipeline

Executes scenarios, captures results, and runs all three automation phases:
1. Auto-feedback (mark good outcomes)
2. Auto-corrections (create from mismatches)
3. Auto-pattern promotion (cluster & create patterns)

Usage:
    python -m scripts.run_scenarios_and_train [--policy minimal] [--domains email notes]
"""

import sys
import os
import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.automated_training import AutomatedTrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_scenarios(policy: str = "minimal", domains: List[str] = None) -> List[Dict]:
    """
    Run scenario runner and capture results
    
    Returns:
        List of scenario results with actual vs expected values
    """
    logger.info("=" * 70)
    logger.info("RUNNING SCENARIOS")
    logger.info("=" * 70)
    
    # Build command
    cmd = [sys.executable, "-m", "scenarios.sim.run_scenarios", "--policy", policy]
    
    if domains:
        cmd.extend(["--domains"] + domains)
    
    # Run scenarios
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300  # 5 min timeout
        )
        
        if result.returncode != 0:
            logger.error(f"Scenario runner failed with code {result.returncode}")
            logger.error(result.stderr)
            return []
        
        # Parse output to extract results
        # This looks for JSON output from the scenario runner
        scenario_results = _extract_scenario_results(result.stdout + result.stderr)
        
        logger.info(f"OK: Scenarios completed: {len(scenario_results)} steps executed")
        return scenario_results
    
    except subprocess.TimeoutExpired:
        logger.error("Scenario runner timed out")
        return []
    except Exception as e:
        logger.error(f"Error running scenarios: {e}")
        return []


def _extract_scenario_results(output: str) -> List[Dict]:
    """
    Extract scenario results from runner output
    
    The scenario runner outputs JSON structures in training_data.jsonl
    We'll read the most recent ones from data/training/auto_generated.jsonl
    """
    training_file = PROJECT_ROOT / "data" / "training" / "auto_generated.jsonl"
    
    results = []
    
    # Get file size before running scenarios to know how many lines to read
    # For now, just read all lines as new scenarios append
    if training_file.exists():
        try:
            with open(training_file, 'r') as f:
                # Read last 100 lines (assuming recent scenario run)
                lines = f.readlines()[-100:]
                for line in lines:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            results.append(data)
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            logger.warning(f"Error reading scenario results: {e}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run scenarios and automated training pipeline"
    )
    parser.add_argument(
        '--policy',
        type=str,
        choices=['default', 'minimal', 'strict'],
        default='minimal',
        help='LLM policy for scenarios'
    )
    parser.add_argument(
        '--domains',
        nargs='+',
        help='Only run scenarios from specific domains'
    )
    parser.add_argument(
        '--skip-scenarios',
        action='store_true',
        help='Skip scenario execution, use existing data'
    )
    parser.add_argument(
        '--feedback-only',
        action='store_true',
        help='Run only auto-feedback phase (no corrections/promotion)'
    )
    
    args = parser.parse_args()
    
    logger.info("INTEGRATED SCENARIO & TRAINING PIPELINE")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    
    # Phase 0: Run scenarios
    if args.skip_scenarios:
        logger.info("Skipping scenario execution (--skip-scenarios)")
        scenario_results = _extract_scenario_results("")
    else:
        scenario_results = run_scenarios(
            policy=args.policy,
            domains=args.domains
        )
    
    if not scenario_results:
        logger.warning("No scenario results to process")
        return
    
    # Phase 1-3: Run automated training pipeline
    pipeline = AutomatedTrainingPipeline()
    
    if args.feedback_only:
        logger.info("\nRunning PHASE 1 only (Auto-Feedback)")
        results = {
            'timestamp': datetime.now().isoformat(),
            'phase1_feedback': pipeline.run_scenario_feedback(scenario_results)
        }
    else:
        logger.info("\nRunning PHASES 1-3 (Full Pipeline)")
        results = pipeline.run_full_pipeline(scenario_results)
    
    # Generate and display report
    report = pipeline.generate_report(results)
    print("\n" + report)
    
    # Save report
    report_dir = PROJECT_ROOT / "data" / "training"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = report_dir / f"pipeline_report_{datetime.now().isoformat().replace(':', '-')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    logger.info(f"\nReport saved to: {report_file}")
    logger.info(f"Completed at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
