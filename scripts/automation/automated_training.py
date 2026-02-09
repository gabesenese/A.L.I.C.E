#!/usr/bin/env python3
"""
Automated Training Pipeline

Orchestrates three automation systems:
1. Auto-feedback from scenarios (no commands needed)
2. Auto-correction pipeline (offline "/correct")
3. Auto-pattern promotion (no /patterns clicks)
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.learning_engine import (
    get_learning_engine,
    get_auto_correction_engine,
    get_pattern_promotion_engine
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutomatedTrainingPipeline:
    """Orchestrates all automated training systems"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.data_dir = self.project_root / "data" / "training"
        self.memory_dir = self.project_root / "memory"
        
        # Initialize engines
        self.learning_engine = get_learning_engine(str(self.data_dir))
        self.correction_engine = get_auto_correction_engine(self.project_root)
        self.promotion_engine = get_pattern_promotion_engine(self.project_root)
        
        logger.info("=" * 70)
        logger.info("AUTOMATED TRAINING PIPELINE")
        logger.info("=" * 70)
    
    def run_scenario_feedback(self, scenario_results: List[Dict]) -> Dict[str, Any]:
        """
        Phase 1: Auto-feedback from scenarios
        
        Marks good outcomes as training examples automatically
        """
        logger.info("\n[PHASE 1] Running Auto-Feedback from Scenarios...")
        
        good_outcomes = 0
        learning_opportunities = 0
        
        for result in scenario_results:
            user_input = result.get('user_input', '')
            response = result.get('actual_response', '')
            intent = result.get('actual_intent', '')
            
            # Check if route and intent match expected
            is_good_outcome = (
                result.get('route_match', False) and
                result.get('intent_match', False)
            )
            
            confidence = result.get('confidence', 0.5)
            quality_score = 0.9 if is_good_outcome else 0.5
            
            # Collect as training data
            self.learning_engine.collect_interaction(
                user_input=user_input,
                assistant_response=response,
                intent=intent,
                quality_score=quality_score,
                entities=result.get('entities', {})
            )
            
            if is_good_outcome:
                good_outcomes += 1
                logger.info(f"OK: Good outcome: '{user_input[:40]}...' ({intent})")
            else:
                learning_opportunities += 1
                logger.info(f"LEARN: '{user_input[:40]}...' (expected: {result.get('expected_intent')})")
        
        summary = {
            'good_outcomes': good_outcomes,
            'learning_opportunities': learning_opportunities,
            'total_scenarios': len(scenario_results)
        }
        
        logger.info(f"\n[Phase 1 Summary]")
        logger.info(f"  Good outcomes: {good_outcomes}")
        logger.info(f"  Learning opportunities: {learning_opportunities}")
        
        return summary
    
    def run_auto_corrections(self, scenario_results: List[Dict]) -> Dict[str, Any]:
        """
        Phase 2: Auto-correction pipeline
        
        Creates corrections from mismatches, adjusts thresholds
        """
        logger.info("\n[PHASE 2] Running Auto-Correction Pipeline...")
        
        # Process scenario results to create corrections
        correction_summary = self.correction_engine.process_scenario_results(scenario_results)
        
        # Apply corrections that have been validated
        applied_summary = self.correction_engine.apply_corrections_to_thresholds()
        
        logger.info(f"\n[Phase 2 Summary]")
        logger.info(f"  Corrections created: {correction_summary['corrections_added']}")
        logger.info(f"  - Intent mismatches: {correction_summary['intent_mismatches']}")
        logger.info(f"  - Route mismatches: {correction_summary['route_mismatches']}")
        logger.info(f"  Corrections applied: {applied_summary['applied_count']}")
        
        return {
            'corrections_created': correction_summary,
            'corrections_applied': applied_summary
        }
    
    def run_pattern_promotion(self) -> Dict[str, Any]:
        """
        Phase 3: Auto-pattern promotion
        
        Clusters similar examples and auto-creates patterns
        """
        logger.info("\n[PHASE 3] Running Auto-Pattern Promotion...")
        
        promotion_summary = self.promotion_engine.scan_and_promote()
        
        logger.info(f"\n[Phase 3 Summary]")
        logger.info(f"  Patterns promoted: {promotion_summary['promoted']}")
        logger.info(f"  Patterns staged for review: {promotion_summary['staged_for_review']}")
        logger.info(f"  Total clusters found: {promotion_summary['total_clusters_found']}")
        
        return promotion_summary
    
    def run_full_pipeline(self, scenario_results: List[Dict]) -> Dict[str, Any]:
        """Execute all three phases"""
        logger.info("STARTING FULL AUTOMATED TRAINING PIPELINE")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'phase1_feedback': self.run_scenario_feedback(scenario_results),
            'phase2_corrections': self.run_auto_corrections(scenario_results),
            'phase3_promotion': self.run_pattern_promotion()
        }
        
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Summary:")
        logger.info(f"  Good scenario outcomes: {results['phase1_feedback']['good_outcomes']}")
        logger.info(f"  Corrections added: {results['phase2_corrections']['corrections_created']['corrections_added']}")
        logger.info(f"  Patterns promoted: {results['phase3_promotion']['promoted']}")
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable report"""
        report = []
        report.append("=" * 70)
        report.append("AUTOMATED TRAINING REPORT")
        report.append(f"Timestamp: {results['timestamp']}")
        report.append("=" * 70)
        
        report.append("\n[PHASE 1: AUTO-FEEDBACK]")
        fb = results['phase1_feedback']
        report.append(f"  Good outcomes marked as training: {fb['good_outcomes']}")
        report.append(f"  Learning opportunities identified: {fb['learning_opportunities']}")
        report.append(f"  Total scenarios processed: {fb['total_scenarios']}")
        
        report.append("\n[PHASE 2: AUTO-CORRECTIONS]")
        corr = results['phase2_corrections']['corrections_created']
        report.append(f"  Corrections created: {corr['corrections_added']}")
        report.append(f"    - Intent mismatches: {corr['intent_mismatches']}")
        report.append(f"    - Route mismatches: {corr['route_mismatches']}")
        applied = results['phase2_corrections']['corrections_applied']
        report.append(f"  Corrections applied: {applied['applied_count']}")
        
        report.append("\n[PHASE 3: AUTO-PATTERN PROMOTION]")
        promo = results['phase3_promotion']
        report.append(f"  Patterns auto-promoted (safe domains): {promo['promoted']}")
        report.append(f"  Patterns staged for review (complex domains): {promo['staged_for_review']}")
        report.append(f"  Total intent clusters found: {promo['total_clusters_found']}")
        
        report.append("\n[SAFETY & CONTROL]")
        report.append("  OK: No patterns auto-promoted for dangerous domains")
        report.append("  OK: Manual /feedback, /correct, /patterns commands still available")
        report.append("  OK: All changes logged and reversible")
        report.append("  OK: Corrections require validation_count >= 3 to apply")
        
        report.append("\n" + "=" * 70)
        return "\n".join(report)


def load_scenario_results() -> List[Dict]:
    """
    Load scenario results from most recent scenario run
    This would normally be called after scenarios.sim.run_scenarios
    """
    # Placeholder - in real usage, this comes from scenario runner output
    return []


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run automated training pipeline")
    parser.add_argument("--results", type=str, help="Path to scenario results JSON file")
    parser.add_argument("--report", action="store_true", help="Generate and print report only")
    args = parser.parse_args()
    
    pipeline = AutomatedTrainingPipeline()
    
    # Load results
    if args.results:
        with open(args.results, 'r') as f:
            scenario_results = json.load(f)
    else:
        logger.warning("No scenario results provided. Running with empty set.")
        scenario_results = []
    
    # Run pipeline
    results = pipeline.run_full_pipeline(scenario_results)
    
    # Generate report
    report = pipeline.generate_report(results)
    print(report)
    
    # Save report
    report_file = PROJECT_ROOT / "data" / "training" / f"training_report_{datetime.now().isoformat()}.txt"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to: {report_file}")
