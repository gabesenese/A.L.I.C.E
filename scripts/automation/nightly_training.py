"""
Nightly Training Pipeline for A.L.I.C.E

Runs automated scenarios, generates training data, and promotes patterns.
Integrates three automation phases:
  1. Auto-feedback: Mark good scenario outcomes as training
  2. Auto-corrections: Create corrections from mismatches
  3. Auto-pattern promotion: Cluster & auto-create patterns

Run this script daily via cron/Task Scheduler:
  - Windows Task Scheduler: Daily at 2 AM
  - Linux cron: 0 2 * * * cd /path/to/alice && python scripts/nightly_training.py
"""

import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Go up 2 levels: automation -> scripts -> project root
sys.path.insert(0, str(PROJECT_ROOT))

TRAINING_DIR = PROJECT_ROOT / "data" / "training"
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(TRAINING_DIR / "nightly_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_nightly_training():
    """Execute the complete nightly training pipeline"""
    
    logger.info("=" * 80)
    logger.info("[NIGHTLY] A.L.I.C.E NIGHTLY TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("=" * 80)
    
    try:
        # Phase 0: Run scenario simulations
        logger.info("\n[PHASE 0] Running Scenario Simulations")
        logger.info("-" * 80)
        
        from scenarios.sim.run_scenarios import ScenarioRunner
        
        runner = ScenarioRunner(
            llm_policy="minimal",  # Use minimal policy to test pattern-first approach
            use_teacher=True  # Enable teacher mode for ideal responses
        )
        
        # Run all scenarios
        runner.run_all()
        
        # Save results
        runner.save_results()
        
        logger.info(f"OK: Scenario simulations complete: {len(runner.results)} steps")
        
        # Convert results to format for automated training pipeline
        scenario_results = []
        for item in runner.training_data:
            scenario_results.append({
                'user_input': item.get('user_input', ''),
                'expected_intent': item.get('expected_intent', ''),
                'actual_intent': item.get('actual_intent', ''),
                'expected_route': item.get('expected_route', ''),
                'actual_route': item.get('actual_route', ''),
                'actual_response': item.get('alice_response', ''),
                'intent_match': item.get('intent_match', False),
                'route_match': item.get('route_match', False),
                'domain': item.get('domain', ''),
                'confidence': 0.7,  # Placeholder
                'entities': {}
            })
        
        # Phase 1-3: Run automated training pipeline
        logger.info("\n[PHASE 1-3] Running Automated Training Pipeline")
        logger.info("-" * 80)
        
        from scripts.automated_training import AutomatedTrainingPipeline
        
        pipeline = AutomatedTrainingPipeline()
        pipeline_results = pipeline.run_full_pipeline(scenario_results)
        
        # Phase 3b: Train ML models on the data we just collected
        logger.info("\n[PHASE 3b] Training Machine Learning Models")
        logger.info("-" * 80)
        
        from ai.models.ml_models import (
            get_router_classifier,
            get_intent_refiner,
            get_pattern_clusterer
        )
        
        try:
            # Load training logs
            training_file = PROJECT_ROOT / "data" / "training" / "training_data.jsonl"
            if training_file.exists():
                training_logs = []
                with open(training_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        if line.strip():
                            try:
                                training_logs.append(json.loads(line))
                            except:
                                pass
                
                if training_logs:
                    # Train router classifier
                    router = get_router_classifier()
                    router_data = [
                        {
                            'user_input': log.get('user_input', ''),
                            'intent': log.get('actual_intent', ''),
                            'confidence': log.get('confidence', 0.5),
                            'has_domain_keywords': any(
                                kw in log.get('user_input', '').lower()
                                for kw in ['email', 'weather', 'notes', 'calendar', 'time']
                            ),
                            'has_vague_pattern': 'vague' in log.get('actual_intent', '').lower(),
                            'actual_route': log.get('actual_route', 'LLM_FALLBACK')
                        }
                        for log in training_logs[-1000:]  # Use last 1000 examples
                    ]
                    if router_data:
                        trained = router.train(router_data)
                        if trained:
                            logger.info(f"[ML] Router classifier trained on {len(router_data)} examples")
                        else:
                            logger.info("[ML] Router classifier skipped (insufficient class diversity)")
                    
                    # Train intent refiner
                    refiner = get_intent_refiner()
                    corrections_file = PROJECT_ROOT / "memory" / "corrections.json"
                    if corrections_file.exists():
                        with open(corrections_file, 'r') as f:
                            corrections = json.load(f)
                        correction_data = [
                            {
                                'user_input': c.get('user_input', ''),
                                'original_intent': c.get('actual_intent', ''),
                                'corrected_intent': c.get('expected_intent', ''),
                                'confidence': c.get('confidence', 0.5)
                            }
                            for c in corrections
                            if c.get('correction_type') and 'intent' in c.get('correction_type', '')
                        ]
                        if correction_data:
                            refiner.train(correction_data)
                            logger.info(f"[ML] Intent refiner trained on {len(correction_data)} corrections")
                    
                    # Train pattern clusterer on fallback interactions
                    clusterer = get_pattern_clusterer()
                    fallback_logs = [
                        log for log in training_logs
                        if log.get('actual_route') == 'LLM_FALLBACK'
                    ]
                    if len(fallback_logs) >= 10:
                        clusterer.train(fallback_logs[-500:])
                        logger.info(f"[ML] Pattern clusterer trained on {len(fallback_logs)} fallback interactions")
        
        except Exception as e:
            logger.warning(f"[ML Training] Error training ML models: {e}")
        
        # Phase 4: Analyze real interaction fallbacks (if using LLM)
        logger.info("\n[PHASE 4] Analyzing Real Interaction Fallbacks")
        logger.info("-" * 80)
        
        from ai.training.teacher_loop import TeacherLoop
        
        try:
            teacher = TeacherLoop()
            
            # Analyze last 24 hours of real interactions
            suggestions = teacher.analyze_fallbacks(lookback_hours=24)
            
            logger.info(f"Found {len(suggestions)} learning opportunities from real interactions")
            
            # Auto-learn high-confidence patterns
            learned_count = teacher.auto_learn_high_confidence(suggestions)
            
            logger.info(f"OK: Auto-learned {learned_count} patterns from real interactions")
        except Exception as e:
            logger.warning(f"[Note] Real interaction analysis skipped: {e}")
            learned_count = 0
        
        # Phase 5: Generate summary report
        logger.info("\n[PHASE 5] Training Summary")
        logger.info("=" * 80)
        
        logger.info(f"Simulation & Automation Results:")
        logger.info(f"  - Scenarios Run: {len(runner.results)} steps")
        logger.info(f"  - Good outcomes marked as training: {pipeline_results['phase1_feedback']['good_outcomes']}")
        logger.info(f"  - Corrections created: {pipeline_results['phase2_corrections']['corrections_created']['corrections_added']}")
        logger.info(f"  - Patterns promoted: {pipeline_results['phase3_promotion']['promoted']}")
        logger.info(f"  - Patterns staged for review: {pipeline_results['phase3_promotion']['staged_for_review']}")
        logger.info(f"\nReal Interaction Analysis:")
        logger.info(f"  - Patterns auto-learned: {learned_count}")
        logger.info("=" * 80)
        logger.info(f"OK: Nightly training complete at: {datetime.now().isoformat()}")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Nightly training failed: {e}", exc_info=True)
        return False


def main():
    """Main entry point"""
    success = run_nightly_training()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
