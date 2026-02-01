"""
Nightly Training Pipeline for A.L.I.C.E

Runs automated scenarios, generates training data, and promotes patterns.
This is the "Tony Stark" overnight improvement system.

Run this script daily via cron/Task Scheduler:
  - Windows Task Scheduler: Daily at 2 AM
  - Linux cron: 0 2 * * * cd /path/to/alice && python scripts/nightly_training.py
"""

import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/training/nightly_training.log"),
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
        # Step 1: Run scenario simulations
        logger.info("\n[STEP 1] Running Scenario Simulations")
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
        
        logger.info(f"[OK] Scenario simulations complete")
        
        # Step 2: Promote patterns from logs
        logger.info("\n[STEP 2] Promoting Patterns from Simulation Logs")
        logger.info("-" * 80)
        
        from ai.promote_patterns import PatternPromoter
        
        promoter = PatternPromoter(
            min_frequency=3,  # Pattern must appear at least 3 times
            min_teacher_consistency=0.8,  # Teacher must be 80% consistent
            min_alice_agreement=0.7  # Alice should agree with teacher 70% of time
        )
        
        # Analyze logs
        log_file = Path("data/training/auto_generated.jsonl")
        candidates = promoter.analyze_logs(log_file)
        
        logger.info(f"Found {len(candidates)} pattern candidates")
        
        # Promote patterns with auto-apply enabled
        promoted_count = promoter.promote_patterns(
            candidates,
            auto_apply=True  # Auto-promote high-confidence patterns
        )
        
        logger.info(f"[OK] Pattern promotion complete ({promoted_count} patterns promoted)")
        
        # Step 3: Analyze LLM fallbacks from real interactions (if any)
        logger.info("\n[STEP 3] Analyzing Real Interaction Fallbacks")
        logger.info("-" * 80)
        
        from ai.teacher_loop import TeacherLoop
        
        teacher = TeacherLoop()
        
        # Analyze last 24 hours of real interactions
        suggestions = teacher.analyze_fallbacks(lookback_hours=24)
        
        logger.info(f"Found {len(suggestions)} learning opportunities from real interactions")
        
        # Auto-learn high-confidence patterns
        learned_count = teacher.auto_learn_high_confidence(suggestions)
        
        logger.info(f"[OK] Auto-learned {learned_count} patterns from real interactions")
        
        # Step 4: Generate summary report
        logger.info("\n[STEP 4] Training Summary")
        logger.info("=" * 80)
        logger.info(f"Simulation Results:")
        logger.info(f"  - Scenarios Run: {len(runner.results)} steps")
        logger.info(f"  - Training Data Generated: {len(runner.training_data)} interactions")
        logger.info(f"  - Patterns Promoted: {promoted_count}")
        logger.info(f"\nReal Interaction Analysis:")
        logger.info(f"  - LLM Fallbacks Found: {len(suggestions)}")
        logger.info(f"  - Patterns Auto-Learned: {learned_count}")
        logger.info("=" * 80)
        logger.info(f"[OK] Nightly training complete at: {datetime.now().isoformat()}")
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
