"""
Start Automation
Initializes and starts nightly audit scheduler (run after test passes)
"""

import sys
import os
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    """Start automation scheduler"""
    
    print("\n" + "="*70)
    print("STARTING NIGHTLY AUDIT AUTOMATION")
    print("="*70)
    
    # Initialize components
    print("\n[1/2] Initializing components...")
    try:
        from app.alice import ALICE
        from ai.llm_engine import LocalLLMEngine, LLMConfig
        from ai.ollama_teacher import create_teacher
        from ai.ollama_auditor import create_auditor
        from ai.ollama_scorer import create_scorer
        from ai.ollama_feedback_injector import create_injector
        from ai.metric_tracker import create_tracker
        from ai.nightly_audit_scheduler import create_scheduler
        from ai.audit_config_optimizer import create_optimizer
        
        alice = ALICE(debug=False)
        llm = LocalLLMEngine(config=LLMConfig(model="llama3.1:8b"))
        
        scheduler = create_scheduler(
            alice,
            create_teacher(llm),
            create_auditor(llm),
            create_scorer(),
            create_injector(),
            create_tracker()
        )
        
        optimizer = create_optimizer()
        
        print("✓ ALICE initialized")
        print("✓ LLM engine ready")
        print("✓ Teacher (query generator) ready")
        print("✓ Auditor (grader) ready")
        print("✓ Scorer (signal generator) ready")
        print("✓ Injector (training pipeline) ready")
        print("✓ Tracker (metrics) ready")
        print("✓ Optimizer (auto-tune) ready")
    
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Start scheduler
    print("\n[2/2] Starting scheduler...")
    try:
        schedule_thread = scheduler.start_scheduler(hour=2, minute=0)
        
        print("✓ Scheduler started")
        print("\nCONFIGURATION:")
        status = scheduler.scheduler.get_status()
        print(f"  - Status: Running")
        print(f"  - Schedule: Daily at 02:00 (2 AM)")
        print(f"  - Scheduled jobs: {status['scheduled_jobs']}")
        print(f"  - Next run: {status['next_run']}")
        
    except Exception as e:
        print(f"✗ Failed to start scheduler: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("✓ AUTOMATION STARTED")
    print("="*70)
    
    print("\nWHAT HAPPENS NIGHTLY (2 AM):")
    print("  1. Generate test queries for all domains (teacher)")
    print("  2. Get Alice responses for each query")
    print("  3. Audit responses across dimensions (auditor)")
    print("  4. Score audits and generate training signals (scorer)")
    print("  5. Inject signals into training data (injector)")
    print("  6. Track pre-training metrics (tracker)")
    print("  7. Fine-tune model on domain-specific data")
    print("  8. Compare post-training metrics")
    print("  9. Auto-adjust parameters if needed (optimizer)")
    
    print("\nMONITOR PROGRESS:")
    print("  - View metrics: data/training/metrics/domain_metrics.jsonl")
    print("  - View audit feedback: data/training/audit_feedback.jsonl")
    print("  - View domain datasets: data/training/{domain}_feedback.json")
    print("  - Run: python scripts/monitor_audit_progress.py")
    
    print("\nTO STOP SCHEDULER:")
    print("  - Kill this process (Ctrl+C)")
    print("  - Or call: scheduler.stop_scheduler()")
    
    print("\n" + "="*70 + "\n")
    
    # Keep running
    try:
        import time
        print("✓ Scheduler running in background. Press Ctrl+C to stop.\n")
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\n\nStopping scheduler...")
        scheduler.stop_scheduler()
        print("✓ Scheduler stopped")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
