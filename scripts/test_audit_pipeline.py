"""
Quick Audit Pipeline Test
Verifies full pipeline works before setting up automation
"""

import sys
import os
import logging
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run audit pipeline test"""
    
    print("\n" + "="*70)
    print("OLLAMA AUDIT PIPELINE TEST")
    print("="*70)
    
    # Step 1: Initialize Alice and LLM
    print("\n[1/3] Initializing ALICE and LLM...")
    try:
        from app.alice import ALICE
        from ai.llm_engine import LocalLLMEngine, LLMConfig
        
        alice = ALICE(debug=False)
        llm = LocalLLMEngine(config=LLMConfig(model="llama3.1:8b"))
        print("✓ Alice initialized")
        print("✓ LLM engine ready")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return False
    
    # Step 2: Run test pipeline
    print("\n[2/3] Running end-to-end audit cycle...")
    print("     (Testing weather and email domains)")
    try:
        from ai.test_audit_cycle import test_full_audit_pipeline
        
        results = test_full_audit_pipeline(
            alice,
            llm,
            domains=['weather', 'email'],
            skills_per_domain=1,
            queries_per_skill=2
        )
        
        if results['status'] != 'complete':
            print(f"✗ Test failed: {results.get('status')}")
            return False
        
        print(f"✓ Pipeline test complete")
        print(f"  - Domains tested: {results['domains_tested']}")
        print(f"  - Total queries: {results['total_queries']}")
        print(f"  - Total audits: {results['total_audits']}")
        print(f"  - Signals generated: {results['total_signals']}")
    
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Show results
    print("\n[3/3] Analyzing results...")
    try:
        import json
        
        results_file = Path("data/training/test_results.json")
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
            
            print("\n" + "="*70)
            print("TEST RESULTS SUMMARY")
            print("="*70)
            
            for domain, domain_data in data.get('domain_results', {}).items():
                print(f"\n{domain.upper()}")
                print(f"  Skills tested: {domain_data['skills_tested']}")
                print(f"  Queries: {domain_data['queries']}")
                print(f"  Audits: {domain_data['audits']}")
                print(f"  Signals: {domain_data['signals']}")
                print(f"  Avg score: {domain_data['avg_score']:.2f}/5.0")
            
            print("\n" + "="*70)
            print("OUTPUT ARTIFACTS")
            print("="*70)
            print(f"✓ Test results: data/training/test_results.json")
            print(f"✓ Feedback log: data/training/audit_feedback.jsonl")
            print(f"✓ Domain datasets: data/training/{{domain}}_feedback.json")
            
    except Exception as e:
        print(f"✗ Failed to show results: {e}")
        return False
    
    print("\n" + "="*70)
    print("✓ PIPELINE TEST SUCCESSFUL")
    print("="*70)
    
    print("\nNEXT STEPS:")
    print("  1. Review results in: data/training/test_results.json")
    print("  2. Check domain feedback in: data/training/*_feedback.json")
    print("  3. If satisfied, run: python scripts/start_automation.py")
    print("\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
