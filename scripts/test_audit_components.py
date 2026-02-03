"""
Simple Audit Component Test
Tests audit modules without full ALICE initialization
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_audit_components():
    """Test individual audit components"""
    
    print("\n" + "="*70)
    print("AUDIT COMPONENT TEST")
    print("="*70)
    
    # Test 1: Teaching spec
    print("\n[1/5] Testing teaching specification...")
    try:
        from ai.ollama_teaching_spec import TEACHING_VECTORS, get_domain_vectors
        
        domains = list(TEACHING_VECTORS.keys())
        print(f"✓ Loaded {len(domains)} domains: {', '.join(domains)}")
        
        for domain in domains[:2]:
            vectors = get_domain_vectors(domain)
            print(f"  - {domain}: {len(vectors)} skills")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test 2: Auditor spec
    print("\n[2/5] Testing audit dimensions...")
    try:
        from ai.ollama_auditor_spec import AUDIT_DIMENSIONS, ScoringDimension
        
        domains_audited = list(AUDIT_DIMENSIONS.keys())
        print(f"✓ Loaded {len(domains_audited)} audited domains: {', '.join(domains_audited)}")
        
        for domain in domains_audited[:2]:
            dims = AUDIT_DIMENSIONS[domain]
            print(f"  - {domain}: {len(dims)} dimensions")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test 3: Scorer
    print("\n[3/5] Testing scorer...")
    try:
        from ai.ollama_scorer import create_scorer
        
        scorer = create_scorer()
        print(f"✓ Scorer initialized")
        print(f"  Signals created: 0")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test 4: Feedback injector
    print("\n[4/5] Testing feedback injector...")
    try:
        from ai.ollama_feedback_injector import create_injector
        
        injector = create_injector()
        print(f"✓ Feedback injector initialized")
        print(f"  Feedback log: data/training/audit_feedback.jsonl")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test 5: Metric tracker
    print("\n[5/5] Testing metric tracker...")
    try:
        from ai.metric_tracker import create_tracker
        
        tracker = create_tracker()
        print(f"✓ Metric tracker initialized")
        print(f"  Metrics file: data/training/metrics/domain_metrics.jsonl")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    print("\n" + "="*70)
    print("✓ ALL AUDIT COMPONENTS FUNCTIONAL")
    print("="*70)
    
    print("\nNEXT STEPS:")
    print("  1. Audit components are ready to use")
    print("  2. Build a simple test with mock responses")
    print("  3. Or integrate with full ALICE when network is stable")
    print("\n")
    
    return True


if __name__ == "__main__":
    success = test_audit_components()
    sys.exit(0 if success else 1)
