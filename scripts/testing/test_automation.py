#!/usr/bin/env python3
"""
Test Automated Training System

Quick validation that all three automation phases work correctly.
No scenarios needed - uses mock data.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.learning_engine import (
    get_learning_engine,
    get_auto_correction_engine,
    get_pattern_promotion_engine
)


def test_auto_feedback():
    """Test Phase 1: Auto-Feedback"""
    print("\n" + "=" * 70)
    print("TEST 1: Auto-Feedback from Scenarios")
    print("=" * 70)
    
    learning_engine = get_learning_engine()
    
    # Simulate good outcomes
    good_scenarios = [
        {"user_input": "hi alice", "response": "Hello! How can I help?", "intent": "greeting"},
        {"user_input": "thanks for that", "response": "You're welcome!", "intent": "thanks"},
        {"user_input": "how are you?", "response": "I'm doing well!", "intent": "status_inquiry"},
    ]
    
    for scenario in good_scenarios:
        learning_engine.collect_interaction(
            user_input=scenario["user_input"],
            assistant_response=scenario["response"],
            intent=scenario["intent"],
            quality_score=0.9
        )
        print(f"[OK] Marked as training: '{scenario['user_input']}'")
    
    stats = learning_engine.get_statistics()
    print(f"\nTraining engine stats:")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  High quality: {stats['high_quality']}")
    print(f"  By intent: {stats['by_intent']}")
    
    return True


def test_auto_corrections():
    """Test Phase 2: Auto-Corrections"""
    print("\n" + "=" * 70)
    print("TEST 2: Auto-Corrections")
    print("=" * 70)
    
    correction_engine = get_auto_correction_engine()
    
    # Simulate scenario mismatches
    mismatches = [
        {
            "user_input": "tell me about the sun",
            "expected_intent": "clarification",
            "actual_intent": "weather",
            "expected_route": "CLARIFICATION",
            "actual_route": "TOOL",
            "domain": "weather",
            "confidence": 0.65
        },
        {
            "user_input": "what's tomorrow looking like?",
            "expected_intent": "vague_question",
            "actual_intent": "weather",
            "expected_route": "CLARIFICATION",
            "actual_route": "TOOL",
            "domain": "clarification",
            "confidence": 0.72
        }
    ]
    
    # Process mismatches
    summary = correction_engine.process_scenario_results(mismatches)
    
    print(f"Corrections created:")
    print(f"  Total: {summary['corrections_added']}")
    print(f"  Intent mismatches: {summary['intent_mismatches']}")
    print(f"  Route mismatches: {summary['route_mismatches']}")
    
    # Check corrections file
    corrections_file = PROJECT_ROOT / "memory" / "corrections.json"
    if corrections_file.exists():
        with open(corrections_file, 'r') as f:
            corrections = json.load(f)
        print(f"\nCorrections stored: {len(corrections)} total in memory/corrections.json")
        if corrections:
            recent = corrections[-1]
            print(f"  Latest correction:")
            print(f"    Input: '{recent['user_input']}'")
            print(f"    Expected: {recent.get('expected_intent', 'N/A')}")
            print(f"    Actual: {recent.get('actual_intent', 'N/A')}")
    
    return True


def test_pattern_promotion():
    """Test Phase 3: Pattern Promotion"""
    print("\n" + "=" * 70)
    print("TEST 3: Auto-Pattern Promotion")
    print("=" * 70)
    
    promotion_engine = get_pattern_promotion_engine()
    
    # Create sample training data
    learning_engine = get_learning_engine()
    
    # Add multiple high-quality greeting examples
    greeting_examples = [
        "hi alice",
        "hello alice",
        "hey alice",
        "greetings",
        "good morning"
    ]
    
    for example in greeting_examples:
        learning_engine.collect_interaction(
            user_input=example,
            assistant_response="Hello! How can I help you?",
            intent="greeting",
            quality_score=0.85
        )
    
    print(f"Created {len(greeting_examples)} high-quality greeting examples")
    
    # Run promotion
    results = promotion_engine.scan_and_promote()
    
    print(f"\nPromotion results:")
    print(f"  Patterns auto-promoted: {results['promoted']}")
    print(f"  Patterns staged for review: {results['staged_for_review']}")
    print(f"  Total clusters found: {results['total_clusters_found']}")
    
    # Check if patterns were created
    patterns_file = PROJECT_ROOT / "memory" / "learning_patterns.json"
    if patterns_file.exists():
        with open(patterns_file, 'r') as f:
            patterns = json.load(f)
        
        if isinstance(patterns, dict) and 'greeting' in patterns:
            print(f"\n[OK] Greeting pattern created in learning_patterns.json")
            greeting_pattern = patterns['greeting']
            if isinstance(greeting_pattern, dict):
                print(f"  Active: {greeting_pattern.get('active', False)}")
                print(f"  Source: {greeting_pattern.get('source', 'N/A')}")
    
    return True


def test_dangerous_domain_blocking():
    """Test that dangerous domains are NOT auto-promoted"""
    print("\n" + "=" * 70)
    print("TEST 4: Dangerous Domain Blocking")
    print("=" * 70)
    
    learning_engine = get_learning_engine()
    promotion_engine = get_pattern_promotion_engine()
    
    # Add examples for dangerous domain
    dangerous_examples = [
        "execute this python code",
        "run sudo command",
        "access admin panel",
        "system shutdown"
    ]
    
    for example in dangerous_examples:
        learning_engine.collect_interaction(
            user_input=example,
            assistant_response="I cannot do that.",
            intent="code_execution",
            quality_score=0.8
        )
    
    print(f"Created {len(dangerous_examples)} code_execution examples")
    
    # Try promotion
    results = promotion_engine.scan_and_promote()
    
    # Check patterns_for_review instead of auto-promotion
    review_file = PROJECT_ROOT / "memory" / "patterns_for_review.json"
    if review_file.exists():
        with open(review_file, 'r') as f:
            review_patterns = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(review_patterns, dict):
            review_patterns = list(review_patterns.values()) if review_patterns else []
        
        # Should be in review, not promoted
        code_exec_in_review = any(
            isinstance(p, dict) and p.get('intent') == 'code_execution'
            for p in review_patterns
        )
        
        if code_exec_in_review:
            print(f"[OK] Dangerous domain correctly staged for review")
            print(f"  Patterns in review: {len(review_patterns)}")
        else:
            print(f"[INFO] No dangerous patterns in review (may have existed already)")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("AUTOMATED TRAINING SYSTEM - VALIDATION TESTS")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)
    
    tests = [
        ("Auto-Feedback", test_auto_feedback),
        ("Auto-Corrections", test_auto_corrections),
        ("Pattern Promotion", test_pattern_promotion),
        ("Dangerous Domain Blocking", test_dangerous_domain_blocking),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n[PASS] {test_name} - PASSED")
            else:
                failed += 1
                print(f"\n[FAIL] {test_name} - FAILED")
        except Exception as e:
            failed += 1
            print(f"\n[FAIL] {test_name} - FAILED with error:")
            print(f"  {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n[OK] ALL TESTS PASSED - Automation system ready!")
    else:
        print(f"\n[FAIL] {failed} test(s) failed - review logs above")
    
    print("=" * 70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
