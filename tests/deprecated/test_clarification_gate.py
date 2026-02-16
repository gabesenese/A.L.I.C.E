"""
Test Clarification Gate Implementation
Verifies routing decisions for vague vs. clear questions
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai.router import RequestRouter

def test_clarification_gate():
    """Test clarification gate with various inputs"""
    router = RequestRouter()
    
    test_cases = [
        # (user_input, intent, confidence, should_clarify, description)
        (
            "i have a question about the sun",
            "weather_query",
            0.59,
            True,
            "Vague pattern 'question about' + low confidence → should clarify"
        ),
        (
            "what's the temperature in New York",
            "weather_query",
            0.85,
            False,
            "High confidence + domain keywords (temperature, New York) → execute tool"
        ),
        (
            "tell me about weather",
            "weather_query",
            0.62,
            True,
            "Vague pattern 'tell me about' + low confidence → should clarify"
        ),
        (
            "check my email",
            "email_read",
            0.88,
            False,
            "High confidence + clear intent → execute tool"
        ),
        (
            "i'm curious about the weather outside",
            "weather_query",
            0.68,
            False,
            "Domain keywords (weather, outside) despite low confidence → execute tool"
        ),
        (
            "what about my calendar",
            "calendar_read",
            0.65,
            True,
            "Vague pattern 'what about' + low confidence + only 1 keyword → should clarify"
        ),
        (
            "schedule a meeting tomorrow at 2pm",
            "calendar_create",
            0.92,
            False,
            "High confidence + multiple calendar keywords → execute tool"
        ),
        (
            "i wanted to ask about files",
            "file_search",
            0.58,
            False,
            "Has domain keyword 'file' but low confidence + vague pattern 'ask about' → should clarify"
        ),
    ]
    
    print("=" * 80)
    print("CLARIFICATION GATE TEST")
    print("=" * 80)
    print()
    
    passed = 0
    failed = 0
    
    for user_input, intent, confidence, expected_clarify, description in test_cases:
        # Call should_clarify
        result = router.should_clarify(
            intent=intent,
            confidence=confidence,
            entities={},
            user_text=user_input
        )
        
        # Check result
        status = "PASS" if result == expected_clarify else "FAIL"
        if result == expected_clarify:
            passed += 1
        else:
            failed += 1
        
        # Print result
        print(f"{status}")
        print(f"  Input: '{user_input}'")
        print(f"  Intent: {intent} (conf={confidence})")
        print(f"  Expected: {'CLARIFY' if expected_clarify else 'EXECUTE'}")
        print(f"  Got: {'CLARIFY' if result else 'EXECUTE'}")
        print(f"  Reason: {description}")
        print()
    
    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)
    
    return failed == 0


def test_routing_priority():
    """Test full routing with clarification gate"""
    router = RequestRouter()
    
    test_cases = [
        (
            "i have a question about the sun",
            "weather_query",
            0.59,
            "CONVERSATIONAL",
            "Should route to CONVERSATIONAL for clarification"
        ),
        (
            "what's the temperature in New York",
            "weather_query",
            0.85,
            "TOOL",
            "Should route to TOOL with clear domain keywords"
        ),
        (
            "hi alice",
            "greeting",
            0.95,
            "CONVERSATIONAL",
            "Should route to CONVERSATIONAL for greeting pattern"
        ),
        (
            "i wanted to ask about files",
            "file_search",
            0.58,
            "CONVERSATIONAL",
            "Should route to CONVERSATIONAL - low confidence, no domain keywords detected"
        ),
    ]
    
    print()
    print("=" * 80)
    print("ROUTING PRIORITY TEST")
    print("=" * 80)
    print()
    
    passed = 0
    failed = 0
    
    for user_input, intent, confidence, expected_decision, description in test_cases:
        # Route the request
        result = router.route(
            intent=intent,
            confidence=confidence,
            entities={},
            context={},
            user_text=user_input
        )
        
        # Check result
        decision_name = result.decision.value.upper()
        status = "PASS" if decision_name == expected_decision else "FAIL"
        if decision_name == expected_decision:
            passed += 1
        else:
            failed += 1
        
        # Print result
        print(f"{status}")
        print(f"  Input: '{user_input}'")
        print(f"  Intent: {intent} (conf={confidence})")
        print(f"  Expected: {expected_decision}")
        print(f"  Got: {decision_name}")
        print(f"  Reasoning: {result.reasoning}")
        print(f"  Metadata: {result.metadata}")
        print()
    
    print("=" * 80)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "CLARIFICATION GATE TEST SUITE" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Run tests
    test1_passed = test_clarification_gate()
    test2_passed = test_routing_priority()
    
    # Final summary
    print()
    print("=" * 80)
    if test1_passed and test2_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED")
    print("=" * 80)
    print()
    
    sys.exit(0 if (test1_passed and test2_passed) else 1)
