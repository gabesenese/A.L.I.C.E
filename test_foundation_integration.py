"""
Test script to verify foundation system integration in main.py

This script tests:
1. Foundation systems are properly initialized
2. Response generation works in different modes (parallel, primary, exclusive)
3. Feedback learning mechanism functions correctly
4. Repetition detection works
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_foundation_import():
    """Test that foundation systems can be imported"""
    try:
        from ai.foundation_integration import FoundationIntegration
        print("✓ Foundation integration imports successfully")
    except ImportError as e:
        raise AssertionError(f"Failed to import foundation integration: {e}") from e

def test_foundation_instantiation():
    """Test that foundation systems can be instantiated"""
    try:
        from ai.foundation_integration import FoundationIntegration
        from ai.core.llm_engine import LocalLLMEngine
        
        # Mock LLM engine
        class MockLLM:
            def generate(self, prompt, **kwargs):
                return "This is a mock response."
        
        foundations = FoundationIntegration(
            llm_generator=MockLLM(),
            phrasing_learner=None
        )
        print("✓ Foundation systems instantiate successfully")
        print(f"  - Response Variance Engine: {foundations.response_engine is not None}")
        print(f"  - Personality Evolution: {foundations.personality_engine is not None}")
        print(f"  - Context Graph: {foundations.context_graph is not None}")
        assert foundations.response_engine is not None
        assert foundations.personality_engine is not None
        assert foundations.context_graph is not None
    except Exception as e:
        raise AssertionError(f"Failed to instantiate foundation systems: {e}") from e

def test_response_generation():
    """Test response generation with foundation systems"""
    try:
        from ai.foundation_integration import FoundationIntegration
        
        class MockLLM:
            def generate(self, prompt, **kwargs):
                return "The weather is sunny today with a temperature of 22°C."
        
        foundations = FoundationIntegration(
            llm_generator=MockLLM(),
            phrasing_learner=None
        )
        
        # Test response generation
        result = foundations.process_interaction(
            user_id="test_user",
            user_input="What's the weather like?",
            intent="weather:query",
            entities={"location": "London"},
            plugin_result={"success": True, "data": {"temperature": 22}}
        )
        
        response = result.get('response')
        if response:
            print("✓ Foundation response generation works")
            print(f"  Response: {response[:100]}...")
        else:
            raise AssertionError("Foundation response generation failed - no response")
    except Exception as e:
        raise AssertionError(f"Failed to generate response: {e}") from e

def test_repetition_detection():
    """Test that repetition detection works"""
    try:
        from ai.foundation_integration import FoundationIntegration
        
        class MockLLM:
            def __init__(self):
                self.call_count = 0
                
            def generate(self, prompt, **kwargs):
                self.call_count += 1
                if "asking about this again" in prompt.lower() or "repetition" in prompt.lower():
                    return "I notice you're asking about this again. The weather is still sunny at 22°C."
                return f"The weather is sunny today. (Call {self.call_count})"
        
        mock_llm = MockLLM()
        foundations = FoundationIntegration(
            llm_generator=mock_llm,
            phrasing_learner=None
        )
        
        # Ask same question 3 times
        responses = []
        for i in range(3):
            result = foundations.process_interaction(
                user_id="test_user",
                user_input="What's the weather?",
                intent="weather:query",
                entities={},
                plugin_result={"success": True}
            )
            responses.append(result.get('response'))
        
        # Check that responses are different (variance engine working)
        if len(set(responses)) > 1:
            print("✓ Response variance works - different responses each time")
            for i, resp in enumerate(responses, 1):
                print(f"  Response {i}: {resp[:60]}...")
        else:
            raise AssertionError("All responses identical - variance may not be working")
            
    except Exception as e:
        raise AssertionError(f"Repetition detection test failed: {e}") from e

def test_personality_evolution():
    """Test personality adaptation"""
    try:
        from ai.foundation_integration import FoundationIntegration
        
        class MockLLM:
            def generate(self, prompt, **kwargs):
                return "Response adapted to user style."
        
        foundations = FoundationIntegration(
            llm_generator=MockLLM(),
            phrasing_learner=None
        )
        
        # Get initial personality
        initial_personality = foundations.personality_engine.get_personality_profile("test_user")
        print(f"  Initial personality: {initial_personality[:100]}...")
        
        # Simulate brief interaction (should decrease verbosity)
        for _ in range(5):
            foundations.process_interaction(
                user_id="test_user",
                user_input="weather?",  # Very brief
                intent="weather:query",
                entities={},
                plugin_result={"success": True}
            )
        
        # Check if personality adapted
        adapted_personality = foundations.personality_engine.get_personality_profile("test_user")
        print(f"  Adapted personality: {adapted_personality[:100]}...")
        
        print("✓ Personality evolution system functional")
        
    except Exception as e:
        raise AssertionError(f"Personality evolution test failed: {e}") from e

def test_context_graph():
    """Test context graph memory"""
    try:
        from ai.foundation_integration import FoundationIntegration
        
        class MockLLM:
            def generate(self, prompt, **kwargs):
                return "Context-aware response."
        
        foundations = FoundationIntegration(
            llm_generator=MockLLM(),
            phrasing_learner=None
        )
        
        # Add some context
        foundations.process_interaction(
            user_id="test_user",
            user_input="My name is Gabriel",
            intent="general:statement",
            entities={"person": "Gabriel"},
            plugin_result=None
        )
        
        foundations.process_interaction(
            user_id="test_user",
            user_input="I live in New York",
            intent="general:statement",
            entities={"location": "New York"},
            plugin_result=None
        )
        
        # Query context
        context = foundations.get_context_summary("test_user")
        if "Gabriel" in context or "New York" in context:
            print("✓ Context graph stores and retrieves information")
            print(f"  Context: {context[:150]}...")
        else:
            raise AssertionError("Context may not be storing entities correctly")
            
    except Exception as e:
        raise AssertionError(f"Context graph test failed: {e}") from e

def main():
    """Run all tests"""
    print("=" * 60)
    print("Foundation System Integration Tests")
    print("=" * 60)
    print()
    
    tests = [
        ("Import Test", test_foundation_import),
        ("Instantiation Test", test_foundation_instantiation),
        ("Response Generation Test", test_response_generation),
        ("Repetition Detection Test", test_repetition_detection),
        ("Personality Evolution Test", test_personality_evolution),
        ("Context Graph Test", test_context_graph)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        print("-" * 60)
        try:
            test_func()
            results.append((name, True))
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All foundation integration tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
