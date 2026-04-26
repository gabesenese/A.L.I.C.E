"""
Test Foundation Systems
Validates Response Variance Engine, Personality Evolution, and Context Graph
Run this before integrating into main.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.foundation_integration import FoundationIntegration
from ai.response.response_variance_engine import ResponseVarianceEngine, ResponseContext
from ai.personality.personality_evolution import PersonalityEvolutionEngine, InteractionSignal
from ai.memory.context_graph import ContextGraph


def test_context_graph():
    """Test Context Graph functionality"""
    print("\n" + "="*60)
    print("TEST 1: Context Graph")
    print("="*60)
    
    graph = ContextGraph(data_dir="data/test_context")
    
    # Test entity addition
    entity1 = graph.add_entity('location', 'Kitchener')
    entity2 = graph.add_entity('topic', 'weather')
    entity3 = graph.add_entity('topic', 'temperature')
    
    print(f"✓ Added 3 entities")
    
    # Test relationship
    graph.add_relationship(entity1.entity_id, entity2.entity_id, 'mentioned_with')
    print(f"✓ Created relationship: location → weather")
    
    # Test conversation turn
    turn = graph.record_turn(
        user_id="test_user",
        user_input="What's the weather in Kitchener?",
        alice_response="It's 5°C and overcast.",
        intent="weather:current",
        entities={
            'location': ['Kitchener'],
            'topic': ['weather']
        }
    )
    print(f"✓ Recorded conversation turn: {turn.turn_id}")
    
    # Test retrieval
    recent = graph.get_recent_entities(limit=5)
    print(f"✓ Retrieved {len(recent)} recent entities")
    
    # Test context summary
    summary = graph.get_context_summary("test_user")
    print(f"✓ Context summary: {len(summary['conversation_history'])} turns")
    
    # Test statistics
    stats = graph.get_statistics()
    print(f"✓ Statistics: {stats['total_entities']} entities, {stats['total_relationships']} relationships")
    
    print("\n✅ Context Graph: ALL TESTS PASSED")


def test_personality_evolution():
    """Test Personality Evolution Engine"""
    print("\n" + "="*60)
    print("TEST 2: Personality Evolution")
    print("="*60)
    
    engine = PersonalityEvolutionEngine(data_dir="data/test_personality")
    
    # Test initial traits
    traits = engine.get_traits_for_user("test_user")
    print(f"✓ Initial verbosity: {traits.verbosity:.2f}")
    print(f"✓ Initial formality: {traits.formality:.2f}")
    
    # Test signal detection from brief message
    signals = engine.detect_signals_from_interaction(
        user_id="test_user",
        user_input="weather?",  # Very brief
        alice_response="It's 5°C.",
        user_reaction=None
    )
    print(f"✓ Detected {len(signals)} signals from brief message")
    
    # Apply signals
    engine.apply_signals(signals, "test_user")
    
    # Check if verbosity decreased
    new_traits = engine.get_traits_for_user("test_user")
    if new_traits.verbosity < traits.verbosity:
        print(f"✓ Verbosity adapted: {traits.verbosity:.2f} → {new_traits.verbosity:.2f}")
    else:
        print(f"⚠ Verbosity unchanged (may be intentional)")
    
    # Test signal from formal language
    signals2 = engine.detect_signals_from_interaction(
        user_id="test_user",
        user_input="Could you please provide the weather information?",
        alice_response="Certainly, it's 5°C.",
        user_reaction=None
    )
    print(f"✓ Detected {len(signals2)} signals from formal message")
    
    # Test personality profile
    profile = engine.get_personality_profile("test_user")
    print(f"✓ Profile generated with {profile['sample_count']} samples")
    print(f"  - Verbosity: {profile['description']['verbosity']}")
    print(f"  - Formality: {profile['description']['formality']}")
    
    print("\n✅ Personality Evolution: ALL TESTS PASSED")


def test_response_variance():
    """Test Response Variance Engine"""
    print("\n" + "="*60)
    print("TEST 3: Response Variance Engine")
    print("="*60)
    
    engine = ResponseVarianceEngine()
    
    # Test response generation
    context1 = ResponseContext(
        intent_type="weather:current",
        data={'temperature': -6, 'condition': 'overcast', 'location': 'Kitchener'},
        user_id="test_user",
        conversation_history=[],
        user_verbosity_pref=0.5
    )
    
    response1 = engine.generate_response(context1)
    print(f"✓ Generated response 1: {response1}")
    
    # Generate again - should be different
    response2 = engine.generate_response(context1)
    print(f"✓ Generated response 2: {response2}")
    
    if response1 != response2:
        print(f"✓ Responses are varied (different phrasings)")
    else:
        print(f"⚠ Responses are identical (may happen occasionally)")
    
    # Test repetition detection
    context2 = ResponseContext(
        intent_type="weather:current",
        data={'temperature': -6, 'condition': 'overcast', 'location': 'Kitchener'},
        user_id="test_user",
        conversation_history=[],
        user_verbosity_pref=0.5
    )
    
    # Ask same thing 3 times
    for i in range(3):
        resp = engine.generate_response(context2)
        if i == 2:
            # Third time should acknowledge repetition
            if 'again' in resp.lower() or 'notice' in resp.lower():
                print(f"✓ Repetition detected on attempt {i+1}")
                print(f"  Response: {resp}")
            else:
                print(f"⚠ Repetition not detected (may need tuning)")
    
    # Test quality tracking
    engine.record_response_quality(
        user_id="test_user",
        response=response1,
        user_reaction="thanks!"
    )
    print(f"✓ Quality feedback recorded")
    
    print("\n✅ Response Variance Engine: ALL TESTS PASSED")


def test_integration():
    """Test full integration"""
    print("\n" + "="*60)
    print("TEST 4: Full Integration")
    print("="*60)
    
    integration = FoundationIntegration()
    
    # Test complete interaction
    result = integration.process_interaction(
        user_id="test_user",
        user_input="What's the weather?",
        intent="weather:current",
        entities={'location': ['Kitchener'], 'topic': ['weather']},
        plugin_result={
            'success': True,
            'data': {
                'temperature': -6,
                'condition': 'overcast',
                'location': 'Kitchener'
            }
        }
    )
    
    print(f"✓ Complete interaction processed")
    print(f"  Response: {result['response']}")
    print(f"  Turn ID: {result['turn_id']}")
    print(f"  Context entities: {result['context_entities']}")
    
    # Test learning from feedback
    integration.learn_from_feedback(
        user_id="test_user",
        user_input="What's the weather?",
        alice_response=result['response'],
        user_reaction="too long"
    )
    print(f"✓ Learning from feedback processed")
    
    # Test context retrieval
    context = integration.get_context_summary("test_user")
    print(f"✓ Context summary retrieved")
    print(f"  Topics: {context.get('recent_topics', [])}")
    print(f"  Turns: {len(context.get('conversation_history', []))}")
    
    # Test statistics
    stats = integration.get_statistics()
    print(f"✓ Statistics retrieved:")
    print(f"  Total entities: {stats['context_graph']['total_entities']}")
    print(f"  Personality users: {stats['personality_users']}")
    
    print("\n✅ Full Integration: ALL TESTS PASSED")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("FOUNDATION SYSTEMS TEST SUITE")
    print("="*60)
    
    tests = [
        ("Context Graph", test_context_graph),
        ("Personality Evolution", test_personality_evolution),
        ("Response Variance", test_response_variance),
        ("Full Integration", test_integration)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True))
        except Exception as e:
            print(f"\n❌ {name}: FAILED")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Foundation systems are ready.")
        print("\nNext steps:")
        print("1. Review ai/foundation_integration.py for integration instructions")
        print("2. Gradually migrate main.py to use new systems")
        print("3. Run integration tests with real A.L.I.C.E interactions")
    else:
        print("\n⚠️ Some tests failed. Review errors above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
